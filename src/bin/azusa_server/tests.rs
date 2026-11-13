//! 路由级回归：非法请求不破坏索引，快照不可混用，候选预算不被截断。
use super::*;
use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

/// 使用真实 Router 和 JSON 提取器，避免直接调用 handler 掩盖协议错误。
async fn call(app: &Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string())).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

/// 生成二维向量和稳定 chunk 标记，便于检查 ID 与正文来源的对应关系。
fn record(id: u64) -> Value {
    json!({"id": id, "vector": [1.0, id as f32 / 100.0], "metadata": {"chunk_id": format!("chunk-{id}")}})
}

/// 生成连续 ID 的整批装载请求；不同指纹表示不同客户端快照身份。
fn snapshot(fingerprint: &str, count: u64) -> Value {
    json!({"dataset_fingerprint": fingerprint, "records": (0..count).map(record).collect::<Vec<_>>()})
}

#[tokio::test]
/// 旧式单条插入、重复 ID 拒绝、搜索和回读仍组成正常闭环。
async fn legacy_insert_search_and_record_still_work() {
    let app = app();
    assert_eq!(call(&app, "POST", "/insert", record(7)).await.0, StatusCode::CREATED);
    assert_eq!(call(&app, "POST", "/insert", record(7)).await.0, StatusCode::CONFLICT);
    let (status, result) = call(&app, "POST", "/search", json!({"vector":[1.0,0.07],"k":1})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["results"][0]["id"], 7);
    assert_eq!(call(&app, "GET", "/record/7", Value::Null).await.1["metadata"]["chunk_id"], "chunk-7");
    assert_eq!(call(&app, "GET", "/record/99", Value::Null).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
/// 每个非法查询后再次验证健康检查和正常搜索，防止只断言错误码。
async fn invalid_requests_leave_index_healthy_and_unchanged() {
    let app = app();
    assert_eq!(call(&app, "POST", "/insert", record(0)).await.0, StatusCode::CREATED);
    for body in [
        json!({"vector":[],"k":1}),
        json!({"vector":[1.0],"k":1}),
        json!({"vector":[1.0,0.0],"k":0}),
        json!({"vector":[1.0,0.0],"k":1,"ef_search":0}),
        json!({"vector":[1.0,0.0],"k":true}),
        json!({"vector":[1.0,0.0],"k":-1}),
        json!({"vector":[1e30,0.0],"k":1}),
        json!({"vector":[1e300,0.0],"k":1}),
    ] {
        let (status, error) = call(&app, "POST", "/search", body).await;
        assert!(status.is_client_error(), "{error}");
        assert!(error["error"]["code"].is_string());
        let (status, health) = call(&app, "GET", "/health", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health["record_count"], 1);
        assert_eq!(call(&app, "POST", "/search", json!({"vector":[1.0,0.0],"k":1})).await.0, StatusCode::OK);
    }
    for vector in [json!([]), json!([1.0]), json!([1e30,0.0])] {
        let (status, _) = call(&app, "POST", "/insert", json!({"id":1,"vector":vector})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
    assert_eq!(call(&app, "GET", "/health", Value::Null).await.1["record_count"], 1);
    assert_eq!(call(&app, "POST", "/insert", record(1)).await.0, StatusCode::CREATED);
}

#[tokio::test]
/// 批次中任意记录非法时，数量、维度和指纹均不得发生部分更新。
async fn batch_validation_is_atomic_and_allows_retry_after_failure() {
    let app = app();
    let fingerprint = "a".repeat(64);
    for records in [
        json!([record(0), {"id":1,"vector":[1.0]}]),
        json!([record(0), record(0)]),
        json!([record(1)]),
        json!([{"id":0,"vector":[]}]),
        json!([{"id":0,"vector":[1e30,0.0]}]),
    ] {
        assert_eq!(call(&app, "POST", "/index/load", json!({"dataset_fingerprint":fingerprint,"records":records})).await.0, StatusCode::BAD_REQUEST);
        let health = call(&app, "GET", "/health", Value::Null).await.1;
        assert_eq!(health["record_count"], 0);
        assert!(health["dimension"].is_null());
        assert!(health["dataset_fingerprint"].is_null());
    }
    assert_eq!(call(&app, "POST", "/index/load", snapshot(&fingerprint, 2)).await.0, StatusCode::OK);
}

#[tokio::test]
/// 验证相同快照幂等，不同快照、伪装的重复装载和后续追加均被拒绝。
async fn snapshot_identity_is_required_and_cannot_be_replaced_or_appended() {
    let app = app();
    let first = "a".repeat(64);
    let second = "b".repeat(64);
    assert_eq!(call(&app, "POST", "/index/load", snapshot(&first, 2)).await.0, StatusCode::OK);
    assert_eq!(call(&app, "POST", "/index/load", snapshot(&first, 2)).await.0, StatusCode::OK);
    for body in [snapshot(&second, 2), snapshot(&first, 3)] {
        assert_eq!(call(&app, "POST", "/index/load", body).await.0, StatusCode::CONFLICT);
    }
    let mut altered = snapshot(&first, 2);
    altered["records"][0]["vector"] = json!([0.0,1.0]);
    assert_eq!(call(&app, "POST", "/index/load", altered).await.0, StatusCode::CONFLICT);
    assert_eq!(call(&app, "POST", "/insert", record(2)).await.0, StatusCode::CONFLICT);
    for fingerprint in [Value::Null, json!(second)] {
        assert_eq!(call(&app, "POST", "/search", json!({"vector":[1.0,0.0],"k":1,"dataset_fingerprint":fingerprint})).await.0, StatusCode::CONFLICT);
    }
    let (status, result) = call(&app, "POST", "/search", json!({"vector":[1.0,0.0],"k":1,"dataset_fingerprint":first})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["results"][0]["id"], 0);
    assert_eq!(call(&app, "GET", "/health", Value::Null).await.1["record_count"], 2);
}

#[tokio::test]
/// 空快照也有身份；手工非空索引不能被新客户端冒认为已知快照。
async fn empty_snapshot_is_bound_and_manual_index_is_not_adopted() {
    let app = app();
    let first = "a".repeat(64);
    assert_eq!(call(&app, "POST", "/index/load", snapshot(&first, 0)).await.0, StatusCode::OK);
    let (status, result) = call(&app, "POST", "/search", json!({"vector":[1.0],"k":1,"dataset_fingerprint":first})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(result["results"], json!([]));
    assert_eq!(call(&app, "POST", "/insert", record(0)).await.0, StatusCode::CONFLICT);
    let manual = super::app();
    assert_eq!(call(&manual, "POST", "/insert", record(0)).await.0, StatusCode::CREATED);
    assert_eq!(call(&manual, "POST", "/index/load", snapshot(&first, 1)).await.0, StatusCode::CONFLICT);
}

#[tokio::test]
/// 使用超过旧固定容量的 k，确认服务按 k 扩大 ef_search，而不是静默截断到 16。
async fn candidate_budget_covers_k_above_sixteen() {
    let app = app();
    let fingerprint = "a".repeat(64);
    assert_eq!(call(&app, "POST", "/index/load", snapshot(&fingerprint, 25)).await.0, StatusCode::OK);
    for ef_search in [Value::Null, json!(1), json!(16), json!(40)] {
        let (status, result) = call(&app, "POST", "/search", json!({"vector":[1.0,0.0],"k":20,"ef_search":ef_search,"dataset_fingerprint":fingerprint})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result["results"].as_array().unwrap().len(), 20);
    }
}

#[test]
/// 直接覆盖 JSON 无法表达的 NaN/Infinity，确认数值校验本身也拒绝它们。
fn non_finite_values_are_rejected_before_index_access() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(validate_vector(&[value, 0.0], None).is_err());
    }
}
