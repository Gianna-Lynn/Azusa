//! 本地内存 HNSW 服务：验证向量边界，绑定不可变快照，并保留手工插入演示。
use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use azusa::index::hnsw::HnswIndex;
use azusa::types::{Metric, VectorRecord};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, MutexGuard};

#[tokio::main]
/// 启动本机内存服务；监听地址仅从当前进程配置读取，不修改系统配置。
async fn main() {
    let address = std::env::var("AZUSA_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".to_string());
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    println!("Azusa listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app()).await.unwrap();
}

/// 每个 Router 独享状态，测试无需启动或改写用户正在使用的服务。
fn app() -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/insert", post(insert_handler))
        .route("/index/load", post(load_handler).layer(DefaultBodyLimit::max(64 * 1024 * 1024)))
        .route("/search", post(search_handler))
        .route("/record/:id", get(record_handler))
        .with_state(AppState(Arc::new(Mutex::new(IndexState::new()))))
}

#[derive(Clone)]
/// 在请求间共享同一索引；克隆的是 Arc，而不是索引和向量数据。
struct AppState(Arc<Mutex<IndexState>>);

/// 将索引、维度与数据身份放在同一把锁下，避免只更新其中一部分。
struct IndexState {
    index: HnswIndex,
    dimension: Option<usize>,
    dataset_fingerprint: Option<String>,
}

impl IndexState {
    /// 创建尚未绑定数据的空索引，允许旧式插入或首次完整快照装载。
    fn new() -> Self {
        Self {
            index: HnswIndex::new(Metric::Cosine, 0.5),
            dimension: None,
            dataset_fingerprint: None,
        }
    }

    /// 从同一次加锁读取中生成健康快照，数量和指纹不会来自不同版本。
    fn health(&self) -> HealthResponse {
        HealthResponse {
            status: "ok",
            record_count: self.index.len(),
            dimension: self.dimension,
            dataset_fingerprint: self.dataset_fingerprint.clone(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
/// 向量查询参数；指纹必须与当前索引一致，预算在 handler 中校验。
struct SearchRequest {
    vector: Vec<f32>,
    k: usize,
    ef_search: Option<usize>,
    dataset_fingerprint: Option<String>,
}

/// 指纹由可信本地客户端计算，和完整快照一起装载；不是认证或真实性证明。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadRequest {
    dataset_fingerprint: String,
    records: Vec<VectorRecord>,
}

#[derive(Serialize)]
/// 一条近邻命中，score 越大表示 Cosine 相似度越高。
struct SearchHit {
    id: u64,
    score: f32,
}

#[derive(Serialize)]
/// 保留原有 results 外壳，方便旧演示调用方读取。
struct SearchResponse {
    results: Vec<SearchHit>,
}

#[derive(Serialize)]
/// 当前索引的运行状态；未绑定快照或尚无维度时对应字段为 null。
struct HealthResponse {
    status: &'static str,
    record_count: usize,
    dimension: Option<usize>,
    dataset_fingerprint: Option<String>,
}

#[derive(Serialize)]
/// HTTP 错误的统一外壳，不将内部堆栈发送给调用方。
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
/// 稳定错误码供程序分类，中文 message 用于本地调试。
struct ErrorDetail {
    code: &'static str,
    message: &'static str,
}

type ApiError = (StatusCode, Json<ErrorBody>);

/// 将状态码和公开错误描述组装成 Axum 可直接返回的 JSON 响应。
fn error(status: StatusCode, code: &'static str, message: &'static str) -> ApiError {
    (status, Json(ErrorBody { error: ErrorDetail { code, message } }))
}

/// 锁损坏时显式报告不可用，不让后续请求再次因 unwrap 而 panic。
fn lock_state(state: &AppState) -> Result<MutexGuard<'_, IndexState>, ApiError> {
    state.0.lock().map_err(|_| error(
        StatusCode::SERVICE_UNAVAILABLE, "index_unavailable", "索引不可用，请重启服务",
    ))
}

/// 类型、JSON、非有限数字和错误字段由提取器拒绝，并统一成结构化错误。
fn request_json<T>(payload: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    payload.map(|Json(value)| value).map_err(|rejection| error(
        rejection.status(), "invalid_json", "请求必须符合 JSON 字段、类型与数值契约",
    ))
}

/// 在接触索引前校验维度和数值范围，避免 f32 距离计算溢出及断言破坏索引。
fn validate_vector(vector: &[f32], dimension: Option<usize>) -> Result<(), ApiError> {
    if vector.is_empty() || dimension.is_some_and(|dim| vector.len() != dim) {
        return Err(error(StatusCode::BAD_REQUEST, "invalid_dimension", "向量必须非空且与索引维度一致"));
    }
    let squared_norm: f64 = vector.iter().map(|&x| f64::from(x).powi(2)).sum();
    if vector.iter().any(|x| !x.is_finite()) || squared_norm > f64::from(f32::MAX) / 4.0 {
        return Err(error(StatusCode::BAD_REQUEST, "invalid_vector", "向量必须有限且不能导致 f32 距离计算溢出"));
    }
    Ok(())
}

/// 只读取运行状态，不加载数据；锁损坏时沿用结构化 503 错误。
async fn health_handler(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    Ok(Json(lock_state(&state)?.health()))
}

/// 旧演示接口只用于未绑定快照的索引；不得追加修改已发布的数据身份。
async fn insert_handler(
    State(state): State<AppState>,
    payload: Result<Json<VectorRecord>, JsonRejection>,
) -> Result<StatusCode, ApiError> {
    let record = request_json(payload)?;
    let mut current = lock_state(&state)?;
    if current.dataset_fingerprint.is_some() {
        return Err(error(StatusCode::CONFLICT, "snapshot_immutable", "快照索引不可追加，请重启后装载新数据"));
    }
    validate_vector(&record.vector, current.dimension)?;
    if current.index.get_record_by_id(record.id).is_some() {
        return Err(error(StatusCode::CONFLICT, "duplicate_id", "记录 ID 已存在"));
    }
    current.dimension = Some(record.vector.len());
    current.index.insert_record(record, 8, 2, 4);
    Ok(StatusCode::CREATED)
}

/// 先验证整个批次，再构建并发布；失败不留下半个索引，也不覆盖已有索引。
async fn load_handler(
    State(state): State<AppState>,
    payload: Result<Json<LoadRequest>, JsonRejection>,
) -> Result<Json<HealthResponse>, ApiError> {
    let request = request_json(payload)?;
    if request.dataset_fingerprint.len() != 64
        || !request.dataset_fingerprint.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(error(StatusCode::BAD_REQUEST, "invalid_fingerprint", "指纹必须是小写 SHA-256 十六进制字符串"));
    }
    let dimension = request.records.first().map(|r| r.vector.len());
    for (id, record) in request.records.iter().enumerate() {
        if record.id != id as u64 {
            return Err(error(StatusCode::BAD_REQUEST, "invalid_record_id", "快照 ID 必须按顺序从零连续编号"));
        }
        validate_vector(&record.vector, dimension)?;
    }
    let mut current = lock_state(&state)?;
    if !current.index.is_empty() || current.dataset_fingerprint.is_some() {
        // 重复装载仅允许相同身份和相同实际记录；并发相同装载可以安全复用。
        let matches = current.dataset_fingerprint.as_deref() == Some(&request.dataset_fingerprint)
            && current.index.len() == request.records.len()
            && request.records.iter().all(|record| current.index.get_record_by_id(record.id)
                .is_some_and(|stored| serde_json::to_value(stored).ok() == serde_json::to_value(record).ok()));
        if matches {
            return Ok(Json(current.health()));
        }
        return Err(error(StatusCode::CONFLICT, "dataset_mismatch", "索引数据身份冲突，请重启服务"));
    }
    let mut replacement = IndexState::new();
    replacement.dimension = dimension;
    replacement.index.batch_insert(request.records, 8, 2, 4);
    replacement.dataset_fingerprint = Some(request.dataset_fingerprint);
    *current = replacement;
    Ok(Json(current.health()))
}

/// 在同一次持锁期间检查身份并执行查询，防止检查和检索使用不同数据。
///
/// 非法维度、非有限数值和零预算在调用 HNSW 之前拒绝；ef 至少覆盖 k。
async fn search_handler(
    State(state): State<AppState>,
    payload: Result<Json<SearchRequest>, JsonRejection>,
) -> Result<Json<SearchResponse>, ApiError> {
    let request = request_json(payload)?;
    if request.k == 0 || request.ef_search == Some(0) {
        return Err(error(StatusCode::BAD_REQUEST, "invalid_search_budget", "k 和 ef_search 必须是正整数"));
    }
    let mut current = lock_state(&state)?;
    if request.dataset_fingerprint != current.dataset_fingerprint {
        return Err(error(StatusCode::CONFLICT, "dataset_mismatch", "查询数据身份与服务索引不一致"));
    }
    validate_vector(&request.vector, current.dimension)?;
    let ef_search = request.ef_search.unwrap_or(16).max(request.k);
    let results = current.index
        .search_knn_with_scores(&request.vector, request.k, ef_search)
        .into_iter().map(|(id, score)| SearchHit { id, score }).collect();
    Ok(Json(SearchResponse { results }))
}

/// 按精确数字 ID 返回原始向量记录；不存在返回 404，不做模糊匹配。
async fn record_handler(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<VectorRecord>, ApiError> {
    let current = lock_state(&state)?;
    current.index.get_record_by_id(id).cloned().map(Json).ok_or_else(||
        error(StatusCode::NOT_FOUND, "record_not_found", "记录不存在"))
}

#[cfg(test)]
#[path = "azusa_server/tests.rs"]
mod tests;
