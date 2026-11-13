//! 本地内存 HNSW 服务：验证向量边界，绑定不可变快照，并保留手工插入演示。
//!
//! axum 是 Rust 的 HTTP web 框架（tokio 官方出品）。
//! 以一次请求的旅程为主线，看 axum 分别在哪个环节起作用：
//!
//! 客户端 HTTP 请求
//!   │ ① 路由匹配（axum：`Router` + `.route()`）：/search + POST → search_handler
//!   │ ② 提取器解析（axum：提取器）：`Json<SearchRequest>` 解请求体、
//!   │        `State` 注入共享索引、`Path(id)` 取路径参数；
//!   │        解析失败 → `JsonRejection` → 结构化错误
//!   │ ③ handler 跑业务逻辑（与 axum 无关）：指纹校验、向量校验、HNSW 查询
//!   │        （索引算法在 azusa 库 src/index/ 里）
//!   │ ④ handler 返回 `Result<Json<..>, ApiError>`
//!   │ ⑤ 序列化（axum）：Ok → 200 + JSON；Err → 对应状态码 + 错误 JSON
//!   ▼
//! 客户端 HTTP 响应
//!
//! 另外，/index/load 的 64MB 请求体上限是 axum 的 `layer` 中间件，在请求进入 handler 之前生效。
use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use azusa::index::hnsw::HnswIndex;
use azusa::types::{Metric, VectorRecord};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, MutexGuard};

// #[tokio::main] 把 main 包进 tokio 异步运行时，函数体里才能使用 .await。
#[tokio::main]
/// 启动本机内存服务；监听地址仅从当前进程配置读取，不修改系统配置。
///
/// # Panics
/// 绑定监听地址、读取实际监听地址或运行服务失败时 panic。
async fn main() {
    // 监听地址优先读环境变量 AZUSA_BIND_ADDR；没设置就用默认的 127.0.0.1:3000。
    // 127.0.0.1 只监听本机回环地址，不对外网开放。
    let address = std::env::var("AZUSA_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".to_string());
    // 绑定 TCP 端口并进入监听；绑定失败（比如端口被占用）会直接 panic 退出。
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    // local_addr() 是实际生效的监听地址（配置 0 号端口时由系统分配），打印出来方便排查。
    println!("Azusa listening on {}", listener.local_addr().unwrap());
    // 把监听器交给 axum，用 app() 构造的路由开始处理 HTTP 请求。
    // serve 会一直运行到进程退出，正常关闭时才返回 Ok。
    axum::serve(listener, app()).await.unwrap();
}

/// 每个 Router 独享状态，测试无需启动或改写用户正在使用的服务。
///
/// # Returns
/// 带独立空索引状态和 HTTP 路由的 Router。
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
/// 在请求间共享同一索引, 访问时加锁. 克隆的是 Arc，而不是索引和向量数据。
struct AppState(Arc<Mutex<IndexState>>);

/// 将索引、维度与数据身份放在同一把锁下，避免只更新其中一部分。
struct IndexState {
    index: HnswIndex,
    dimension: Option<usize>,
    dataset_fingerprint: Option<String>,
}

impl IndexState {
    /// 创建尚未绑定数据的空索引，允许旧式插入或首次完整快照装载。
    ///
    /// # Returns
    /// 使用 Cosine 度量、尚无维度和数据指纹的索引状态。
    fn new() -> Self {
        Self {
            index: HnswIndex::new(Metric::Cosine, 0.5),
            dimension: None,
            dataset_fingerprint: None,
        }
    }

    /// 从同一次加锁读取中生成健康快照，数量和指纹不会来自不同版本。
    ///
    /// # Returns
    /// 包含状态、记录数量、维度和数据指纹的健康快照。
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
///
/// # Arguments
/// - `status`: HTTP 状态码；
/// - `code`: 稳定错误码；
/// - `message`: 公开的错误描述。
///
/// # Returns
/// 状态码与 JSON 错误体组成的 `ApiError`。
fn error(status: StatusCode, code: &'static str, message: &'static str) -> ApiError {
    (status, Json(ErrorBody { error: ErrorDetail { code, message } }))
}

/// 锁损坏时显式报告不可用，不让后续请求再次因 unwrap 而 panic。
///
/// # Arguments
/// - `state`: 共享索引状态。
///
/// # Returns
/// 成功时返回锁守卫，锁损坏时返回结构化 503 错误。
fn lock_state(state: &AppState) -> Result<MutexGuard<'_, IndexState>, ApiError> {
    state.0.lock().map_err(|_| error(
        StatusCode::SERVICE_UNAVAILABLE, "index_unavailable", "索引不可用，请重启服务",
    ))
}

/// 类型、JSON、非有限数字和错误字段由提取器拒绝，并统一成结构化错误。
///
/// # Arguments
/// - `payload`: JSON 提取结果或提取失败原因。
///
/// # Returns
/// 成功时返回解析后的值，失败时返回结构化错误。
fn request_json<T>(payload: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    payload.map(|Json(value)| value).map_err(|rejection| error(
        rejection.status(), "invalid_json", "请求必须符合 JSON 字段、类型与数值契约",
    ))
}

/// 在接触索引前校验维度和数值范围，避免 f32 距离计算溢出及断言破坏索引。
///
/// # Arguments
/// - `vector`: 待校验的非空向量；
/// - `dimension`: 已知的索引维度；存在时必须与向量长度一致。
///
/// # Returns
/// 校验通过返回 `Ok(())`，否则返回维度或数值错误。
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
///
/// # Arguments
/// - `state`: 共享索引状态。
///
/// # Returns
/// 成功时返回 JSON 健康快照，失败时返回结构化错误。
async fn health_handler(State(state): State<AppState>) -> Result<Json<HealthResponse>, ApiError> {
    Ok(Json(lock_state(&state)?.health()))
}

/// 旧演示接口只用于未绑定快照的索引；不得追加修改已发布的数据身份。
///
/// # Arguments
/// - `state`: 共享索引状态；
/// - `payload`: 待插入记录的 JSON 提取结果。
///
/// # Returns
/// 成功时返回 201 状态码，解析、校验或插入冲突时返回结构化错误。
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

/// 加载一批带数据指纹的完整快照，并返回当前索引的健康状态。
///
/// 函数会先校验快照指纹、记录 ID 和向量；首次加载时构建并发布新索引(Rust实现的HNSW图结构)。
/// 如果服务已加载相同快照，则直接复用；如果快照不同，则拒绝覆盖已有索引。
async fn load_handler(
    State(state): State<AppState>,
    payload: Result<Json<LoadRequest>, JsonRejection>,
) -> Result<Json<HealthResponse>, ApiError> {
    // 1.解析请求并校验快照指纹、记录 ID 和向量维度，确保整个批次合法。
    let request = request_json(payload)?;  // 问号?的作用: 成功就继续，失败就提前返回错误。
    if request.dataset_fingerprint.len() != 64
        || !request.dataset_fingerprint.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))  // 判断c是不是接下来的其中之一: 0 1 2 ... 9 a b c d e f
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
    // 2.已有索引只能复用完全相同的快照，避免覆盖或混入不同数据。

    //判断服务当前是否已经装载过快照，或者已经留下快照身份信息。
    if !current.index.is_empty() || current.dataset_fingerprint.is_some() {
        // 只有指纹、记录数量和每条记录都相同，才把这次请求视为重复装载。
        let matches = current.dataset_fingerprint.as_deref() == Some(&request.dataset_fingerprint)
            // 比较快照指纹：as_deref() 将 Option<&String> 转成 Option<&str>，
            // 再与请求中的指纹比较，确认两批数据具有相同身份。
            && current.index.len() == request.records.len()
            // 比较记录数量，避免仅凭指纹或部分记录就错误复用已有索引。
            && request.records.iter().all(|record| current.index.get_record_by_id(record.id)
                // all() 要求每一条请求记录都满足后面的条件；只要一条不满足，结果就是 false。
                .is_some_and(|stored| {
                    // is_some_and() 要求索引中确实存在相同 ID 的记录，
                    // 并且该记录序列化后的完整内容与请求记录相同。
                    serde_json::to_value(stored).ok() == serde_json::to_value(record).ok()
                }));
        if matches {
            // 相同快照无需重复构建索引，直接返回当前健康状态。
            return Ok(Json(current.health()));
        }
        // 已有索引但数据不一致时拒绝覆盖，要求重启服务后再装载新快照。
        return Err(error(StatusCode::CONFLICT, "dataset_mismatch", "索引数据身份冲突，请重启服务"));
    }
    // 3.服务当前没有加载过快照, 所以第一次加载快照.
    // 所有校验通过后一次性构建并发布新索引，失败时不会留下半成品。
    let mut replacement = IndexState::new();    // 创建一个新的空索引状态
    replacement.dimension = dimension;                      // 设置维度为请求中记录的维度
    replacement.index.batch_insert(request.records, 8, 2, 4);   // 把请求中的所有记录批量插入 HNSW 索引
    replacement.dataset_fingerprint = Some(request.dataset_fingerprint);    // 保存这批数据的指纹
    *current = replacement;                                 // 用新索引替换当前索引状态，完成装载
    Ok(Json(current.health()))
}

/// 在同一次持锁期间检查身份并执行查询，防止检查和检索使用不同数据。
///
/// 处理流程：解析 JSON → 拒绝零预算 → 加锁 → 核对数据指纹 → 校验 query 向量
/// → HNSW 检索 → 包装成 SearchResponse 返回。任何一步失败都通过 ApiError
/// 变成带稳定错误码的 HTTP 响应，而不是 panic。
///
/// 非法维度、非有限数值和零预算在调用 HNSW 之前拒绝；ef_search 至少覆盖 k。
async fn search_handler(
    State(state): State<AppState>,
    payload: Result<Json<SearchRequest>, JsonRejection>,
) -> Result<Json<SearchResponse>, ApiError> {
    // 第一步：解包请求。JSON 不合法时 request_json 已转成结构化错误，? 直接返回 4xx。
    let request = request_json(payload)?;

    // 第二步：拒绝零预算。k 是必须返回的结果数，ef_search 是搜索预算，都不能为 0。
    if request.k == 0 || request.ef_search == Some(0) {
        return Err(error(StatusCode::BAD_REQUEST, "invalid_search_budget", "k 和 ef_search 必须是正整数"));
    }

    // 第三步：加锁并核对身份。指纹检查、向量校验和检索必须在同一把锁内完成，
    // 防止中途被 /index/load 换掉数据，出现"检查是一批、检索是另一批"的错配。
    let mut current = lock_state(&state)?;
    // 客户端拿旧指纹来查新数据，说明它手上的 embedding 列表已经过期，直接 409。
    if request.dataset_fingerprint != current.dataset_fingerprint {
        return Err(error(StatusCode::CONFLICT, "dataset_mismatch", "查询数据身份与服务索引不一致"));
    }

    // 第四步：校验 query 向量。维度一致、数值有限、平方和不能大到让 f32 溢出。
    validate_vector(&request.vector, current.dimension)?;

    // 第五步：确定搜索预算。没传用默认 16；不能小于 k，否则召回不满 k 个。
    let ef_search = request.ef_search.unwrap_or(16).max(request.k);

    // 第六步：HNSW 检索。返回 (id, 分数) 列表，转成 JSON 里的 {"id": ..., "score": ...}。
    // id 是装载时的插入下标，Python 侧拿它当列表位置用。
    let results = current.index
        .search_knn_with_scores(&request.vector, request.k, ef_search)
        .into_iter().map(|(id, score)| SearchHit { id, score }).collect();

    // 第七步：包装响应。200 + {"results": [...]}。
    Ok(Json(SearchResponse { results }))
}

/// 按精确数字 ID 返回原始向量记录；不存在返回 404，不做模糊匹配。
///
/// # Arguments
/// - `state`: 共享索引状态；
/// - `id`: 路径中的记录 ID。
///
/// # Returns
/// 成功时返回 JSON 向量记录，记录不存在或锁不可用时返回结构化错误。
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
