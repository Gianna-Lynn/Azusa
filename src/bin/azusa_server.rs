use std::sync::{Arc, Mutex};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Router, Json};
use axum::extract::{State, Path};
use azusa::index::hnsw::HnswIndex;
use azusa::types::{Metric, VectorRecord};
use serde::{Serialize, Deserialize};


#[tokio::main]
async fn main() {
    // 启动服务器
    let index = HnswIndex::new(Metric::Cosine, 0.5);

    let state = AppState{
        index: Arc::new(Mutex::new(index)),
    };

    // 创建 axum router(一张表)
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/insert", post(insert_handler))
        .route("/search",post(search_handler))
        .route("/record/:id", get(record_handler))
        .with_state(state);

    // 启动监听
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

}

//服务共享状态
#[derive(Clone)]
struct AppState{
    index: Arc<Mutex<HnswIndex>>,
}

// SearchRequest是客户端发给 /search 的JSON, 所以需要能够从JSON反序列化.
// 对应于POST的body
#[derive(Deserialize)]
struct SearchRequest{
    vector: Vec<f32>,
    k: usize,
    ef_search: Option<usize>,
}

// SearchHit和SearchResponse是服务器返回给客户端的JSON, 所以需要能够序列化为JSON.

///返回一条命中记录(id, score)
#[derive(Serialize)]
struct SearchHit{
    id: u64,
    score: f32,
}

// 对应接口返回值
///返回一堆命中记录 Vec<(id, score)>
#[derive(Serialize)]
struct SearchResponse{
    results: Vec<SearchHit>,
}

// HealthResponse也返回JSON
// 对应接口返回值
/// record_count是HNSW表中存在的节点数
#[derive(Serialize)]
struct HealthResponse{
    status: String,
    record_count: usize,
}

// 定义handler
// handler就是 "某个URL被访问时实际执行的函数"
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse>{
                      //State(state): State<AppState>的意思是: 从axum注入的全局状态中拿到AppState.
    let index = state.index.lock().unwrap();

    Json(
        HealthResponse { 
            status: "ok".to_string(), 
            record_count: index.len(), 
        }
    )
}

// insert_handler:
// 从 Json<VectorRecord> 拿请求体
// lock index
// 检查 id 是否重复
// insert_record
// 返回成功
async fn insert_handler(
    State(state): State<AppState>,
    Json(record): Json<VectorRecord>,
)->StatusCode{

    let mut index = state.index.lock().unwrap();

    if index.get_record_by_id(record.id).is_some(){
        return StatusCode::CONFLICT;
    }

    index.insert_record(
        record,
        8,
        2,
        4,
    );

    StatusCode::CREATED
}



// search_handler:
// 从 Json<SearchRequest> 拿请求体
// lock index
// 调用 search_knn_with_scores
// 包装成 SearchResponse
async fn search_handler(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>
) -> Json<SearchResponse>{
    let mut index = state.index.lock().unwrap();
    let res = index.search_knn_with_scores(
        &request.vector,
        request.k,
        request.ef_search.unwrap_or(16), //search_knn_with_scores 返回 Vec<(Id, f32)>
    ).iter().map(|(idx, score)|{
            SearchHit{
                id: *idx,
                score: *score,
            }
        }
    ).collect();
    Json(
        SearchResponse {
            results: res,
        }
    )
}

// record_handler:
// 从 Path<u64> 拿 URL 里的 id
// lock index
// get_record_by_id
// 找到就返回 Json(record)
// 找不到就 404
async fn record_handler(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<VectorRecord>, StatusCode>{
    let index = state.index.lock().unwrap();


    if let Some(record) = index.get_record_by_id(id){
        Ok(
            Json(record.clone())
        )
    }   
    else{
        Err(
          StatusCode::NOT_FOUND // 404 Not found
        ) 
    }

}
