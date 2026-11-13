//! Isolated benchmark HTTP worker: the same Azusa cosine metric for exact and HNSW.
//! Reads normalized little-endian f32 matrices; no production protocol changes.
use axum::{extract::State, routing::{get, post}, Json, Router};
use azusa::{distance::metric_similarity, index::hnsw::HnswIndex, types::Metric};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, io::Write, sync::{Arc, Mutex}, time::Instant};

#[derive(Deserialize)]
/// Freeze vector file locations and graph budgets for one independent graph.
struct Config {
    base: String, dim: usize, count: usize, m: usize, ef_construction: usize,
    /// Optional JSON permutation of original row IDs; IDs never become insertion positions.
    #[serde(default)]
    order: Option<String>,
}

/// Reject duplicate, missing or out-of-range IDs before building a shuffled graph.
fn validate_order(order: &[usize], count: usize) -> Result<(), &'static str> {
    let unique: std::collections::HashSet<_> = order.iter().copied().collect();
    if order.len() != count || unique.len() != count || order.iter().any(|&id| id >= count) {
        return Err("order must be a complete permutation of original row IDs");
    }
    Ok(())
}

/// Own identical base vectors and a single graph; requests are deliberately sequential.
struct Data { base: Vec<Vec<f32>>, index: HnswIndex, build_ms: f64 }

#[derive(Deserialize)]
/// One precomputed query vector and explicit result/search budgets.
struct Query { vector: Vec<f32>, k: usize, ef: usize, exact: bool }

/// Read a complete f32 matrix and reject malformed or nonfinite inputs.
fn read_vectors(path: &str, dim: usize, count: usize) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if dim == 0 || count == 0 || bytes.len() != dim.checked_mul(count).and_then(|n| n.checked_mul(4)).ok_or("size overflow")? {
        return Err("invalid matrix size".into());
    }
    let mut result = Vec::with_capacity(count);
    for row in bytes.chunks_exact(dim * 4) {
        let vector: Vec<f32> = row.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        let norm: f32 = vector.iter().map(|x| x*x).sum();
        if vector.iter().any(|x| !x.is_finite()) || (norm - 1.0).abs() > 0.001 {
            return Err("vectors must be finite and normalized".into());
        }
        result.push(vector);
    }
    Ok(result)
}

/// Exact cosine top-k with deterministic ID tie-break, using Azusa's distance code.
fn exact(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
    let mut scores: Vec<_> = base.iter().enumerate().map(|(i,v)| (i as u64, metric_similarity(v, query, Metric::Cosine))).collect();
    let compare = |a: &(u64,f32), b: &(u64,f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    scores.select_nth_unstable_by(k-1, compare);
    scores.truncate(k);
    scores.sort_unstable_by(compare);
    scores.into_iter().map(|x| x.0).collect()
}

/// Execute one query, reporting native computation separately from client HTTP timing.
async fn search(State(state): State<Arc<Mutex<Data>>>, Json(query): Json<Query>) -> Result<Json<Value>, axum::http::StatusCode> {
    let mut data = state.lock().map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    if query.k == 0 || query.k > data.base.len() || query.ef < query.k
        || query.vector.len() != data.base[0].len() || query.vector.iter().any(|x| !x.is_finite()) {
        return Err(axum::http::StatusCode::BAD_REQUEST);
    }
    let started = Instant::now();
    let ids = if query.exact { exact(&data.base, &query.vector, query.k) }
        else { data.index.search_knn(&query.vector, query.k, query.ef) };
    Ok(Json(json!({"ids":ids, "native_ms":started.elapsed().as_secs_f64()*1000.0})))
}

/// Expose count and build duration to prove that the intended full graph is ready.
async fn health(State(state): State<Arc<Mutex<Data>>>) -> Json<Value> {
    let data = state.lock().unwrap();
    Json(json!({"count":data.base.len(), "dimension":data.base[0].len(), "build_ms":data.build_ms}))
}

#[tokio::main(flavor = "current_thread")]
/// Build once, log incremental progress, and bind loopback port zero for this run only.
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config: Config = serde_json::from_slice(&fs::read(std::env::args().nth(1).ok_or("config required")?)?)?;
    if config.m < 2 || config.ef_construction < config.m { return Err("invalid graph budgets".into()); }
    let base = read_vectors(&config.base, config.dim, config.count)?;
    let order: Vec<usize> = if let Some(path) = &config.order {
        serde_json::from_slice(&fs::read(path)?)?
    } else { (0..base.len()).collect() };
    validate_order(&order, base.len())?;
    let mut index = HnswIndex::new(Metric::Cosine, 0.5);
    let started = Instant::now();
    for (i, &id) in order.iter().enumerate() {
        index.insert(id as u64, base[id].clone(), config.ef_construction, config.m, config.m*2);
        if (i+1) % 1000 == 0 {
            println!("BUILD {} / {} elapsed_s={:.2}", i+1, base.len(), started.elapsed().as_secs_f64());
            std::io::stdout().flush()?;
        }
    }
    let build_ms = started.elapsed().as_secs_f64()*1000.0;
    let state = Arc::new(Mutex::new(Data {base,index,build_ms}));
    let app = Router::new().route("/health",get(health)).route("/search",post(search)).with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    println!("READY {} build_ms={}", listener.local_addr()?, build_ms);
    std::io::stdout().flush()?;
    axum::serve(listener,app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    /// Permuting insertion order must preserve external IDs and reject malformed permutations.
    fn shuffled_ids_and_invalid_orders() {
        assert!(validate_order(&[2,0,1],3).is_ok());
        for invalid in [vec![0,0,2],vec![0,1],vec![0,1,3]] {
            assert!(validate_order(&invalid,3).is_err());
        }
        let base=[vec![1.,0.],vec![0.,1.],vec![-1.,0.]];
        let mut graph=HnswIndex::new(Metric::Cosine,0.5);
        for id in [2,0,1] {graph.insert(id as u64,base[id].clone(),20,4,8);}
        assert_eq!(graph.search_knn(&vec![1.,0.],3,20),vec![0,1,2]);
    }
    #[test]
    /// Verify cosine ordering and exact tie-breaking before full-corpus execution.
    fn cosine_tie_order() {
        let base=vec![vec![1.,0.],vec![0.,1.],vec![1.,0.]];
        assert_eq!(exact(&base,&[1.,0.],3),vec![0,2,1]);
        let mut graph=HnswIndex::new(Metric::Cosine,0.5);
        for (i,v) in base.iter().enumerate() {graph.insert(i as u64,v.clone(),20,4,8);}
        let ids=graph.search_knn(&vec![1.,0.],3,20);
        assert_eq!(ids.len(),3);
        assert_eq!(ids.into_iter().collect::<std::collections::HashSet<_>>(),[0,1,2].into_iter().collect());
    }
}
