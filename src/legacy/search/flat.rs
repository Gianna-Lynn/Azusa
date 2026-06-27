//! legacy 线性扫描搜索基线，主要用于 smoke test 和对照实验。
use crate::distance::cosine_similarity;
use crate::legacy::storage::inmemory::InMemoryVectorStore;
use crate::types::{SearchResult, Vector};

/// 执行 legacy 线性扫描搜索。
///
/// 早期版本的线性扫描搜索实现，主要用于测试和基线对比。
/// 它属于 legacy 路径，不是当前 HNSW 主线搜索实现。
pub fn search_flat(store: &InMemoryVectorStore, query: &Vector, k: usize) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = store
        .records()
        .iter()
        .map(|rec| {
            let score = cosine_similarity(query, &rec.vector);
            SearchResult { id: rec.id, score }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(k);
    results
}
