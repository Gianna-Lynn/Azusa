/********************************************** Search **********************************************/

use crate::distance::cosine_similarity;
use crate::storage::inmemory::InMemoryVectorStore;
use crate::types::{SearchResult, Vector};

// 这个文件是早期版本的线性扫描搜索实现，主要用于测试和基线对比。正式版本的搜索实现会放在 hnsw_impl.rs 中。
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
