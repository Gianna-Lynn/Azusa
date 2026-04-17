/********************************************** Search **********************************************/

use crate::distance::cosine_similarity;
use crate::storage::inmemory::InMemoryVectorStore;
use crate::types::{SearchResult, Vector};

pub fn search_flat(
    store: &InMemoryVectorStore,
    query: &Vector,
    k: usize,
) -> Vec<SearchResult> {
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
