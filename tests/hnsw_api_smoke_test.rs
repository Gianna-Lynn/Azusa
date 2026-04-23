use azusa::{
    index::hnsw::HnswIndex,
    types::{Metadata, Metric, VectorRecord},
};

#[test]
fn search_returns_id_and_score_and_get_by_id_returns_metadata() {
    let mut index = HnswIndex::new(Metric::Cosine, 0.5);

    index.batch_insert(
        vec![
            VectorRecord {
                id: 1,
                vector: vec![1.0, 0.0],
                metadata: Metadata {
                    doc_id: Some("doc-1".to_string()),
                    chunk_id: Some("chunk-1".to_string()),
                    source: Some("unit-test".to_string()),
                    page: Some(3),
                },
            },
            VectorRecord {
                id: 2,
                vector: vec![0.0, 1.0],
                metadata: Metadata {
                    doc_id: Some("doc-2".to_string()),
                    chunk_id: Some("chunk-2".to_string()),
                    source: Some("unit-test".to_string()),
                    page: Some(7),
                },
            },
        ],
        8,
        2,
        4,
    );

    let results = index.search(&vec![1.0, 0.0], 2, 8);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 >= results[1].1);

    let metadata = index.get_by_id(1).expect("metadata should exist");
    assert_eq!(metadata.doc_id.as_deref(), Some("doc-1"));
    assert_eq!(metadata.chunk_id.as_deref(), Some("chunk-1"));
    assert_eq!(metadata.source.as_deref(), Some("unit-test"));
    assert_eq!(metadata.page, Some(3));
}
