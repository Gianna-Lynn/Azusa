//! HNSW 公共 API 的烟雾测试。
use azusa::{
    index::hnsw::HnswIndex,
    types::{Metadata, Metric, VectorRecord},
};

#[test]
// 验证一个空索引在查询时不会报错，而是直接返回空结果。
fn search_on_empty_index_returns_empty() {
    let mut index = HnswIndex::new(Metric::Cosine, 0.5);

    let results = index.search_knn(&vec![1.0, 0.0], 5, 8);
    assert!(results.is_empty(), "empty index should return empty results");
}

#[test]
// 验证当 k 大于数据集规模时，搜索结果会自然收敛到全部节点。
fn search_with_k_larger_than_dataset_returns_all_nodes() {
    let mut index = HnswIndex::new(Metric::Cosine, 0.5);

    index.batch_insert(
        vec![
            VectorRecord {
                id: 1,
                vector: vec![1.0, 0.0],
                metadata: Metadata::default(),
            },
            VectorRecord {
                id: 2,
                vector: vec![0.0, 1.0],
                metadata: Metadata::default(),
            },
            VectorRecord {
                id: 3,
                vector: vec![0.5, 0.5],
                metadata: Metadata::default(),
            },
        ],
        8,
        2,
        4,
    );

    let results = index.search_knn(&vec![1.0, 0.0], 10, 8);
    assert_eq!(results.len(), 3, "k should be clipped to dataset size");
}

#[test]
#[should_panic(expected = "dimension mismatch")]
// 验证空 query 会因为维度不匹配而触发 panic，避免静默返回错误结果。
fn search_with_empty_query_panics_dimension_mismatch() {
    let mut index = HnswIndex::new(Metric::Cosine, 0.5);
    index.batch_insert(
        vec![VectorRecord {
            id: 1,
            vector: vec![1.0, 0.0],
            metadata: Metadata::default(),
        }],
        8,
        2,
        4,
    );

    let _ = index.search_knn(&vec![], 1, 8);
}

#[test]
#[should_panic(expected = "duplicate id is not allowed")]
// 验证 batch_insert 会拒绝重复 id，确保索引内部主键唯一。
fn batch_insert_rejects_duplicate_ids() {
    let mut index = HnswIndex::new(Metric::Cosine, 0.5);

    index.batch_insert(
        vec![
            VectorRecord {
                id: 1,
                vector: vec![1.0, 0.0],
                metadata: Metadata::default(),
            },
            VectorRecord {
                id: 1,
                vector: vec![0.0, 1.0],
                metadata: Metadata::default(),
            },
        ],
        8,
        2,
        4,
    );
}

#[test]
// 验证搜索结果会返回 id/score，同时 metadata 也能按 id 正确取回。
fn search_returns_id_and_score_and_get_metadata_by_id_returns_metadata() {
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

    let results = index.search_knn_with_scores(&vec![1.0, 0.0], 2, 8);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 >= results[1].1);

    let metadata = index.get_metadata_by_id(1).expect("metadata should exist");
    assert_eq!(metadata.doc_id.as_deref(), Some("doc-1"));
    assert_eq!(metadata.chunk_id.as_deref(), Some("chunk-1"));
    assert_eq!(metadata.source.as_deref(), Some("unit-test"));
    assert_eq!(metadata.page, Some(3));
}
