//! 使用三篇自编短文本和预设向量演示 HNSW 公共 API，不加载模型或修改课程数据。

use azusa::{
    index::hnsw::HnswIndex,
    types::{Metadata, Metric, VectorRecord},
};

/// 运行索引、检索和按 ID 回读的小闭环；原文留在应用侧，由稳定 ID 关联。
fn main() {
    // 向量只为展示接口而手工设置，不表达模型的语义编码能力。
    let documents = [
        (
            1,
            "Azusa 的 Rust HNSW 索引保存在内存中，进程重启后需要重建。",
            vec![1.0, 0.0, 0.0],
        ),
        (
            2,
            "RRF 使用排名融合 Dense 与 BM25 的候选结果。",
            vec![0.0, 1.0, 0.0],
        ),
        (
            3,
            "Kaiji 通过工具适配器调用 Azusa 的检索与证据读取接口。",
            vec![0.0, 0.0, 1.0],
        ),
    ];
    let records = documents
        .iter()
        .map(|(id, _, vector)| VectorRecord {
            id: *id,
            vector: vector.clone(),
            metadata: Metadata {
                doc_id: Some(format!("demo-doc-{id}")),
                chunk_id: Some(format!("demo-{id}")),
                source: Some("retrieval_demo".to_string()),
                page: None,
            },
        })
        .collect();

    let mut index = HnswIndex::new(Metric::Cosine, 0.5);
    index.batch_insert(records, 8, 2, 4);
    println!(
        "已索引 {} 篇示例文本（预设向量，非模型编码）。",
        index.len()
    );
    println!("查询：Azusa 的索引重启后会保留吗？");

    let hits = index.search_knn_with_scores(&vec![1.0, 0.0, 0.0], 1, 16);
    let (id, score) = hits.first().expect("示例索引应返回一条命中");
    assert_eq!(*id, 1, "预设查询应命中内存索引说明");
    let record = index.get_record_by_id(*id).expect("命中 ID 应可回读");
    let (_, text, _) = documents
        .iter()
        .find(|(document_id, _, _)| document_id == id)
        .expect("命中 ID 应对应一篇示例原文");

    println!(
        "命中：id={}, score={:.3}, chunk_id={}",
        record.id,
        score,
        record
            .metadata
            .chunk_id
            .as_deref()
            .expect("示例应包含 chunk_id")
    );
    println!("原文：{text}");
    println!("已按 ID 取回同一条向量记录与来源信息。");
}
