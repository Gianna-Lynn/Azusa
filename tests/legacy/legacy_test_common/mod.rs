// Shared helpers for legacy handcrafted tests.
// 这些 helper 只服务于旧版手搓数据集测试，不属于正式测试基础设施。

use azusa::{
    distance::similarity,
    index::hnsw::HnswIndex,
    types::{Id, Metric, Vector},
};

use super::legacy_handcrafted_cases::HnswTestCase;

const TEST_METRIC: Metric = Metric::Cosine;

// 参数说明:
// - id: 新节点的唯一标识符
// - data: 新节点的向量数据
// - ef_construction: 在构建过程中搜索层时使用的参数，控制搜索的宽度, 也就是论文中的ef_construction
//                    至于所谓的ef_search不会出现在这里, 是用于搜索函数的.
// - max_neighbors_to_select: 每层为新节点挑选的最大邻居数.
// - max_neighbors_to_keep: 每层允许保留的最大邻居数. 在插入新节点时, 如果某个节点的邻居数超过这个上限, 就需要进行邻居选择和替换.

/// id: u64, data: Vec<f32>, lvl: usize
/// 构图 helper, 改进自 hnsw_test_v1.rs, 建图的时候引入 METRIC
/// 人为指定每个节点所在的层次.
pub fn build_index(
    nodes_data: Vec<(u64, Vec<f32>, usize)>,
    ef_construction: usize,
    max_neighbors_to_select: usize,
    max_neighbors_to_keep: usize,
) -> HnswIndex {
    let mut index = HnswIndex::new_with_metric(TEST_METRIC);
    for (id, data, level) in nodes_data.iter() {
        index.insert_with_fixed_level_for_legacy(
            *id,
            data.clone(),
            ef_construction,
            max_neighbors_to_select,
            max_neighbors_to_keep,
            level,
        );
    }
    index
}

// debug: 这个地方的expected_list最好完全独立于build_index.
//        也就是说, 不要从index中取结果, 而是直接从HnswTestCase中取结果.
pub fn ideal_answer_generator(case: &HnswTestCase, query: &Vector, top_k: usize) -> Vec<Id> {
    let mut nodes_data_vec = case.nodes_data.clone();
    nodes_data_vec.sort_by(|node1, node2| {
        let score1 = similarity(query, &node1.1, TEST_METRIC);
        let score2 = similarity(query, &node2.1, TEST_METRIC);
        match score2.partial_cmp(&score1).unwrap() {
            std::cmp::Ordering::Equal => node1.0.cmp(&node2.0),
            other => other,
        }
    });

    let mut id_vec: Vec<Id> = Vec::new();
    for (idx, node) in nodes_data_vec.iter().enumerate() {
        if idx == top_k {
            break;
        }
        id_vec.push(node.0);
    }
    id_vec
}
