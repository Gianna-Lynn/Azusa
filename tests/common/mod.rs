use toy_vector_db::{
    distance::similarity,
    index::hnsw::HnswIndex,
    types::{Id, Metric, Vector},
};

use super::datasets::HnswTestCase;

const TEST_METRIC: Metric = Metric::Cosine;

// 参数说明:
// - id: 新节点的唯一标识符
// - data: 新节点的向量数据
// - ef_construction: 在构建过程中搜索层时使用的参数，控制搜索的宽度, 也就是论文中的ef_construction
//                    至于所谓的ef_search不会出现在这里, 是用于搜索函数的.
// - m: 每层连接的最大邻居数. 在插入新节点时, 每层最多连接m个邻居.
// - m_max: 每层允许的最大邻居数. 在插入新节点时, 如果某个节点的邻居数超过m_max, 就需要进行邻居选择和替换.

/// id: u64, data: Vec<f32>, lvl: usize
/// 构图 helper, 改进自 hnsw_test_v1.rs, 建图的时候引入 METRIC
/// 人为指定每个节点所在的层次.
pub fn build_index(
    nodes_data: Vec<(u64, Vec<f32>, usize)>,
    ef_construction: usize,
    m: usize,
    m_max: usize,
) -> HnswIndex {
    let mut index = HnswIndex::new_with_metric(TEST_METRIC);
    for (id, data, level) in nodes_data.iter() {
        index.insert_fixed_level(*id, data.clone(), ef_construction, m, m_max, level);
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
