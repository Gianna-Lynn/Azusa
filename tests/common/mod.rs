use toy_vector_db::{
    index::hnsw::HnswIndex,
    types::{Vector, Id, Metric},
    distance::similarity
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
/// v1 版本的构图 helper, 复制自hnsw_test_v1.rs
/// 人为指定每个节点所在的层次.
pub fn build_index_v1(nodes_data: Vec<(u64, Vec<f32>, usize)>, ef_construction: usize, m: usize, m_max: usize) -> HnswIndex {
    let mut index = HnswIndex::new_with_metric(TEST_METRIC);
    for (id, data, level) in nodes_data.iter() {
        index.insert_v1_fixed_level(*id, data.clone(), ef_construction, m, m_max, level);
    }
    index
}

// debug: 这个地方的expected_list最好完全独立于build_index_v1.
//        也就是说, 不要从index中取结果, 而是直接从HnswTestCase中取结果.
pub fn ideal_answer_generator(case: &HnswTestCase, query: &Vector, top_k: usize) -> Vec<Id>{
    // 按暴力 similarity 排序后的理想 id 列表
    // - 遍历 case 的 nodes_data
    // - 算 query 和每个点的 similarity
    // - 排序
    // - 取前 k
    // - 你必须给 tie 一个稳定规则，不然实验结果会漂

    // let mut nodes_vec = index.get_nodes().clone(); 已经弃用的版本, 从index中取数据.现在已经改成nodes_data_vec
    let mut nodes_data_vec = case.nodes_data.clone();
            // 数据示例
            // nodes_data: vec![
            //     (1, vec![1.0, 2.0, 3.0], 3),
            //     (2, vec![2.0, 3.0, 4.0], 4),
            // ],
    // note: 访问元组的第一个元素,需要用.1, 第二个元素.2
    nodes_data_vec.sort_by(|node1, node2|{
        let score1 = similarity(query, &node1.1, TEST_METRIC);
        let score2 = similarity(query, &node2.1, TEST_METRIC);
        match score2.partial_cmp(&score1).unwrap(){
            std::cmp::Ordering::Equal => node1.0.cmp(&node2.0),
            other => other,
        }
    });
    let mut id_vec: Vec<Id> = Vec::new();
    for( idx, node) in nodes_data_vec.iter().enumerate(){
        if idx == top_k {break;}
        id_vec.push(node.0);
    }
    return id_vec;
}

