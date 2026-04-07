mod datasets;
use datasets::*;
use toy_vector_db::{
    index::hnsw::HnswIndex,
    types::{Vector, Id, Metric},
    distance::similarity
};
const TEST_METRIC: Metric = Metric::L2;


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
fn build_index_v1(nodes_data: Vec<(u64, Vec<f32>, usize)>, ef_construction: usize, m: usize, m_max: usize) -> HnswIndex {
    let mut index = HnswIndex::new();
    for (id, data, level) in nodes_data.iter() {
        index.insert_v1_fixed_level(*id, data.clone(), ef_construction, m, m_max, level);
    }
    index
}

// debug: 这个地方的expected_list最好完全独立于build_index_v1.
//        也就是说, 不要从index中取结果, 而是直接从HnswTestCase中取结果.
//     
fn donotuse_ideal_answer_generator(index: &HnswIndex, query: &Vector, top_k: usize) -> Vec<Id>{

    // 按暴力 similarity 排序后的理想 id 列表
    // - 遍历 case 的 nodes_data
    // - 算 query 和每个点的 similarity
    // - 排序
    // - 取前 k
    // - 你必须给 tie 一个稳定规则，不然实验结果会漂

    assert!(top_k > 0, "in function ideal_answer_generator, expected top_k > 0");

    let mut nodes_vec = index.get_nodes().clone();
    nodes_vec.sort_by(|node1, node2|{
        let score1 = similarity(query, node1.get_data(), TEST_METRIC);
        let score2 = similarity(query, node2.get_data(), TEST_METRIC);
        match score2.partial_cmp(&score1).unwrap(){
            std::cmp::Ordering::Equal => node1.get_id().cmp(node2.get_id()),
            other => other,
        }
    });
    let mut id_vec: Vec<Id> = Vec::new();
    for( idx, node) in nodes_vec.iter().enumerate(){
        if idx == top_k {break;}
        id_vec.push(*node.get_id());
    }
    return id_vec;
}


fn ideal_answer_generator(case: &HnswTestCase, query: &Vector, top_k: usize) -> Vec<Id>{
    // 按暴力 similarity 排序后的理想 id 列表
    // - 遍历 case 的 nodes_data
    // - 算 query 和每个点的 similarity
    // - 排序
    // - 取前 k
    // - 你必须给 tie 一个稳定规则，不然实验结果会漂
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

// 简化 selected_cases() 的宏：自动从函数名生成case_name字符串
macro_rules! add_test_cases {
    ($($case_fn:ident),+ $(,)?) => {{
        let cases = vec![$($case_fn()),+];
        let case_names = vec![$(stringify!($case_fn).to_string()),+];
        (cases, case_names)
    }};
}

fn selected_cases() -> (Vec<datasets::HnswTestCase>, Vec<String>){
    add_test_cases!(
        // =============== 第一轮添加的case ===============
        // unique_ranking_case,
        // multilevel_case,
        // high_dimension_case,
        // negative_coordinates_case,
        
        // =============== 第二轮添加的case ===============
        // two_node_case,
        // three_node_case,
        // dense_2d_case,
        // flat_2d_case,
        // greedy_chain_case,
        // search_layer_case,
        // greedy_stop_case,
        // single_node_case,
        // beginning_is_the_best_case,
        // identical_vectors_case,
        // duplicate_distance_case,
        // collinear_points_case,
        // clustered_distribution_case,
        // near_zero_distance_case,
        // k_larger_than_dataset_case,
        // extreme_values_case,
        // tightly_packed_case,
        // sparse_vectors_case,
        // bridge_trap_case,

        // =============== 第三轮添加的case ===============
        // weak_bridge_two_clusters_case,
        // crowded_near_neighbors_case,
        // single_bridge_chain_case,
        // highdim_bridge_case,
        // 第三轮的结论: 
        // 当前 harder cases 下，ef_search=4 不足以稳定返回完整 top-k
        // ef_search=8 可恢复正确结果
        // ef_search=16 暂未表现出明显高于 8 的收益
        // 当前样例下，m=2 与 m=4 尚未表现出显著差异

        // =============== 第四轮：中等规模数据集（Agent生成）===============
        // 目的：逼出 m 参数的信号差异
        // 特点：50-100个点，结构可解释，m=2 vs m=4/8 应有明显差异
        // sparse_four_clusters_case,
        // dense_single_cluster_case,
        // pyramid_hierarchy_case,
        // layered_strips_case,
        // radial_star_case,

        // =============== 第五轮：多团串联模板（手搓）===============
        // 目的：逼出 m 参数的信号差异
        multi_cluster_chain_case,
    )
}

#[test] #[ignore] 
fn experiment_param_sweep(){
    // 固定一些参数
    let ef_construction = 10;
    let ms = [2, 4];
    // let ef_searches = [4, 8, 16];
    let ef_searches = [16];

    // 然后两层循环：
    // 外层扫 m
    for m in ms{
        // 中间自己算 m_max
        let m_max = 2 * m;
        // 内层扫 ef_search
        for ef_search in ef_searches{
            // improve: 在这里补一个参数标题打印.
            println!("\n========== ef_construction={}, m={}, m_max={}, ef_search={} ==========", ef_construction, m, 2*m, ef_search);

            // debug: 统计量的初始化应出现在这里.对于每一组参数(m, ef_search),对应一些统计量.

            let mut total = 0;      //这组参数 (m, ef_search) 一共跑了多少个 case。
            let mut top1_hit= 0;    //第一个结果命中的 case 个数
            let mut exact_hit= 0; //有多少个 case 完全匹配
            
            
            
            
            let (cases, cases_names) = selected_cases(); //这一步是解构
            
            // debug: 不能直接对 tuple 迭代，要先解构再 zip 配对.
            // for (case,case_names) in selected_cases() {

            // debug: zip:是拉链的意思, 把两个序列一一配对.
            for (case, case_name) in cases.into_iter().zip(cases_names.into_iter()) {

                total += 1; //这组参数 (m, ef_search) 一共跑了多少个 case。
                // Task 6：在每组参数里跑所有 selected cases
                // 1
                // 初始化统计量：
                // debug: 统计量的初始化并不在这里.对于每一组参数(m, ef_search),对应一些统计量.


                // 2
                // 对每个 case：
                // 算 expected
                // 用当前参数建图
                // 用当前 ef_search 调 search_knn_v1_from_entry_for_experiment(...)
                // 比较结果

                // debug: 如果先let index, 后let expected_list: 前者利用值传递拿走了case.nodes_data, 后者想要整体借用case, rust不允许"部分字段已经被搬走后还想借用整个结构体"
                //        解决方法是交换这两行, 先let expected_list借用, 后拿走case.nodes_data
                //        解决方法更正: 第二步拿走case.nodes_data还不行,改成了clone()
                let expected_list = ideal_answer_generator(&case, &case.query, case.k);
                
                let index = build_index_v1(case.nodes_data.clone(), ef_construction, m, m_max);
                // debug: 这个地方的expected_list最好完全独立于build_index_v1.
                // let expected_list = ideal_answer_generator(&index, &case.query, case.k);
                let search_result_list = index.search_knn_v1_from_entry_for_experiment(case.entry_id, &case.query,  case.k, ef_search);
                let search_result_list_len = search_result_list.len();
                // 3
                // 定义两个判断：
                let mut top1_ok: bool = false;    //这个 case 的第一个结果是否命中
                let mut exact_ok: bool = false;   //这个 case 的整个结果列表是否完全一致

                // debug: 改成宽容的逻辑，不用 assert! 打断整趟实验
                //        空列表时，top1_ok = false；长度不等时，exact_ok = false；继续打印
                if !expected_list.is_empty() && !search_result_list.is_empty() {
                    if expected_list[0] == search_result_list[0] {
                        top1_ok = true; 
                        top1_hit += 1;
                    }
                }

                // 4
                // 更新统计量
                // debug: rust 允许使用 == 直接对比两个list, 返回一个布尔值, 不需要一位一位对比.
                // debug: exact_hit 和 exact_ok 一开始因为变量作用不清晰所以完全会错了意.
                // let len = expected_list.len();
                //  for i in 0 ..= len-1 {
                //    if expected_list[i] == search_result[i]{
                //         exact_hit += 1;
                //     }
                // }
                // if exact_hit == len {exact_ok = true;}

                if expected_list == search_result_list{
                    exact_ok = true;
                    exact_hit += 1;
                }


                // improve: 添加一个新的试验指标 recall@k = 预测结果中与理想top-k重叠的数量 / k
                use std::collections::HashSet;
                let expected_set: HashSet<Id> = expected_list.iter().cloned().collect();
                let overlap_count = search_result_list.iter()
                    .filter(|id| expected_set.contains(id))
                    .count();
                // .iter() 是对列表的迭代器，.filter() 是过滤器，保留那些在 expected_set 中的 id，最后 .count() 计算重叠的数量。
                
                let recall_at_k_score = overlap_count as f32 / case.k as f32;

                // 5
                // 打印：case 名字, top1 是否对,exact 是否对, expected 什么, got 什么
                println!("  [{}]", case_name);
                println!("    top1_ok={}, exact_ok={}, search_result_list_len={},  recall@{}={}", top1_ok, exact_ok, search_result_list_len, case.k, recall_at_k_score);
                println!("    expected={:?}", expected_list);
                println!("    got={:?}", search_result_list);
                
            }
        
        // Task 7：打印每组参数的 summary, 一共 <m x ef_search> 这么多组.
        // m, m_max, ef_construction, ef_search, top1_hit / total, exact_hit / total
        println!("---");
        println!("SUMMARY: ef_construction={}, m={}, m_max={}, ef_search={}", ef_construction, m, m_max, ef_search);
        println!("  top1_hit / total = [{}/{}]", top1_hit, total);
        println!("  exact_hit / total = [{}/{}]", exact_hit, total);
        println!("===\n");
        }
    }
}