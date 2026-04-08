//use crate::sift1m_recall::load_sift_bytes;

// 文件	                    用途
// sift_base.fvecs	        数据库向量（100万个128维向量）— 你要建索引的数据
// sift_query.fvecs	        查询向量（10000个128维向量）— 用来测试搜索效果
// sift_learn.fvecs	        学习向量（100000个128维向量）— 可选的训练集
// sift_groundtruth.ivecs	标准答案（查询的真值） — 每个查询的top-k正确答案
// 用 sift_base 建立HNSW索引
// 用 sift_query 去查询
// 拿结果和 sift_groundtruth 比较，计算recall@k（召回率）

mod sift1m_recall;

use crate::sift1m_recall::*;
use toy_vector_db::{
    distance::similarity,
    index::hnsw::{HnswIndex, HnswNode},
    types::Metric,
};
const TEST_METRIC: Metric = Metric::L2;



#[test]
pub fn sift1m_recall_smoke_test(){
    let limit: usize = 10;
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function sift1m_recall_smoke_test, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes,limit);
    
    // println!("base_vecs is {:?}",base_vecs);
    // println!("base_vecs.len() is {}", base_vecs.len());  //==limit
    // println!("base_vecs[0].len() is {}", base_vecs[0].len()); //==128
    
    // Vec<Vec<f32>> 没有实现 Display trait，只实现了 Debug trait。 
    // {} 不可行
    // {:?} — Debug 打印（单行）
    // {:#?} — Debug 打印（漂亮格式，换行）

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function sift1m_recall_smoke_test, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes,limit);

    // println!("query_vecs is {:?}",query_vecs);
    // println!("query_vecs.len() is {}", query_vecs.len());  //==limit
    // println!("query_vecs[0].len() is {}", query_vecs[0].len()); //==128

    let query = query_vecs[0].clone();
    let k = 5;
    let top_k =  exact_top_k(&base_vecs, &query, k, TEST_METRIC);
    let index = build_index(&base_vecs);
    let search_result = index.search_knn_v1(&query, k, 10);

    //打印向量本体
    //println!("query is {:?}",query);
    //println!("base_vecs is {:?}",base_vecs);
    println!("top-{} results:{:?}", k, top_k);
    println!("hnsw-{} results:{:?}", k, search_result);

}


pub fn exact_top_k(base_vecs: &Vec<Vec<f32>>, query: &Vec<f32>, k: usize, metric: Metric)->Vec<(u64, f32)>{
    let mut results = Vec::new();
    for(idx, base_vec) in base_vecs.iter().enumerate(){
        let score = similarity(&query, &base_vec , metric);
        results.push((idx as u64, score)); //(idx, score)
    }
    results.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    let top_k = results.iter().take(k).copied().collect::<Vec<_>>();
    //.iter()把results变成一个 "一个个看元素"的迭代器, .take(k)只取前k个, 
    //.collect::<Vec<_>>()把前面取得的这些元素重新收集成一个Vec,具体的元素让Rust自己推断.

    // .iter().take(k) 产生的是 Iterator<&(u64, f32)>（引用），
    // .copied() 会把引用转成实际的值，然后才能正确推断为 Vec<(u64, f32)>。
    // (u64, f32) 是 Copy 类型（两个字段都是 Copy）.
    //               Copy 是 Rust 的一个特征（trait），表示这个类型可以通过简单的按位复制来完整复制，无需特殊处理。静态的数据类型基本都是Copy
    // 对于 Copy 类型，.copied() 明确表达"复制这个 Copy 值"，更高效且符合语义
    // .clone() 虽然也能用，但对 Copy 类型来说是多余的
    top_k
}
// TODO: 以下建图代码由CODEX完成
fn build_index(base_vecs: &[Vec<f32>]) -> HnswIndex {
    let mut index = HnswIndex::new_with_metric(TEST_METRIC);

    for (id, base_vec) in base_vecs.iter().enumerate() {
        index.insert(HnswNode::new(id as u64, base_vec.clone(), 0));
    }

    index
}

#[test]
fn test_hnsw_in_sift1m(){
    let limit: usize = 10;       // limit控制了我们从sift1m的文件里, 只读前多少条向量.
    let ef_search = 7;    // ef_search控制了我们在hnsw里搜索的时候, 最多看多少个节点. 这个参数越大, 搜索效果越好, 但是搜索速度越慢.

    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function test_hnsw_in_sift1m, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes,limit);
    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function test_hnsw_in_sift1m, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes,limit);
    let query = query_vecs[0].clone();
    let index =  build_index(&base_vecs);
    let brute_force = exact_top_k(&base_vecs, &query, limit, TEST_METRIC);
        //brute_force: Vec<(Id, score)> 
    let search_result_ids = index.search_knn_v1(&query, limit, ef_search);
        //search_result_ids: Vec<Id>
    let brute_force_ids: Vec<_> = brute_force.iter().map(|(id,_)|   *id).collect();
    assert_eq!(brute_force_ids[0], search_result_ids[0]);
    println!("{:#?}", search_result_ids);
    println!("{:#?}", brute_force_ids);
}
