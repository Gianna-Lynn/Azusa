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
use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::sift1m_recall::*;
use toy_vector_db::{
    distance::similarity, index::hnsw::{HnswIndex}, types::Metric
};
const TEST_METRIC: Metric = Metric::L2;


// base_limit   控制了我们从sift1m的文件里, 只读前多少条向量.
// k            用来表达 "取得前几个最相似的结果."
// ef_search    控制了我们在hnsw里搜索的时候, 最多看多少个节点. 这个参数越大, 搜索效果越好, 但是搜索速度越慢.
// build_index 的参数也统一放在这里，方便你在顶部一键调参。

type Sift1mSweepParams = (usize, usize, usize, usize);

const SMOKE_TEST_PARAMS: &[Sift1mSweepParams] = &[
    (10, 5, 10, 5), // (limit, k, ef_search, query_limit)
];
const HNSW_IN_SIFT1M_PARAMS: &[Sift1mSweepParams] = &[
    // 想做 sweep 的时候，继续往这里加 tuple 就行。
    // (base_limit, k, ef_search, query_limit)
    // (200, 5, 40, 5), 
    // (200, 5, 50, 5), 
    // (200, 5, 80, 5), 

    // (500, 5, 80, 5),     
    // (500, 5, 120, 5),
    // (500, 5, 160, 5),
    // (500, 5, 200, 5), 

    // (1000, 5, 20, 5),
    // (1000, 5, 40, 5),
    // (1000, 5, 60, 5),
    // (1000, 5, 80, 5),

    (1000, 5, 40, 5),
];


#[derive(Debug, Clone, Copy)]
struct HnswBuildParams {
    ef_construction: usize,
    m: usize,
    m_max: usize,
}

#[derive(Debug, Clone)]
struct Sift1mEfSearchSweepGroup {
    base_limit: usize,
    k: usize,
    ef_search: &'static [usize],
    query_limit: usize,
}




const HNSW_IN_SIFT1M_PARAMS_GROUP: &[Sift1mEfSearchSweepGroup] = &[
    Sift1mEfSearchSweepGroup {
        base_limit: 10000,
        k: 5,
        ef_search: &[60, 80],
        query_limit: 10,
    },
];

const DEFAULT_HNSW_BUILD_PARAMS: HnswBuildParams = HnswBuildParams {
    ef_construction: 80,
    m: 4,
    m_max: 8,
};


/**************************  测试主体  **************************/
#[test]
fn test_hnsw_in_sift1m(){
    for case in HNSW_IN_SIFT1M_PARAMS_GROUP {
        // for &ef_search_num in case.ef_search{
        //     run_hnsw_in_sift1m_case(case.base_limit, case.k, ef_search_num, case.query_limit)
        // }
        run_hnsw_in_sift1m_case(case.base_limit, case.k, case.ef_search, case.query_limit)
    }
}

fn run_hnsw_in_sift1m_case(
    base_limit: usize,
    k: usize,
    ef_search: &'static [usize],
    query_limit: usize,) {
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function run_hnsw_in_sift1m_case, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes,base_limit);
    let build_start = Instant::now();
    let index =  build_index(&base_vecs, DEFAULT_HNSW_BUILD_PARAMS);
    println!("base_limit={} build_index took: {:?}", base_limit, build_start.elapsed());

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function run_hnsw_in_sift1m_case, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes,query_limit);

    for ef_search_num in ef_search{
        let mut cnt = 0;
        let mut total_recall_at_k:f32 = 0.0;
        let mut total_exact_at_k = 0;
        println!("--- base_limit={}, ef_search={} ---", base_limit, ef_search_num);
        for query_vec in query_vecs.iter(){
            
            cnt = cnt + 1;

            let query = query_vec.clone();

            let brute_force = exact_top_k(&base_vecs, &query, k, TEST_METRIC);
            let search_start = Instant::now();
            let search_result_ids = index.search_knn_v1(&query, k, *ef_search_num);
            println!("  query[{}/{}] ef_search={} search_knn_v1 took: {:?}", cnt, query_limit, ef_search_num, search_start.elapsed());
            let brute_force_ids: Vec<_> = brute_force.iter().map(|(id,_)|   *id).collect();

            let brute_set: HashSet<_> = brute_force_ids.iter().copied().collect();
            let overlap: Vec<_> = search_result_ids
                .iter()
                .copied()
                .filter(|id| brute_set.contains(id))
                .collect();
            let overlap_num = overlap.len();

            let recall_at_k = overlap_num as f32 / k as f32;
            let exact_at_k = brute_force_ids.len() >= k
                                && search_result_ids.len() >= k
                                && &brute_force_ids[..k] == &search_result_ids[..k];
            total_recall_at_k += recall_at_k;
            if exact_at_k{
                total_exact_at_k += 1;
            }
        }
        // 关于top-k中涉及到的两个指标:
        // recall@k : 前k个命中了多少个. 不要求顺序一致, 只要求id部分或全部重合. 可以是一个计数值或者百分比分数.
        // exact@k  : 前k个是否完全一致. 要求顺序一致, 要求id全部重合. 是一个布尔值.
        println!("---");
        println!("SUMMARY: base_limit={}, k={}, ef_search={}, query_limit={}", base_limit, k, ef_search_num, query_limit);
        println!("  recall@{} = [{}/{}]", k , total_recall_at_k, cnt);
        println!("  exact@{}  = [{}/{}]", k , total_exact_at_k, cnt);
        println!("---\n");
    }
}



#[test]
pub fn sift1m_recall_smoke_test(){
    for &(limit, k, ef_search, query_limit) in SMOKE_TEST_PARAMS {
        run_sift1m_recall_case("smoke", limit, k, ef_search, query_limit);
    }
}

fn run_sift1m_recall_case(
    case_name: &str,
    limit: usize,
    k: usize,
    ef_search: usize,
    query_limit: usize,) {
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function run_sift1m_recall_case, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes,limit);
    
    // println!("base_vecs is {:?}",base_vecs);
    // println!("base_vecs.len() is {}", base_vecs.len());  //==limit
    // println!("base_vecs[0].len() is {}", base_vecs[0].len()); //==128
    
    // Vec<Vec<f32>> 没有实现 Display trait，只实现了 Debug trait。 
    // {} 不可行
    // {:?} — Debug 打印（单行）
    // {:#?} — Debug 打印（漂亮格式，换行）

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function run_sift1m_recall_case, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes,query_limit);

    // println!("query_vecs is {:?}",query_vecs);
    // println!("query_vecs.len() is {}", query_vecs.len());  //==limit
    // println!("query_vecs[0].len() is {}", query_vecs[0].len()); //==128

    let query = query_vecs[0].clone();
    let top_k =  exact_top_k(&base_vecs, &query, k, TEST_METRIC);
    let build_start = Instant::now();
    let index = build_index(&base_vecs, DEFAULT_HNSW_BUILD_PARAMS);
    println!("build_index took: {:?}", build_start.elapsed());
    let search_start = Instant::now();
    let search_result = index.search_knn_v1(&query, k, ef_search);
    println!("search_knn_v1 took: {:?}", search_start.elapsed());

    //打印向量本体
    //println!("query is {:?}",query);
    //println!("base_vecs is {:?}",base_vecs);
    println!("top-{} results:{:?}", k, top_k);
    println!("hnsw-{} results:{:?}", k, search_result);
    println!("case={} done", case_name);

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

fn build_index(base_vecs: &[Vec<f32>], params: HnswBuildParams) -> HnswIndex {
    let mut index = HnswIndex::new_with_metric(TEST_METRIC);
    println!(
        "insert_v1 params: ef_construction={}, m={}, m_max={}",
        params.ef_construction,
        params.m,
        params.m_max
    );

    for (id, base_vec) in base_vecs.iter().enumerate() {
        // 认为每一条向量的数组下标序号就是它的id.
        index.insert_v1(
            id as u64,
            base_vec.clone(),
            params.ef_construction,
            params.m,
            params.m_max,
        );
    }
    index
}




#[test]
fn test_hnsw_in_sift1m_groundtruth(){
    for &(base_limit, k, ef_search, query_limit) in HNSW_IN_SIFT1M_PARAMS {
        run_hnsw_in_sift1m_case_groundtruth(base_limit, k, ef_search, query_limit);
    }
}

fn run_hnsw_in_sift1m_case_groundtruth(
    base_limit: usize,
    k: usize,
    ef_search: usize,
    query_limit: usize, ) {
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function run_hnsw_in_sift1m_case, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes,base_limit);
    let index =  build_index(&base_vecs, DEFAULT_HNSW_BUILD_PARAMS);

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function run_hnsw_in_sift1m_case, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes,query_limit);

    let groundtruth_bytes = load_sift_bytes("data//sift1m//sift//sift_groundtruth.ivecs")
        .expect("in function run_hnsw_in_sift1m_case, load query_bytes error");    
    let groundtruth_vecs = read_uvecs(&groundtruth_bytes,query_limit);
    // println!("{:#?}", groundtruth_vecs);
    


    let mut valid_query_cnt = 0;
    let mut total_recall_at_k:f32 = 0.0;
    let mut total_exact_at_k = 0;
    let mut total_search_time = Duration::default();
    for(query_idx,  query_vec) in query_vecs.iter().enumerate(){
        // query没有单独的ID. query的序号就是它的隐式ID
        // groundtruth 也按照同样的序号对齐.
        // 在循环当中, 我们认为这个序号就是query_idx.

        // 给一条 query 的 groundtruth，取得前 k 个整数 id的方法.
        let groundtruth_vec = &groundtruth_vecs[query_idx];   //使用引用&: 只需要读取, 不需要修改, 不要从大Vec中移出元素
        // println!("groundtruth_vec : {:#?}", groundtruth_vec);
        let groundtruth_vec_top_k : Vec<_>= groundtruth_vec
            .iter()
            .copied()
            .filter(|&id| id < base_limit as u32)
            .take(k)
            .collect();
        
        println!("groundtruth_vec_top_k : {:#?}", groundtruth_vec_top_k);
        //只有当 groundtruth_vec_top_k.len() == k 时，才把这条 query 纳入统计
        if groundtruth_vec_top_k.len() != k{
            println!("Skip this query: groundtruth_vec_top_k.len() != k");
            continue;
        }     
        // valid_query_cnt 表示“真正纳入 recall/exact 统计的 query 数”   
        valid_query_cnt += 1;
        let search_start = Instant::now();
        let search_result_ids = index.search_knn_v1(&query_vec, k, ef_search);
        println!("query[{}] search took: {:?}", query_idx, search_start.elapsed());
        total_search_time += search_start.elapsed();
        let search_result_ids_u32: Vec<u32> = search_result_ids.iter().map(|&id| id as u32).collect();

        let groundtruth_set_top_k: HashSet<_> = groundtruth_vec_top_k.iter().copied().collect();
        let overlap: Vec<_> = search_result_ids_u32
            .iter()
            .copied()
            .filter(|id| groundtruth_set_top_k.contains(id))
            .collect();
        let overlap_num = overlap.len();

        let recall_at_k = overlap_num as f32 / k as f32;
        let exact_at_k = groundtruth_vec_top_k.len() >= k
                            && search_result_ids.len() >= k
                            && &groundtruth_vec_top_k[..k] == &search_result_ids_u32[..k];
        
        total_recall_at_k += recall_at_k;
        if exact_at_k{
            total_exact_at_k += 1;
        }
    }


    // 关于top-k中涉及到的两个指标:
    // recall@k : 前k个命中了多少个. 不要求顺序一致, 只要求id部分或全部重合. 可以是一个计数值或者百分比分数.
    // exact@k  : 前k个是否完全一致. 要求顺序一致, 要求id全部重合. 是一个布尔值.
    
    println!("---");
    println!("SUMMARY: base_limit={}, k={}, ef_search={}, query_limit={}, valid_query_cnt={}", base_limit, k,ef_search, query_limit, valid_query_cnt);
    
    if valid_query_cnt == 0{
        println!("no valid queries under current base_limit");
    }
    else {
        println!("  valid_query_cnt = [{}]", valid_query_cnt);
        println!("  total_search_time = {:?}", total_search_time);
        println!("  avg_search_time   = {:?}", total_search_time / valid_query_cnt as u32);
        println!("  recall@{} = [{}/{}]", k , total_recall_at_k, valid_query_cnt);
        println!("  average recall@{} = [{}]", k, total_recall_at_k / valid_query_cnt as f32);
        println!("  exact@{}  = [{}/{}]", k , total_exact_at_k, valid_query_cnt);
        println!("---\n");
    }
}
