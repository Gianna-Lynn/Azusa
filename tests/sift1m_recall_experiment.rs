// SIFT1M recall experiments and smoke checks.
// 这份文件更接近小规模 benchmark / 实验脚本，不是纯粹的单元测试。

// 文件	                    用途
// sift_base.fvecs	        数据库向量（100万个128维向量）— 你要建索引的数据
// sift_query.fvecs	        查询向量（10000个128维向量）— 用来测试搜索效果
// sift_learn.fvecs	        学习向量（100000个128维向量）— 可选的训练集
// sift_groundtruth.ivecs	标准答案（查询的真值） — 每个查询的top-k正确答案
// 用 sift_base 建立HNSW索引
// 用 sift_query 去查询
// 拿结果和 sift_groundtruth 比较，计算recall@k（召回率）

#[path = "sift1m_test_utils.rs"]
mod sift1m_test_utils;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use crate::sift1m_test_utils::*;
use azusa::{distance::similarity, index::hnsw::HnswIndex, types::Metric};
const TEST_METRIC: Metric = Metric::L2;

// base_limit   控制了我们从sift1m的文件里, 只读前多少条向量.
// k            用来表达 "取得前几个最相似的结果."
// ef_search    控制了我们在hnsw里搜索的时候, 最多看多少个节点. 这个参数越大, 搜索效果越好, 但是搜索速度越慢.
// query_limit  控制了我们从sift1m的文件里, 只读前多少条查询向量. 这个参数越大, 实验结果越稳定, 但是测试时间越长.

/**************************  参数配置  **************************/
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

const HNSW_IN_SIFT1M_PARAMS_GROUP: &[Sift1mEfSearchSweepGroup] = &[Sift1mEfSearchSweepGroup {
    base_limit: 100000,
    k: 5,
    ef_search: &[60, 80, 100, 150, 200, 300],
    query_limit: 100,
}];

const DEFAULT_HNSW_BUILD_PARAMS: HnswBuildParams = HnswBuildParams {
    ef_construction: 80,
    m: 4,
    m_max: 8,
};

/**************************  测试主体  **************************/
#[test]
fn test_hnsw_in_sift1m() {
    run_sift1m_experiments(HNSW_IN_SIFT1M_PARAMS_GROUP);
}

fn run_sift1m_experiments(cases: &[Sift1mEfSearchSweepGroup]) {
    for case in cases {
        run_sift1m_experiment_case(case);
    }
}

fn run_sift1m_experiment_case(case: &Sift1mEfSearchSweepGroup) {
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function run_sift1m_experiment_case, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes, case.base_limit);

    let build_start = Instant::now();
    let mut index = build_index(&base_vecs, DEFAULT_HNSW_BUILD_PARAMS);
    println!(
        "base_limit={} build_index took: {:?}",
        case.base_limit,
        build_start.elapsed()
    );
    index.print_timings();

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function run_sift1m_experiment_case, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes, case.query_limit);

    let mut all_brute_force_ids = Vec::with_capacity(case.query_limit);
    println!(
        "Pre-calculating brute-force ground truth for {} queries...",
        case.query_limit
    );
    let brute_force_start = Instant::now();
    for query_vec in &query_vecs {
        let brute_force = exact_top_k(&base_vecs, query_vec, case.k, TEST_METRIC);
        let brute_force_ids: Vec<_> = brute_force.iter().map(|(id, _)| *id).collect();
        all_brute_force_ids.push(brute_force_ids);
    }
    println!(
        "Ground truth calculation took: {:?}",
        brute_force_start.elapsed()
    );

    for &ef_search in case.ef_search {
        let mut total_recall_at_k: f32 = 0.0;
        let mut total_exact_at_k = 0;
        let mut total_search_time = Duration::default();
        println!(
            "--- base_limit={}, ef_search={} ---",
            case.base_limit, ef_search
        );
        for (idx, query_vec) in query_vecs.iter().enumerate() {
            let search_start = Instant::now();
            let search_result_ids = index.search_knn(query_vec, case.k, ef_search);
            total_search_time += search_start.elapsed();
            let brute_force_ids = &all_brute_force_ids[idx];

            let brute_set: HashSet<_> = brute_force_ids.iter().copied().collect();
            let overlap_num = search_result_ids
                .iter()
                .filter(|id| brute_set.contains(id))
                .count();

            let recall_at_k = overlap_num as f32 / case.k as f32;
            let exact_at_k = brute_force_ids.len() >= case.k
                && search_result_ids.len() >= case.k
                && &brute_force_ids[..case.k] == &search_result_ids[..case.k];
            total_recall_at_k += recall_at_k;
            if exact_at_k {
                total_exact_at_k += 1;
            }
        }

        println!("---");
        println!(
            "SUMMARY: base_limit={}, k={}, ef_search={}, query_limit={}",
            case.base_limit, case.k, ef_search, case.query_limit
        );
        println!("  total_search_time = {:?}", total_search_time);
        println!(
            "  avg_search_time   = {:?}",
            total_search_time / case.query_limit as u32
        );
        println!(
            "  recall@{} = [{}/{}]",
            case.k, total_recall_at_k, case.query_limit
        );
        println!(
            "  exact@{}  = [{}/{}]",
            case.k, total_exact_at_k, case.query_limit
        );
        println!("---\n");
    }
}

// 大橘: 优化函数签名与代码规范.
// 将函数参数从 &Vec<T> 改为更通用的切片 &[T], 增强函数适用性.
pub fn exact_top_k(
    base_vecs: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<(u64, f32)> {
    let mut results = Vec::with_capacity(base_vecs.len());
    for (idx, base_vec) in base_vecs.iter().enumerate() {
        // 大橘: 优化代码规范.
        // similarity 函数的参数是 &[f32], query 和 base_vec 可以直接传入, Rust 会自动解引用.
        // 原: let score = similarity(&query, &base_vec , metric);
        let score = similarity(query, base_vec, metric);
        results.push((idx as u64, score));
    }

    // 大橘: 优化性能.
    // 原本是对所有结果进行 O(N log N) 的全排序, 然后取前 k 个.
    // 当 base_vecs 很大时 (例如 SIFT1M 的 100 万), 这个开销非常大.
    // 现改为使用 `select_nth_unstable_by`, 这是一个 O(N) 的部分排序算法,
    // 它能保证第 k 个元素就位, 并且所有前 k 个元素都在它前面.
    // 之后再对这 k 个元素进行一次 O(k log k) 的排序, 总体性能大幅提升.
    if results.len() > k {
        // O(N) 操作: 将第 k 个位置的元素就位.
        // 排序规则: score 降序, score 相同则 id 升序, 保持结果可复现。
        results.select_nth_unstable_by(k, |a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
        // 截断, 只保留前面未经排序的 k 个元素.
        results.truncate(k);
    }
    // O(k log k) 操作: 对最终的 top-k 结果进行精确排序.
    results.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
        other => other,
    });
    results
}

fn build_index(base_vecs: &[Vec<f32>], params: HnswBuildParams) -> HnswIndex {
    let mut index = HnswIndex::new_with_metric(TEST_METRIC);
    println!(
        "insert params: ef_construction={}, m={}, m_max={}",
        params.ef_construction, params.m, params.m_max
    );

    for (id, base_vec) in base_vecs.iter().enumerate() {
        // 认为每一条向量的数组下标序号就是它的id.
        index.insert(
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
fn sift1m_recall_smoke_test() {
    let smoke_cases = [Sift1mEfSearchSweepGroup {
        base_limit: 10,
        k: 5,
        ef_search: &[10],
        query_limit: 5,
    }];
    run_sift1m_experiments(&smoke_cases);
}

// =========================
// Dead Code: Groundtruth path
// 当前主测试主体不再使用官方 groundtruth。
// 这一段只作为历史参考保留，避免和当前 SIFT1M sweep 主路径混在一起。
// =========================
#[allow(dead_code)]
fn run_hnsw_in_sift1m_case_groundtruth(
    base_limit: usize,
    k: usize,
    ef_search: usize,
    query_limit: usize,
) {
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function run_hnsw_in_sift1m_case, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes, base_limit);
    let mut index = build_index(&base_vecs, DEFAULT_HNSW_BUILD_PARAMS);
    index.print_timings();

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function run_hnsw_in_sift1m_case, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes, query_limit);

    let groundtruth_bytes = load_sift_bytes("data//sift1m//sift//sift_groundtruth.ivecs")
        .expect("in function run_hnsw_in_sift1m_case, load query_bytes error");
    let groundtruth_vecs = read_uvecs(&groundtruth_bytes, query_limit);
    // println!("{:#?}", groundtruth_vecs);

    let mut valid_query_cnt = 0;
    let mut total_recall_at_k: f32 = 0.0;
    let mut total_exact_at_k = 0;
    let mut total_search_time = Duration::default();
    for (query_idx, query_vec) in query_vecs.iter().enumerate() {
        // query没有单独的ID. query的序号就是它的隐式ID
        // groundtruth 也按照同样的序号对齐.
        // 在循环当中, 我们认为这个序号就是query_idx.

        // 给一条 query 的 groundtruth，取得前 k 个整数 id的方法.
        let groundtruth_vec = &groundtruth_vecs[query_idx]; //使用引用&: 只需要读取, 不需要修改, 不要从大Vec中移出元素
                                                            // println!("groundtruth_vec : {:#?}", groundtruth_vec);
        let groundtruth_vec_top_k: Vec<_> = groundtruth_vec
            .iter()
            .copied()
            .filter(|&id| id < base_limit as u32)
            .take(k)
            .collect();

        println!("groundtruth_vec_top_k : {:#?}", groundtruth_vec_top_k);
        //只有当 groundtruth_vec_top_k.len() == k 时，才把这条 query 纳入统计
        if groundtruth_vec_top_k.len() != k {
            println!("Skip this query: groundtruth_vec_top_k.len() != k");
            continue;
        }
        // valid_query_cnt 表示“真正纳入 recall/exact 统计的 query 数”
        valid_query_cnt += 1;
        let search_start = Instant::now();
        let search_result_ids = index.search_knn(&query_vec, k, ef_search);
        println!(
            "query[{}] search took: {:?}",
            query_idx,
            search_start.elapsed()
        );
        total_search_time += search_start.elapsed();
        let search_result_ids_u32: Vec<u32> =
            search_result_ids.iter().map(|&id| id as u32).collect();

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
        if exact_at_k {
            total_exact_at_k += 1;
        }
    }

    // 关于top-k中涉及到的两个指标:
    // recall@k : 前k个命中了多少个. 不要求顺序一致, 只要求id部分或全部重合. 可以是一个计数值或者百分比分数.
    // exact@k  : 前k个是否完全一致. 要求顺序一致, 要求id全部重合. 是一个布尔值.

    println!("---");
    println!(
        "SUMMARY: base_limit={}, k={}, ef_search={}, query_limit={}, valid_query_cnt={}",
        base_limit, k, ef_search, query_limit, valid_query_cnt
    );

    if valid_query_cnt == 0 {
        println!("no valid queries under current base_limit");
    } else {
        println!("  valid_query_cnt = [{}]", valid_query_cnt);
        println!("  total_search_time = {:?}", total_search_time);
        println!(
            "  avg_search_time   = {:?}",
            total_search_time / valid_query_cnt as u32
        );
        println!(
            "  recall@{} = [{}/{}]",
            k, total_recall_at_k, valid_query_cnt
        );
        println!(
            "  average recall@{} = [{}]",
            k,
            total_recall_at_k / valid_query_cnt as f32
        );
        println!(
            "  exact@{}  = [{}/{}]",
            k, total_exact_at_k, valid_query_cnt
        );
        println!("---\n");
    }
}
