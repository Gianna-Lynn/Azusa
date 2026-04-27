// SIFT1M recall experiments and smoke checks.
// 这份文件更接近小规模 benchmark / 实验脚本，不是纯粹的单元测试。

// 文件	                    用途
// sift_base.fvecs	        数据库向量（100万个128维向量）— 你要建索引的数据
// sift_query.fvecs	        查询向量（10000个128维向量）— 用来测试搜索效果
// sift_learn.fvecs	        学习向量（100000个128维向量）— 可选的训练集
// sift_groundtruth.ivecs	标准答案（查询的真值） — 每个查询的top-k正确答案
// 用 sift_base 建立HNSW索引
// 用 sift_query 去查询
// 拿结果和 sift_groundtruth 比较，计算 top-k 重合比例和 exact@k 命中率

#[path = "sift1m_test_utils.rs"]
mod sift1m_test_utils;
use crate::sift1m_test_utils::*;
use azusa::{distance::similarity, index::hnsw::HnswIndex, types::Metric};
use std::collections::HashSet;
use std::time::{Duration, Instant};
const TEST_METRIC: Metric = Metric::L2;

// base_limit   控制了我们从sift1m的文件里, 只读前多少条向量.
// k            用来表达 "取得前几个最相似的结果."
// ef_search    控制了我们在hnsw里搜索的时候, 最多看多少个节点. 这个参数越大, 搜索效果越好, 但是搜索速度越慢.
// query_limit  控制了我们从sift1m的文件里, 只读前多少条查询向量. 这个参数越大, 实验结果越稳定, 但是测试时间越长.

/**************************  参数配置  **************************/
// 控制HNSW图怎么构建: 连边宽度、每层保留边数、以及随机层级分布。
#[derive(Debug, Clone, Copy)]
struct HnswIndexBuildConfig {
    ef_construction: usize,
    m: usize,
    m_max: usize,
    level_sample_probability: f64,
}

const HNSW_BUILD_CONFIG: HnswIndexBuildConfig = HnswIndexBuildConfig {
    ef_construction: 80,
    m: 4,
    m_max: 8,
    level_sample_probability: 0.5,
};

// 控制SIFT1M实验规模和查询阶段ef_search扫参。
#[derive(Debug, Clone)]
struct Sift1mExperimentConfig {
    base_limit: usize,
    k: usize,
    ef_search: &'static [usize],
    query_limit: usize,
    reference_source: ReferenceSource,
}

const SIFT1M_EXPERIMENTS: &[Sift1mExperimentConfig] = &[Sift1mExperimentConfig {
    base_limit: 100000,
    k: 5,
    ef_search: &[60, 80, 100, 150, 200, 300],
    query_limit: 100,
    reference_source: ReferenceSource::OfficialGroundTruth,
}];

#[derive(Debug, Clone, Copy)]
enum ReferenceSource {
    OfficialGroundTruth,
    BruteForce,
}

/**************************  测试主体  **************************/
#[test]
fn test_hnsw_in_sift1m() {
    for experiment_config in SIFT1M_EXPERIMENTS {
        run_sift1m_experiment(experiment_config);
    }
}

fn run_sift1m_experiment(config: &Sift1mExperimentConfig) {
    run_sift1m_experiment_with_reference_source(config, config.reference_source);
}

fn run_sift1m_experiment_with_reference_source(
    config: &Sift1mExperimentConfig,
    reference_source: ReferenceSource,
) {
    let base_bytes = load_sift_bytes("data//sift1m//sift//sift_base.fvecs")
        .expect("in function run_sift1m_experiment, load base_bytes error");
    let base_vecs = read_fvecs(&base_bytes, config.base_limit);

    let build_start = Instant::now();
    let mut index = build_index(&base_vecs, HNSW_BUILD_CONFIG);
    println!(
        "base_limit={} build_index took: {:?}",
        config.base_limit,
        build_start.elapsed()
    );
    index.print_timings();

    let query_bytes = load_sift_bytes("data//sift1m//sift//sift_query.fvecs")
        .expect("in function run_sift1m_experiment, load query_bytes error");
    let query_vecs = read_fvecs(&query_bytes, config.query_limit);

    let reference_top_k_ids =
        build_reference_top_k_ids(config, &base_vecs, &query_vecs, reference_source);

    for &ef_search in config.ef_search {
        run_sift1m_eval_pass(
            &mut index,
            config,
            &query_vecs,
            &reference_top_k_ids,
            ef_search,
        );
    }
}

fn build_reference_top_k_ids(
    config: &Sift1mExperimentConfig,
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    reference_source: ReferenceSource,
) -> Vec<Option<Vec<u32>>> {
    match reference_source {
        ReferenceSource::BruteForce => {
            println!(
                "Pre-calculating brute-force ground truth for {} queries...",
                query_vecs.len()
            );
            let brute_force_start = Instant::now();
            let references = query_vecs
                .iter()
                .map(|query_vec| {
                    let brute_force = exact_top_k(base_vecs, query_vec, config.k, TEST_METRIC);
                    Some(brute_force.iter().map(|(id, _)| *id as u32).collect())
                })
                .collect();
            println!(
                "Ground truth calculation took: {:?}",
                brute_force_start.elapsed()
            );
            references
        }
        ReferenceSource::OfficialGroundTruth => {
            println!(
                "Loading official ground truth for {} queries...",
                query_vecs.len()
            );
            let groundtruth_start = Instant::now();
            let groundtruth_bytes = load_sift_bytes("data//sift1m//sift//sift_groundtruth.ivecs")
                .expect("in function build_reference_top_k_ids, load groundtruth_bytes error");
            let groundtruth_vecs = read_uvecs(&groundtruth_bytes, query_vecs.len());
            let references = groundtruth_vecs
                .into_iter()
                .map(|groundtruth_vec| {
                    let groundtruth_vec_top_k: Vec<u32> = groundtruth_vec
                        .into_iter()
                        .filter(|&id| id < config.base_limit as u32)
                        .take(config.k)
                        .collect();
                    if groundtruth_vec_top_k.len() == config.k {
                        Some(groundtruth_vec_top_k)
                    } else {
                        None
                    }
                })
                .collect();
            println!(
                "Official ground truth loading took: {:?}",
                groundtruth_start.elapsed()
            );
            references
        }
    }
}

fn run_sift1m_eval_pass(
    index: &mut HnswIndex,
    config: &Sift1mExperimentConfig,
    query_vecs: &[Vec<f32>],
    reference_top_k_ids: &[Option<Vec<u32>>],
    ef_search: usize,
) {
    let mut total_recall_at_k: f32 = 0.0;
    let mut total_exact_at_k = 0;
    let mut total_search_time = Duration::default();
    let mut valid_query_cnt = 0usize;

    println!(
        "--- base_limit={}, ef_search={} ---",
        config.base_limit, ef_search
    );

    for (query_idx, (query_vec, reference_top_k_ids)) in query_vecs
        .iter()
        .zip(reference_top_k_ids.iter())
        .enumerate()
    {
        let Some(reference_top_k_ids) = reference_top_k_ids.as_ref() else {
            println!("Skip query[{}]: no valid reference top-k", query_idx);
            continue;
        };

        valid_query_cnt += 1;
        let search_start = Instant::now();
        let search_result_ids = index.search_knn(query_vec, config.k, ef_search);
        total_search_time += search_start.elapsed();

        let search_result_ids_u32: Vec<u32> =
            search_result_ids.iter().map(|&id| id as u32).collect();
        let reference_set_top_k: HashSet<_> = reference_top_k_ids.iter().copied().collect();
        let overlap_num = search_result_ids_u32
            .iter()
            .filter(|id| reference_set_top_k.contains(id))
            .count();

        let recall_at_k = overlap_num as f32 / config.k as f32;
        let exact_at_k = search_result_ids_u32.len() >= config.k
            && &reference_top_k_ids[..config.k] == &search_result_ids_u32[..config.k];
        total_recall_at_k += recall_at_k;
        if exact_at_k {
            total_exact_at_k += 1;
        }
    }

    println!("---");
    println!(
        "SUMMARY: base_limit={}, k={}, ef_search={}, query_limit={}, valid_query_cnt={}",
        config.base_limit, config.k, ef_search, config.query_limit, valid_query_cnt
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
            config.k, total_recall_at_k, valid_query_cnt
        );
        println!(
            "  average recall@{} = [{}]",
            config.k,
            total_recall_at_k / valid_query_cnt as f32
        );
        println!(
            "  exact@{}  = [{}/{}]",
            config.k, total_exact_at_k, valid_query_cnt
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

        // select_nth_unstable_by 的排序规则是: 以第 k 个元素为基准, 将所有比它更 "大" 的元素放在前面, 比它更 "小" 的元素放在后面.
        // unstable 的意思是: 对于 score 相同的元素, 它们之间的相对顺序不保证稳定.
        // 在比较函数里加入 id 的比较, score 降序, score 相同则 id 升序, 保持结果可复现。
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

fn build_index(base_vecs: &[Vec<f32>], config: HnswIndexBuildConfig) -> HnswIndex {
    let mut index = HnswIndex::new(TEST_METRIC, config.level_sample_probability);
    println!(
        "HNSW build config: ef_construction={}, m={}, m_max={}, level_sample_probability={}",
        config.ef_construction, config.m, config.m_max, config.level_sample_probability
    );

    for (id, base_vec) in base_vecs.iter().enumerate() {
        // 认为每一条向量的数组下标序号就是它的id.
        index.insert(
            id as u64,
            base_vec.clone(),
            config.ef_construction,
            config.m,
            config.m_max,
        );
    }
    index
}

#[test]
fn sift1m_recall_smoke_test() {
    let smoke_config = Sift1mExperimentConfig {
        base_limit: 10,
        k: 5,
        ef_search: &[10],
        query_limit: 5,
        reference_source: ReferenceSource::BruteForce,
    };
    run_sift1m_experiment_with_reference_source(&smoke_config, ReferenceSource::BruteForce);
}
