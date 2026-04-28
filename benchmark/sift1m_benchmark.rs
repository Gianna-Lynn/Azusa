//! SIFT1M benchmark harness，用于比较 flat 和 HNSW 的表现。
use azusa::distance::similarity;
use azusa::index::hnsw::HnswIndex;
use azusa::types::Metric;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkAlgorithm {
    Flat,
    Hnsw,
}

impl BenchmarkAlgorithm {
    fn as_str(self) -> &'static str {
        match self {
            BenchmarkAlgorithm::Flat => "flat",
            BenchmarkAlgorithm::Hnsw => "hnsw",
        }
    }
}

#[derive(Debug)]
struct BuildConfig {
    m: usize,
    m_max: usize,
    ef_construction: usize,
    level_sample_probability: f64,
}

#[derive(Debug)]
struct EvalConfig {
    base_limits: Vec<usize>,
    query_limit: usize,
    ks: Vec<usize>,
    ef_search_values: Vec<usize>,
    algorithms: Vec<BenchmarkAlgorithm>,
}

#[derive(Debug)]
struct BenchmarkConfig {
    build: BuildConfig,
    eval: EvalConfig,
}

#[derive(Debug)]
struct BenchmarkRow {
    algorithm: BenchmarkAlgorithm,
    base_limit: usize,
    k: usize,
    ef_search: Option<usize>,
    valid_query_count: usize,
    total_search_time: Duration,
    avg_search_time: Duration,
    average_recall: f32,
    exact_count: usize,
    build_time: Option<Duration>,
}

#[derive(Debug, Default, Clone)]
struct EvalAccumulator {
    total_recall_at_k: f32,
    exact_count: usize,
    total_search_time: Duration,
    valid_query_count: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            m: 4,
            m_max: 8,
            ef_construction: 80,
            level_sample_probability: 0.5,
        }
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            base_limits: vec![100_000],
            query_limit: 100,
            ks: vec![5],
            ef_search_values: vec![60, 100, 300],
            algorithms: vec![BenchmarkAlgorithm::Hnsw],
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            build: BuildConfig::default(),
            eval: EvalConfig::default(),
        }
    }
}

fn main() {
    // 入口函数：按配置把 SIFT1M benchmark 从“数据读取 -> 评测 -> 输出报告”完整跑一遍。
    let config = parse_args();

    // 打印最终配置，方便在命令行里确认这次 benchmark 实际用了哪些参数。
    println!("{config:#?}");

    // 预先把原始二进制文件读进内存，后面每个 base_limit / query_limit 都从这些字节里切片读取。
    let base_bytes = fs::read("data//sift1m//sift//sift_base.fvecs").unwrap();
    let query_bytes = fs::read("data//sift1m//sift//sift_query.fvecs").unwrap();
    let groundtruth_bytes = fs::read("data//sift1m//sift//sift_groundtruth.ivecs").unwrap();

    // 只读取需要参与评测的 query 子集，避免 query_limit 较小时还去解析完整文件。
    let query_vecs = read_fvecs(&query_bytes, config.eval.query_limit);

    // 这里保存所有最终结果行，最后统一导出 CSV / Markdown。
    let mut benchmark_rows = Vec::new();

    // 外层按 base_limit 逐个评测：同一批 query 会在不同 base 规模下重复测试。
    for &base_limit in &config.eval.base_limits {
        // 从原始 base 数据里切出当前规模的数据集。
        let base_vecs = read_fvecs(&base_bytes, base_limit);

        // 过滤掉大于 base 数据规模的 k，避免后面构造 reference 或统计时越界。
        let usable_ks: Vec<usize> = config
            .eval
            .ks
            .iter()
            .copied()
            .filter(|&k| k <= base_vecs.len())
            .collect();

        if usable_ks.is_empty() {
            println!(
                "skip base_limit={} because every k is larger than the dataset size {}",
                base_limit,
                base_vecs.len()
            );
            continue;
        }

        // 这次评测里真正会用到的最大 k，用它可以复用同一次搜索结果。
        let max_k = *usable_ks.iter().max().unwrap();

        // 预先为每个 query 准备参考答案前缀，后面 Flat 和 HNSW 都会复用。
        let reference_top_k_ids =
            build_reference_top_k_ids(&groundtruth_bytes, &base_vecs, &query_vecs, max_k);

        // 每个算法单独跑一轮，便于对比暴力基线和 HNSW 的表现。
        for &algorithm in &config.eval.algorithms {
            match algorithm {
                BenchmarkAlgorithm::Flat => {
                    // Flat 路径：直接暴力搜索，不依赖索引。
                    let rows = run_flat_pass(
                        base_limit,
                        &base_vecs,
                        &query_vecs,
                        &reference_top_k_ids,
                        &usable_ks,
                        max_k,
                    );
                    for row in &rows {
                        print_row(row);
                    }
                    benchmark_rows.extend(rows);
                }
                BenchmarkAlgorithm::Hnsw => {
                    // HNSW 路径：先建索引，再在多个 ef_search 下分别测试。
                    let build_start = Instant::now();
                    let mut hnsw_index = build_index(&base_vecs, &config.build);
                    let build_time = build_start.elapsed();
                    println!(
                        "build_index took {:?} for base_limit={}",
                        build_time, base_limit
                    );
                    hnsw_index.print_timings();

                    let rows = run_hnsw_pass(
                        base_limit,
                        &mut hnsw_index,
                        &query_vecs,
                        &reference_top_k_ids,
                        &usable_ks,
                        max_k,
                        &config.eval.ef_search_values,
                        build_time,
                    );
                    for row in &rows {
                        print_row(row);
                    }
                    benchmark_rows.extend(rows);
                }
            }
        }
    }

    // 全部评测结束后再统一写报告，避免中途多次落盘。
    write_reports(&config, &benchmark_rows);
}

fn build_reference_top_k_ids(
    groundtruth_bytes: &[u8],
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<u32>> {
    // 从官方 ground truth 中提取每个 query 的前 k 个正确答案；
    // 如果文件里的 id 超出了当前 base 子集范围，就回退到暴力精确搜索。
    let groundtruth_vecs = read_uvecs(groundtruth_bytes, query_vecs.len());
    groundtruth_vecs
        .into_iter()
        .zip(query_vecs.iter())
        .map(|(groundtruth_vec, query_vec)| {
            let top_k_ids: Vec<u32> = groundtruth_vec
                .into_iter()
                .filter(|&id| id < base_vecs.len() as u32)
                .take(k)
                .collect();
            if top_k_ids.len() == k {
                top_k_ids
            } else {
                let brute_force = exact_top_k(base_vecs, query_vec, k, Metric::L2);
                brute_force.iter().map(|(id, _)| *id as u32).collect()
            }
        })
        .collect()
}

fn exact_top_k(base_vecs: &[Vec<f32>], query: &[f32], k: usize, metric: Metric) -> Vec<(u64, f32)> {
    // 对 base 全量扫一遍，计算每个向量与 query 的相似度，并返回排序后的 top-k。
    let mut results = Vec::with_capacity(base_vecs.len());
    for (idx, base_vec) in base_vecs.iter().enumerate() {
        let score = similarity(query, base_vec, metric);
        results.push((idx as u64, score));
    }

    if results.len() > k {
        results.select_nth_unstable_by(k, |a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
        results.truncate(k);
    }
    results.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
        other => other,
    });
    results
}

fn build_index(base_vecs: &[Vec<f32>], build: &BuildConfig) -> HnswIndex {
    // 按给定的构建参数，把 base 向量逐条插入 HNSW 索引。
    let mut index = HnswIndex::new(Metric::L2, build.level_sample_probability);

    // 这里仍然假设 sift1m 的 id 就是向量下标。
    for (id, base_vec) in base_vecs.iter().enumerate() {
        index.insert(
            id as u64,
            base_vec.clone(),
            build.ef_construction,
            build.m,
            build.m_max,
        );
    }
    index
}

/// 执行 Flat baseline 的整条评测流程。
///
/// 这条路径不依赖 HNSW 索引，而是直接对 `base_vecs` 做暴力全量搜索：
/// 1. 对每个 query 调用 `exact_top_k`，拿到前 `max_k` 个结果；
/// 2. 把结果 id 转成 `u32`，与预先准备好的 ground truth 前缀做对比；
/// 3. 把 recall、exact 命中数、耗时等指标累积到 `EvalAccumulator`；
/// 4. 最后把所有 `k` 对应的累积结果转成 `BenchmarkRow`。
///
/// 之所以先算 `max_k`，是因为同一次搜索结果可以复用到多个 `k`，
/// 避免为 `k=5`、`k=10` 等不同评测点重复做暴力搜索。
fn run_flat_pass(
    base_limit: usize,
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    reference_top_k_ids: &[Vec<u32>],
    ks: &[usize],
    max_k: usize,
) -> Vec<BenchmarkRow> {
    let mut accumulators = vec![EvalAccumulator::default(); ks.len()];
    
    // 遍历每个 query 和它对应的 ground truth 前缀，执行暴力搜索，并把结果灌进累积器。
    for (query_vec, reference_ids) in query_vecs.iter().zip(reference_top_k_ids.iter()) {
        // Flat baseline 的核心就是“每个 query 都在整个 base 集上做精确搜索”。
        let search_start = Instant::now();
        // 这里直接拿到前 max_k 个结果，后面再按实际的 k 取前缀对比。
        let search_result_ids = exact_top_k(base_vecs, query_vec, max_k, Metric::L2);
        let elapsed = search_start.elapsed();
        let search_result_ids_u32: Vec<u32> =
            search_result_ids.iter().map(|(id, _)| *id as u32).collect();

        // 把这次查询的结果和 ground truth 前缀对比，更新每个 k 对应的累积统计。
        update_accumulators(
            ks,
            reference_ids,
            &search_result_ids_u32,
            elapsed,
            &mut accumulators,
        );
    }

    // 搜集完所有 query 的统计信息后，再统一转成输出行。
    build_rows(
        BenchmarkAlgorithm::Flat,
        base_limit,
        None,
        None,
        ks,
        accumulators,
    )
}

/// 执行 HNSW 评测流程。
///
/// 这条路径和 Flat baseline 的区别在于：
/// - 搜索不是直接扫全量 base，而是调用 `hnsw_index.search_knn`；
/// - 同一份索引会被多个 `ef_search` 重复使用，用来观察召回率和速度的权衡；
/// - `build_time` 会被带到结果行里，方便把“建索引成本”也算进报告。
///
/// 这里按 `ef_search_values` 逐个跑一遍，每个 `ef_search` 都独立统计一组
/// `EvalAccumulator`，这样最后每个 `k` 都会生成一行对应的 benchmark 结果。
fn run_hnsw_pass(
    base_limit: usize,
    hnsw_index: &mut HnswIndex,
    query_vecs: &[Vec<f32>],
    reference_top_k_ids: &[Vec<u32>],
    ks: &[usize],
    max_k: usize,
    ef_search_values: &[usize],
    build_time: Duration,
) -> Vec<BenchmarkRow> {
    let mut rows = Vec::new();

    // 遍历每个 ef_search，执行搜索并更新累积统计。每个 ef_search 都会生成一组结果行，最后一起返回。
    for &ef_search in ef_search_values {
        // 每个 ef_search 都单独统计，避免不同搜索宽度之间互相污染。
        let mut accumulators = vec![EvalAccumulator::default(); ks.len()];
        // 遍历每个 query 和它对应的 ground truth 前缀，执行 HNSW 搜索，并把结果灌进累积器。
        for (query_vec, reference_ids) in query_vecs.iter().zip(reference_top_k_ids.iter()) {
            let search_start = Instant::now();
            let search_result_ids = hnsw_index.search_knn(query_vec, max_k, ef_search);
            let elapsed = search_start.elapsed();
            let search_result_ids_u32: Vec<u32> =
                search_result_ids.iter().map(|&id| id as u32).collect();

            // 把这次查询的结果和 ground truth 前缀对比，更新每个 k 对应的累积统计。
            update_accumulators(
                ks,
                reference_ids,
                &search_result_ids_u32,
                elapsed,
                &mut accumulators,
            );
        }//end for each query

        // 每个 ef_search 都会生成一组结果行，最后一起返回。
        rows.extend(build_rows(
            BenchmarkAlgorithm::Hnsw,
            base_limit,
            Some(ef_search),
            Some(build_time),
            ks,
            accumulators,
        ));
    }//end for each ef_search

    rows

}

/// 把单个 query 的搜索结果灌进累积器。
///
/// 这个函数只负责“把一次查询的观测值记账”，不做任何输出：
/// - `valid_query_count` 累加，表示参与统计的 query 数量；
/// - `total_search_time` 累加，后面可以算平均耗时；
/// - `total_recall_at_k` 累加，后面再除以 query 数量得到平均 recall；
/// - `exact_count` 记录“前 k 个结果和 ground truth 前缀完全一致”的次数。
///
/// 注意这里同时支持多个 `k`，所以一个 query 的结果只会被处理一次，
/// 但会分别更新 `k=5`、`k=10` 之类不同评测点对应的累积器。
fn update_accumulators(
    ks: &[usize],
    reference_top_k_ids: &[u32],
    search_result_ids: &[u32],
    elapsed: Duration,
    accumulators: &mut [EvalAccumulator],
) {
    // 为了高效判断搜索结果里哪些 id 命中了 ground truth 前 k 个，这里把 ground truth 前 k 个 id 放进一个 HashSet。
    let reference_set_top_k: HashSet<_> = reference_top_k_ids.iter().copied().collect();

    // 遍历每个 k 和对应的累积器，更新统计信息。
    for (k, accumulator) in ks.iter().copied().zip(accumulators.iter_mut()) {
        accumulator.valid_query_count += 1; // 这个 query 参与了统计，累加器的有效查询数加一。
        accumulator.total_search_time += elapsed; // 这个 query 的搜索耗时累加到总耗时里。

        // `search_result_ids` 可能比 k 短，所以这里只能按实际长度取前缀。
        let search_len = search_result_ids.len().min(k);
        let search_prefix = &search_result_ids[..search_len];   // 这个地方不能直接取 k 个，因为搜索结果可能不足 k 个；但 ground truth 前缀是有保证的，直接取 k 个就行。
        let reference_prefix = &reference_top_k_ids[..k];   // 这个地方可以直接取k个

        // recall@k 的定义：搜索结果 top-k 与 ground truth top-k 的重合比例。
        let overlap_count = search_prefix
            .iter()
            .filter(|id| reference_set_top_k.contains(id))
            .count();
        accumulator.total_recall_at_k += overlap_count as f32 / k as f32;

        // exact@k 表示搜索结果前 k 个 ID 是否和 ground truth 前 k 个 ID 完全一致。
        let exact_at_k = search_result_ids.len() >= k && reference_prefix == search_prefix;
        if exact_at_k {
            accumulator.exact_count += 1;
        }
    }
}

/// 把一组累积统计转换成最终可打印/可导出的 benchmark 行。
///
/// 输入的 `accumulators` 是按 `ks` 一一对应的，因此这个函数的工作就是：
/// 1. 把每个 `k` 的总耗时、平均耗时、平均 recall、exact 次数装进 `BenchmarkRow`；
/// 2. 为同一种算法、同一个 base_limit、同一次 build_time 生成多行结果；
/// 3. 让上层调用者不必关心“如何从统计值拼成表格行”。
///
/// 这样 Flat 和 HNSW 两条路径都可以复用同一套输出结构，避免重复逻辑。
fn build_rows(
    algorithm: BenchmarkAlgorithm,
    base_limit: usize,
    ef_search: Option<usize>,
    build_time: Option<Duration>,
    ks: &[usize],
    accumulators: Vec<EvalAccumulator>,
) -> Vec<BenchmarkRow> {
    // 是基于 `ks` 和 `accumulators` 的一一对应关系，把每个 k 和它的累积统计配对起来，生成对应的结果行。
    ks.iter()
        .copied()
        .zip(accumulators.into_iter())
        .map(|(k, accumulator)| {
            // 如果没有有效 query，避免除零；这时耗时和召回都记为默认值。

            // 平均搜索时间的定义是“所有有效查询的总搜索时间除以有效查询数”，
            let avg_search_time = if accumulator.valid_query_count == 0 {
                Duration::default()
            } else {
                accumulator.total_search_time / accumulator.valid_query_count as u32
            };

            // 平均 recall@k 的定义是“每个有效查询的 recall@k 的平均值”，
            let average_recall = if accumulator.valid_query_count == 0 {
                0.0
            } else {
                accumulator.total_recall_at_k / accumulator.valid_query_count as f32
            };

            // 把这一行的所有信息装进 BenchmarkRow，返回给上层调用者。
            BenchmarkRow {
                algorithm,
                base_limit,
                k,
                ef_search,
                valid_query_count: accumulator.valid_query_count,
                total_search_time: accumulator.total_search_time,
                avg_search_time,
                average_recall,
                exact_count: accumulator.exact_count,
                build_time,
            }
        })
        .collect()
}

fn print_row(row: &BenchmarkRow) {
    // 把一条 benchmark 结果打印到终端，便于快速查看当前配置下的表现。
    println!("---");
    println!("algorithm = {}", row.algorithm.as_str());
    println!("base_limit = {}", row.base_limit);
    println!("k = {}", row.k);
    if let Some(ef_search) = row.ef_search {
        println!("ef_search = {}", ef_search);
    }
    if let Some(build_time) = row.build_time {
        println!("build_time = {:?}", build_time);
    }
    println!("valid_query_count = {}", row.valid_query_count);
    println!("total_search_time = {:?}", row.total_search_time);
    println!("avg_search_time = {:?}", row.avg_search_time);
    println!(
        "average recall@{} = {:.2}%",
        row.k,
        row.average_recall * 100.0
    );
    println!(
        "exact@{} = {}/{}",
        row.k, row.exact_count, row.valid_query_count
    );
}

// 把所有 benchmark 行写成 CSV 和 Markdown 两种格式，方便后续分析和展示。
fn write_reports(config: &BenchmarkConfig, results: &[BenchmarkRow]) {
    // 将所有 benchmark 结果导出为 CSV 和 Markdown，方便后续分析和贴到文档里。
    let output_dir = "target/benchmark";
    fs::create_dir_all(output_dir).unwrap();

    let csv_path = format!("{output_dir}/sift1m.csv");
    let md_path = format!("{output_dir}/sift1m.md");

    let mut csv = String::from(
        "algorithm,base_limit,query_limit,k,m,m_max,ef_construction,level_sample_probability,build_time,ef_search,valid_query_count,total_search_time,avg_search_time,recall,exact_count\n",
    );
    for result in results {
        let ef_search = result
            .ef_search
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let build_time = result
            .build_time
            .map(|value| format!("{:?}", value))
            .unwrap_or_else(|| "-".to_string());

        let fields = vec![
            result.algorithm.as_str().to_string(),
            result.base_limit.to_string(),
            config.eval.query_limit.to_string(),
            result.k.to_string(),
            config.build.m.to_string(),
            config.build.m_max.to_string(),
            config.build.ef_construction.to_string(),
            config.build.level_sample_probability.to_string(),
            build_time,
            ef_search,
            result.valid_query_count.to_string(),
            format!("{:?}", result.total_search_time),
            format!("{:?}", result.avg_search_time),
            format!("{:.6}", result.average_recall),
            result.exact_count.to_string(),
        ];
        csv.push_str(&fields.join(","));
        csv.push('\n');
    }

    let mut markdown = String::new();
    markdown.push_str("# SIFT1M Benchmark\n\n");
    markdown.push_str(&format!(
        "- base_limits: `{:?}`\n- query_limit: `{}`\n- ks: `{:?}`\n- algorithms: `{:?}`\n- m: `{}`\n- m_max: `{}`\n- ef_construction: `{}`\n- level_sample_probability: `{}`\n\n",
        config.eval.base_limits,
        config.eval.query_limit,
        config.eval.ks,
        config.eval.algorithms,
        config.build.m,
        config.build.m_max,
        config.build.ef_construction,
        config.build.level_sample_probability,
    ));
    markdown.push_str(
        "| algorithm | base limit | k | ef_search | build time | valid queries | total search time | avg search time | recall | exact |\n",
    );
    markdown.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for result in results {
        let ef_search = result
            .ef_search
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let build_time = result
            .build_time
            .map(|value| format!("{:?}", value))
            .unwrap_or_else(|| "-".to_string());
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:?} | {:?} | {:.2}% | {}/{} |\n",
            result.algorithm.as_str(),
            result.base_limit,
            result.k,
            ef_search,
            build_time,
            result.valid_query_count,
            result.total_search_time,
            result.avg_search_time,
            result.average_recall * 100.0,
            result.exact_count,
            result.valid_query_count,
        ));
    }

    fs::write(&csv_path, csv).unwrap();
    fs::write(&md_path, markdown).unwrap();
    println!("wrote {csv_path}");
    println!("wrote {md_path}");
}

fn parse_args() -> BenchmarkConfig {
    // 解析命令行参数，支持 base/query/k/ef_search/m/mode 等评测开关。
    let args: Vec<String> = env::args().skip(1).collect();
    let mut config = BenchmarkConfig::default();

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--base-limit" {
            config.eval.base_limits = vec![parse_usize(&args[i + 1])];
            i += 2;
        } else if args[i] == "--base-limits" {
            config.eval.base_limits = parse_usize_list(&args[i + 1]);
            i += 2;
        } else if args[i] == "--query-limit" {
            config.eval.query_limit = parse_usize(&args[i + 1]);
            i += 2;
        } else if args[i] == "--k" {
            config.eval.ks = vec![parse_usize(&args[i + 1])];
            i += 2;
        } else if args[i] == "--ks" {
            config.eval.ks = parse_usize_list(&args[i + 1]);
            i += 2;
        } else if args[i] == "--m" {
            config.build.m = parse_usize(&args[i + 1]);
            config.build.m_max = config.build.m * 2;
            i += 2;
        } else if args[i] == "--ef-search" {
            config.eval.ef_search_values = parse_usize_list(&args[i + 1]);
            i += 2;
        } else if args[i] == "--mode" {
            config.eval.algorithms = parse_algorithms(&args[i + 1]);
            i += 2;
        } else if args[i] == "--ef-construction" {
            config.build.ef_construction = parse_usize(&args[i + 1]);
            i += 2;
        } else if args[i] == "--p" {
            config.build.level_sample_probability = parse_f64(&args[i + 1]);
            i += 2;
        } else {
            panic!("unknown argument: {}", args[i]);
        }
    }

    config
}

/// 解析逗号分隔的 `usize` 列表。
///
/// 支持像 `--base-limits 1000,10000`、`--ks 1,5,10`、`--ef-search 60,100,300`
/// 这样的新参数格式。
///
/// 解析规则比较宽松：
/// - 允许末尾多一个逗号的空项被忽略；
/// - 每个非空项都必须能成功解析成 `usize`，否则直接 panic；
/// - 返回顺序与输入顺序一致，不做排序。
fn parse_usize_list(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter(|item| !item.is_empty())
        .map(parse_usize)
        .collect()
}

/// 解析 benchmark 算法模式。
///
/// 新的 `--mode` 支持下面几种输入：
/// - `flat`：只跑暴力 baseline；
/// - `hnsw`：只跑 HNSW；
/// - `both`：同时跑 Flat 和 HNSW；
/// - 也支持逗号列表，比如 `flat,hnsw`，方便和别的参数保持一致。
///
/// 这里会做两层处理：
/// 1. 先把字符串拆成多个 token；
/// 2. 再把 token 映射成 `BenchmarkAlgorithm`，并在最后去重。
///
/// 这样可以容忍 `both,flat` 之类的重复写法，而不会影响最终执行顺序。
fn parse_algorithms(raw: &str) -> Vec<BenchmarkAlgorithm> {
    let mut algorithms = Vec::new();
    for item in raw.split(',').filter(|item| !item.is_empty()) {
        match item {
            "flat" => algorithms.push(BenchmarkAlgorithm::Flat),
            "hnsw" => algorithms.push(BenchmarkAlgorithm::Hnsw),
            "both" => {
                algorithms.push(BenchmarkAlgorithm::Flat);
                algorithms.push(BenchmarkAlgorithm::Hnsw);
            }
            other => panic!("unknown benchmark mode: {}", other),
        }
    }

    if algorithms.is_empty() {
        panic!("benchmark mode cannot be empty");
    }

    algorithms.dedup();
    algorithms
}

fn parse_usize(raw: &str) -> usize {
    // 把字符串参数转成 usize，解析失败就直接 panic。
    raw.parse::<usize>().unwrap()
}

fn parse_f64(raw: &str) -> f64 {
    // 把字符串参数转成 f64，通常用于概率或浮点型阈值。
    raw.parse::<f64>().unwrap()
}

pub fn read_u32(input: &mut &[u8]) -> u32 {
    // 从字节流中按小端序读取一个 u32，并把读取位置向后移动 4 字节。
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    *input = rest;
    u32::from_le_bytes(int_bytes.try_into().unwrap())
}

pub fn read_f32(input: &mut &[u8]) -> f32 {
    // 从字节流中按小端序读取一个 f32，并把读取位置向后移动 4 字节。
    let (float_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    *input = rest;
    f32::from_le_bytes(float_bytes.try_into().unwrap())
}

pub fn read_one_fvec(input: &mut &[u8]) -> Vec<f32> {
    // 读取一条 fvecs 向量：先读维度，再连续读出该维度个 f32。
    let dim = read_u32(input) as usize;
    let mut vec_f32: Vec<f32> = Vec::new();
    for _ in 0..dim {
        if input.len() < std::mem::size_of::<f32>() {
            panic!("In function read_one_vec, got a vector without enough dim");
        }
        let tmp = read_f32(input);
        vec_f32.push(tmp);
    }
    vec_f32
}

pub fn read_fvecs(bytes: &[u8], limit: usize) -> Vec<Vec<f32>> {
    // 从 fvecs 原始字节流中最多读取 limit 条向量。
    let mut input = bytes;
    let mut res: Vec<Vec<f32>> = Vec::with_capacity(limit);
    for _ in 0..limit {
        if input.len() < std::mem::size_of::<u32>() {
            break;
        }
        let tmp = read_one_fvec(&mut input);
        if tmp.is_empty() {
            break;
        }
        res.push(tmp);
    }
    res
}

pub fn read_one_uvec(input: &mut &[u8]) -> Vec<u32> {
    // 读取一条 ivecs/uvecs 风格向量：先读维度，再连续读出该维度个 u32。
    let dim = read_u32(input) as usize;
    let mut vec_u32 = Vec::with_capacity(dim);
    for _ in 0..dim {
        if input.len() < std::mem::size_of::<u32>() {
            panic!("In function read_one_uvec, got a vector without enough dim");
        }
        vec_u32.push(read_u32(input));
    }
    vec_u32
}

pub fn read_uvecs(bytes: &[u8], limit: usize) -> Vec<Vec<u32>> {
    // 从 ivecs/uvecs 原始字节流中最多读取 limit 条向量。
    let mut input = bytes;
    let mut res = Vec::with_capacity(limit);
    for _ in 0..limit {
        if input.len() < std::mem::size_of::<u32>() {
            break;
        }
        let tmp = read_one_uvec(&mut input);
        if tmp.is_empty() {
            break;
        }
        res.push(tmp);
    }
    res
}
