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
    let config = parse_args();

    println!("{config:#?}");

    let base_bytes = fs::read("data//sift1m//sift//sift_base.fvecs").unwrap();
    let query_bytes = fs::read("data//sift1m//sift//sift_query.fvecs").unwrap();
    let groundtruth_bytes = fs::read("data//sift1m//sift//sift_groundtruth.ivecs").unwrap();
    let query_vecs = read_fvecs(&query_bytes, config.eval.query_limit);

    let mut benchmark_rows = Vec::new();

    for &base_limit in &config.eval.base_limits {
        let base_vecs = read_fvecs(&base_bytes, base_limit);
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

        let max_k = *usable_ks.iter().max().unwrap();
        let reference_top_k_ids =
            build_reference_top_k_ids(&groundtruth_bytes, &base_vecs, &query_vecs, max_k);

        for &algorithm in &config.eval.algorithms {
            match algorithm {
                BenchmarkAlgorithm::Flat => {
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

    write_reports(&config, &benchmark_rows);
}

fn build_reference_top_k_ids(
    groundtruth_bytes: &[u8],
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<u32>> {
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

fn run_flat_pass(
    base_limit: usize,
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    reference_top_k_ids: &[Vec<u32>],
    ks: &[usize],
    max_k: usize,
) -> Vec<BenchmarkRow> {
    let mut accumulators = vec![EvalAccumulator::default(); ks.len()];

    for (query_vec, reference_ids) in query_vecs.iter().zip(reference_top_k_ids.iter()) {
        let search_start = Instant::now();
        let search_result_ids = exact_top_k(base_vecs, query_vec, max_k, Metric::L2);
        let elapsed = search_start.elapsed();
        let search_result_ids_u32: Vec<u32> =
            search_result_ids.iter().map(|(id, _)| *id as u32).collect();

        update_accumulators(
            ks,
            reference_ids,
            &search_result_ids_u32,
            elapsed,
            &mut accumulators,
        );
    }

    build_rows(
        BenchmarkAlgorithm::Flat,
        base_limit,
        None,
        None,
        ks,
        accumulators,
    )
}

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

    for &ef_search in ef_search_values {
        let mut accumulators = vec![EvalAccumulator::default(); ks.len()];
        for (query_vec, reference_ids) in query_vecs.iter().zip(reference_top_k_ids.iter()) {
            let search_start = Instant::now();
            let search_result_ids = hnsw_index.search_knn(query_vec, max_k, ef_search);
            let elapsed = search_start.elapsed();
            let search_result_ids_u32: Vec<u32> =
                search_result_ids.iter().map(|&id| id as u32).collect();

            update_accumulators(
                ks,
                reference_ids,
                &search_result_ids_u32,
                elapsed,
                &mut accumulators,
            );
        }

        rows.extend(build_rows(
            BenchmarkAlgorithm::Hnsw,
            base_limit,
            Some(ef_search),
            Some(build_time),
            ks,
            accumulators,
        ));
    }

    rows
}

fn update_accumulators(
    ks: &[usize],
    reference_top_k_ids: &[u32],
    search_result_ids: &[u32],
    elapsed: Duration,
    accumulators: &mut [EvalAccumulator],
) {
    let reference_set_top_k: HashSet<_> = reference_top_k_ids.iter().copied().collect();

    for (k, accumulator) in ks.iter().copied().zip(accumulators.iter_mut()) {
        accumulator.valid_query_count += 1;
        accumulator.total_search_time += elapsed;

        let search_len = search_result_ids.len().min(k);
        let search_prefix = &search_result_ids[..search_len];
        let reference_prefix = &reference_top_k_ids[..k];

        let overlap_count = search_prefix
            .iter()
            .filter(|id| reference_set_top_k.contains(id))
            .count();
        accumulator.total_recall_at_k += overlap_count as f32 / k as f32;

        let exact_at_k = search_result_ids.len() >= k && reference_prefix == search_prefix;
        if exact_at_k {
            accumulator.exact_count += 1;
        }
    }
}

fn build_rows(
    algorithm: BenchmarkAlgorithm,
    base_limit: usize,
    ef_search: Option<usize>,
    build_time: Option<Duration>,
    ks: &[usize],
    accumulators: Vec<EvalAccumulator>,
) -> Vec<BenchmarkRow> {
    ks.iter()
        .copied()
        .zip(accumulators.into_iter())
        .map(|(k, accumulator)| {
            let avg_search_time = if accumulator.valid_query_count == 0 {
                Duration::default()
            } else {
                accumulator.total_search_time / accumulator.valid_query_count as u32
            };
            let average_recall = if accumulator.valid_query_count == 0 {
                0.0
            } else {
                accumulator.total_recall_at_k / accumulator.valid_query_count as f32
            };

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

fn write_reports(config: &BenchmarkConfig, results: &[BenchmarkRow]) {
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

fn parse_usize_list(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter(|item| !item.is_empty())
        .map(parse_usize)
        .collect()
}

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
    raw.parse::<usize>().unwrap()
}

fn parse_f64(raw: &str) -> f64 {
    raw.parse::<f64>().unwrap()
}

pub fn read_u32(input: &mut &[u8]) -> u32 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    *input = rest;
    u32::from_le_bytes(int_bytes.try_into().unwrap())
}

pub fn read_f32(input: &mut &[u8]) -> f32 {
    let (float_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    *input = rest;
    f32::from_le_bytes(float_bytes.try_into().unwrap())
}

pub fn read_one_fvec(input: &mut &[u8]) -> Vec<f32> {
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
