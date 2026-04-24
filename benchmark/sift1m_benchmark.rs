use azusa::distance::similarity;
use azusa::index::hnsw::HnswIndex;
use azusa::types::Metric;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::time::{Duration, Instant}; //用来计时

#[derive(Debug)]
struct BenchmarkConfig {
    base_limit: usize,
    query_limit: usize,
    k: usize,
    m: usize,
    ef_search_values: Vec<usize>,
    ef_construction: usize,
    level_sample_probability: f64,
}

#[derive(Debug)]
struct BenchmarkResult {
    ef_search: usize,
    valid_query_count: usize,
    total_search_time: Duration,
    avg_search_time: Duration,
    average_recall: f32,
    exact_count: usize,
}

impl Default for BenchmarkConfig {
    //给BenchmarkConfig定义一个默认配置.
    fn default() -> Self {
        Self {
            base_limit: 100_000,
            query_limit: 100,
            k: 5,
            m: 4,
            ef_search_values: vec![60, 100, 300],
            ef_construction: 80,
            level_sample_probability: 0.5,
        }
    }
}

fn main() {
    let config = parse_args();

    println!("{config:#?}"); //:? 是 Debug 打印  //:#? 是漂亮换行版 Debug 打印
                             // 等价于 println!("{:#?}", config);
    let base_bytes = fs::read("data//sift1m//sift//sift_base.fvecs").unwrap();
    let query_bytes = fs::read("data//sift1m//sift//sift_query.fvecs").unwrap();
    let groundtruth_bytes = fs::read("data//sift1m//sift//sift_groundtruth.ivecs").unwrap();
    let base_vecs = read_fvecs(&base_bytes, config.base_limit);
    let query_vecs = read_fvecs(&query_bytes, config.query_limit);
    let reference_top_k_ids =
        build_reference_top_k_ids(&groundtruth_bytes, &base_vecs, &query_vecs, config.k);

    let build_start = Instant::now();
    let mut hnsw_index = HnswIndex::new(Metric::L2, config.level_sample_probability);

    // 注意: 此处仅适用于sift1m中的id等于index的情况. 如果要更换其他数据集(乱序id),需要重写获取id的逻辑.
    for (id, vect) in base_vecs.iter().enumerate() {
        hnsw_index.insert(
            id as u64,
            vect.clone(), // 注意这里使用的是成本很高的clone
            config.ef_construction,
            config.m,
            config.m * 2, //此处m_max用m的2倍权且代替
        );
    }
    let build_time = build_start.elapsed();
    println!("build_index took {:?}", build_time);
    hnsw_index.print_timings();

    let mut benchmark_results = Vec::new();
    for &ef_search in &config.ef_search_values {
        let result = run_eval_pass(
            &mut hnsw_index,
            &query_vecs,
            &reference_top_k_ids,
            config.k,
            ef_search,
        );
        print_result(config.k, &result);
        benchmark_results.push(result);
    }

    write_reports(&config, build_time, &benchmark_results);
}

fn build_reference_top_k_ids(
    groundtruth_bytes: &[u8],
    base_vecs: &[Vec<f32>],
    query_vecs: &[Vec<f32>],
    k: usize,
) -> Vec<Option<Vec<u32>>> {
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
                Some(top_k_ids)
            } else {
                let brute_force = exact_top_k(base_vecs, query_vec, k, Metric::L2);
                Some(brute_force.iter().map(|(id, _)| *id as u32).collect())
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

fn run_eval_pass(
    hnsw_index: &mut HnswIndex,
    query_vecs: &[Vec<f32>],
    reference_top_k_ids: &[Option<Vec<u32>>],
    k: usize,
    ef_search: usize,
) -> BenchmarkResult {
    let mut total_recall_at_k: f32 = 0.0;
    let mut exact_count = 0usize;
    let mut total_search_time = Duration::default();
    let mut valid_query_count = 0usize;

    for (query_vec, reference_top_k_ids) in query_vecs.iter().zip(reference_top_k_ids.iter()) {
        let Some(reference_top_k_ids) = reference_top_k_ids.as_ref() else {
            continue;
        };

        valid_query_count += 1;
        let search_start = Instant::now();
        let search_result_ids = hnsw_index.search_knn(query_vec, k, ef_search);
        total_search_time += search_start.elapsed();

        let search_result_ids_u32: Vec<u32> =
            search_result_ids.iter().map(|&id| id as u32).collect();
        let reference_set_top_k: HashSet<_> = reference_top_k_ids.iter().copied().collect();
        let overlap_count = search_result_ids_u32
            .iter()
            .filter(|id| reference_set_top_k.contains(id))
            .count();

        total_recall_at_k += overlap_count as f32 / k as f32;

        let exact_at_k = search_result_ids_u32.len() >= k
            && &reference_top_k_ids[..k] == &search_result_ids_u32[..k];
        if exact_at_k {
            exact_count += 1;
        }
    }

    let avg_search_time = if valid_query_count == 0 {
        Duration::default()
    } else {
        total_search_time / valid_query_count as u32
    };
    let average_recall = if valid_query_count == 0 {
        0.0
    } else {
        total_recall_at_k / valid_query_count as f32
    };

    BenchmarkResult {
        ef_search,
        valid_query_count,
        total_search_time,
        avg_search_time,
        average_recall,
        exact_count,
    }
}

fn print_result(k: usize, result: &BenchmarkResult) {
    println!("---");
    println!("ef_search = {}", result.ef_search);
    println!("valid_query_count = {}", result.valid_query_count);
    println!("total_search_time = {:?}", result.total_search_time);
    println!("avg_search_time = {:?}", result.avg_search_time);
    println!(
        "average recall@{} = {:.2}%",
        k,
        result.average_recall * 100.0
    );
    println!(
        "exact@{} = {}/{}",
        k, result.exact_count, result.valid_query_count
    );
}

fn write_reports(config: &BenchmarkConfig, build_time: Duration, results: &[BenchmarkResult]) {
    let output_dir = "target/benchmark";
    fs::create_dir_all(output_dir).unwrap();

    let csv_path = format!("{output_dir}/sift1m.csv");
    let md_path = format!("{output_dir}/sift1m.md");

    let mut csv = String::from(
        "base_limit,query_limit,k,m,m_max,ef_construction,level_sample_probability,build_time,ef_search,valid_query_count,total_search_time,avg_search_time,recall,exact_count\n",
    );
    for result in results {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{:?},{},{},{:?},{:?},{:.6},{}\n",
            config.base_limit,
            config.query_limit,
            config.k,
            config.m,
            config.m * 2,
            config.ef_construction,
            config.level_sample_probability,
            build_time,
            result.ef_search,
            result.valid_query_count,
            result.total_search_time,
            result.avg_search_time,
            result.average_recall,
            result.exact_count,
        ));
    }

    let mut markdown = String::new();
    markdown.push_str("# SIFT1M Benchmark\n\n");
    markdown.push_str(&format!(
        "- base_limit: `{}`\n- query_limit: `{}`\n- k: `{}`\n- m: `{}`\n- m_max: `{}`\n- ef_construction: `{}`\n- level_sample_probability: `{}`\n- build_time: `{:?}`\n\n",
        config.base_limit,
        config.query_limit,
        config.k,
        config.m,
        config.m * 2,
        config.ef_construction,
        config.level_sample_probability,
        build_time,
    ));
    markdown.push_str(
        "| ef_search | valid queries | total search time | avg search time | recall | exact |\n",
    );
    markdown.push_str("| ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for result in results {
        markdown.push_str(&format!(
            "| {} | {} | {:?} | {:?} | {:.2}% | {}/{} |\n",
            result.ef_search,
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

    // 示例命令行: cargo run --bin sift1m_benchmark -- --base-limit 1000 --query-limit 10
    // 对应参数数组:
    // args[0] = "--base-limit"
    // args[1] = "1000"
    // args[2] = "--query-limit"
    // args[3] = "10"

    let mut config = BenchmarkConfig::default();

    let mut i = 0;

    while i < args.len() {
        if args[i] == "--base-limit" {
            //没参数时 args[0] 不存在。需要额外保护条件.
            config.base_limit = parse_usize(&args[i + 1]); //把第二个参数字符串 "1000" 解析成 usize 数字。
            i += 2;
        } else if args[i] == "--query-limit" {
            config.query_limit = parse_usize(&args[i + 1]);
            i += 2;
        } else if args[i] == "--k" {
            config.k = parse_usize(&args[i + 1]);
            i += 2;
        } else if args[i] == "--m" {
            config.m = parse_usize(&args[i + 1]);
            i += 2;
        } else if args[i] == "--ef-search" {
            config.ef_search_values = args[i + 1]
                .split(',') //把字符串按逗号切开
                .map(|item| parse_usize(item))
                //map(parse_usize)也对,不必显式地写出闭包.
                .collect(); //收集成为Vec<usize>
            i += 2;
        } else if args[i] == "--ef-construction" {
            config.ef_construction = parse_usize(&args[i + 1]);
            i += 2;
        } else if args[i] == "--p" {
            config.level_sample_probability = parse_f64(&args[i + 1]);
            i += 2;
        } else {
            panic!("unknown argument: {}", args[i]);
        }
    }
    config
}

fn parse_usize(raw: &str) -> usize {
    // raw:&str 表示接收一个字符串切片, 比如"1000"
    raw.parse::<usize>().unwrap()
}

fn parse_f64(raw: &str) -> f64 {
    // raw:&str 表示接收一个字符串切片, 比如"1000"
    raw.parse::<f64>().unwrap()
}

// 直接复用自 tests\sift1m_recall_experiment.rs
pub fn read_u32(input: &mut &[u8]) -> u32 {
    //input: &mut &[u8] 是一个“可移动的读指针”
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>()); //split_at(4) 从前面切 4 个字节
    *input = rest; //*input = rest 表示读指针往后移动
    u32::from_le_bytes(int_bytes.try_into().unwrap()) //from_le_bytes 按小端格式把 4 字节转成 u32
}

//基于上面的read_u32, 改写成read_f32, 直接复用自 tests\sift1m_recall_experiment.rs
pub fn read_f32(input: &mut &[u8]) -> f32 {
    let (float_bytes, rest) = input.split_at(size_of::<u32>());
    *input = rest;
    f32::from_le_bytes(float_bytes.try_into().unwrap())
}

// 直接复用自 tests\sift1m_recall_experiment.rs
pub fn read_one_fvec(input: &mut &[u8]) -> Vec<f32> {
    let dim = read_u32(input) as usize;
    let mut vec_f32: Vec<f32> = Vec::new();
    for _ in 0..dim {
        if input.len() < size_of::<f32>() {
            panic!("In function read_one_vec, got a vector without enough dim");
        }
        let tmp = read_f32(input);
        vec_f32.push(tmp);
    }
    vec_f32
}

// 直接复用自 tests\sift1m_recall_experiment.rs
// bytes: &[u8]：传入整个文件字节
// limit: usize：只读前多少条
// Vec<Vec<f32>>：很多条向量，每条向量是一个 Vec<f32>
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
        if input.len() < size_of::<u32>() {
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
