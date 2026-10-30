//! 在独立进程上对同一输入测量 Flat/HNSW，输出逐查询观测；不改检索服务协议。

use azusa::{distance::metric_similarity, index::hnsw::HnswIndex, types::Metric};
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashSet, env, fs, time::Instant};

#[derive(Deserialize)]
/// 冻结单个实验单元的向量、插入顺序和索引预算。
struct Input {
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    order: Vec<usize>,
    algorithm: String,
    k: usize,
    ef_search: usize,
    m: usize,
    ef_construction: usize,
}

/// 对维度、有限数值和完整排列先校验，防止错误输入进入索引断言边界。
fn validate(input: &Input) -> Result<(), &'static str> {
    let dim = input.base.first().map_or(0, Vec::len);
    if dim == 0 || input.queries.is_empty() || input.k == 0 || input.k > input.base.len()
        || input.ef_search < input.k || input.m < 2 || input.ef_construction < input.m
        || !["flat", "hnsw"].contains(&input.algorithm.as_str())
    {
        return Err("invalid experiment budgets or algorithm");
    }
    if input.base.iter().chain(&input.queries).any(|v| v.len() != dim
        || v.iter().any(|x| !x.is_finite() || x.abs() > 1e6)) {
        return Err("invalid vector dimension or value");
    }
    let order: HashSet<_> = input.order.iter().copied().collect();
    if input.order.len() != input.base.len() || order.len() != input.base.len()
        || order.iter().any(|&id| id >= input.base.len()) {
        return Err("insertion order must be a complete permutation");
    }
    Ok(())
}

/// 使用与图相同的 L2 分数做精确 top-k；分数相同时按 ID 排序，避免不稳定金标。
fn exact(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
    let mut scores: Vec<_> = base.iter().enumerate()
        .map(|(id, vector)| (id as u64, metric_similarity(vector, query, Metric::L2))).collect();
    let compare = |a: &(u64, f32), b: &(u64, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    scores.select_nth_unstable_by(k - 1, compare);
    scores.truncate(k);
    scores.sort_unstable_by(compare);
    scores.into_iter().map(|item| item.0).collect()
}

#[cfg(windows)]
/// 读取操作系统的峰值 Working Set；包含整个工作进程，不冒充纯索引内存。
fn peak_rss_bytes() -> Option<usize> {
    #[repr(C)]
    struct Counters { cb: u32, faults: u32, values: [usize; 8] }
    #[link(name = "kernel32")]
    extern "system" { fn GetCurrentProcess() -> *mut std::ffi::c_void; }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(process: *mut std::ffi::c_void, counters: *mut Counters, size: u32) -> i32;
    }
    let mut counters = Counters { cb: std::mem::size_of::<Counters>() as u32, faults: 0, values: [0; 8] };
    // 指针只指向当前栈上的足长 C 布局缓冲区，句柄由系统提供且不归调用方关闭。
    let result = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if result != 0 { Some(counters.values[0]) } else { None }
}

#[cfg(not(windows))]
/// Linux 读取 VmHWM；其他平台显式给空值，不能把未测内存写成零。
fn peak_rss_bytes() -> Option<usize> {
    fs::read_to_string("/proc/self/status").ok()?.lines()
        .find(|line| line.starts_with("VmHWM:"))?.split_whitespace().nth(1)?
        .parse::<usize>().ok().map(|kb| kb * 1024)
}

/// 单元入口只接受一个输入 JSON 路径；预热不计查询延迟，保留所有原始排名和耗时。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("input JSON path required")?;
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    validate(&input)?;
    let start = Instant::now();
    let mut index = if input.algorithm == "hnsw" {
        let mut index = HnswIndex::new(Metric::L2, 0.5);
        for &id in &input.order {
            index.insert(id as u64, input.base[id].clone(), input.ef_construction, input.m, input.m * 2);
        }
        Some(index)
    } else { None };
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    // 每单元在全量查询上预热一次；查询顺序、数据和度量由上层固定。
    for query in &input.queries {
        if let Some(index) = &mut index { index.search_knn(query, input.k, input.ef_search); }
        else { exact(&input.base, query, input.k); }
    }
    let mut observations = Vec::new();
    for query in &input.queries {
        let start = Instant::now();
        let ids = if let Some(index) = &mut index { index.search_knn(query, input.k, input.ef_search) }
                  else { exact(&input.base, query, input.k) };
        observations.push(json!({"ids": ids, "latency_ms": start.elapsed().as_secs_f64() * 1000.0}));
    }
    println!("{}", json!({"algorithm": input.algorithm, "build_ms": build_ms,
        "peak_rss_bytes": peak_rss_bytes(), "queries": observations}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// 精确金标必须遵循 L2 与 ID tie-break，k=1 与全部数据都可运行。
    fn exact_ranking_and_ties() {
        let base = vec![vec![2.0], vec![0.0], vec![0.0]];
        assert_eq!(exact(&base, &[0.0], 1), vec![1]);
        assert_eq!(exact(&base, &[0.0], 3), vec![1, 2, 0]);
    }

    #[test]
    /// 重复排列、错误维度及 ef<k 必须在建图前拒绝。
    fn invalid_input_is_rejected() {
        let mut input = Input { base: vec![vec![1.0], vec![2.0]], queries: vec![vec![1.0]],
            order: vec![0, 1], algorithm: "hnsw".into(), k: 1, ef_search: 8, m: 4, ef_construction: 40 };
        assert!(validate(&input).is_ok());
        input.order = vec![0, 0];
        assert!(validate(&input).is_err());
        input.order = vec![0, 1];
        input.queries = vec![vec![1.0, 2.0]];
        assert!(validate(&input).is_err());
        input.queries = vec![vec![1.0]];
        input.ef_search = 0;
        assert!(validate(&input).is_err());
    }
}
