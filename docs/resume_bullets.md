# Azusa 简历描述

## 中文版

- 使用 Rust 实现基于 HNSW 的向量检索原型，支持向量记录插入、top-k 近邻搜索、metadata 回查，并通过 HTTP JSON 接口提供最小服务化调用闭环。
- 基于 SIFT1M 官方 ground truth 搭建可重复的 Recall@k、Exact@k 与延迟评测流程，支持参数化运行及 CSV/Markdown 结果输出。
- 设计 PowerShell HTTP smoke 流程，覆盖健康检查、插入、检索、记录回查与缺失记录等核心接口行为。

## English Version

- Built a Rust vector-retrieval prototype based on HNSW, supporting vector insertion, top-k nearest-neighbor search, metadata lookup, and a minimal HTTP JSON service.
- Developed a reproducible SIFT1M evaluation workflow using official ground truth, with Recall@k, Exact@k, latency measurement, parameterized runs, and CSV/Markdown reports.
- Created a PowerShell HTTP smoke workflow covering health checks, insertion, search, record retrieval, and missing-record behavior.
