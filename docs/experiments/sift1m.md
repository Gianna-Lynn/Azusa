# SIFT1M 历史向量近邻实验

以下结果来自 2026 年 4—5 月的历史记录，本次仅整理文档，没有重新执行性能实验。

## 轻量 Smoke Benchmark

用于快速验证 benchmark harness 能跑通，并写出 `target/benchmark/sift1m.md`。

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

说明：

- `base_limit=1000`：只读取前 1000 条 base 向量。
- `query_limit=10`：只评估前 10 条 query。
- `mode=hnsw`：只跑 HNSW，不跑 flat baseline。
- 这是 smoke benchmark，不代表最终性能。

输出位置：

```text
target/benchmark/sift1m.csv
target/benchmark/sift1m.md
```

2026-05-06 历史 smoke：

```text
date: 2026-05-06
command: cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
build_time: 88.6562ms
valid_query_count: 10
total_search_time: 409.8us
avg_search_time: 40.98us
recall@5: 98.00%
exact@5: 9/10
output: target/benchmark/sift1m.md
```

## SIFT1M Benchmark 命令

日常中等规模复测命令：

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 100000 --query-limit 100 --ks 5 --ef-search 60,100,300 --m 4 --ef-construction 80
```

全量 1M 的复现命令（建图耗时较长，按需运行）：

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limits 1000000 --ks 1,5,10 --ef-search 60,100,300 --query-limit 100 --m 8,16,32 --ef-construction 100,200
```

## 已有结果摘要

### 2026-04-22：100k 调参落点

记录位置：

```text
develop 分支的 docs/archive/perf_logs.md
```

摘要：

| ef_search | avg_search_time | recall@5 | exact@5 |
| ---: | ---: | ---: | ---: |
| 60 | 129.318us | 87.0% | 62% |
| 100 | 228.890us | 92.2% | 75% |
| 300 | 522.015us | 97.0% | 89% |

说明：

- 这轮覆盖 `base_limit=100000`、`query_limit=100`、`k=5`。
- 这是当前轻量调参时优先参考的结果。

### 2026-04-25：已有 1M sweep 记录

记录位置：

```text
develop 分支的 docs/archive/perf_logs.md
target/benchmark/sweeps/sift1m_hnsw_sweep_1m_20260425_195847/
```

摘要：

| config | build time | ef_search=60 recall@10 / exact@10 | ef_search=100 recall@10 / exact@10 | ef_search=300 recall@10 / exact@10 |
| --- | ---: | ---: | ---: | ---: |
| m8_efc100 | 1251.370706978s | 89.40% / 49/100 | 94.90% / 68/100 | 99.40% / 91/100 |
| m16_efc100 | 3055.649133404s | 97.50% / 76/100 | 99.20% / 89/100 | 99.80% / 94/100 |
| m16_efc200 | 4668.71236162s | 97.80% / 78/100 | 99.20% / 90/100 | 99.90% / 95/100 |

说明：

- `m32_efc200` 在启动后受服务器硬盘挂载故障影响，没有完整结果。
- 这组全量数据按历史记录引用，不代表当前版本重测。

## 当前限制

- SIFT1M 原始数据位于 `data/sift1m/sift/`，不属于仓库内可移植的小样本数据。
- benchmark 输出写入 `target/benchmark/`，该目录不提交。
- 轻量 smoke 只能证明流程可运行，不能代表最终召回率或性能。
- 全量 1M benchmark 建图耗时较长，且依赖本机/服务器资源，需具备对应资源后单独执行。

历史完整参数日志保存在 develop；摘要保留相关配置、日期及失败配置说明。原始数据和大型产物不随仓库分发。
