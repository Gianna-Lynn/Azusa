# Azusa Performance Logs

这份文件只记录值得留档的实验汇总结果。
原始终端输出继续放在 `target/logs/` 下，并保持每次覆盖。

## 2026-04-18

Experiment: `SIFT1M recall sweep`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- output log: `target/logs/sift1m_recall_experiment.out`

### Results

| ef_search | recall@5 | exact@5 |
| --------: | -------: | ------: |
|        60 |    72.8% |     35% |
|        80 |    77.0% |     44% |
|       100 |    79.6% |     47% |
|       150 |    83.2% |     52% |
|       200 |    86.8% |     60% |
|       300 |    90.6% |     70% |

### Notes

- 这次记录保留的是汇总表，不包含逐 query 明细。
- 日常调参时优先看 `target/logs/sift1m_recall_experiment.out`。
- 只有确认值得保留的一版结果，才追加到这份文档里。

## 2026年4月19日11:40:03

Experiment: `SIFT1M recall sweep with timing`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- output log: `docs/result.out`

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 30.1282601s |
| ground truth time | 1.2903312s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 2.56s | 8.6% |
| search_layer | 25.39s | 85.3% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          21.6837ms |         216.837us |    72.8% |     35% |
|        80 |          28.7138ms |         287.138us |    77.0% |     44% |
|       100 |          34.1943ms |         341.943us |    79.6% |     47% |
|       150 |          43.6116ms |         436.116us |    83.2% |     52% |
|       200 |          57.7896ms |         577.896us |    86.8% |     60% |
|       300 |          81.7277ms |         817.277us |    90.6% |     70% |

### Notes

- 这版结果补充了建图时间、ground truth 计算时间，以及插入阶段热点分布。
- `search_layer` 依然占插入阶段主要耗时，约为总插入时间的 `85.3%`。
- `ef_search` 从 `60` 提高到 `300` 时，`recall@5` 从 `72.8%` 提升到 `90.6%`，同时平均查询耗时从 `216.837us` 增长到 `817.277us`。

## 2026-04-20

Experiment: `SIFT1M recall sweep after removing neighbor to_vec`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- change: `search_layer` 保留 workspace/stamp 设计, 去掉邻居 `to_vec()`

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 30.2196078s |
| ground truth time | 1.3316127s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 2.65s | 8.9% |
| search_layer | 25.37s | 84.9% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          19.7407ms |         197.407us |    72.8% |     35% |
|        80 |          23.6708ms |         236.708us |    77.0% |     44% |
|       100 |          27.9301ms |         279.301us |    79.6% |     47% |
|       150 |          39.6365ms |         396.365us |    83.2% |     52% |
|       200 |          49.8616ms |         498.616us |    86.8% |     60% |
|       300 |          71.1145ms |         711.145us |    90.6% |     70% |

### Notes

- 这版结果用于保留“workspace/stamp 保持不变, 只去掉邻居 `to_vec()`”后的基线。
- 和 2026年4月19日11:40:03 那版相比, 建图时间基本回到同一水平, 查询耗时在各档 `ef_search` 下整体更低。
