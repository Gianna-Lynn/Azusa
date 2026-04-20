# Azusa Performance Logs

这份文件只记录值得留档的实验汇总结果。
原始终端输出继续放在 `target/logs/` 下，并保持每次覆盖。

## Overview

| version | change | build_index | search_layer | ef=100 avg | ef=300 avg | recall@5 / exact@5 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-20 | 图内部从 `Id` 驱动改为 `index` 驱动 | 19.8201135s | 15.56s | 241.834us | 575.874us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-20 | 去掉 `result_set` 热路径维护 | 27.9259546s | 22.96s | 281.949us | 719.258us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-20 | 去掉邻居 `to_vec()` 复制 | 30.2196078s | 25.37s | 279.301us | 711.145us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-19 | 带 timing 的基线版本 | 30.1282601s | 25.39s | 341.943us | 817.277us | 79.6% / 47%, 90.6% / 70% |

### Overview Notes

- `build_index` 和 `search_layer` 越低越好。
- 查询侧只摘了 `ef_search=100` 和 `300` 两档，分别代表中档与高档搜索开销。
- 当前留档版本里，`Id -> index` 内部重构这一版整体最好，且 `recall@5` / `exact@5` 没有回退。

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

## 2026-04-20

Experiment: `SIFT1M recall sweep after removing result_set hot-path maintenance`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- change: `search_layer` 改为仅用 `result_heap` 维护结果集, 去掉 `result_set + lazy deletion` 热路径维护

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 27.9259546s |
| ground truth time | 1.2853574s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 2.71s | 9.8% |
| search_layer | 22.96s | 83.3% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          18.7443ms |         187.443us |    72.8% |     35% |
|        80 |          25.6416ms |         256.416us |    77.0% |     44% |
|       100 |          28.1949ms |         281.949us |    79.6% |     47% |
|       150 |          39.3195ms |         393.195us |    83.2% |     52% |
|       200 |          52.2370ms |         522.370us |    86.8% |     60% |
|       300 |          71.9258ms |         719.258us |    90.6% |     70% |

### Notes

- 这版结果用于记录“去掉 `result_set` 热路径维护, 改为 `result_heap` 单独维护结果集”后的表现。
- 和同日“去掉邻居 `to_vec()`”那版基线相比, 建图时间从 `30.2196078s` 降到 `27.9259546s`, 约快 `7.6%`。
- `search_layer` 插入耗时从 `25.37s` 降到 `22.96s`, 约快 `9.5%`, 说明 `result_set + lazy deletion` 确实是主要常数热点之一。
- `recall@5` 与 `exact@5` 在各档 `ef_search` 下保持不变, 说明这次优化没有改变搜索语义。
- 查询侧整体接近持平, `ef_search=60` 和 `150` 略快, `80/200/300` 附近有小幅波动, 当前主要收益仍然集中在建图阶段。

## 2026-04-20

Experiment: `SIFT1M recall sweep after switching graph internals from Id to index`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- change: 图内部表示从 `Id` 驱动改为连续 `index` 驱动, 邻接表 / entry point / search_layer / pruning 热路径统一改走 `usize`

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 19.8201135s |
| ground truth time | 1.3273091s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 2.14s | 11.0% |
| search_layer | 15.56s | 79.9% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          15.5512ms |         155.512us |    72.8% |     35% |
|        80 |          20.3955ms |         203.955us |    77.0% |     44% |
|       100 |          24.1834ms |         241.834us |    79.6% |     47% |
|       150 |          34.2195ms |         342.195us |    83.2% |     52% |
|       200 |          41.9431ms |         419.431us |    86.8% |     60% |
|       300 |          57.5874ms |         575.874us |    90.6% |     70% |

### Notes

- 这版结果用于记录“图内部统一改成连续 `index` 驱动”后的第一版性能档案。
- 和上一版“去掉 `result_set` 热路径维护”相比, 建图时间从 `27.9259546s` 降到 `19.8201135s`, 约快 `29.0%`。
- `search_layer` 插入耗时从 `22.96s` 降到 `15.56s`, 约快 `32.2%`, 说明 `Id -> index` 哈希映射确实是热路径里的大常数项。
- 查询侧在各档 `ef_search` 下也同步受益, 例如 `ef_search=300` 时平均查询耗时从 `719.258us` 降到 `575.874us`, 约快 `19.9%`。
- `recall@5` 与 `exact@5` 在各档 `ef_search` 下保持不变, 说明这次重构只改变内部表示, 没有改变搜索语义。
