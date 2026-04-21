# Azusa Performance Logs

这份文件只记录值得留档的实验汇总结果。
原始终端输出继续放在 `target/logs/` 下，并保持每次覆盖。

## Overview

Benchmark config: release, base_limit=100000, query_limit=100, k=5, ef_construction=80, m=4, m_max=8, metric=L2

| version | change | build_index | search_layer | ef=100 avg | ef=300 avg | recall@5 / exact@5 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-21 | 新节点连边和反向边裁剪均接入 heuristic neighbor selection | 26.607152s | 18.56s | 252.872us | 709.235us | 92.2% / 75%, 97.0% / 89% |
| 2026-04-21 | 新节点连边接入 heuristic neighbor selection | 25.227798s | 17.91s | 347.366us | 616.848us | 88.8% / 70%, 96.2% / 84% |
| 2026-04-21 | 相似度缓存改为 workspace epoch 数组并修复 query 边界 | 20.5939304s | 14.31s | 255.632us | 629.796us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-20 | 图内部从 `Id` 驱动改为 `index` 驱动 | 19.8201135s | 15.56s | 241.834us | 575.874us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-20 | 去掉 `result_set` 热路径维护 | 27.9259546s | 22.96s | 281.949us | 719.258us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-20 | 去掉邻居 `to_vec()` 复制 | 30.2196078s | 25.37s | 279.301us | 711.145us | 79.6% / 47%, 90.6% / 70% |
| 2026-04-19 | 带 timing 的基线版本 | 30.1282601s | 25.39s | 341.943us | 817.277us | 79.6% / 47%, 90.6% / 70% |

### Overview Notes

- `build_index` 和 `search_layer` 越低越好。
- 查询侧只摘了 `ef_search=100` 和 `300` 两档，分别代表中档与高档搜索开销。
- 当前留档版本里，2026-04-21 的全量 heuristic neighbor selection 版本进一步提高了 `recall@5` / `exact@5`，代价是建图时间继续上升，查询侧在不同 `ef_search` 档位出现分化。

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

## 2026-04-21

Experiment: `SIFT1M recall sweep after switching similarity cache to workspace epoch arrays and fixing query boundaries`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- command: `cargo test --release test_hnsw_in_sift1m -- --nocapture`
- change: `SearchLayerWorkspace` 增加 `sim_epoch` / `sim_marks` / `sim_values`, `search_layer` / `greedy_search_at_level` / `select_neighbors_simple` / `get_nearest` 改用 workspace 数组缓存相似度, 并在不同 query 入口调用 `reset_similarity_cache` 隔离缓存 epoch

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 20.5939304s |
| ground truth time | 1.3509758s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 1.39s | 7.4% |
| search_layer | 14.31s | 76.0% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          14.0727ms |         140.727us |    72.8% |     35% |
|        80 |          19.2201ms |         192.201us |    77.0% |     44% |
|       100 |          25.5632ms |         255.632us |    79.6% |     47% |
|       150 |          37.4485ms |         374.485us |    83.2% |     52% |
|       200 |          40.6331ms |         406.331us |    86.8% |     60% |
|       300 |          62.9796ms |         629.796us |    90.6% |     70% |

### Notes

- 这版用于记录“相似度缓存从 `HashMap<usize, f32>` 改为 workspace epoch 数组, 并修复不同 query 间 epoch 隔离”后的性能结果。
- 和上一版“图内部从 `Id` 驱动改为 `index` 驱动”相比, 建图时间从 `19.8201135s` 到 `20.5939304s`, 约慢 `3.9%`, 可能包含本次运行波动和更保守的 query 边界 reset 成本。
- `search_layer` 插入耗时从 `15.56s` 降到 `14.31s`, 约快 `8.0%`; 这是本次改动最稳定的正向信号。
- 查询侧 `ef_search=100` 从 `241.834us` 升到 `255.632us`, 约慢 `5.7%`; `ef_search=300` 从 `575.874us` 升到 `629.796us`, 约慢 `9.4%`, 说明保守 reset 后查询侧没有稳定收益。
- `recall@5` 与 `exact@5` 在各档 `ef_search` 下恢复到上一版同一水平, 说明此前未隔离 query epoch 导致的语义回退已经修复。

## 2026-04-21

Experiment: `SIFT1M recall sweep after enabling heuristic neighbor selection for new-node links`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- command: `cargo test --release test_hnsw_in_sift1m -- --nocapture`
- change: 新节点连边从 `select_neighbors_simple` 切到 heuristic neighbor selection, `SearchLayerWorkspace` 托管 `work_heap` / `discard_heap`, heap tie-break 使用 `Reverse<usize>` 保持 index 驱动; 反向边超限裁剪暂时仍走 simple selection

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 25.227798s |
| ground truth time | 1.4356661s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 1.69s | 7.2% |
| search_layer | 17.91s | 76.7% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          24.62ms |         246.2us |    84.6% |     58% |
|        80 |        28.3911ms |       283.911us |    87.6% |     66% |
|       100 |        34.7366ms |       347.366us |    88.8% |     70% |
|       150 |        51.9618ms |       519.618us |    92.8% |     74% |
|       200 |        58.3748ms |       583.748us |    94.8% |     78% |
|       300 |        61.6848ms |       616.848us |    96.2% |     84% |

### Notes

- 和同日 workspace 相似度缓存版本相比, `recall@5` 在各档 `ef_search` 下明显提升, 例如 `ef_search=100` 从 `79.6%` 到 `88.8%`, `ef_search=300` 从 `90.6%` 到 `96.2%`。
- `exact@5` 同样提升明显, `ef_search=300` 从 `70%` 到 `84%`, 说明 heuristic selection 改善的不只是命中数量, 也改善了 top-5 结果集合的整体质量。
- 建图时间从 `20.5939304s` 升到 `25.227798s`, 约慢 `22.5%`; `search_layer` 插入耗时从 `14.31s` 升到 `17.91s`, 约慢 `25.2%`。主要原因是 heuristic selection 会额外计算候选 `e` 与已选邻居 `r` 的相似度, 用更多建图成本换取更分散的边。
- 查询侧中低档 `ef_search` 耗时上升较明显, 如 `ef_search=100` 从 `255.632us` 到 `347.366us`; 高档 `ef_search=300` 与上一版接近, 从 `629.796us` 到 `616.848us`。
- 当前只替换新节点连边选择, 反向边超限裁剪仍保留 simple selection, 便于单独观察 heuristic new-node links 对图质量和成本的影响。

## 2026-04-21

Experiment: `SIFT1M recall sweep after enabling heuristic neighbor selection for new-node links and reverse pruning`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- command: `cargo test --release test_hnsw_in_sift1m -- --nocapture`
- change: 新节点连边和旧节点反向边超限裁剪均从 simple selection 切到 heuristic neighbor selection; workspace 继续托管 `work_heap` / `discard_heap`

### Build Summary

| item | value |
| --- | ---: |
| build_index time | 26.607152s |
| ground truth time | 1.4883383s |
| insert count | 100000 |

### Insert Timings

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 1.72s | 7.0% |
| search_layer | 18.56s | 75.1% |

### Query Results

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |        18.1198ms |       181.198us |    87.0% |     62% |
|        80 |        21.4482ms |       214.482us |    90.0% |     70% |
|       100 |        25.2872ms |       252.872us |    92.2% |     75% |
|       150 |        42.0259ms |       420.259us |    94.4% |     81% |
|       200 |        48.3072ms |       483.072us |    95.8% |     86% |
|       300 |        70.9235ms |       709.235us |    97.0% |     89% |

### Notes

- 和“仅新节点连边 heuristic”版本相比, 全量 heuristic 继续提高质量: `ef_search=100` 的 `recall@5` 从 `88.8%` 到 `92.2%`, `exact@5` 从 `70%` 到 `75%`; `ef_search=300` 的 `recall@5` 从 `96.2%` 到 `97.0%`, `exact@5` 从 `84%` 到 `89%`。
- 建图时间从 `25.227798s` 升到 `26.607152s`, 约慢 `5.5%`; `search_layer` 插入耗时从 `17.91s` 升到 `18.56s`, 约慢 `3.6%`。反向裁剪 heuristic 带来额外候选多样性检查, 但增量成本相对新节点连边 heuristic 更小。
- 查询侧中低档本次反而更快, 例如 `ef_search=100` 从 `347.366us` 降到 `252.872us`, 同时质量更高; 高档 `ef_search=300` 从 `616.848us` 升到 `709.235us`, 说明全量 heuristic 改变了图结构后, 不同搜索宽度下的探索路径成本不同。
- 和同日 workspace 相似度缓存版本相比, `ef_search=300` 的 `recall@5` 从 `90.6%` 提升到 `97.0%`, `exact@5` 从 `70%` 提升到 `89%`, 是当前留档版本里质量最高的一版。
