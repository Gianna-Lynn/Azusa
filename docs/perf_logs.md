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
