# 检索评测与证据验证

## 无模型小闭环

从 `rag/` 使用项目的 Python 环境运行：

```powershell
python run.py benchmark-lab --smoke --top-k 1,3
```

命令使用 `tests/fixtures/retrieval_smoke_embedding.jsonl` 的六条合成短文本和
`retrieval_smoke_eval.jsonl` 的三条问题，只执行真实 BM25。
向量字段是占位零向量；不加载 embedding/reranker，不要求 GPU、Rust 服务或 API Key。
这三题仅验证指标、报告和复现路径，不能用于宣称真实检索质量或性能提升。
`--smoke` 拒绝其他模式，避免将占位向量包装成 dense 实验。

默认创建 `output/eval_runs/runs/benchmark-<UTC时间>-<随机后缀>/`，终端打印实际路径。
每次输出包括：

- `benchmark.json`：输入字节 SHA-256、代码提交和源码哈希、环境配置、汇总行、逐题排名与 trace、失败坐标。
- `benchmark.csv`：固定列，成功与失败分别用 `status=success/failed` 标记，失败保留 `error`。
- `benchmark.md`：指标表、P50/P95 口径、输入身份与 Failed cells。

只有完整矩阵全部成功才返回退出码 0；失败单元格不会因其他成功而消失。
指定 `--output-dir` 时目录必须为空；写入函数还会独占创建各结果文件，拒绝覆盖。
三个文件没有跨文件事务保证，写入中途失败会以异常结束，不报告成功。

## 指标边界

- Hit@k：前 k 项中是否存在相关证据。
- Recall@k：命中的唯一相关 chunk 数 / 全部相关 chunk 数。
- MRR：各题第一个相关结果的倒数排名的平均值。
- binary nDCG@k：DCG 按 `1/log2(rank+1)` 折扣；IDCG 使用 `min(请求k, 相关项数)`。
  少返回结果不能缩小分母。例如请求 k=3、相关项有两条、只返回第一条时，nDCG 约为
  `1 / (1 + 1/log2(3)) = 0.6131`，不能记为 1。
- 为兼容旧的直接两参数函数调用，未显式传 k 时才采用列表长度；正式评测始终传请求 k。
- 重复检索 ID、重复语料 chunk 身份或不在语料中的金标直接失败，不生成虚高分数。
- 每个模式/k 单元格先执行一次显式预热，预热耗时与 trace 单独记录；不计入逐题延迟。
- 逐题延迟包含编码、客户端快照身份检查、HTTP、检索和重排；BM25 包含每次重新建索引。
  P50/P95 使用 nearest-rank。固定顺序、单次查询与小样本不能证明尾延迟 SLA，跨机器不可直接比较。

## 实际后端与追溯

`hybrid_rerank` 只是请求模式，不能证明模型实际成功运行。每题 trace 记录：

- `rerank_backend=model`：模型返回数量正确且有限的分数。
- `rerank_backend=rule_fallback`：模型异常、关闭或分数非法时，使用规则排序，并保留原因。

预热后端也独立记录，汇总保留各后端的题数。规则回退成绩不能写成模型重排成绩。
其他模式报告 `not_used`，不把“没有执行重排”误认为模型成功。

Provenance 记录提交、dirty 状态、参与运行的 Python 源文件哈希、输入文件绝对路径和
字节哈希、Python/依赖版本、模型配置、已加载模型的实际设备。不会读取或保存任意环境
变量、API Key 或令牌。未校验模型权重字节；配置路径相同不能证明权重版本相同。
公开性能结论前仍须固定模型版本、数据许可、独立测试集、机器资源与重复测量条件。

输入在加载前后、各单元格执行前后核对；发现字节变化时对应单元格记为失败。
输入检查本身位于逐查询计时之外，查询内部的快照身份检查仍计入延迟。

## 自定义数据与普通评测

```powershell
python run.py benchmark-lab --variant sample=tests/fixtures/retrieval_smoke_embedding.jsonl,tests/fixtures/retrieval_smoke_eval.jsonl --modes bm25 --top-k 1,3
python run.py eval --mode bm25 --top-k 3 --embedding-path tests/fixtures/retrieval_smoke_embedding.jsonl --eval-path tests/fixtures/retrieval_smoke_eval.jsonl
```

普通 `eval` 也默认使用新的 `runs/` 目录，输出每模式结果及包含成功/失败清单的
`evaluation.json`；失败退出非零。旧 `--output-name` 仅作为实验标签，显式
`--keep-run-copy` 在该次独立目录中额外保留命名副本，不再隐式更新 `latest/`。
`workflow`、Search Agent 等旧命令有各自的输出策略，本次没有改写它们。

真实 dense 模式需要当前 Rust 服务与本地模型。可在启动 Python 前通过当前进程配置覆盖：

| 配置名 | 用途 |
| --- | --- |
| `AZUSA_EMBEDDING_MODEL` | 本地 embedding 模型路径 |
| `AZUSA_EMBEDDING_DEVICE` | embedding 请求设备 |
| `AZUSA_RERANK_MODEL` | 本地 reranker 路径 |
| `AZUSA_RERANK_DEVICE` | reranker 请求设备 |
| `AZUSA_SERVER_URL` | Rust 服务地址 |

这些是可选读取项，不要求修改 Windows 用户或系统环境变量；不设置时沿用本机原配置。
切换数据 variant 时，Rust 不可变快照不会自动重置。BM25/brute 可独立比较多份数据；
HNSW/hybrid 应为不同快照分别启动独立服务或独立执行，冲突保留为失败，不能静默清空。

## 验证命令

```powershell
python -m pytest tests/test_eval_metrics.py tests/test_benchmark_lab.py tests/test_retrieval_evaluation.py -q -p no:cacheprovider -o pythonpath=src --basetemp=.cache/p0b-local-run
```

以上直接检查短排名分母、重复证据、缺失金标、输入被替换、模式失败、实际回退、
三格式失败保留及拒绝覆盖。需要复验 HTTP 协议时，按 [HTTP API](http_api.md) 构建服务并运行对应测试。


## Agent 回归与引用案例

`rag/data/eval/agent_eval.jsonl` 包含 15 条工程样例：single-hop、rewrite、multi-hop
各 3 条，证据不足 2 条，引用 badcase 4 条。它用于行为回归，不是开放域能力 benchmark。

2026-07-13 的历史 BM25 基线：类型与 hop 数通过率均为 1.0，平均证据 recall 为
0.8846；开启本地 Answerer 后，citation precision 为 0.6000、recall 为 0.5667。
这些数字没有在本次文档整理中重测，不能当作当前版本的新结果。

代表案例来自 2026-06-24 的历史 workflow：题目询问将比分追至 10–7 的达阵球员，
答案给出 Jonathan Stewart，但引用指向 halftime show 正文。该例说明答案正确不等于
证据正确。历史逐题原文保存在 develop；代码和测试继续在当前分支维护。

审查时分别检查检索是否命中相关证据、引用是否对齐预期 chunk，以及自然语言答案是否
完整并受正文支持。LLM 辅助审查不能替代金标、客观指标和人工抽样。
