# Azusa 改进日志

> 按时间倒序记录每一次可复现的改进。原始 archive 报告保留在 `docs/archive/` 和 `rag/docs/archive/` 中不变，本文件是它们的聚合索引。

---

## 2026-07-13 — Stage 3 P1/P2：Agent Eval 与可信 Citation

### 改动

- 剥离尚未完成的 PyO3 试验代码，继续使用已经稳定工作的 Rust HTTP 服务接入 Python RAG。
- 新增 `rag/data/eval/agent_eval.jsonl`，提供 15 条 Stage 3 Agent 回归样例。
- 新增 `python run.py agent-eval`，统一保存 query plan、retrieval hops、fused evidence、回答、citation 和耗时 trace。
- Answerer 对外固定输出“结论、证据、引用来源、不确定性”四段结构。
- 移除“模型缺少 citation 时强制回填 top-1”的策略，只接受模型明确给出且属于当前 evidence 的 chunk id。
- 修复只有 citation 没有答案字段时误把引用行当作答案的问题，新增 `answer_missing` 状态并保留有效证据供人工审核。
- 为 Agent、Answerer、Agent Eval 和 RRF 补充 20 个 Python 单元测试，新增及改动函数均带中文功能说明。
- 单独格式化 `src/index/hnsw_impl/search.rs`，只调整换行、空白和注释尾部空格，不改变 HNSW 语义。

### Agent Eval 样例分布

| 类别 | 数量 |
| --- | ---: |
| single-hop | 3 |
| query rewrite | 3 |
| multi-hop | 3 |
| insufficient evidence | 2 |
| citation badcase | 4 |

### 复现命令

仅验证查询规划、检索和 evidence fusion：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent-eval --mode bm25 --top-k 5
```

同时验证本地 0.5B Answerer 和 citation：

```powershell
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent-eval --mode bm25 --top-k 5 --with-answer
```

### 2026-07-13 BM25 基线

| 指标 | 结果 |
| --- | ---: |
| query type accuracy | 1.0000 |
| hop count pass rate | 1.0000 |
| average evidence recall | 0.8846 |
| citation precision | 0.6000 |
| citation recall | 0.5667 |

这组结果是工程回归基线，不是公开 benchmark。当前 0.5B Answerer 仍会漏答、错实体或漏 citation；系统现在会保留这些失败并明确不确定性，不再通过伪造 top-1 citation 掩盖问题。

### 验证

- `cargo fmt --check`：通过
- `cargo check`：通过
- `cargo test`：5 项通过
- Python `unittest`：20 项通过
- 单条真实模型 smoke：`When were stromules discovered?` 正确回答 `1962年`，并引用对应 evidence chunk

---

## 2026-07-10 — RRF Fusion 替换 Naive Hybrid Merge

### 改动

`rag/src/azusa_rag/search_hybrid.py` — `hybrid_merge()` 替换为 `rrf_fusion()`。

|  | 旧 | 新 |
|---|---|---|
| 策略 | dense-first, sparse 追补 | Reciprocal Rank Fusion (k=60) |
| 函数签名 | `hybrid_merge(dense_ids, sparse_ids, top_k)` | `rrf_fusion(*ranked_lists, top_k, k=60)` |
| 多路支持 | 固定 dense + sparse | 可变参数，任意多路 |

### RRF 原理

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

- k=60（TREC 等 IR 实验常用值）
- 只用 rank，天然跨检索器可比
- dense (cosine) 和 sparse (BM25) 的原始分数分布完全不同，归一化不靠谱；RRF 绕开了这个问题
- 同分时按 id 升序 tie-break，保证确定性

### 为什么换

旧 merge 逻辑 `dense 优先 → sparse 追补` 的问题：
- sparse 的正确结果在 dense 全错时会被排在所有 dense 结果后面
- top-k 截断直接丢弃 sparse 里的正确答案
- hybrid 指标退化到和 HNSW 一样（avg_hit@k=0.9333, avg_recall@k=0.9, avg_mrr=0.9）

### Eval 对比

- eval_path: `rag/data/eval/min_eval.jsonl`
- query_count: `15`
- top_k: `3`
- 服务: `cargo run --bin azusa_server`
- 评测命令: `python run.py eval --mode all --top-k 3`

#### 核心指标

| mode | 指标 | 旧 merge (6/24) | RRF (7/10) | 变化 |
|---|---|---|---|---|
| hybrid | avg_hit@k | 0.9333 | **1.0000** | +0.0667 |
| hybrid | avg_recall@k | 0.9000 | **1.0000** | +0.1000 |
| hybrid | avg_mrr | 0.9000 | **1.0000** | +0.1000 |
| hybrid_rerank | avg_hit@k | 1.0000 | 1.0000 | — |
| hybrid_rerank | avg_recall@k | 1.0000 | 1.0000 | — |
| hybrid_rerank | avg_mrr | 1.0000 | 1.0000 | — |
| brute/hnsw/bm25 | 全部 | 不变 | 不变 | — |

#### 修复的关键 case

**`min_eval_query_13`** — Gothic Revival 问题：
- Query: "Which artist who had a major influence on the Gothic Revival is represented in the V&A's British galleries?"
- HNSW (dense) 全错，返回了不相关的 chunk
- BM25 (sparse) top-1 命中 gold chunk
- 旧 merge: dense 的 3 个错误结果排前面，sparse 正确答案排第 4，被 top-3 截断 → **hit=0**
- RRF: BM25 rank-1 贡献 1/61，dense 没贡献，gold chunk 进入 top-3 → **hit=1**

**`min_eval_query_3`** — JR 常磐缓行线：
- `cat_notes_3_chunk_002` 在 BM25 中排第 2，旧 merge 未补入 → recall=0.5
- RRF 补入 top-3 → **recall=1.0**

**`min_eval_query_7`** — 登机牌段落：
- gold chunk 在 BM25 rank-1，旧 merge 被 dense 结果压到第 2 → MRR=0.5
- RRF 排到第 1 → **MRR=1.0**

#### 延迟

hybrid 延迟无显著变化：旧 130.1ms → 新 132.1ms（+1.5%）。

#### 面试叙事

> "在改成 RRF 之前，hybrid 的指标和 HNSW 一模一样——本质上是 HNSW 的慢速版。原因在于融合策略是 dense 优先、sparse 追补。当 dense 全错时，sparse 里的正确答案被 top-k 截断丢弃。改成 RRF 之后，两路检索器的 rank 被平等对待，hybrid 从『最差 dense 模式』直接追平了 hybrid_rerank 的满分指标，而延迟只有 rerank 管线的 1/3。这说明融合策略往往比加更多检索器更重要。"

---

## 2026-06-24 — Stage 2 评测收口

以下为 `rag/docs/archive/eval_results_2026-06-24.md` 原始内容。

> generated_at: `2026-06-24T16:43:52+08:00`
> workflow_name: `stage2_final`
> eval_path: `F:\Repositories\Azusa\rag\data\eval\min_eval.jsonl`
> query_count: `15`
> top_k: `3`

### Reproduce

在本机第一个 PowerShell 窗口启动 Azusa server：

```powershell
cd F:\Repositories\Azusa
cargo run --bin azusa_server
```

在第二个 PowerShell 窗口运行最终 Stage 2 workflow：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run_workflow.py --output-name stage2_final --answer-limit 15 --publish-stage2-docs
```

### Steps

- `rebuild_artifacts`: status=`skipped` requested=`False`
- `import_hf_eval`: status=`skipped` requested=`False`

### Mode Guide

| mode | family | label | notes |
| --- | --- | --- | --- |
| brute | dense-only | Brute Force Dense | 直接用向量相似度暴力搜索, 作为最朴素的 dense baseline. |
| hnsw | dense-only | HNSW Dense | 通过 Rust HNSW 服务执行近似向量检索, 用来观察 ANN 近似检索是否保持效果. |
| bm25 | sparse-only | BM25 Sparse | 只依赖词面匹配的 BM25 baseline, 适合观察专有名词、年份和字面命中收益. |
| hybrid | dense+sparse | Hybrid | 先召回 dense 和 BM25 sparse, 再按当前 merge 策略合并去重. |
| hybrid_rerank | dense+sparse+rerank | Hybrid Rerank | 在 hybrid 候选集基础上再做 rerank, 用来观察排序纠偏收益. |

### Metric Ranking

| mode | family | avg_hit@k | avg_recall@k | avg_mrr | avg latency ms | p95 latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid_rerank | dense+sparse+rerank | 1.0000 | 1.0000 | 1.0000 | 462.211 | 528.411 |
| brute | dense-only | 1.0000 | 0.9667 | 0.9667 | 118.729 | 136.467 |
| bm25 | sparse-only | 1.0000 | 0.9667 | 0.9667 | 9.225 | 10.047 |
| hnsw | dense-only | 0.9333 | 0.9000 | 0.9000 | 118.289 | 130.827 |
| hybrid | dense+sparse | 0.9333 | 0.9000 | 0.9000 | 130.140 | 149.795 |

### Delta Vs Baseline

- baseline_mode: `brute`

| mode | hit delta | recall delta | mrr delta |
| --- | --- | --- | --- |
| brute | 0.0000 | 0.0000 | 0.0000 |
| hnsw | -0.0667 | -0.0667 | -0.0667 |
| bm25 | 0.0000 | 0.0000 | 0.0000 |
| hybrid | -0.0667 | -0.0667 | -0.0667 |
| hybrid_rerank | 0.0000 | 0.0333 | 0.0333 |

### Most Divergent Cases

| query_id | hit spread | recall spread | mrr spread |
| --- | --- | --- | --- |
| min_eval_query_13 | 1.0000 | 1.0000 | 1.0000 |
| min_eval_query_11 | 0.0000 | 0.0000 | 0.5000 |
| min_eval_query_7 | 0.0000 | 0.0000 | 0.5000 |
| min_eval_query_14 | 0.0000 | 0.5000 | 0.0000 |
| min_eval_query_3 | 0.0000 | 0.5000 | 0.0000 |

#### min_eval_query_13

- query: Which artist who had a major influence on the Gothic Revival is represented in the V&A's British galleries?

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_000', 'autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_001'] |
| hnsw | 0.0000 | 0.0000 | 0.0000 | ['autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_004', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_000', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_003'] |
| bm25 | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000'] |
| hybrid | 0.0000 | 0.0000 | 0.0000 | ['autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_004', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_000', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_003'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000', 'cat_notes_3_chunk_007'] |

#### min_eval_query_11

- query: In what year did Massachusetts first require children to be educated in schools?

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_002'] |
| hnsw | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_002'] |
| bm25 | 1.0000 | 1.0000 | 0.5000 | ['autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0014_57275f6ef1498d1400e8f709_chunk_001'] |
| hybrid | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_002'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_007'] |

#### min_eval_query_7

- query: 放在可以看见的地方吧
这样大家都会更尊重登机牌

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 0.5000 | ['cat_notes_2_chunk_005', 'cat_notes_2_chunk_004', 'cat_notes_2_chunk_001'] |
| hnsw | 1.0000 | 1.0000 | 0.5000 | ['cat_notes_2_chunk_005', 'cat_notes_2_chunk_004', 'cat_notes_2_chunk_001'] |
| bm25 | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_2_chunk_004', 'cat_notes_2_chunk_001', 'cat_notes_3_chunk_002'] |
| hybrid | 1.0000 | 1.0000 | 0.5000 | ['cat_notes_2_chunk_005', 'cat_notes_2_chunk_004', 'cat_notes_2_chunk_001'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_2_chunk_004', 'cat_notes_2_chunk_000', 'cat_notes_2_chunk_002'] |

#### min_eval_query_14

- query: In the 1890s, which regional colleges and universities did the University of Chicago affiliate with?

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001'] |
| hnsw | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001'] |
| bm25 | 1.0000 | 0.5000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0013_572867543acd2414000df9a1_chunk_000'] |
| hybrid | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_005'] |

#### min_eval_query_3

- query: JR常磐缓行线
常规意义上的直通运转

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 0.5000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_001', 'cat_notes_3_chunk_005'] |
| hnsw | 1.0000 | 0.5000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_001', 'cat_notes_3_chunk_005'] |
| bm25 | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_002', 'cat_notes_3_chunk_001'] |
| hybrid | 1.0000 | 0.5000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_001', 'cat_notes_3_chunk_005'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_002', 'cat_notes_3_chunk_001'] |

### All-Fail Cases

- 没有出现所有 mode 全部 miss 的 case。

### Answer Checks

- answer_mode: `hybrid_rerank`
- manual_review_coverage: `15/15`

| answer metric | value |
| --- | ---: |
| answer accuracy | 0.4000 |
| citation support rate | 0.6667 |
| avg completeness (0-2) | 0.9333 |
| avg faithfulness (0-2) | 1.6667 |
| avg citation-ID precision | 0.8000 |
| avg citation-ID recall | 0.8000 |
| avg answer latency ms | 2723.387 |

- badcase_type_counts: `{'incomplete_answer': 7, 'partial_answer': 1, 'over_citation': 1, 'wrong_answer': 1, 'wrong_citation': 2, 'wrong_entity': 1}`

#### min_eval_query_1

- query: 有时只需要一本足够厚的书
欢迎加入正确座位秩序共同体
- reference_answer: 真正的边界不需要大声，有时只需要一本足够厚的书；14C 乘客随后被欢迎加入正确座位秩序共同体。
- notes: 同一主题句跨越相邻两个 chunk，适合验证 chunk 级召回是否稳定。
- answer: 答案：教宗猫
引用：cat_notes_1_chunk_001, cat_notes_1_chunk_002, cat_notes_2_chunk_004
- citations: ['cat_notes_1_chunk_001', 'cat_notes_1_chunk_002', 'cat_notes_2_chunk_004']
- citation_id_precision/recall: 0.6667/1.0000
- answer_latency_ms: 1517.550
- manual_review: {'query_id': 'min_eval_query_1', 'reviewed_answer': '答案：教宗猫\n引用：cat_notes_1_chunk_001, cat_notes_1_chunk_002, cat_notes_2_chunk_004', 'answer_correct': False, 'citation_supported': True, 'completeness_score': 0, 'faithfulness_score': 2, 'badcase_type': 'incomplete_answer', 'review_notes': '答案只给出角色名，没有回答两段语义；引用包含金标 chunk，但还回填了无关的 cat_notes_2_chunk_004。'}
- top1: cat_notes_1_chunk_001

#### min_eval_query_2

- query: 人生有时需要击中重点
教宗猫以极高礼貌完成座位确认
- reference_answer: 教宗猫用极高的礼貌处理占座问题，并说棒球棒提醒人们人生有时需要击中重点。
- notes: 两句都落在同一个 chunk，适合做最小精确命中测试。
- answer: 答案：极高礼貌
引用：cat_notes_2_chunk_002
- citations: ['cat_notes_2_chunk_002']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 2127.735
- manual_review: {'query_id': 'min_eval_query_2', 'reviewed_answer': '答案：极高礼貌\n引用：cat_notes_2_chunk_002', 'answer_correct': False, 'citation_supported': True, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'incomplete_answer', 'review_notes': '答案抓住礼貌这一关键词，但没有说明占座处理和击中重点的含义。'}
- top1: cat_notes_2_chunk_002

#### min_eval_query_3

- query: JR常磐缓行线
常规意义上的直通运转
- reference_answer: JR 常磐缓行线与东京地铁千代田线直通；JR 与都营地下铁之间常规意义上的直通运转非常少见。
- notes: query 同时覆盖 JR-地铁直通正例与都营不常规直通说明。
- answer: 答案：常规意义上的直通运转
引用：cat_notes_3_chunk_002
- citations: ['cat_notes_3_chunk_002']
- citation_id_precision/recall: 1.0000/0.5000
- answer_latency_ms: 2092.966
- manual_review: {'query_id': 'min_eval_query_3', 'reviewed_answer': '答案：常规意义上的直通运转\n引用：cat_notes_3_chunk_002', 'answer_correct': False, 'citation_supported': True, 'completeness_score': 0, 'faithfulness_score': 2, 'badcase_type': 'incomplete_answer', 'review_notes': '只复述问题短语，没有回答常磐缓行线与千代田线直通，也漏掉第一条支撑证据。'}
- top1: cat_notes_3_chunk_000

#### min_eval_query_4

- query: 精神上，当然是精神上
懂事是好事
- reference_answer: 旁边乘客猫表示精神上理解，其他乘客也称他懂事，并认为懂事是好事。
- notes: 验证第一段客舱秩序场景能被精确召回。
- answer: 答案：懂事是好事
引用：cat_notes_1_chunk_000
- citations: ['cat_notes_1_chunk_000']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 1467.070
- manual_review: {'query_id': 'min_eval_query_4', 'reviewed_answer': '答案：懂事是好事\n引用：cat_notes_1_chunk_000', 'answer_correct': True, 'citation_supported': True, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'partial_answer', 'review_notes': '核心结论正确且引用正确，但没有完整复述场景关系。'}
- top1: cat_notes_1_chunk_000

#### min_eval_query_5

- query: 很多觉醒都始于一个问题
也可能始于一个答案
- reference_answer: 教宗猫说很多觉醒始于一个问题，也可能始于一个答案。
- notes: 验证短尾金句型 query 能命中结尾 chunk。
- answer: 答案：很多觉醒都始于一个问题。
引用：cat_notes_1_chunk_002
- citations: ['cat_notes_1_chunk_002']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 1663.863
- manual_review: {'query_id': 'min_eval_query_5', 'reviewed_answer': '答案：很多觉醒都始于一个问题。\n引用：cat_notes_1_chunk_002', 'answer_correct': False, 'citation_supported': True, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'incomplete_answer', 'review_notes': '漏掉了也可能始于一个答案这一后半句。'}
- top1: cat_notes_1_chunk_002

#### min_eval_query_6

- query: 您的座位是 17A
主也为您预备了窗景
- reference_answer: 空乘确认西装猫的座位是靠窗的 17A，教宗猫说主也为他预备了窗景。
- notes: 验证中段座位纠纷处理结果的精确命中。
- answer: 答案：17A
引用：cat_notes_2_chunk_003, cat_notes_2_chunk_005, cat_notes_1_chunk_000
- citations: ['cat_notes_2_chunk_003', 'cat_notes_2_chunk_005', 'cat_notes_1_chunk_000']
- citation_id_precision/recall: 0.3333/1.0000
- answer_latency_ms: 1594.401
- manual_review: {'query_id': 'min_eval_query_6', 'reviewed_answer': '答案：17A\n引用：cat_notes_2_chunk_003, cat_notes_2_chunk_005, cat_notes_1_chunk_000', 'answer_correct': True, 'citation_supported': True, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'over_citation', 'review_notes': '座位号正确，但漏掉靠窗和窗景信息，并回填了两条不必要引用。'}
- top1: cat_notes_2_chunk_003

#### min_eval_query_7

- query: 放在可以看见的地方吧
这样大家都会更尊重登机牌
- reference_answer: 教宗猫让棒球棒放在大家看得见的地方，因为这样大家会更尊重登机牌。
- notes: 验证棒球棒存在感与登机牌秩序段落。
- answer: 答案：放得当心
引用：cat_notes_2_chunk_004
- citations: ['cat_notes_2_chunk_004']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 1655.721
- manual_review: {'query_id': 'min_eval_query_7', 'reviewed_answer': '答案：放得当心\n引用：cat_notes_2_chunk_004', 'answer_correct': False, 'citation_supported': False, 'completeness_score': 0, 'faithfulness_score': 0, 'badcase_type': 'wrong_answer', 'review_notes': '答案在证据中没有对应表达，也没有回答为什么要放在看得见的地方。'}
- top1: cat_notes_2_chunk_004

#### min_eval_query_8

- query: 第一，不要占座
17A 的窗景其实也不错
- reference_answer: 西装猫总结不要占座，并发现 17A 的窗景其实也不错。
- notes: 验证降落前总结段落的精确召回。
- answer: 答案：第一，不要占座。
引用：cat_notes_2_chunk_003
- citations: ['cat_notes_2_chunk_003']
- citation_id_precision/recall: 0.0000/0.0000
- answer_latency_ms: 2527.651
- manual_review: {'query_id': 'min_eval_query_8', 'reviewed_answer': '答案：第一，不要占座。\n引用：cat_notes_2_chunk_003', 'answer_correct': True, 'citation_supported': False, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'wrong_citation', 'review_notes': '答案主体正确但不完整；所引 chunk 不包含该总结，正确证据是 cat_notes_2_chunk_005。'}
- top1: cat_notes_2_chunk_005

#### min_eval_query_9

- query: 也愿登机牌与你同在
座位号也很重要咪
- reference_answer: 空乘猫回应愿登机牌与你同在；故事总结爱与宽恕重要，但座位号也很重要。
- notes: 验证尾声短 chunk 在 top-k 中可被稳定命中。
- answer: 答案：也愿登机牌与你同在
引用：cat_notes_2_chunk_006
- citations: ['cat_notes_2_chunk_006']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 2059.659
- manual_review: {'query_id': 'min_eval_query_9', 'reviewed_answer': '答案：也愿登机牌与你同在\n引用：cat_notes_2_chunk_006', 'answer_correct': False, 'citation_supported': True, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'incomplete_answer', 'review_notes': '引用正确，但漏掉座位号也很重要这一总结。'}
- top1: cat_notes_2_chunk_006

#### min_eval_query_10

- query: JR埼京线 ↔ 临海线
它不是东京地铁
- reference_answer: JR 埼京线与临海线相互直通，但临海线由东京临海高速铁道运营，不属于东京地铁。
- notes: 验证 JR 与临海线直通但不属于东京地铁的说明。
- answer: 答案：JR埼京线 ↔ 临海线
引用：cat_notes_3_chunk_003
- citations: ['cat_notes_3_chunk_003']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 2039.493
- manual_review: {'query_id': 'min_eval_query_10', 'reviewed_answer': '答案：JR埼京线 ↔ 临海线\n引用：cat_notes_3_chunk_003', 'answer_correct': False, 'citation_supported': True, 'completeness_score': 1, 'faithfulness_score': 2, 'badcase_type': 'incomplete_answer', 'review_notes': '只给出线路关系，漏掉临海线运营方以及它不是东京地铁的关键辨析。'}
- top1: cat_notes_3_chunk_003

#### min_eval_query_11

- query: In what year did Massachusetts first require children to be educated in schools?
- reference_answer: 1852
- notes: SQuAD fact lookup; title=Private_school.
- answer: 答案：1852年
引用：autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000
- citations: ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 4779.590
- manual_review: {'query_id': 'min_eval_query_11', 'reviewed_answer': '答案：1852年\n引用：autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'answer_correct': True, 'citation_supported': True, 'completeness_score': 2, 'faithfulness_score': 2, 'badcase_type': None, 'review_notes': '答案与引用均正确。'}
- top1: autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000

#### min_eval_query_12

- query: When were stromules discovered?
- reference_answer: 1962
- notes: SQuAD fact lookup; title=Chloroplast.
- answer: 答案：stromules were discovered in 1962.
引用：autoevaluate_squad_sample_validation_0001_57296de03f37b3190047839e_chunk_002
- citations: ['autoevaluate_squad_sample_validation_0001_57296de03f37b3190047839e_chunk_002']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 4990.621
- manual_review: {'query_id': 'min_eval_query_12', 'reviewed_answer': '答案：stromules were discovered in 1962.\n引用：autoevaluate_squad_sample_validation_0001_57296de03f37b3190047839e_chunk_002', 'answer_correct': True, 'citation_supported': True, 'completeness_score': 2, 'faithfulness_score': 2, 'badcase_type': None, 'review_notes': '答案与引用均正确。'}
- top1: autoevaluate_squad_sample_validation_0001_57296de03f37b3190047839e_chunk_002

#### min_eval_query_13

- query: Which artist who had a major influence on the Gothic Revival is represented in the V&A's British galleries?
- reference_answer: Horace Walpole
- notes: SQuAD entity lookup; title=Victoria_and_Albert_Museum.
- answer: 答案：Thomas Chippendale
引用：autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002
- citations: ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002']
- citation_id_precision/recall: 1.0000/1.0000
- answer_latency_ms: 1528.476
- manual_review: {'query_id': 'min_eval_query_13', 'reviewed_answer': '答案：Thomas Chippendale\n引用：autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'answer_correct': False, 'citation_supported': False, 'completeness_score': 0, 'faithfulness_score': 0, 'badcase_type': 'wrong_entity', 'review_notes': '正确答案是 Horace Walpole；模型从同一 chunk 中选了另一个人物。'}
- top1: autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002

#### min_eval_query_14

- query: In the 1890s, which regional colleges and universities did the University of Chicago affiliate with?
- reference_answer: Des Moines College, Kalamazoo College, Butler University, Stetson University, and later Shimer College.
- notes: Reworded SQuAD case to match the passage's 1890s wording and require both adjacent chunks.
- answer: 答案：University of Chicago, Des Moines College, Kalamazoo College, Butler University, Stetson University
引用：autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000
- citations: ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000']
- citation_id_precision/recall: 1.0000/0.5000
- answer_latency_ms: 6632.066
- manual_review: {'query_id': 'min_eval_query_14', 'reviewed_answer': '答案：University of Chicago, Des Moines College, Kalamazoo College, Butler University, Stetson University\n引用：autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'answer_correct': False, 'citation_supported': False, 'completeness_score': 1, 'faithfulness_score': 1, 'badcase_type': 'incomplete_answer', 'review_notes': '错误地把 University of Chicago 列入附属院校，并漏掉后续的 Shimer College 与第二条证据。'}
- top1: autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000

#### min_eval_query_15

- query: Who scored the touchdown that cut the score to 10-7?
- reference_answer: Jonathan Stewart
- notes: SQuAD fact lookup; title=Super_Bowl_50.
- answer: 答案：Jonathan Stewart
引用：autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000
- citations: ['autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000']
- citation_id_precision/recall: 0.0000/0.0000
- answer_latency_ms: 4173.942
- manual_review: {'query_id': 'min_eval_query_15', 'reviewed_answer': '答案：Jonathan Stewart\n引用：autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000', 'answer_correct': True, 'citation_supported': False, 'completeness_score': 2, 'faithfulness_score': 2, 'badcase_type': 'wrong_citation', 'review_notes': '答案正确，但引用指向 halftime show 语料；正确证据是 autoevaluate_squad_sample_validation_0004_56d729180d65d21400198427_chunk_000。'}
- top1: autoevaluate_squad_sample_validation_0004_56d729180d65d21400198427_chunk_000

---

## 2026-05-13 — HF Eval Import + Hybrid Rerank + Chunk Coverage Fix

以下为 `rag/docs/archive/commit_report_2026-05-13_hf_eval_rerank.md` 原始内容。

### 背景

这次提交围绕 `rag/` 子项目做了三组互相关联的增强:

1. 接入一个可快速扩展测试范围的 Hugging Face 小型评测数据入口。
2. 为 `hybrid_rerank` 模式接入模型 reranker, 并保留规则 fallback。
3. 修复 chunker 在自然边界截断后可能跳过文本区间的问题。

这三部分合在一起, 让当前最小 RAG 链路不再只依赖手写的少量样本, 也让 hybrid rerank 模式在真实问答样本上更容易观察效果。

### 主要改动

#### 1. 新增 Hugging Face 样本导入器

新增文件:

- `rag/src/azusa_rag/hf_dataset_import.py`
- `rag/run_import_hf_eval.py`

实现内容:

- 通过 Hugging Face `datasets-server` 的 rows API 拉取 SQuAD 风格小样本。
- 默认数据集为 `autoevaluate/squad-sample`。
- 自动生成:
  - `data/raw/*.txt`
  - `data/eval/hf_squad_sample_eval.jsonl`
  - 少量 `data/queries/*.txt`
- 导入时会基于当前 chunk 规则, 把答案字符区间映射成 `relevant_chunk_ids`, 从而直接兼容现有 `run_eval.py`。

设计原因:

- 不依赖本地 `datasets` 包, 减少环境要求。
- 保持现有 `raw txt -> chunk -> embedding -> search/eval` 主链路不变。
- 先从小样本做起, 便于快速 smoke test 和 bad case 观察。

#### 2. 接入最小 reranker 能力

新增文件:

- `rag/src/azusa_rag/reranker.py`

改动文件:

- `rag/src/azusa_rag/search.py`
- `rag/src/azusa_rag/config.py`

实现内容:

- 新增延迟加载的 reranker 模块:
  - 按配置加载 `AutoTokenizer`
  - 按配置加载 `AutoModelForSequenceClassification`
  - 支持 `cuda -> cpu` 回退
  - 按 batch 对 `(query, passage)` 对打分
- 在 `search.py` 的 rerank 逻辑中:
  - 优先使用模型 reranker 分数重排
  - 若模型不可用 / 推理失败 / 返回数量不匹配, 自动 fallback 到规则打分
- `hybrid_rerank` 的候选集合从原先较小范围扩大到:
  - `candidate_k = min(len(embedding_records), max(20, top_k * 5))`

设计原因:

- 让 `hybrid_rerank` 不再只是纯规则重排。
- 同时保留 fallback, 保证环境不完整时功能仍可运行。
- 扩大候选集合后, reranker 才有机会把原本 dense/sparse 排位靠后的正确项提上来。

#### 3. 修复 chunk 覆盖 bug, 并暴露 span 信息

改动文件:

- `rag/src/azusa_rag/chunker.py`

实现内容:

- 拆出 `_validate_chunk_params(...)`
- 拆出 `_strip_span(...)`
- 新增 `chunk_text_with_spans(...)`
- 修正 chunk 推进逻辑:
  - 旧逻辑: 命中自然边界后仍按固定 `step` 推进
  - 新逻辑: 从 `end - overlap_size` 继续推进
- `chunk_document(...)` 改为复用带 span 的切块结果

关键问题:

- 旧逻辑在自然边界收缩 chunk 终点后, 可能导致下一块从更靠后的位置开始, 从而跳过中间一小段文本。
- 这会直接影响:
  - 检索覆盖率
  - eval gold 映射
  - reranker 候选质量

修复效果:

- chunk 之间不再因为边界收缩产生"漏切区间"。
- HF 样本中的答案字符区间可以稳定映射到对应 chunk。

#### 4. 文档与忽略规则更新

改动文件:

- `rag/README.md`
- `rag/.gitignore`

实现内容:

- README 补充 HF 导入脚本的使用方式与输出说明。
- `rag/.gitignore` 增加 `*.egg-info/`, 避免把打包生成物里的 `*.txt` 错误纳入版本控制。

### 注释补强

这次对新加代码补了统一风格的说明:

- 为新函数补充了 `"""功能注释"""`。
- 为关键逻辑补了少量行内注释, 重点解释:
  - 为什么保留 span
  - 为什么扩候选再 rerank
  - 为什么要做 reranker fallback
  - 为什么 hybrid 也要先确保 HNSW 索引已加载

涉及文件:

- `rag/src/azusa_rag/chunker.py`
- `rag/src/azusa_rag/hf_dataset_import.py`
- `rag/src/azusa_rag/reranker.py`
- `rag/src/azusa_rag/search.py`
- `rag/run_import_hf_eval.py`

### 验证记录

本地完成过的验证:

1. CLI / 环境检查
   - 确认当前环境可用 `hf`
   - 确认 `ChuoEnv` 中可执行 `hf --version`

2. 语法检查
   - `python -m py_compile` 校验通过:
     - `chunker.py`
     - `hf_dataset_import.py`
     - `run_import_hf_eval.py`
     - `reranker.py`
     - `search.py`

3. Hugging Face 导入验证
   - 实际运行 `python run_import_hf_eval.py --length 2`
   - 成功导入 2 条 `autoevaluate/squad-sample` 样本
   - 成功生成 `hf_squad_sample_eval.jsonl`

4. bug 复现与修复验证
   - 对失败样本检查了 answer span 和 chunk span
   - 确认修复前存在覆盖缺口
   - 修复后确认答案 `1962` 可以命中正确 chunk

### 当前注意事项

1. `rag/src/azusa_rag/config.py` 当前使用的是本地绝对模型路径:
   - `D:\Chuo_Data\Models\Qwen3-Embedding-0.6B`
   - `D:\Chuo_Data\Models\bge-reranker-v2-m3`

这适合当前机器直接跑, 但换机器时需要手动调整。

2. 由于 chunk 逻辑已变化, 旧的:
   - `output/artifacts/chunks.jsonl`
   - `output/artifacts/embedding.jsonl`

  最好重新生成, 否则可能与新的 eval gold 或 rerank 预期不一致。

3. HF 导入脚本当前面向 SQuAD 风格数据结构:
   - `question`
   - `context`
   - `answers.answer_start`
   - `answers.text`

如果未来换别的数据集, 可能需要补一层格式适配。

### 建议后续工作

1. 增加一个"一键导入 -> 重建 chunk -> 重建 embedding -> 评测"的整链脚本。
2. 把本地绝对模型路径逐步收敛到环境变量或配置文件覆盖机制。
3. 在 `run_eval.py` 中增加不同 mode 的结果对比输出, 方便直接比较 rerank 是否带来收益。

---

## 2026-05 前后 — DuRetrieval Plugin 设计草案

以下为 `rag/docs/archive/du_retrieval_plugin.md` 原始内容。

### 0. 结论

`DuRetrieval` 第一版先按 passage-level eval 接入 Azusa。

核心对应关系：

```json
{
  "query_id": "问题身份证",
  "query": "问题正文",
  "relevant_chunk_ids": ["正确段落身份证1", "正确段落身份证2"]
}
```

数据集字段可以先按这套人话理解：

- `corpus.id` = 段落身份证
- `corpus.text` = 段落正文
- `queries.id` = 问题身份证
- `queries.text` = 问题正文
- `qrels.qid` = 哪个问题
- `qrels.pid` = 哪个段落是正确答案
- `qrels.score` = 相关性分数，第一版先把 `score > 0` 当作相关

第一版不额外切 chunk。也就是说，`relevant_chunk_ids` 这个字段名暂时沿用 Azusa 现有 eval 格式，但里面装的是 DuRetrieval 的 passage id。

### 1. 目标

把 `C-MTEB/DuRetrieval` 接成 Azusa 的第一套正式中文 retrieval benchmark。

这一步的重点是先让公开数据集进入现有评测链路：

- 能生成 Azusa 可读的 corpus 产物
- 能生成 Azusa 可读的 eval jsonl
- 能用 `run_eval.py` 比较 `brute / hnsw / bm25 / hybrid / hybrid_rerank`

### 2. 输入

第一版 importer 需要读三份数据：

- `C-MTEB/DuRetrieval` 的 `corpus`
- `C-MTEB/DuRetrieval` 的 `queries`
- `C-MTEB/DuRetrieval-qrels` 的 `dev`

字段映射：

| 来源 | 字段 | Azusa 第一版含义 |
| --- | --- | --- |
| `corpus` | `id` | retrieval unit id |
| `corpus` | `text` | retrieval unit text |
| `queries` | `id` | `query_id` |
| `queries` | `text` | `query` |
| `qrels` | `qid` | 对齐 `query_id` |
| `qrels` | `pid` | 进入 `relevant_chunk_ids` |
| `qrels` | `score` | `score > 0` 视为 relevant |

### 3. 输出

第一版 importer 先产出三类文件。

#### 3.1 Corpus 产物

```text
rag/output/artifacts/du_retrieval_passages.jsonl
```

每行表示一个 passage-level retrieval unit：

```json
{
  "doc_id": "du_retrieval",
  "chunk_id": "corpus.id",
  "chunk_index": 0,
  "source": "C-MTEB/DuRetrieval",
  "page": null,
  "text": "corpus.text"
}
```

#### 3.2 Eval 产物

```text
rag/data/eval/du_retrieval_eval.jsonl
```

每行对应一个 query：

```json
{
  "query_id": "queries.id",
  "query": "queries.text",
  "relevant_chunk_ids": ["qrels.pid"],
  "notes": "dataset=C-MTEB/DuRetrieval; split=dev"
}
```

#### 3.3 Query sample

```text
rag/data/queries/du_retrieval_query_001.txt
```

### 4. 第一版暂不做

- 不做 chunk-level gold 映射
- 不接 answer generation
- 不一次接多个 benchmark
- 不为了 DuRetrieval 重构 search 主链路
- 不先追全量大规模实验，先跑小样本闭环

### 5. 下一步具体要干嘛

#### Step 1: 写 importer 骨架

```
rag/src/azusa_rag/du_retrieval_import.py
rag/run_import_du_retrieval.py
```

#### Step 2: 先做小样本

- corpus limit: `1000`
- query limit: `20`
- qrels 只保留这些 query 对应的相关 pid

#### Step 3: 接入 embedding 流程

#### Step 4: 跑最小 eval

```powershell
python run_eval.py --mode bm25 --eval-path data\eval\du_retrieval_eval.jsonl
python run_eval.py --mode all --eval-path data\eval\du_retrieval_eval.jsonl
```

### 6. 成功标准

- importer 能稳定生成 corpus 和 eval 文件
- eval 文件里的 `query_id / query / relevant_chunk_ids` 正确
- embedding 产物里的 `chunk_id` 能对上 `relevant_chunk_ids`
- `run_eval.py` 能在 DuRetrieval 小样本上跑完
- 至少能得到一份 `eval_result_bm25.json`

---

## 2026-04 — SIFT1M HNSW 性能档案

以下为 `docs/archive/perf_logs.md` 原始内容。

> 这份文件只记录值得留档的实验汇总结果。

### 2026-04-25 — SIFT1M Full 1M HNSW Sweep

Experiment: `SIFT1M full 1M HNSW sweep on official groundtruth`

- base_limit: `1000000`
- query_limit: `100`
- ks: `[1, 5, 10]`
- algorithms: `[Hnsw]`
- sweep order: `m8_efc100` -> `m16_efc100` -> `m16_efc200` -> `m32_efc200`
- metric: `L2`
- command: `cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limits 1000000 --ks 1,5,10 --ef-search 60,100,300 --query-limit 100 --m 8,16,32 --ef-construction 100,200`
- change: 这轮把 benchmark 切到全量 `1,000,000` base，并在服务器上用 `nohup` 分批跑 sweep; 前三组完成，第四组在启动点被服务器硬盘挂载故障中断

**Timeline:**

| time (Asia/Shanghai) | event |
| --- | --- |
| 2026-04-25 19:58:47 | sweep started |
| 2026-04-25 20:19:49 | `m8_efc100` completed |
| 2026-04-25 21:10:48 | `m16_efc100` completed |
| 2026-04-25 22:28 | `m16_efc200` completed; `m32_efc200` started |
| 2026-04-29 | server mount recovered locally, results copied back into the archive |

**Sweep Summary:**

| config | build time | `ef_search=60` recall@10 / exact@10 | `ef_search=100` recall@10 / exact@10 | `ef_search=300` recall@10 / exact@10 |
| --- | ---: | ---: | ---: | ---: |
| `m8_efc100` | `1251.370706978s` | `89.40% / 49/100` | `94.90% / 68/100` | `99.40% / 91/100` |
| `m16_efc100` | `3055.649133404s` | `97.50% / 76/100` | `99.20% / 89/100` | `99.80% / 94/100` |
| `m16_efc200` | `4668.71236162s` | `97.80% / 78/100` | `99.20% / 90/100` | `99.90% / 95/100` |

**Notes:**

- `m=16` 相比 `m=8` 的建图时间明显上升，但 recall 和 exact 也更稳，尤其是在 `ef_search=60` 的低档位下差距最明显。
- `ef_construction=200` 进一步抬高建图成本，`m16_efc200` 的 build time 是三组里最高的一档，但查询质量也达到了当前结果里的最好水平。
- `ef_search=300` 时三组都已经接近饱和，后两组主要改善 exact@10，而 recall@10 只剩小幅提升。
- `m32_efc200` 只写到了启动点，没有产出完整结果；这组结果不纳入汇总表。
- 这次 sweep 的完整原始输出保留在 `target/benchmark/sweeps/sift1m_hnsw_sweep_1m_20260425_195847/`。

### 2026-04-22 — Full 1M Baseline

Experiment: `SIFT1M recall sweep on full 1M base with official groundtruth`

- base_limit: `1000000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- command: `cargo test --release test_hnsw_in_sift1m -- --nocapture`
- change: 将主测试切到官方 `sift_groundtruth.ivecs` 参考答案，并把 `base_limit` 提升到 `1000000`

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 392.8317633s |
| ground truth time | 1.2822ms |
| insert count | 1000000 |

**Insert Timings:**

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 34.18s | 8.8% |
| search_layer | 327.47s | 83.8% |

**Query Results:**

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          25.7881ms |         257.881us |    68.2% |     34% |
|        80 |          26.7733ms |         267.733us |    75.2% |     38% |
|       100 |          29.9425ms |         299.425us |    78.4% |     41% |
|       150 |          39.8614ms |         398.614us |    84.0% |     53% |
|       200 |          48.3276ms |         483.276us |    87.6% |     61% |
|       300 |          66.9292ms |         669.292us |    91.2% |     68% |

**Notes:**

- 这是第一份真正基于全量 `1,000,000` base 的官方 groundtruth 结果，可以作为后续全量优化的正式 baseline。
- 这次 `valid_query_cnt=100`, 说明在全量 base 下官方 groundtruth 的前 `k=5` 都能完整对齐，不再需要跳过 query。
- 相比 `100000` base 的结果, 查询侧曲线更像真实全量 SIFT1M 的负载形态, 后续调参应优先参考这一版。

### 2026-04-22 — L2 mul_add 回归修复

Experiment: `SIFT1M recall sweep for the 2026-04-22 HNSW tuning batch`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- change: 这一轮先后包含 `search_layer` 语义对齐、候选分数复用、L2 内部快路径、层级采样参数化、重复边调试护栏、旧辅助状态清理，最终把 `distance::l2_squared_unchecked` 里的 `mul_add` 改回普通乘加

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 25.0554423s |
| ground truth time | 1.4226337s |
| insert count | 100000 |

**Insert Timings:**

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 1.89s | 7.6% |
| search_layer | 20.64s | 83.0% |

**Query Results:**

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |          12.9318ms |         129.318us |    87.0% |     62% |
|        80 |          18.4962ms |         184.962us |    90.0% |     70% |
|       100 |          22.8890ms |         228.890us |    92.2% |     75% |
|       150 |          33.9553ms |         339.553us |    94.4% |     81% |
|       200 |          40.0978ms |         400.978us |    95.8% |     86% |
|       300 |          52.2015ms |         522.015us |    97.0% |     89% |

**Notes:**

- 这次结果确认了 `mul_add` 在当前平台上是负优化，改回普通乘加后，建图和查询都明显恢复。
- `recall@5` / `exact@5` 保持与前一版一致，说明这是纯性能回归修复，不影响搜索语义。
- `ef_search=300` 的平均查询时间回落到 `522.015us`, 比之前那次 `mul_add` 版本明显更快。
- 这份记录对应的是 4.22 当天整组 HNSW 调整的最终落点，而不是单独一个回退动作。

### 2026-04-21 — Heuristic Neighbor Selection (Full)

Experiment: `SIFT1M recall sweep after enabling heuristic neighbor selection for new-node links and reverse pruning`

- base_limit: `100000`
- k: `5`
- query_limit: `100`
- ef_construction: `80`
- m: `4`
- m_max: `8`
- metric: `L2`
- change: 新节点连边和旧节点反向边超限裁剪均从 simple selection 切到 heuristic neighbor selection

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 26.607152s |
| ground truth time | 1.4883383s |
| insert count | 100000 |

**Insert Timings:**

| phase | time | share |
| --- | ---: | ---: |
| greedy_search | 1.72s | 7.0% |
| search_layer | 18.56s | 75.1% |

**Query Results:**

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --------: | ----------------: | --------------: | -------: | ------: |
|        60 |        18.1198ms |       181.198us |    87.0% |     62% |
|        80 |        21.4482ms |       214.482us |    90.0% |     70% |
|       100 |        25.2872ms |       252.872us |    92.2% |     75% |
|       150 |        42.0259ms |       420.259us |    94.4% |     81% |
|       200 |        48.3072ms |       483.072us |    95.8% |     86% |
|       300 |        70.9235ms |       709.235us |    97.0% |     89% |

**Notes:**

- 和"仅新节点连边 heuristic"版本相比, 全量 heuristic 继续提高质量: `ef_search=100` 的 `recall@5` 从 `88.8%` 到 `92.2%`, `exact@5` 从 `70%` 到 `75%`; `ef_search=300` 的 `recall@5` 从 `96.2%` 到 `97.0%`, `exact@5` 从 `84%` 到 `89%`。
- 建图时间从 `25.227798s` 升到 `26.607152s`, 约慢 `5.5%`; `search_layer` 插入耗时从 `17.91s` 升到 `18.56s`, 约慢 `3.6%`。
- 查询侧中低档本次反而更快, 同时质量更高; 高档 `ef_search=300` 从 `616.848us` 升到 `709.235us`, 说明全量 heuristic 改变了图结构后, 不同搜索宽度下的探索路径成本不同。
- 和同日 workspace 相似度缓存版本相比, `ef_search=300` 的 `recall@5` 从 `90.6%` 提升到 `97.0%`, `exact@5` 从 `70%` 提升到 `89%`, 是当前留档版本里质量最高的一版。

### 2026-04-21 — Heuristic Neighbor Selection (New-Node Only)

Experiment: `SIFT1M recall sweep after enabling heuristic neighbor selection for new-node links`

- base_limit: `100000`, k: `5`, query_limit: `100`, ef_construction: `80`, m: `4`, m_max: `8`, metric: `L2`
- change: 新节点连边从 `select_neighbors_simple` 切到 heuristic neighbor selection; 反向边超限裁剪暂时仍走 simple selection

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 25.227798s |
| insert count | 100000 |

**Query Results (excerpt):**

| ef_search | avg_search_time | recall@5 | exact@5 |
| --------: | --------------: | -------: | ------: |
|       100 |         347.366us |    88.8% |     70% |
|       300 |         616.848us |    96.2% |     84% |

**Notes:**

- `recall@5` 在各档 `ef_search` 下明显提升, 例如 `ef_search=100` 从 `79.6%` 到 `88.8%`, `ef_search=300` 从 `90.6%` 到 `96.2%`。
- 建图时间约慢 `22.5%`; `search_layer` 插入耗时约慢 `25.2%`。主要原因是 heuristic selection 会额外计算候选与已选邻居的相似度, 用更多建图成本换取更分散的边。

### 2026-04-21 — Workspace Epoch 相似度缓存

Experiment: `SIFT1M recall sweep after switching similarity cache to workspace epoch arrays and fixing query boundaries`

- base_limit: `100000`, k: `5`, query_limit: `100`, ef_construction: `80`, m: `4`, m_max: `8`, metric: `L2`
- change: `SearchLayerWorkspace` 增加 `sim_epoch` / `sim_marks` / `sim_values`, 改用 workspace 数组缓存相似度, 并在不同 query 入口调用 `reset_similarity_cache` 隔离缓存 epoch

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 20.5939304s |
| insert count | 100000 |

**Query Results (excerpt):**

| ef_search | avg_search_time | recall@5 | exact@5 |
| --------: | --------------: | -------: | ------: |
|       100 |         255.632us |    79.6% |     47% |
|       300 |         629.796us |    90.6% |     70% |

**Notes:**

- `search_layer` 插入耗时从 `15.56s` 降到 `14.31s`, 约快 `8.0%`。
- 查询侧保守 reset 后没有稳定收益, `ef_search=100` 约慢 `5.7%`, `ef_search=300` 约慢 `9.4%`。
- `recall@5` 与 `exact@5` 恢复到上一版同一水平, 说明此前未隔离 query epoch 导致的语义回退已修复。

### 2026-04-20 — Id → Index 内部表示重构

Experiment: `SIFT1M recall sweep after switching graph internals from Id to index`

- base_limit: `100000`, k: `5`, query_limit: `100`, ef_construction: `80`, m: `4`, m_max: `8`, metric: `L2`
- change: 图内部表示从 `Id` 驱动改为连续 `index` 驱动, 邻接表 / entry point / search_layer / pruning 热路径统一改走 `usize`

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 19.8201135s |
| insert count | 100000 |

**Query Results (excerpt):**

| ef_search | avg_search_time | recall@5 | exact@5 |
| --------: | --------------: | -------: | ------: |
|       100 |         241.834us |    79.6% |     47% |
|       300 |         575.874us |    90.6% |     70% |

**Notes:**

- 建图时间从 `27.93s` 降到 `19.82s`, 约快 `29.0%`。
- `search_layer` 插入耗时从 `22.96s` 降到 `15.56s`, 约快 `32.2%`。
- 查询侧 `ef_search=300` 从 `719.258us` 降到 `575.874us`, 约快 `19.9%`。
- `recall@5` 与 `exact@5` 保持不变, 说明这次重构只改变内部表示, 没有改变搜索语义。

### 2026-04-20 — 去掉 result_set 热路径维护

Experiment: `SIFT1M recall sweep after removing result_set hot-path maintenance`

- base_limit: `100000`, k: `5`, query_limit: `100`, ef_construction: `80`, m: `4`, m_max: `8`, metric: `L2`
- change: `search_layer` 改为仅用 `result_heap` 维护结果集, 去掉 `result_set + lazy deletion` 热路径维护

**Notes:**

- 建图时间从 `30.22s` 降到 `27.93s`, 约快 `7.6%`。
- `search_layer` 插入耗时从 `25.37s` 降到 `22.96s`, 约快 `9.5%`。
- `recall@5` 与 `exact@5` 保持不变, 说明优化没有改变搜索语义。

### 2026-04-20 — 去掉邻居 to_vec()

Experiment: `SIFT1M recall sweep after removing neighbor to_vec`

- change: `search_layer` 保留 workspace/stamp 设计, 去掉邻居 `to_vec()`

**Notes:**

- 和基线相比, 建图时间基本回到同一水平, 查询耗时在各档 `ef_search` 下整体更低。

### 2026-04-19 — 带 Timing 基线

Experiment: `SIFT1M recall sweep with timing`

- base_limit: `100000`, k: `5`, query_limit: `100`, ef_construction: `80`, m: `4`, m_max: `8`, metric: `L2`

**Build Summary:**

| item | value |
| --- | ---: |
| build_index time | 30.1282601s |
| insert count | 100000 |

**Query Results (excerpt):**

| ef_search | avg_search_time | recall@5 | exact@5 |
| --------: | --------------: | -------: | ------: |
|        60 |         216.837us |    72.8% |     35% |
|       100 |         341.943us |    79.6% |     47% |
|       300 |         817.277us |    90.6% |     70% |

**Notes:**

- `search_layer` 占插入阶段主要耗时，约为 `85.3%`。
- `ef_search` 从 `60` 提高到 `300` 时，`recall@5` 从 `72.8%` 提升到 `90.6%`，同时平均查询耗时从 `216.837us` 增长到 `817.277us`。

### 2026-04-18 — 首次 SIFT1M Recall Sweep

Experiment: `SIFT1M recall sweep`

- base_limit: `100000`, k: `5`, query_limit: `100`, ef_construction: `80`, m: `4`, m_max: `8`, metric: `L2`

| ef_search | recall@5 | exact@5 |
| --------: | -------: | ------: |
|        60 |    72.8% |     35% |
|        80 |    77.0% |     44% |
|       100 |    79.6% |     47% |
|       150 |    83.2% |     52% |
|       200 |    86.8% |     60% |
|       300 |    90.6% |     70% |

### 100k Overview（调优汇总）

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

---

## 2026-04 — Stage 1 收束

以下为 `docs/archive/stage1_summary.md` 原始内容。

Stage 1 的目标是把 Azusa 从一个可运行的 HNSW 原型，整理成一个可以验证、演示和说明的独立 artifact。本阶段已经完成，不继续扩展核心功能。

### 已完成内容

- Rust HNSW 索引支持向量记录插入、批量插入、top-k 检索和按 id 取回 metadata。
- SIFT1M benchmark 支持官方 ground truth、Recall@k、Exact@k，以及 CSV/Markdown 输出。
- Rust HTTP 服务提供 `health -> insert -> search -> record` 最小闭环。
- Python/RAG 可以通过 HTTP 调用 Azusa，但 RAG、Hybrid、Rerank 和 Agent 属于后续阶段，不计入 Stage 1 验收。

### 交付物

- 项目说明：`docs/AZUSA_README.md`
- HTTP demo：`docs/azusa_server_readme.md`
- HTTP smoke：`scripts/http_smoke_test.ps1`
- 接口决策：`docs/interface_decision.md`
- Benchmark 说明：`docs/benchmark.md`
- 历史性能记录：`docs/archive/perf_logs.md`
- 简历描述：`docs/archive/resume_bullets.md`

### 2026-06-24 本机复验

```powershell
cargo run --bin azusa_server
powershell -ExecutionPolicy Bypass -File .\scripts\http_smoke_test.ps1
cargo test --test hnsw_api_smoke_test
cargo test sift1m_recall_smoke_test -- --nocapture
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

结果：

| base_limit | query_limit | k | ef_search | Recall@5 | Exact@5 | Avg search time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 10 | 5 | 60 | 98.00% | 9/10 | 116.07 us |

### 阶段边界

Stage 1 到此冻结。后续评测收口属于 Stage 2；Search Agent 与 Tool Calling 属于 Stage 3。除非出现明确回归，不再为 Stage 1 修改 HNSW、HTTP 接口或核心数据结构。

---

## 2026-04 — 接口选型决策

以下为 `docs/archive/interface_decision.md` 原始内容。

### 决策

当前选择：**Rust HTTP 服务**。

服务入口：`src/bin/azusa_server.rs`

默认监听地址：`http://127.0.0.1:3000`

当前最小接口：

```text
GET  /health
POST /insert
POST /search
GET  /record/{id}
```

### 为什么选择 Rust HTTP 服务

- 可以直接复用现有 Rust library 和 `HnswIndex`，不需要额外跨语言绑定层。
- 调试路径简单：`cargo run --bin azusa_server` 启动，`curl` 或脚本即可验证闭环。
- 对 Python/RAG 调用友好：调用方只需要发 HTTP JSON 请求。
- 跨语言边界清晰：后续 Python、前端、RAG pipeline 都可以通过同一组 HTTP 接口接入。
- 适合 demo：可以明确展示「插入 -> 搜索 -> 取回记录」这条最小链路。

### 暂不选择的方案

- **PyO3**：需要额外处理 Python wheel、ABI、环境和打包问题；调试边界会从 Rust 内部扩大到 Python/Rust 绑定层。
- **gRPC**：需要引入 proto 定义、代码生成和客户端生成流程；对 demo 来说调试门槛更高。
- **Python 临时后端**：会让核心索引逻辑和服务层分散到两种运行时里。

### 降级方案

如果 HTTP 服务来不及稳定交付，保留以下独立 artifact 作为降级交付：

- Rust library 中的 `HnswIndex` 插入和搜索能力。
- 现有 smoke test 展示的基础检索路径。
- `sift1m_benchmark` 和 SIFT1M 实验记录，证明索引可构建、可搜索、可评估。

### 当前接口状态

- `AppState`: `Arc<Mutex<HnswIndex>>`
- 默认参数：`metric=Cosine, level_sample_probability=0.5, ef_construction=8, m=2, m_max=4`
- 这些参数优先服务 demo 和 smoke test，不代表生产配置。

### 当前限制

- 数据只保存在内存里，服务重启后记录丢失。
- 暂无持久化、删除、更新、批量导入和并发压测。
- 错误处理只覆盖最小场景：重复 id、缺失 record、基础 JSON 解析错误。
- API schema 还没有单独发布 OpenAPI 或 protobuf 定义。
