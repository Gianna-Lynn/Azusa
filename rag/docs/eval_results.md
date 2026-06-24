# Workflow Report

- generated_at: `2026-06-24T16:43:52+08:00`
- workflow_name: `stage2_final`
- eval_path: `F:\Repositories\Azusa\rag\data\eval\min_eval.jsonl`
- query_count: `15`
- top_k: `3`

## Reproduce

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

## Steps

- `rebuild_artifacts`: status=`skipped` requested=`False`
- `import_hf_eval`: status=`skipped` requested=`False`

## Mode Guide

| mode | family | label | notes |
| --- | --- | --- | --- |
| brute | dense-only | Brute Force Dense | 直接用向量相似度暴力搜索, 作为最朴素的 dense baseline. |
| hnsw | dense-only | HNSW Dense | 通过 Rust HNSW 服务执行近似向量检索, 用来观察 ANN 近似检索是否保持效果. |
| bm25 | sparse-only | BM25 Sparse | 只依赖词面匹配的 BM25 baseline, 适合观察专有名词、年份和字面命中收益. |
| hybrid | dense+sparse | Hybrid | 先召回 dense 和 BM25 sparse, 再按当前 merge 策略合并去重. |
| hybrid_rerank | dense+sparse+rerank | Hybrid Rerank | 在 hybrid 候选集基础上再做 rerank, 用来观察排序纠偏收益. |

## Metric Ranking

| mode | family | avg_hit@k | avg_recall@k | avg_mrr | avg latency ms | p95 latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid_rerank | dense+sparse+rerank | 1.0000 | 1.0000 | 1.0000 | 462.211 | 528.411 |
| brute | dense-only | 1.0000 | 0.9667 | 0.9667 | 118.729 | 136.467 |
| bm25 | sparse-only | 1.0000 | 0.9667 | 0.9667 | 9.225 | 10.047 |
| hnsw | dense-only | 0.9333 | 0.9000 | 0.9000 | 118.289 | 130.827 |
| hybrid | dense+sparse | 0.9333 | 0.9000 | 0.9000 | 130.140 | 149.795 |

## Delta Vs Baseline

- baseline_mode: `brute`

| mode | hit delta | recall delta | mrr delta |
| --- | --- | --- | --- |
| brute | 0.0000 | 0.0000 | 0.0000 |
| hnsw | -0.0667 | -0.0667 | -0.0667 |
| bm25 | 0.0000 | 0.0000 | 0.0000 |
| hybrid | -0.0667 | -0.0667 | -0.0667 |
| hybrid_rerank | 0.0000 | 0.0333 | 0.0333 |

## Most Divergent Cases

| query_id | hit spread | recall spread | mrr spread |
| --- | --- | --- | --- |
| min_eval_query_13 | 1.0000 | 1.0000 | 1.0000 |
| min_eval_query_11 | 0.0000 | 0.0000 | 0.5000 |
| min_eval_query_7 | 0.0000 | 0.0000 | 0.5000 |
| min_eval_query_14 | 0.0000 | 0.5000 | 0.0000 |
| min_eval_query_3 | 0.0000 | 0.5000 | 0.0000 |

### min_eval_query_13

- query: Which artist who had a major influence on the Gothic Revival is represented in the V&A's British galleries?

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_000', 'autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_001'] |
| hnsw | 0.0000 | 0.0000 | 0.0000 | ['autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_004', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_000', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_003'] |
| bm25 | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000'] |
| hybrid | 0.0000 | 0.0000 | 0.0000 | ['autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_004', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_000', 'autoevaluate_squad_sample_validation_0017_56f86966aef2371900626053_chunk_003'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002', 'autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000', 'cat_notes_3_chunk_007'] |

### min_eval_query_11

- query: In what year did Massachusetts first require children to be educated in schools?

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_002'] |
| hnsw | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_002'] |
| bm25 | 1.0000 | 1.0000 | 0.5000 | ['autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0014_57275f6ef1498d1400e8f709_chunk_001'] |
| hybrid | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_001', 'autoevaluate_squad_sample_validation_0005_57274beff1498d1400e8f5e6_chunk_002'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0000_572759665951b619008f8884_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_007'] |

### min_eval_query_7

- query: 放在可以看见的地方吧
这样大家都会更尊重登机牌

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 0.5000 | ['cat_notes_2_chunk_005', 'cat_notes_2_chunk_004', 'cat_notes_2_chunk_001'] |
| hnsw | 1.0000 | 1.0000 | 0.5000 | ['cat_notes_2_chunk_005', 'cat_notes_2_chunk_004', 'cat_notes_2_chunk_001'] |
| bm25 | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_2_chunk_004', 'cat_notes_2_chunk_001', 'cat_notes_3_chunk_002'] |
| hybrid | 1.0000 | 1.0000 | 0.5000 | ['cat_notes_2_chunk_005', 'cat_notes_2_chunk_004', 'cat_notes_2_chunk_001'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_2_chunk_004', 'cat_notes_2_chunk_000', 'cat_notes_2_chunk_002'] |

### min_eval_query_14

- query: In the 1890s, which regional colleges and universities did the University of Chicago affiliate with?

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001'] |
| hnsw | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001'] |
| bm25 | 1.0000 | 0.5000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0013_572867543acd2414000df9a1_chunk_000'] |
| hybrid | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_004', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_001', 'autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_005'] |

### min_eval_query_3

- query: JR常磐缓行线
常规意义上的直通运转

| mode | hit@k | recall@k | mrr | pred |
| --- | --- | --- | --- | --- |
| brute | 1.0000 | 0.5000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_001', 'cat_notes_3_chunk_005'] |
| hnsw | 1.0000 | 0.5000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_001', 'cat_notes_3_chunk_005'] |
| bm25 | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_002', 'cat_notes_3_chunk_001'] |
| hybrid | 1.0000 | 0.5000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_001', 'cat_notes_3_chunk_005'] |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | ['cat_notes_3_chunk_000', 'cat_notes_3_chunk_002', 'cat_notes_3_chunk_001'] |

## All-Fail Cases

- 没有出现所有 mode 全部 miss 的 case。

## Answer Checks

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

### min_eval_query_1

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

### min_eval_query_2

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

### min_eval_query_3

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

### min_eval_query_4

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

### min_eval_query_5

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

### min_eval_query_6

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

### min_eval_query_7

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

### min_eval_query_8

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

### min_eval_query_9

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

### min_eval_query_10

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

### min_eval_query_11

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

### min_eval_query_12

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

### min_eval_query_13

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

### min_eval_query_14

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

### min_eval_query_15

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
