# Azusa-F1 Representative Cases

这份文档把 Stage 2 workflow 和 Search Agent v0 的结果整理成可讲 case。目标不是证明 Azusa 性能最好，而是把系统行为、失败模式和改进方向讲清楚。

## Case 1: Successful Fact Lookup

- query_id: `min_eval_query_12`
- query: `When were stromules discovered?`
- reference_answer: `1962`
- representative mode: `bm25` / `hybrid_rerank`

关键现象：

- BM25 top-1 命中 `autoevaluate_squad_sample_validation_0001_57296de03f37b3190047839e_chunk_002`。
- 该 chunk 包含 `first observed in 1962`。
- Agent demo 输出 `答案：stromules were discovered in 1962.`，citation 指向同一个 chunk。
- `citation_id_precision/recall` 在 Stage 2 report 中为 `1.0000/1.0000`。

可以这样讲：

```text
This is a clean lexical fact lookup. Sparse retrieval works well because the query contains a rare entity and asks for a year. The answerer only needs to extract the year from the retrieved evidence, so this case validates the simplest end-to-end path: retrieval hit, evidence is sufficient, answer is faithful, citation is correct.
```

说明价值：

- 展示 `raw txt -> chunk -> retrieval -> answer + citation` 的完整闭环。
- 适合作为 README 或面试中的第一个 happy path。

## Case 2: Rerank Recovers A Miss

- query_id: `min_eval_query_13`
- query: `Which artist who had a major influence on the Gothic Revival is represented in the V&A's British galleries?`
- reference_answer: `Horace Walpole`

关键现象：

| mode | hit@k | recall@k | mrr |
| --- | ---: | ---: | ---: |
| brute | 1.0000 | 1.0000 | 1.0000 |
| hnsw | 0.0000 | 0.0000 | 0.0000 |
| bm25 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 0.0000 | 0.0000 | 0.0000 |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 |

可以这样讲：

```text
This case shows why the project keeps both sparse retrieval and rerank. The HNSW/hybrid candidate order missed the gold chunk, while BM25 and hybrid_rerank recovered it. It is a useful disagreement case because it demonstrates that dense-only retrieval is not always enough for entity-heavy questions, and rerank can change the final top-k behavior.
```

注意事项：

- Stage 2 answer still failed with `wrong_entity`: it answered `Thomas Chippendale` instead of `Horace Walpole`.
- 这说明检索命中不等于答案正确。
- 这个 case 很适合解释 RAG evaluation 为什么要拆成 retrieval metrics 和 answer review。

说明价值：

- 展示 mode comparison 的意义。
- 展示 rerank 的收益。
- 同时展示 answer 层仍可能失败，因此需要 badcase taxonomy。

## Case 3: Correct Answer, Wrong Citation

- query_id: `min_eval_query_15`
- query: `Who scored the touchdown that cut the score to 10-7?`
- reference_answer: `Jonathan Stewart`
- badcase_type: `wrong_citation`

关键现象：

- answer 输出 `Jonathan Stewart`，人工 review 标记 `answer_correct=True`。
- citation 指向 `autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000`。
- review notes 说明该 citation 指向 halftime show 语料，正确证据应是 `autoevaluate_squad_sample_validation_0004_56d729180d65d21400198427_chunk_000`。
- `citation_id_precision/recall` 为 `0.0000/0.0000`。

可以这样讲：

```text
This case separates answer correctness from citation faithfulness. The generated answer is right, but the supporting evidence is wrong. In a RAG system, this is still a serious failure because the system cannot justify its answer from the retrieved context.
```

说明价值：

- 展示为什么不能只看 answer accuracy。
- 展示 citation correctness 是独立指标。
- 支撑 `LLM-assisted review + human spot check` 的评测设计。

## Takeaways

- Retrieval metrics answer whether the system found the right evidence.
- Citation metrics answer whether the cited chunk ids match the expected evidence.
- Human review answers whether the natural-language answer is correct, complete, and faithful.
- Badcase taxonomy tells us what to fix next.

当前 Azusa-F1 的最重要价值不是说系统已经很强，而是它已经能把这些行为记录下来、复现出来、解释出来。
