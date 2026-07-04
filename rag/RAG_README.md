# Azusa RAG

最小可运行的 RAG 实验链路：

- `raw txt -> chunks.jsonl -> embedding.jsonl`
- `brute / hnsw / bm25 / hybrid / hybrid_rerank` 检索
- 最小检索评测：`Hit@k / Recall@k / MRR`
- 最小 `answer + citation` 和 Search Agent v0

## 环境

```powershell
conda activate ChuoEnv
cd F:\Repositories\Azusa\rag
```

`hnsw` / `hybrid` / `hybrid_rerank` 需要先启动 Rust 服务：

```powershell
cd F:\Repositories\Azusa
cargo run --bin azusa_server
```

本地模型路径在 `src/azusa_rag/config.py` 里配置，当前默认：

- embedding: `D:\Chuo_Data\Models\Qwen3-Embedding-0.6B`
- reranker: `D:\Chuo_Data\Models\bge-reranker-v2-m3`
- answerer: `D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct`

## 常用命令

统一入口是 `run.py`：

```powershell
python run.py <command> [args]
```

```powershell
python run.py chunk
python run.py embed
python run.py search --mode brute --top-k 3
python run.py search --mode hnsw --top-k 3
python run.py search --mode hybrid_rerank --top-k 3 --with-answer
python run.py agent --mode hybrid_rerank --top-k 5 --with-answer
python run.py eval --mode all --top-k 3
python run.py workflow
python run.py import-hf-eval --length 20
```

常用参数：

- `--query-text "..."`：直接传 query。
- `--query-file path\to\query.txt`：从文件读 query。
- `--keep-run-copy`：额外保留命名副本；默认只覆盖 `latest/`。

## Search Agent v0

Agent v0 是 RAG pipeline 上层的最小编排入口，当前重点不是构造复杂 Agent 框架，而是把一次检索问答过程记录清楚：

```powershell
python run.py agent --mode bm25 --top-k 5 --with-answer --query-file data\queries\autoevaluate_squad_sample_query_002.txt
python run.py agent --mode hybrid_rerank --top-k 5 --with-answer --query-file data\queries\autoevaluate_squad_sample_query_001.txt
```

`bm25` 不需要 Rust 服务；`hybrid_rerank` 需要先启动 `azusa_server`。运行后会生成：

- `output/search_runs/latest/search_agent_v0_bm25.json`
- `output/search_runs/latest/search_agent_v0_hybrid_rerank.json`

Agent JSON 里最值得先看的字段：

- `query_plan`：query 分类、rewrite 和拆分后的 retrieval queries。
- `hops`：每一跳实际检索到的 chunk。
- `fused_evidence`：跨 hop 去重、融合后的证据列表。
- `fusion.retrieval_sources`：结果来自 dense、sparse、dense + sparse 或 bm25 sparse。
- `answer_citations`：answerer 最终引用的 chunk id。

更多说明见 `docs/agent_tool_demo.md`。

## Evaluation Philosophy

Azusa 的评测目标不是宣称 RAG / Agent 达到 SOTA，而是让检索行为、答案忠实性和失败模式变得可观察。当前评测更接近工程回归集和面试说明材料，不是公开排行榜式 benchmark。

评测分三层理解：

- 检索层尽量客观化：使用 `Hit@k`、`Recall@k`、`MRR`、latency 观察不同 mode 是否找到了 gold chunks。
- 答案层半自动化：使用 `citation_id_precision` / `citation_id_recall` 检查引用 id 是否对齐金标，再用人工 review 标注 `answer_correct`、`citation_supported`、`completeness_score`、`faithfulness_score`。
- 分析层面向改进：用 badcase taxonomy 判断下一步该修 chunk、retrieval、rerank、answer 还是 citation。

因此，workflow 报告里的指标用于回答：

- 不同检索策略行为有什么差异？
- 哪些 case 能说明 hybrid / rerank 的收益或误伤？
- answer + citation 的失败模式是什么？
- 这个系统下一步应该怎么改？

大 LLM 可以辅助语义评价，但不作为唯一考官。最终结论应结合客观检索指标、LLM-assisted review 和人工抽样复核。

## 输出

- `output/artifacts/`
  `chunks.jsonl` 和 `embedding.jsonl`
- `output/search_runs/latest/`
  search / agent 的默认覆盖结果
- `output/eval_runs/latest/`
  eval / workflow 的默认覆盖结果
- `data/import_snapshots/hf_eval/latest/`
  HF 样本导入默认输出
- `data/processed/du_retrieval_workflow/latest/`
  DuRetrieval workflow 默认输出

## 可展示资产

- `docs/badcase_v1.md`：Stage 2 badcase taxonomy。
- `docs/agent_tool_demo.md`：Search Agent v0 demo、输出字段和当前限制。
- `docs/azusa_f1_cases.md`：可在 README / 面试 / 简历说明中复用的代表性 case。
- `..\docs\interview_qa.md`：Azusa 项目追问、回答草稿和简历 bullet 草稿。

## 常见问题

- `hnsw` / `hybrid` / `hybrid_rerank` 连不上：确认 `azusa_server` 正在运行。
- 输出写不进去：通常是目标 JSON 被编辑器或其他进程占用。
- 换数据后结果怪：`azusa_server` 是内存索引，换 `embedding.jsonl` 后最好重启服务。
