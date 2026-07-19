# Search Agent v0 Demo

Search Agent v0 是 Azusa RAG pipeline 上的一层轻量编排。它不依赖复杂 Agent 框架，当前目标是把一次问答过程拆成可观察的步骤：query planning、retrieval hops、evidence fusion、answer with citations。

## Positioning

Agent 在 Azusa 里不是主项目核心。主项目核心仍然是自研 Rust HNSW 检索核、Python RAG pipeline、hybrid / rerank 和 eval workflow。Agent v0 的作用是把这些能力封装成一个上层入口，使项目可以用 Agent / Tool Calling 的语言来展示。

当前最小链路：

```text
query -> classify / rewrite -> retrieval hop(s) -> evidence fusion -> answer + citation
```

## Run

`bm25` 模式适合做本地 smoke demo，不需要启动 Rust 服务：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent --mode bm25 --top-k 5 --with-answer --query-file data\queries\autoevaluate_squad_sample_query_002.txt
```

`hybrid_rerank` 模式适合展示 dense + sparse + rerank 的完整链路，需要先启动 `azusa_server`：

```powershell
cd F:\Repositories\Azusa
cargo run --bin azusa_server
```

然后在另一个 PowerShell 窗口运行：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent --mode hybrid_rerank --top-k 5 --with-answer --query-file data\queries\autoevaluate_squad_sample_query_001.txt
```

## Output

默认输出在：

```text
output/search_runs/latest/search_agent_v0_<mode>.json
```

如果当前 Python 进程没有权限写入仓库输出目录，`run_agent.py` 会 fallback 到系统临时目录，并在终端打印实际 `saved to` 路径。

## JSON Fields

`query_plan` 描述 Agent 对用户问题的第一步处理：

- `query_type`：当前规则分类为 `single_hop` 或 `multi_hop`。
- `rewritten_query`：清理空白和尾部标点后的查询。
- `retrieval_queries`：真正送入检索层的一组 query。
- `notes`：本次 query plan 的规则说明。

`hops` 记录每一跳检索的原始结果。当前单跳问题通常只有一个 hop；多跳问题会拆成多个 retrieval query 后分别检索。

`fused_evidence` 是 Agent v0 的主要观察对象。它会把不同 hop 返回的 chunk 按 `chunk_id` 去重，再按命中次数和最佳排名排序。

`fusion` 字段解释每条 evidence 为什么排在这里：

- `hit_count`：这个 chunk 被多少个 hop 命中。
- `best_rank`：这个 chunk 在所有 hop 中出现过的最好排名。
- `hop_indices`：命中过它的 hop 编号。
- `retrieval_sources`：来源标记，例如 `bm25 sparse`、`dense`、`dense + sparse`。

`answer` / `answer_citations` 是 answerer 的输出。它们用于观察本地小模型是否正确使用 evidence，但不应单独作为检索质量证明。

Answerer 对外固定输出四个字段：

```text
结论：<直接回答>
证据：<简短证据摘要>
引用来源：<本次 evidence 中真实存在的 chunk_id>
不确定性：<无，或具体限制>
```

模型未明确给出有效 chunk id 时，系统不会再强制引用 top-1。此时 `answer_citations` 保持为空，并在 `answer_uncertainty` 和 `citation_validation` 中标记需要人工复核。若模型给出了有效 citation 但没有 `答案：`/`结论：` 字段，系统会保留 citation 和证据，同时将状态标记为 `answer_missing`，提示“有引用但没有答案，需要人工审核”。只有答案和 citation 都缺失时，才归为证据不足。

考虑到当前使用的是 0.5B 本地小模型，模型内部只生成“答案 + 显式引用”两行；Python 规范化层再把已引用 chunk 的原文确定性整理成“证据”字段，并补充“不确定性”。这样既降低小模型的格式负担，也不会替模型猜测 citation。

## Stage 3 Agent Eval

P1 使用一份独立的 15 题样例集：

```text
data/eval/agent_eval.jsonl
```

它覆盖：

- 3 个 single-hop case
- 3 个 query rewrite case
- 3 个显式 multi-hop case
- 2 个 insufficient-evidence case
- 4 个 citation badcase

无需加载 Answerer 的检索轨迹 smoke：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent-eval --mode bm25 --top-k 5
```

需要同时验证本地小模型回答和 citation 时：

```powershell
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent-eval --mode bm25 --top-k 5 --with-answer
```

结果默认保存到：

```text
output/eval_runs/latest/agent_eval_<mode>.json
```

每个 case 都保留完整 `trace`，包括 query plan、各跳检索结果、融合证据、最终回答、citation 校验状态和耗时。样例集本身提交到 Git，运行输出继续放在可覆盖的 `output/` 目录。

## Stage 3 P3 实现状态

P3 的最小 Tool Calling runtime 已完成，并已接入 Search Agent 主流程。当前实现包含工具分发、参数校验、结构化错误、一次重试、计时、工具上下文和可序列化的工具 trace。原始作业说明仍保留在：

```text
docs/stage3_p3_assignment.md
```

实现代码位于 `src/azusa_rag/tool_calling.py`，验收测试位于 `tests/test_tool_calling_assignment.py`；测试当前实际执行，不再默认跳过。

## Demo Reading Example

`When were stromules discovered?` 的 bm25 demo 中，top-1 evidence 命中 `...chunk_002`，文本包含 `first observed in 1962`，answer 输出 `stromules were discovered in 1962`，并引用同一个 chunk。

这个 case 适合说明：

- BM25 对年份和实体词问题很直接。
- `fused_evidence` 能展示证据来源。
- answer + citation 可以形成一个可读的端到端闭环。

## Current Limits

- Query classification 和 rewrite 仍是规则版，不是模型规划器。
- Multi-hop split 只覆盖显式连接词和少量中文触发词。
- Citation answer 依赖本地小模型，仍可能出现答案不完整或实体选择错误；系统现在只保证引用 ID 来自本次 evidence，不替模型伪造引用。
- Agent v0 还没有真正 tool calling runtime；当前更接近可观察的 search orchestrator。

## Interview Summary

可以这样介绍：

```text
I wrapped the Azusa RAG pipeline into a lightweight Search Agent v0. It classifies and rewrites the query, runs one or more retrieval hops, fuses repeated evidence, and generates an answer with chunk-level citations. The goal is not to build a full autonomous agent, but to expose retrieval behavior and citation quality in a structured trace.
```
