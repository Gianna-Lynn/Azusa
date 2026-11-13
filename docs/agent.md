# Search Agent 与工具调用

Search Agent 是 RAG 上的一层规则编排，支持查询分类、rewrite、显式多跳检索和证据融合，
随后执行最小工具调用并生成可选回答。它不等同于自主模型规划器。

```text
query -> classify / rewrite -> retrieval hops -> evidence fusion
      -> get_chunk_by_id -> bounded retry -> answer context -> citations / trace
```

## 运行

从 rag 目录执行，需事先准备 embedding 记录：

```powershell
python run.py agent --mode bm25 --top-k 5 --query-text "When were stromules discovered?"
python run.py agent-eval --mode bm25 --top-k 5
```

需要回答时增加 --with-answer 并提供本地 Answerer。BM25 不需要向量模型或 Rust 服务；
其他检索模式的依赖见 [RAG 指南](rag.md)。

## Trace

- query_plan：分类、rewrite 和拆分后的检索查询。
- hops：每跳命中记录；fused_evidence：跨跳融合后的证据。
- fusion.retrieval_sources：dense、sparse 或两者共同来源。
- 工具执行 trace：参数、每次尝试、错误、耗时和最终结果。
- answer_citations：模型明确给出的本轮证据 ID；缺少引用时不回填。

search/agent 默认写入 output/search_runs/latest；agent-eval 默认写入
output/eval_runs/latest，文件占用时以终端打印的实际回退路径为准。

## 工具运行时

唯一工具为 get_chunk_by_id。运行时校验工具名与参数，把成功、未知工具、缺参和
执行错误保存为结构化尝试。默认最多两次尝试，成功即停止，不使用无限重试。
只允许成功工具结果进入答案上下文，失败不能变成引用；工具返回保留原始 chunk_id。

实现见 `rag/src/azusa_rag/tool_calling.py`；既有契约测试
`rag/tests/test_tool_calling_assignment.py` 当前实际运行。小型回归与引用失败见
[评测](evaluation.md)。规则 rewrite、显式连接词拆分和本地小模型均有覆盖边界，
不能由 trace 完整推断答案正确或具备自主推理能力。
