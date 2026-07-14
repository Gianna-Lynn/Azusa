# Azusa

特急Azusa(あずさ、梓)是JR东日本中央本线(特急)列车, 常见于新宿至松本区段.  东京是我心中的理想国. 用这个列车名来命名我的第一个向量数据库项目, 就显得再好不过了.

Azusa 是一个基于 HNSW 的向量检索 / RAG 实验项目。

当前主线分两层：

- Rust HNSW：内存索引、top-k 检索、metadata 查询、HTTP demo 服务。
- Python RAG：chunk -> embedding -> dense/BM25/RRF hybrid -> rerank -> citation answer -> Search Agent。

Rust/Python 当前继续使用 HTTP 接口。未完成的 PyO3 试验已剥离，避免扩展构建问题阻塞 RAG 与 Agent 主线。

## 快速入口

Rust 服务：

```powershell
cargo run --bin azusa_server
```

RAG 子项目：

```powershell
cd rag
python run.py --help
```

Stage 3 Agent 样例评测：

```powershell
cd rag
python run.py agent-eval --mode bm25 --top-k 5
python run.py agent-eval --mode bm25 --top-k 5 --with-answer
```

SIFT1M benchmark：

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

## 当前能力

- `HnswIndex` 支持插入向量记录、搜索 top-k、返回 id / score、按 id 取 metadata。
- `azusa_server` 提供最小 `health -> insert -> search -> record` HTTP 闭环。
- `sift1m_benchmark` 支持 recall@k / exact@k 复测，并输出 Markdown / CSV。
- Hybrid Search 使用 RRF 融合 dense 与 BM25 排名，避免原始分数不可比和 dense-first 截断问题。
- Search Agent v0 支持规则分类、query rewrite、显式 multi-hop retrieval 和 evidence fusion。
- Answerer 固定输出结论、证据、引用来源和不确定性；citation 只接受当前 evidence 中真实存在的 chunk id。
- 15 题 Agent Eval 覆盖 single-hop、query rewrite、multi-hop、证据不足和历史 citation badcase。
- `rag/run.py` 统一调度 chunk、embed、search、eval、workflow、agent 和 agent-eval。

## 文档入口

- RAG 怎么跑：`rag/RAG_README.md`
- Search Agent 与 citation 格式：`rag/docs/agent_tool_demo.md`
- Stage 3 Agent Eval 基线：`rag/docs/stage3_agent_eval.md`
- HTTP API：`docs/http_api.md`
- Benchmark：`docs/benchmark.md`
- 当前 RAG badcase：`rag/docs/badcase_v1.md`
- 聚合改进记录：`CHANGELOG.md`

历史说明、阶段总结、接口取舍和长报告都放在 `docs/archive/` 或 `rag/docs/archive/`。

## 当前限制

- HTTP 服务仍是内存索引，服务重启后记录会丢失。
- 暂无持久化、删除、更新、批量导入和并发压测。
- SIFT1M 原始数据不随仓库提交，benchmark 依赖本机或服务器上的 `data/sift1m/sift/`。
- Query classification 和 multi-hop split 当前仍是确定性规则版，不是模型规划器。
- 本地 0.5B Answerer 仍可能漏答、错实体或漏 citation；系统会保留结构化 trace 和不确定性供人工复核。
- Agent Eval 是工程回归集，不代表公开 benchmark 或生产质量。
- RAG 子项目仍是实验链路，还不是完整应用。
