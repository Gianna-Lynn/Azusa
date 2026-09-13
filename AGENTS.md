# AGENTS.md

本文件是 Azusa 仓库中供 Codex 及其他编码代理遵循的协作说明。修改代码前先阅读本文件、相关模块和测试；保持改动聚焦，优先使用仓库已有的命令、脚本和文档。

## 项目概览

Azusa 是一个由 Rust 向量检索服务和 Python RAG 实验链路组成的研究项目。

- `src/`、`Cargo.toml`：Rust HNSW 索引库与 Axum HTTP 服务。
- `rag/`：Python RAG 子项目，负责知识库同步、切块、向量化、检索、重排、回答、引用和 Agent 编排。
- `tests/`：Rust 公共 API 烟雾测试；`tests/archive/` 保存历史实验和旧实现测试。
- `rag/tests/`：Python RAG、Knowledge Base、过滤器、Agent、Kaiji gateway 和 benchmark lab 测试。
- `docs/`、`rag/docs/`：HTTP API、benchmark、课程任务、失败边界和实验记录。
- `scripts/`：可重复执行的辅助脚本，例如 HTTP smoke test。

主线数据流大致如下：

```text
源文档 -> Knowledge Base manifest -> 稳定 chunks -> embeddings
                                                    |        |
                                                    +-> dense / HNSW
                                                    +-> BM25
                                                         |
                                                    RRF -> rerank
                                                         |
                                                    过滤/ACL
                                                         |
                                                  evidence -> answer / Agent

Kaiji gateway -> search / chunk / health
```

当前仓库在 `main` 分支上承载课程 TODO；`solution/azusa-v2` 是完整参考实现。README 和设计文档中的“已支持”能力需要结合当前分支确认，尤其是 `knowledge_base.py`、`filters.py` 和 `kaiji_gateway.py` 中以 `NotImplementedError` 标记的 TODO，不要未经验证就当作已完成能力。

## 协作与环境约定

- 当前环境是 Windows；优先使用默认的 PowerShell 7，并使用 PowerShell 原生命令处理路径和文件。
- Python 命令在本机优先使用 conda 环境 `ChuoEnv`；如果在 `E5_server` 上运行，优先使用 `KandaEnv`。
- 登录服务器使用 `ssh E5_server`；服务器上禁止使用 `sudo`。
- 修改前先检查 `git status`，保留用户已有改动，不重置、覆盖或删除无关文件。
- 优先修改源码和文档，不要把 `rag/output/`、缓存、临时 benchmark 结果或本地模型提交进仓库。
- 生成或覆盖 `latest/` 目录中的实验结果前，确认这是用户要求的输出；重要结果应记录在对应文档，而不是依赖未提交的临时文件。
- 遇到路径、协议或行为不确定时，先查 `README.md`、`docs/README.md`、`rag/RAG_README.md` 和对应模块测试。

## Rust 服务

### 常用命令

以下命令从仓库根目录 `F:\Repositories\Azusa` 执行：

```powershell
# 运行 Rust 测试
cargo test

# 构建 release 版本
cargo build --release

# 启动 HNSW HTTP 服务，默认监听 127.0.0.1:3000
cargo run --bin azusa_server

# 运行 HTTP 闭环 smoke test（需先在另一个终端启动服务）
pwsh -ExecutionPolicy Bypass -File .\scripts\http_smoke_test.ps1

# 运行 SIFT1M benchmark（原始数据不随仓库提交）
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

### 模块与 HTTP 边界

- `src/index/`：当前 HNSW 实现及 workspace、构建、搜索和访问器逻辑。
- `src/distance.rs`：向量距离/相似度计算。
- `src/types.rs`：`Id`、`Vector`、`VectorRecord`、`Metadata`、`SearchResult` 和 `Metric`。
- `src/legacy/`：旧版实现，仅在明确涉及 legacy 时修改或测试。
- `src/bin/azusa_server.rs`：内存版 Axum 服务；当前索引由 `Arc<Mutex<_>>` 持有。

服务目前提供以下接口：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 返回服务状态和 `record_count` |
| `POST` | `/insert` | 插入一个 `VectorRecord`；重复 ID 返回 `409` |
| `POST` | `/search` | 接收 query vector，返回带 `id`/`score` 的 top-k 结果 |
| `GET` | `/record/:id` | 按 ID 取回完整记录；不存在返回 `404` |

Rust 服务是内存索引，重启后数据丢失；没有 WAL、持久化、删除/更新接口。Python 侧切换 `embedding.jsonl` 后必须重启服务，避免复用上一批数据造成 ID 对不上。

## Python RAG 子项目

### 环境与统一入口

从 `rag/` 目录执行 Python 命令，并先激活环境：

```powershell
conda activate ChuoEnv
cd F:\Repositories\Azusa\rag
pip install -e .
python run.py --help
```

`rag/run.py` 是首选统一入口，同时保留旧的 `run_*.py` 脚本以兼容历史工作流。当前子命令包括：

```powershell
# 基础产物
python run.py chunk
python run.py embed
python run.py kb-sync

# 单次检索与 Agent
python run.py search --mode brute --top-k 3 --query-text "your query"
python run.py search --mode hybrid_rerank --top-k 3 --with-answer --query-text "your query"
python run.py agent --mode hybrid_rerank --top-k 5 --with-answer --query-text "your query"
python run.py agent-eval --mode bm25 --top-k 5

# 评测与完整工作流
python run.py eval --mode all --top-k 3
python run.py workflow
python run.py benchmark-lab --modes brute,bm25 --top-k 3,5

# 数据集导入与 DuRetrieval
python run.py import-hf-eval --length 20
python run.py import-du-retrieval
python run.py download-du-retrieval
python run.py du-retrieval-workflow
```

具体参数以各子命令的 `--help` 为准。`chunk`/`embed` 是兼容的基础流水线；涉及多格式文档、manifest、增量复用和删除语义时，优先检查 `kb-sync` 及 `knowledge_base.py` 的当前分支实现。

### 检索模式与依赖

`rag/src/azusa_rag/modes.py` 是检索模式的单一来源，当前模式为：

| 模式 | 检索方式 | 说明 |
| --- | --- | --- |
| `brute` | dense | 全量向量暴力检索，作为基线 |
| `hnsw` | dense | 通过 HTTP 调用 Rust HNSW 服务 |
| `bm25` | sparse | 使用 `rank_bm25` 做关键词检索 |
| `hybrid` | dense + sparse | HNSW dense 与 BM25 通过 RRF 融合 |
| `hybrid_rerank` | dense + sparse + rerank | 扩大候选集后使用 reranker 重排 |

`hnsw`、`hybrid` 和 `hybrid_rerank` 需要先启动 `azusa_server`；服务默认地址为 `http://127.0.0.1:3000`。服务中已有非空索引但数量与当前 embedding 不一致时，Python 侧会拒绝静默复用，应重启服务。

### Python 关键模块

- `config.py`：路径、模型、向量维度、chunk 参数和 Rust 服务地址的集中配置。
- `schemas.py`：`DocumentRecord`、`ChunkRecord`、`EmbeddingRecord` 等数据结构。
- `loaders.py`、`knowledge_base.py`：读取源文档、生成稳定身份、构建 chunk、维护 manifest 和 embedding 产物。
- `chunker.py`、`embedder.py`：传统切块和 embedding 生成。
- `search.py`：统一检索门面；`search_hnsw.py`、`search_sparse.py`、`search_hybrid.py` 分别负责后端细节。
- `filters.py`：metadata filter 与最小 ACL；当前主线可能仍保留课程 TODO。
- `reranker.py`、`answerer.py`：候选重排、回答生成和 citation 规范化。
- `agent.py`、`tool_calling.py`：query 分类、rewrite、multi-hop、evidence fusion 和最小工具调用运行时。
- `kaiji_gateway.py`：面向 Kaiji 的进程内/HTTP 语义网关，包含参数校验、检索结果映射、chunk 查询和结构化错误。

本机默认模型配置位于 `rag/src/azusa_rag/config.py`：

- embedding：`D:\Chuo_Data\Models\Qwen3-Embedding-0.6B`，维度 `512`，默认设备 `cuda`。
- reranker：`D:\Chuo_Data\Models\bge-reranker-v2-m3`，默认设备 `cuda`。
- answerer：`D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct`（配置在 `answerer.py`）。

模型路径和 CUDA 是本机相关配置；不要把本地模型目录写入仓库，也不要为了通过轻量测试擅自改动默认模型配置。

## 数据与输出路径

以下路径均相对于 `rag/`：

| 路径 | 内容 |
| --- | --- |
| `data/raw/` | 源文档；Knowledge Base 支持 `.txt`、`.md`、`.markdown`、`.html`、`.htm`、`.pdf` |
| `data/queries/` | 单次检索使用的 query 文件 |
| `data/eval/` | 检索和 Agent 回归集 |
| `data/toy/` | DuRetrieval toy 数据 |
| `data/import_snapshots/hf_eval/` | HF 评测样本导入快照 |
| `data/processed/du_retrieval_workflow/` | DuRetrieval workflow 产物 |
| `output/artifacts/` | `chunks.jsonl`、`embedding.jsonl`、`knowledge_base_manifest.json` |
| `output/search_runs/` | search/agent 结果；默认写入 `latest/` |
| `output/eval_runs/` | eval/workflow 结果；默认写入 `latest/` |

`latest/` 是默认覆盖目标，`archive/` 或 `runs/` 用于保留历史副本。数据和模型较大时，先确认用户是否希望重新生成，避免无意覆盖实验资产。

## 测试与验证

Rust：

```powershell
cargo test
cargo test --test hnsw_api_smoke_test
```

Python：从 `rag/` 执行：

```powershell
python -m pytest
```

涉及特定模块时优先运行对应测试，例如：

```powershell
python -m pytest tests/test_knowledge_base.py tests/test_filters.py
python -m pytest tests/test_agent.py tests/test_agent_eval.py tests/test_tool_calling_assignment.py
python -m pytest tests/test_kaiji_gateway.py tests/test_benchmark_lab.py
```

需要真实 embedding、reranker 或 Rust 服务的测试/评测，应先确认模型和服务可用；纯契约测试优先保持轻量，不要为了测试而下载大模型或修改业务行为。

## 修改时的注意事项

- `main` 分支的课程 TODO 和 `solution/azusa-v2` 的完整实现必须区分；修复 TODO 前先看对应任务书、追踪文档和测试契约。
- 稳定 `document_id`、`chunk_id`、内容哈希、manifest 和 embedding 复用关系属于数据契约，修改前要检查向后兼容和删除语义。
- 过滤采用 over-fetch 后过滤时，结果可能少于请求的 `k`；不要把它描述成原生 filtered ANN。
- dense 与 sparse 分数不可直接比较，hybrid 使用 RRF；需要调整融合逻辑时同步检查 source map、trace 和评测。
- HTTP 和 Kaiji gateway 的字段名、状态码、错误码和 `retryable` 语义是外部契约，改动必须同步测试与文档。
- 旧版 `src/legacy/` 和 `tests/archive/` 默认不属于当前主线；只有任务明确涉及时才改动。
- 修改完成后至少运行与改动直接相关的测试，并检查 `git diff`，确认没有混入缓存、生成物或无关格式化。

## 参考入口

- 总览与快速启动：`README.md`
- Rust/Python 项目说明：`docs/README.md`
- RAG 命令与输出说明：`rag/RAG_README.md`
- HTTP 接口：`docs/http_api.md`
- Knowledge Base 设计：`docs/knowledge_base_v1.md`
- Kaiji 集成：`docs/kaiji_integration.md`
- benchmark：`docs/benchmark.md`
- Agent 与工具调用：`rag/docs/agent_tool_demo.md`、`rag/docs/stage3_p3_assignment.md`
- 课程任务与完成追踪：`rag/docs/azusa_v2_assignment.md`、`rag/docs/azusa_v2_traceability.md`
