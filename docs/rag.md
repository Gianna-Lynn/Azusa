# Python RAG 使用指南

需要 Python 3.11+。从仓库根目录进入 rag，使用自己的虚拟环境安装依赖：

```powershell
cd rag
python -m pip install -e .
python run.py --help
```

## 无模型体验

```powershell
python run.py benchmark-lab --smoke --top-k 1,3
```

使用已提交的六条合成文本与三条查询，只执行 BM25，生成独立目录及三格式报告。
向量为占位值，不加载模型、GPU 或 Rust 服务；它验证工程协议，不是质量 benchmark。

## 模型与服务

默认配置针对原开发机的本地模型目录与 CUDA。运行真实向量链路前，核对
`rag/src/azusa_rag/config.py`；以下变量只在当前进程覆盖配置：

| 配置 | 含义 |
| --- | --- |
| AZUSA_EMBEDDING_MODEL / AZUSA_EMBEDDING_DEVICE | embedding 模型路径与设备 |
| AZUSA_RERANK_MODEL / AZUSA_RERANK_DEVICE | reranker 路径与设备 |
| AZUSA_SERVER_URL | Rust 服务地址，默认 http://127.0.0.1:3000 |

embedding 默认为 Qwen3-Embedding-0.6B、512 维；reranker 默认为 bge-reranker-v2-m3。
Answerer 的 Qwen2.5-0.5B-Instruct 路径单独定义在 `answerer.py`，没有上述环境变量入口。
这里不自动下载模型或修改默认配置；具体依赖以 `rag/pyproject.toml` 为准。

模式依赖：BM25 不需要模型或 Rust；brute 需要 embedding；hnsw、hybrid 和
hybrid_rerank 还需从根目录另启 `cargo run --bin azusa_server`。生成回答需 Answerer。

## 构建与查询

准备源文档后，从 rag 目录执行：

```powershell
python run.py kb-sync --source-dir data/raw
python run.py search --mode bm25 --top-k 3 --query-text "Azusa 的索引如何重建？"
python run.py search --mode hybrid_rerank --top-k 3 --with-answer --query-text "Azusa 的索引如何重建？"
```

`kb-sync` 支持 TXT、Markdown、HTML 和 PDF 文本层，默认写入 output/artifacts；
它需要可用的编码模型。初次同步或切换语料后，应重新启动 Rust 服务再装载新快照。
同步产物、编码身份和显式空库删除见 [知识库](knowledge_base.md)。

`chunk` / `embed` 保留基础流水线兼容入口。数据导入还提供 import-hf-eval、
download-du-retrieval、import-du-retrieval 与 du-retrieval-workflow；参数见各命令 --help。

## 输出与评测

| 路径（相对 rag） | 用途 |
| --- | --- |
| output/artifacts/ | 当前 chunks、embeddings 和 manifest |
| output/search_runs/latest/ | search / agent 默认覆盖结果 |
| output/eval_runs/runs/ | eval / benchmark-lab 默认独立实验目录 |
| output/eval_runs/latest/ | workflow / agent-eval 等旧入口的默认结果 |
| data/import_snapshots/hf_eval/latest/ | HF 样本导入快照 |
| data/processed/du_retrieval_workflow/ | DuRetrieval workflow 产物 |

已有输出需要留存时先复制到独立目录；不要把所有命令都当成拒绝覆盖的写入方式。
指标与报告见 [评测](evaluation.md)，编排见 [Agent](agent.md)。

连接失败先检查服务地址；数据更换后结果异常先核对快照身份并重启重建。
Windows 输出文件被占用时应关闭占用程序，再按终端打印的实际路径定位结果。
