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

## 常见问题

- `hnsw` / `hybrid` / `hybrid_rerank` 连不上：确认 `azusa_server` 正在运行。
- 输出写不进去：通常是目标 JSON 被编辑器或其他进程占用。
- 换数据后结果怪：`azusa_server` 是内存索引，换 `embedding.jsonl` 后最好重启服务。
