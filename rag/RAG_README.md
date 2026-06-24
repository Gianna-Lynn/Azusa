# Azusa RAG

最小可运行的 RAG 骨架，当前重点是把这几层接起来：

- `raw txt -> chunks.jsonl -> embedding.jsonl`
- `brute / hnsw / bm25 / hybrid / hybrid_rerank` 检索
- 最小检索评测：`Hit@k / Recall@k / MRR`
- 最小 `answer + citation`

## 怎么看这个目录

- `run_main.py`
  生成 `chunks.jsonl`
- `run_embed.py`
  生成 `embedding.jsonl`
- `run_search.py`
  看单条 query 的检索 / answer 结果
- `run_agent.py`
  跑 Search Agent v0：query classification / query rewrite / multi-hop retrieval / evidence fusion
- `run_eval.py`
  看批量检索指标
- `run_workflow.py`
  一键汇总，能跑就行
- `src/azusa_rag/`
  核心实现

## 前置条件

```powershell
conda activate ChuoEnv
cd F:\Repositories\Azusa\rag
```

如果要跑 `hnsw`、`hybrid`、`hybrid_rerank`，还需要先启动服务：

```powershell
cd F:\Repositories\Azusa
cargo run --bin azusa_server
```

当前本地模型路径：

- embedding: `D:\Chuo_Data\Models\Qwen3-Embedding-0.6B`
- reranker: `D:\Chuo_Data\Models\bge-reranker-v2-m3`
- answerer: `D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct`

## 最常用运行方式

### 1. 生成 artifacts

```powershell
python run_main.py
python run_embed.py
```

产物：

- `output/artifacts/chunks.jsonl`
- `output/artifacts/embedding.jsonl`

### 2. 跑单条检索

```powershell
python run_search.py --mode brute --top-k 3
python run_search.py --mode hnsw --top-k 3
python run_search.py --mode hybrid --top-k 3
python run_search.py --mode hybrid_rerank --top-k 3
python run_search.py --mode hybrid_rerank --top-k 3 --with-answer
```

可选：

- `--query-text "..."` 直接传 query
- `--query-file path\to\query.txt` 从文件读 query
- `--output-name xxx` 避免覆盖默认结果文件

产物：

- `output/search_runs/search_result_<mode>.json`

### 2.5 跑 Search Agent v0

```powershell
python run_agent.py --mode bm25 --top-k 5 --query-text "first question and second question"
python run_agent.py --mode hybrid_rerank --top-k 5 --with-answer
```

产物：

- `output/search_runs/search_agent_v0_<mode>.json`

### 3. 跑批量评测

```powershell
python run_eval.py --mode brute --top-k 3
python run_eval.py --mode bm25 --top-k 3
python run_eval.py --mode all --top-k 3
```

可选：

- `--eval-path data\eval\min_eval.jsonl`
- `--output-name xxx`

产物：

- `output/eval_runs/eval_result_<mode>.json`

### 4. 一键工作流

```powershell
python run_workflow.py
```

常用参数：

```powershell
python run_workflow.py --skip-answer-checks
python run_workflow.py --rebuild-artifacts
python run_workflow.py --output-name my_workflow
```

产物：

- `output/eval_runs/<name>_summary.json`
- `output/eval_runs/<name>_report.md`

### 5. 导入 HF 样本

```powershell
python run_import_hf_eval.py --length 20
```

产物：

- `data/raw/autoevaluate_squad_sample_*.txt`
- `data/eval/hf_squad_sample_eval.jsonl`

## 输出目录

- `output/artifacts/`
  可复用的数据产物
- `output/search_runs/`
  单条 query 结果
- `output/eval_runs/`
  批量评测和 workflow 汇总

## 常见问题

### `hnsw` / `hybrid` / `hybrid_rerank` 连不上

先确认 `azusa_server` 正在运行。

### 输出文件写不进去

通常是目标 JSON 正被编辑器或其他进程占用。换一个 `--output-name` 最快。

### 换了数据后结果怪怪的

`azusa_server` 是内存索引。换了一批新的 `embedding.jsonl` 后，最好重启服务再跑。
