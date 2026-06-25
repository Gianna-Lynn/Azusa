# Azusa RAG

最小可运行的 RAG 骨架，当前重点是把这几层接起来：

- `raw txt -> chunks.jsonl -> embedding.jsonl`
- `brute / hnsw / bm25 / hybrid / hybrid_rerank` 检索
- 最小检索评测：`Hit@k / Recall@k / MRR`
- 最小 `answer + citation`

## 怎么看这个目录

- `run.py`
  统一命令入口，优先使用
- `run_main.py`
  兼容入口：生成 `chunks.jsonl`
- `run_embed.py`
  兼容入口：生成 `embedding.jsonl`
- `run_search.py`
  兼容入口：看单条 query 的检索 / answer 结果
- `run_agent.py`
  兼容入口：跑 Search Agent v0：query classification / query rewrite / multi-hop retrieval / evidence fusion
- `run_eval.py`
  兼容入口：看批量检索指标
- `run_workflow.py`
  兼容入口：一键汇总，能跑就行
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

统一入口：

```powershell
python run.py <command> [args]
```

旧的 `run_*.py` 仍然保留兼容，但后续优先建议使用这个统一入口。

### 1. 生成 artifacts

```powershell
python run.py chunk
python run.py embed
```

产物：

- `output/artifacts/chunks.jsonl`
- `output/artifacts/embedding.jsonl`

### 2. 跑单条检索

```powershell
python run.py search --mode brute --top-k 3
python run.py search --mode hnsw --top-k 3
python run.py search --mode hybrid --top-k 3
python run.py search --mode hybrid_rerank --top-k 3
python run.py search --mode hybrid_rerank --top-k 3 --with-answer
```

可选：

- `--query-text "..."` 直接传 query
- `--query-file path\to\query.txt` 从文件读 query
- `--output-name xxx` 只改逻辑名称；默认仍覆盖 `latest/` 里的结果
- `--keep-run-copy` 额外保留一份命名副本

产物：

- `output/search_runs/latest/search_result_<mode>.json`

`output/search_runs/latest/` 会在首次运行对应命令时自动创建。

### 2.5 跑 Search Agent v0

```powershell
python run.py agent --mode bm25 --top-k 5 --query-text "first question and second question"
python run.py agent --mode hybrid_rerank --top-k 5 --with-answer
```

产物：

- `output/search_runs/latest/search_agent_v0_<mode>.json`

只有显式加 `--keep-run-copy` 时，才会额外保留一份 `output/search_runs/<name>_*.json` 命名副本。

### 3. 跑批量评测

```powershell
python run.py eval --mode brute --top-k 3
python run.py eval --mode bm25 --top-k 3
python run.py eval --mode all --top-k 3
```

可选：

- `--eval-path data\eval\min_eval.jsonl`
- `--output-name xxx` 只改变逻辑名称；默认仍覆盖同一组结果文件

产物：

- `output/eval_runs/latest/eval_result_<mode>.json`

`output/eval_runs/latest/` 会在首次运行对应命令时自动创建。

### 4. 一键工作流

```powershell
python run.py workflow
```

常用参数：

```powershell
python run.py workflow --skip-answer-checks
python run.py workflow --rebuild-artifacts
python run.py workflow --output-name my_workflow
python run.py workflow --output-name my_workflow --keep-run-copy
```

产物：

- `output/eval_runs/latest/summary.json`
- `output/eval_runs/latest/report.md`
- `output/eval_runs/latest/badcase_v1.md`

只有显式加 `--keep-run-copy` 时，才会额外保留一份 `output/eval_runs/<name>_*` 命名副本。

### 5. 导入 HF 样本

```powershell
python run.py import-hf-eval --length 20
```

产物：

- `data/import_snapshots/hf_eval/latest/raw/autoevaluate_squad_sample_*.txt`
- `data/import_snapshots/hf_eval/latest/eval.jsonl`
- `data/import_snapshots/hf_eval/latest/queries/`

如果显式传 `--raw-dir`、`--eval-path`、`--query-dir`，则会写到指定位置；只有显式加 `--keep-run-copy` 时，才会额外保留 `data/import_snapshots/hf_eval/<name>/` 命名副本。

`data/import_snapshots/hf_eval/latest/` 会在首次运行导入命令时自动创建。

## 输出目录

- `output/artifacts/`
  可复用的数据产物
- `output/search_runs/latest/`
  单条 query / agent 结果的默认覆盖目录
- `output/search_runs/archive/`
  历史 search / agent 输出留档
- `output/eval_runs/latest/`
  批量评测和 workflow 汇总的默认覆盖目录
- `output/eval_runs/archive/`
  历史 eval / workflow 输出留档
- `data/import_snapshots/hf_eval/latest/`
  HF 样本导入脚本的默认覆盖目录
- `data/import_snapshots/hf_eval/runs/`
  HF 样本导入脚本的命名运行副本（首次保留命名副本时自动创建）
- `data/processed/du_retrieval_workflow/latest/`
  DuRetrieval workflow 的默认覆盖型产物目录（首次运行对应 workflow 时自动创建）
- `data/processed/du_retrieval_workflow/runs/`
  DuRetrieval workflow 的命名运行副本（首次保留命名副本时自动创建）

## 常见问题

### `hnsw` / `hybrid` / `hybrid_rerank` 连不上

先确认 `azusa_server` 正在运行。

### 输出文件写不进去

通常是目标 JSON 正被编辑器或其他进程占用。优先关闭占用文件；只有需要保留命名批次时再加 `--keep-run-copy`。

### 换了数据后结果怪怪的

`azusa_server` 是内存索引。换了一批新的 `embedding.jsonl` 后，最好重启服务再跑。
