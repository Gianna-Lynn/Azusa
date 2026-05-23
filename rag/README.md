# 项目概览

这是一个最小可运行的 RAG 骨架，当前已经打通：

- `raw txt -> chunks.jsonl -> embedding.jsonl`
- `brute / hnsw / hybrid / hybrid_rerank` 四种检索模式
- 最小检索评测：`Hit@k / Recall@k / MRR`
- 最小 `answer + citation` 生成链路

当前项目的重点不是“做一个很强的大模型问答系统”，而是把下面几层接起来：

1. 文本切 chunk
2. chunk 做 embedding
3. 检索 / hybrid / rerank
4. 把 top-k 证据交给一个轻量 LLM，生成最小 `answer + citation`

# 当前依赖的本地模型

- embedding：`D:\Chuo_Data\Models\Qwen3-Embedding-0.6B`
- reranker：`D:\Chuo_Data\Models\bge-reranker-v2-m3`
- answerer：`D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct`

# 目录说明

- `data/raw/`：原始输入文档
- `data/eval/`：评测集 jsonl
- `data/queries/`：默认 query 文本
- `output/artifacts/`：长期复用的数据产物，如 `chunks.jsonl`、`embedding.jsonl`
- `output/search_runs/`：单条 query 检索 / answer 结果
- `output/eval_runs/`：批量评测结果
- `output/scratch/`：预留给临时调试产物
- `src/azusa_rag/`：核心代码
- `run_search.py`：单条 query 检索 / answer 验证入口
- `run_eval.py`：检索评测入口
- `run_import_hf_eval.py`：导入 HF QA 样本的入口
- `run_workflow.py`：按 README 工作流一键执行并汇总结果的入口

# 输出目录约定

当前新文件的生成逻辑按“用途分层”来收：

- `output/artifacts/`
  - 放会被后续流程复用的正式数据产物
  - 例如：`chunks.jsonl`、`embedding.jsonl`
- `output/search_runs/`
  - 放 `run_search.py` 跑出来的单条 query 结果
  - 例如：`answer_massachusetts_hybrid_rerank.json`
- `output/eval_runs/`
  - 放 `run_eval.py` 跑出来的批量评测结果
  - 例如：`eval_result_hybrid_rerank.json`
- `output/scratch/*_legacy/`
  - 放历史验证文件、重跑快照和旧命名结果
  - 正式目录里只保留当前最值得直接打开看的文件

这样做的目的，是把“基础数据产物”“单次检索观察”“批量评测结果”分开，避免所有 JSON 平铺在一个目录里越来越难认。

另外，当前脚本默认采用“固定文件名并允许覆盖”的策略：

- `run_search.py`
  - 默认覆盖：`output/search_runs/search_result_<mode>.json`
- `run_eval.py`
  - 默认覆盖：`output/eval_runs/eval_result_<mode>.json`

如果你明确想保留一次特殊实验快照，再手动传 `--output-name xxx`。

# 一键工作流入口

如果你不想手动逐条执行 README 里的命令、再分别打开多个 JSON 做肉眼对比，可以直接跑：

```powershell
cd F:\Repositories\Azusa\rag
python run_workflow.py
```

这个脚本会复用现有的检索评测和 answer 验证逻辑，并把关注点尽量收敛到两份核心产物：

- `output/eval_runs/workflow_summary.json`
  - 结构化汇总四个 mode 的检索指标、delta、差异最大的 case、answer spot checks
- `output/eval_runs/workflow_report.md`
  - 供人类快速扫读的结论报告

默认行为：

- 直接使用现成的 `output/artifacts/embedding.jsonl`
- 使用 `data/eval/min_eval.jsonl`
- 对 `brute / hnsw / hybrid / hybrid_rerank` 四种 mode 做汇总比较
- 额外对 `hybrid_rerank` 跑 3 条代表性 `answer + citation` spot checks

可选参数示例：

```powershell
python run_workflow.py --rebuild-artifacts
python run_workflow.py --import-hf-eval --eval-path data\eval\hf_squad_sample_eval.jsonl
python run_workflow.py --skip-answer-checks
python run_workflow.py --output-name hf_workflow --eval-path data\eval\hf_squad_sample_eval.jsonl
```

建议把它理解成：

- `run_search.py`：单题精看
- `run_eval.py`：纯检索指标
- `run_workflow.py`：一键生成“这轮实验到底谁更好”的总结

# 先决条件

## 1. 进入环境

```powershell
conda activate ChuoEnv
cd F:\Repositories\Azusa\rag
```

## 2. 启动 azusa_server

如果要跑 `hnsw`、`hybrid`、`hybrid_rerank`，必须先启动 Rust 检索服务。

```powershell
cd F:\Repositories\Azusa
cargo run --bin azusa_server
```

说明：

- `azusa_server` 当前使用内存索引
- 服务重启后索引会清空
- 如果换了一批新的 `embedding.jsonl` 或新的数据集，最好先重启服务，再重新跑检索，避免旧索引和当前 Python 侧记录错位

# 验证顺序

下面这套顺序是当前最推荐的验证逻辑。照着跑，能比较清楚地判断：

- 数据管线有没有通
- 检索质量如何
- rerank 是否带来收益
- `answer + citation` 是否真的建立在检索证据上

---

## 第 1 步：生成 chunk 和 embedding

如果你已经有现成的 `output/artifacts/chunks.jsonl` 和 `output/artifacts/embedding.jsonl`，这一步可以跳过。

```powershell
cd F:\Repositories\Azusa\rag
python run_main.py
python run_embed.py
```

### 产物

- `output/artifacts/chunks.jsonl`
- `output/artifacts/embedding.jsonl`

### 怎么看

`chunks.jsonl` 每行应包含：

- `doc_id`
- `chunk_id`
- `chunk_index`
- `source`
- `text`
- `page`

`embedding.jsonl` 每行应包含：

- `doc_id`
- `chunk_id`
- `chunk_index`
- `source`
- `embedding`
- `page`
- `text`

### 能得出什么结论

- 如果 `chunks.jsonl` 正常生成，说明文档读取和 chunk 切分通了
- 如果 `embedding.jsonl` 正常生成，说明 embedding 链路通了

---

## 第 2 步：导入一小批 HF QA 样本

这一套样本比较适合验证英文事实题的检索和 `answer + citation`。

说明：

- 这一步需要网络访问 Hugging Face 的 rows API
- 如果本机网络权限被拦，导入会失败；这不影响本地已有数据上的检索与评测

```powershell
cd F:\Repositories\Azusa\rag
python run_import_hf_eval.py --length 20
```

### 产物

- `data/raw/autoevaluate_squad_sample_*.txt`
- `data/eval/hf_squad_sample_eval.jsonl`
- `data/queries/*.txt`

### 怎么看

- `hf_squad_sample_eval.jsonl` 里每行应有：
  - `query_id`
  - `query`
  - `relevant_chunk_ids`
  - `notes`

### 能得出什么结论

- 这说明当前 chunk 规则和评测 gold 的映射是通的
- 后续你可以直接拿这些 query 做检索与 answer 验证

---

## 第 3 步：先验证检索，不带 answer

建议先跑一轮不带 LLM 的检索，确认 top-k 证据本身是否靠谱。

```powershell
cd F:\Repositories\Azusa\rag
python run_search.py --mode brute --top-k 3
python run_search.py --mode hnsw --top-k 3
python run_search.py --mode hybrid --top-k 3
python run_search.py --mode hybrid_rerank --top-k 3
```

### 产物

- `output/search_runs/search_result_brute.json`
- `output/search_runs/search_result_hnsw.json`
- `output/search_runs/search_result_hybrid.json`
- `output/search_runs/search_result_hybrid_rerank.json`

### 怎么看

重点看每个 JSON 里的：

- `query`
- `results`
- `results[].chunk_id`
- `results[].preview`
- `results[].full_text`
- `results[].retrieval_source`（`hybrid` / `hybrid_rerank` 时）

### 怎么比对

- `brute`：作为最直接的向量检索基线
- `hnsw`：看是否和 brute 的前排结果接近
- `hybrid`：看 sparse 补进来后，是否出现更符合字面的候选
- `hybrid_rerank`：看 rerank 后前排结果是否更聚焦问题本身

### 能得出什么结论

- 如果 `brute` 和 `hnsw` 大体一致，说明 Rust HNSW 桥接是可信的
- 如果 `hybrid` 比 `dense only` 更常捞到字面关键句，说明 sparse 补充是有意义的
- 如果 `hybrid_rerank` 能把更相关的 chunk 提前，说明 reranker 起作用了

如果你想额外保留一次检索快照，再手动传：

- `--output-name verify_brute`
- `--output-name verify_hybrid_rerank`

---

## 第 4 步：跑检索评测

```powershell
cd F:\Repositories\Azusa\rag
python run_eval.py --mode all --top-k 3
```

### 产物

- `output/eval_runs/eval_result_brute.json`
- `output/eval_runs/eval_result_hnsw.json`
- `output/eval_runs/eval_result_hybrid.json`
- `output/eval_runs/eval_result_hybrid_rerank.json`

### 怎么看

重点看：

- `avg_hit_at_k`
- `avg_recall_at_k`
- `avg_mrr`
- `cases`

每个 `cases[]` 里重点看：

- `query_id`
- `relevant_chunk_ids`
- `retrieved_chunk_ids`
- `matched_chunk_ids`
- `hit_at_k`
- `recall_at_k`
- `mrr`

### 怎么比对

- 横向比较四个 mode 的 `avg_recall_at_k`
- 横向比较四个 mode 的 `avg_mrr`
- 找出 `hybrid_rerank` 比 `hybrid` 更好的 case
- 找出仍然失败的 case，看是检索漏召回，还是排序不理想

### 能得出什么结论

- `avg_recall_at_k` 更高：说明召回更完整
- `avg_mrr` 更高：说明正确 chunk 排得更靠前
- 如果 `hybrid_rerank` 明显优于 `hybrid`，说明 rerank 确实带来收益

如果你想额外保留一次评测快照，再手动传：

- `--output-name serial_verify_eval`
- `--output-name ten_query_eval`

---

## 第 5 步：验证最小 answer + citation

这一步用于确认：小 LLM 不是在“自由聊天”，而是在“基于检索证据组织一句答案并给出 citation”。

### 推荐先跑的 3 条 query

```powershell
cd F:\Repositories\Azusa\rag
python run_search.py --mode hybrid_rerank --top-k 3 --with-answer --output-name answer_massachusetts --query-text "In what year did Massachusetts first require children to be educated in schools?"
python run_search.py --mode hybrid_rerank --top-k 3 --with-answer --output-name answer_jr_tokyo_metro --query-text "JR常磐缓行线和东京地铁是什么关系？"
python run_search.py --mode hybrid_rerank --top-k 3 --with-answer --output-name answer_insufficient --query-text "这份材料有没有明确提到哈佛大学校长的名字？"
```

### 产物

- `output/search_runs/answer_massachusetts_hybrid_rerank.json`
- `output/search_runs/answer_jr_tokyo_metro_hybrid_rerank.json`
- `output/search_runs/answer_insufficient_hybrid_rerank.json`

### 怎么看

重点看 JSON 顶层新增的这些字段：

- `answer`
- `answer_raw`
- `answer_citations`
- `answer_context`

同时继续看：

- `results`
- `results[].chunk_id`
- `results[].full_text`

### 怎么比对

#### A. 验证 answer 是否来自 evidence

看 `answer` 里的关键信息，能不能在 `answer_context` 或被引用的 `results[].full_text` 里找到。

例子：

- 如果 `answer` 是 `答案：1852年`
- 那你应当能在被引用的 chunk 文本里看到 `1852`

#### B. 验证 citation 是否引用了真实 chunk

看 `answer_citations` 中的每个 `chunk_id`：

- 是否真的出现在 `results[].chunk_id` 中
- 对应的 `full_text` 是否真的支撑答案

#### C. 验证 evidence 不足时是否保守

对“证据不足题”：

- 理想输出是 `答案：证据不足`
- 或至少不要编造一个 evidence 中不存在的实体答案

### 能得出什么结论

- 如果 `answer` 正确且 `answer_citations` 对应到真实证据，说明最小 `answer + citation` 链路已经成立
- 如果 `answer_raw` 偶尔格式飘，但 `answer` 经过后处理后稳定，说明这版系统已经具备最小可用性
- 如果 answer 经常对，但 citation 对不上，说明检索结果没问题，answerer 的 prompt / 后处理还需要继续收

如果你只想保留“当前最新版带 answer 的结果”，可以不传 `--output-name`，直接覆盖：

- `output/search_runs/search_result_hybrid_rerank.json`

---

## 第 6 步：比较“只检索”和“检索 + answer”的差别

推荐拿同一条 query 各跑一次：

```powershell
python run_search.py --mode hybrid_rerank --top-k 3 --output-name compare_retrieval_only --query-text "JR常磐缓行线和东京地铁是什么关系？"
python run_search.py --mode hybrid_rerank --top-k 3 --with-answer --output-name compare_with_answer --query-text "JR常磐缓行线和东京地铁是什么关系？"
```

### 产物

- `output/search_runs/compare_retrieval_only_hybrid_rerank.json`
- `output/search_runs/compare_with_answer_hybrid_rerank.json`

### 怎么比对

- 前者主要看 `results`
- 后者除了 `results`，还看 `answer / answer_citations`

### 能得出什么结论

- 前者回答的是：`系统捞到了哪些证据`
- 后者回答的是：`系统基于这些证据组织出了什么答案`

这就是“检索实验”与“最小 RAG demo”的区别。

# 当前最推荐的手工验收方法

如果你只想花几分钟快速确认当前系统有没有工作，按下面这个最小清单来：

1. 启动 `azusa_server`
2. 跑一条不带 answer 的 `hybrid_rerank`
3. 跑一条带 answer 的 `hybrid_rerank`
4. 打开输出 JSON，对照：
   - `results[].chunk_id`
   - `results[].full_text`
   - `answer`
   - `answer_citations`

只要你能确认：

- top-k 检索结果本身相关
- answer 里的关键信息能在 cited chunk 中找到

就可以得出一个当前阶段很重要的结论：

**这个项目已经不是只有 retrieval 结果，而是已经有一个最小可运行的 `answer + citation` RAG v0 了。**

# 当前已知限制

- `answerer` 使用的是很小的 `Qwen2.5-0.5B-Instruct`，格式稳定性依赖 prompt + 后处理共同约束
- `answer_context` 当前会直接保留 top-k 全部证据文本，结果文件会比较长
- `run_eval.py` 当前仍然只评检索层，不评 `answer + citation`
- 当前最小评测更适合验证“检索 + 证据对齐”，还不是完整答案质量评测

# 常见问题

## 1. `hnsw` / `hybrid` / `hybrid_rerank` 报连接错误

大概率是没启动 `azusa_server`。

## 2. 输出文件写不进去

如果某个 JSON 正被编辑器或其他进程占用，可以换一个 `--output-name` 再跑一次。

## 3. answer 偶尔格式不稳

这是小模型常见现象。当前版本会把模型原始输出写进 `answer_raw`，同时用后处理收成较稳定的：

- `answer`
- `answer_citations`

## 4. 为什么 answer 看起来只是“说几句话”

因为当前 answerer 的职责不是替代检索，而是：

- 根据 top-k 证据组织一句自然语言回答
- 把答案和 chunk 证据对齐

检索 / hybrid / rerank 依然是这套系统里更核心的部分。
