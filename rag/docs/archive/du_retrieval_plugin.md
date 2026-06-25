# DuRetrieval Plugin Draft

## 0. 结论

`DuRetrieval` 第一版先按 passage-level eval 接入 Azusa。

核心对应关系：

```json
{
  "query_id": "问题身份证",
  "query": "问题正文",
  "relevant_chunk_ids": ["正确段落身份证1", "正确段落身份证2"]
}
```

数据集字段可以先按这套人话理解：

- `corpus.id` = 段落身份证
- `corpus.text` = 段落正文
- `queries.id` = 问题身份证
- `queries.text` = 问题正文
- `qrels.qid` = 哪个问题
- `qrels.pid` = 哪个段落是正确答案
- `qrels.score` = 相关性分数，第一版先把 `score > 0` 当作相关

第一版不额外切 chunk。也就是说，`relevant_chunk_ids` 这个字段名暂时沿用 Azusa 现有 eval 格式，但里面装的是 DuRetrieval 的 passage id。

## 1. 目标

把 `C-MTEB/DuRetrieval` 接成 Azusa 的第一套正式中文 retrieval benchmark。

这一步的重点是先让公开数据集进入现有评测链路：

- 能生成 Azusa 可读的 corpus 产物
- 能生成 Azusa 可读的 eval jsonl
- 能用 `run_eval.py` 比较 `brute / hnsw / bm25 / hybrid / hybrid_rerank`

## 2. 输入

第一版 importer 需要读三份数据：

- `C-MTEB/DuRetrieval` 的 `corpus`
- `C-MTEB/DuRetrieval` 的 `queries`
- `C-MTEB/DuRetrieval-qrels` 的 `dev`

字段映射：

| 来源 | 字段 | Azusa 第一版含义 |
| --- | --- | --- |
| `corpus` | `id` | retrieval unit id |
| `corpus` | `text` | retrieval unit text |
| `queries` | `id` | `query_id` |
| `queries` | `text` | `query` |
| `qrels` | `qid` | 对齐 `query_id` |
| `qrels` | `pid` | 进入 `relevant_chunk_ids` |
| `qrels` | `score` | `score > 0` 视为 relevant |

## 3. 输出

第一版 importer 先产出三类文件。

### 3.1 Corpus 产物

建议产物：

```text
rag/output/artifacts/du_retrieval_passages.jsonl
```

每行表示一个 passage-level retrieval unit。字段先对齐 Azusa 后续 embedding / search 需要的最小结构：

```json
{
  "doc_id": "du_retrieval",
  "chunk_id": "corpus.id",
  "chunk_index": 0,
  "source": "C-MTEB/DuRetrieval",
  "page": null,
  "text": "corpus.text"
}
```

这里的 `chunk_id` 暂时就是 passage id。`chunk_index` 第一版可以用导入顺序填。

### 3.2 Eval 产物

建议产物：

```text
rag/data/eval/du_retrieval_eval.jsonl
```

每行对应一个 query：

```json
{
  "query_id": "queries.id",
  "query": "queries.text",
  "relevant_chunk_ids": ["qrels.pid"],
  "notes": "dataset=C-MTEB/DuRetrieval; split=dev"
}
```

如果一个 query 有多个相关 passage，就把多个 `pid` 都放进 `relevant_chunk_ids`。

### 3.3 Query sample

可选产物：

```text
rag/data/queries/du_retrieval_query_001.txt
```

先抽少量 query，方便手动跑 `run_search.py` 看结果。

## 4. 第一版暂不做

- 不做 chunk-level gold 映射
- 不接 answer generation
- 不一次接多个 benchmark
- 不为了 DuRetrieval 重构 search 主链路
- 不先追全量大规模实验，先跑小样本闭环

## 5. 下一步具体要干嘛

### Step 1: 写 importer 骨架

建议新增：

```text
rag/src/azusa_rag/du_retrieval_import.py
rag/run_import_du_retrieval.py
```

第一版只负责：

- 读取少量 corpus rows
- 读取少量 queries rows
- 读取 qrels rows
- 写出 `du_retrieval_passages.jsonl`
- 写出 `du_retrieval_eval.jsonl`

### Step 2: 先做小样本

第一轮不要全量导入。建议先用：

- corpus limit: `1000`
- query limit: `20`
- qrels 只保留这些 query 对应的相关 pid

目标是先验证格式和评测链路。

### Step 3: 接入 embedding 流程

第一版可以先让 `du_retrieval_passages.jsonl` 变成新的 embedding 输入来源。

这里要确认一件事：

- 复用现有 `EmbeddingRecord` 字段
- 让检索结果里的 `chunk_id` 能和 `du_retrieval_eval.jsonl` 的 `relevant_chunk_ids` 对上

### Step 4: 跑最小 eval

目标命令形态先设计成：

```powershell
python run_eval.py --mode bm25 --eval-path data\eval\du_retrieval_eval.jsonl
```

后续再跑：

```powershell
python run_eval.py --mode all --eval-path data\eval\du_retrieval_eval.jsonl
```

## 6. 成功标准

第一版成功不看分数高低，只看链路是否闭合：

- importer 能稳定生成 corpus 和 eval 文件
- eval 文件里的 `query_id / query / relevant_chunk_ids` 正确
- embedding 产物里的 `chunk_id` 能对上 `relevant_chunk_ids`
- `run_eval.py` 能在 DuRetrieval 小样本上跑完
- 至少能得到一份 `eval_result_bm25.json`
