# 项目概览

最小 RAG 启动骨架，当前已经打通 `raw txt -> chunks.jsonl -> embedding.jsonl`

# 当前目标

- 读取 `data/raw/` 中的原始文档
- 将文档切分为 chunk
- 为每个 chunk 补充基础 metadata
- 输出到 `output/chunks.jsonl`
- 将 chunk 编码为 embedding
- 输出到 `output/embedding.jsonl`

# 当前范围

- 只支持最小链路
- 先以 `txt` 文件为主
- 已接入最小 embedding 链路
- 暂不接入 hybrid、rerank、eval
- 暂不调用 Azusa Rust HTTP 服务

# 相似度口径

- Azusa 主项目早期更偏向 cosine 语义
- 随着 SIFT1M 等向量检索实验推进, 主线 ANN / benchmark 口径逐步转向 L2
- 当前 Azusa RAG 子模块在文本 embedding 场景下, 先采用 `L2 normalize + cosine` 的常见路线

# 目录说明

- `data/raw/`：原始输入文档
- `data/processed/`：中间处理产物
- `output/`：最终输出结果
- `src/azusa_rag/`：RAG 启动代码

# 启动标准

- 能从 `data/raw/` 读取一个测试文本
- 能切出若干 chunk
- 能生成 `chunks.jsonl`
- 能把 `chunks.jsonl` 编码为 `embedding.jsonl`

# 运行方式

脚本运行
```bash
cd rag
python run_main.py
python run_embed.py
```
模块运行
```powershell
cd rag
$env:PYTHONPATH='src'
python -m azusa_rag.main
python -m azusa_rag.embed_main
```

# 输出结果
- 输出文件：`output/chunks.jsonl`
- 每行一个 JSON 对象，包含：
  - `doc_id`：来源文档 ID
  - `chunk_id`：唯一标识符，形如 `cat_notes_chunk_000`
  - `chunk_index`：chunk 在文档中的顺序索引, 形如 `0`, `1`, `2` ...
  - `source`：来源文件名
  - `text`：chunk 文本内容
  - `page`：页码占位，当前 `txt` 输入下通常为 `null`
- 输出文件：`output/embedding.jsonl`
- 每行一个 JSON 对象，包含：
  - `doc_id`：来源文档 ID
  - `chunk_id`：对应的 chunk 标识符
  - `chunk_index`：chunk 在文档中的顺序索引
  - `source`：来源文件名
  - `embedding`：模型输出的向量
  - `page`：页码占位，当前 `txt` 输入下通常为 `null`
  - `text`：原始 chunk 文本内容
