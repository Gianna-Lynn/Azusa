# 项目概览

最小 RAG 启动骨架，当前的目标是 `raw document -> chunks.jsonl`

# 当前目标

- 读取 `data/raw/` 中的原始文档
- 将文档切分为 chunk
- 为每个 chunk 补充基础 metadata
- 输出到 `output/chunks.jsonl`

# 当前范围

- 只支持最小链路
- 先以 `txt` 文件为主
- 暂不接入 embedding、hybrid、rerank、eval
- 暂不调用 Azusa Rust HTTP 服务

# 目录说明

- `data/raw/`：原始输入文档
- `data/processed/`：中间处理产物
- `output/`：最终输出结果
- `src/azusa_rag/`：RAG 启动代码

# 启动标准

- 能从 `data/raw/` 读取一个测试文本
- 能切出若干 chunk
- 能生成 `chunks.jsonl`

# 运行方式

脚本运行
```bash
cd rag
python run_main.py
```
模块运行
```powershell
cd rag
$env:PYTHONPATH='src'
python -m azusa_rag.main
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

