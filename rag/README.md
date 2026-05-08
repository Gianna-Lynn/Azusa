# 这个小项目的目的
最小RAG启动骨架, 当前的目标是raw document -> chunks.jsonl

当前目标:
- 读取'rag/data/raw'中的原始文档
- 将文档切分为chunk
- 为每个chunk补充基础的metadata
- 输出到'rag/output/chunks.jsonl'

当前范围:

- 只支持最小链路
- 先看 'txt' 文件为主
- 暂不接入embedding, hybrid, rerank, eval
- 暂不调用 Azusa Rust HTTP 服务

目录说明:

- 'data/raw': 原始输入文档
- 'data/processed/': 中间处理产物
- 'output/': 最终输出结果
- 'src/azusa_rag': RAG启动代码

启动标准:

- 能从'raw/'读取一个测试文本
- 能切出若干 chunk
- 能生成 'chunks.jsonl'