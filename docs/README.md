# 文档导航

| 目的 | 文档 |
| --- | --- |
| 运行 Python RAG、配置模型与查看输出 | [RAG 指南](rag.md) |
| 理解数据流和设计取舍 | [架构](architecture.md) |
| 调用 Rust 服务 | [HTTP API](http_api.md) |
| 构建与更新知识库 | [Knowledge Base](knowledge_base.md) |
| 运行评测、理解指标和案例 | [评测](evaluation.md) |
| 使用查询编排与工具调用 | [Search Agent](agent.md) |
| 对接上层应用 | [Kaiji 网关](kaiji_integration.md) |
| 理解适用范围和失败 | [限制与失败模式](limitations.md) |

## 实验报告

- [DuRetrieval 中文文本检索](experiments/duretrieval.md)：BM25、精确向量、HNSW 与插入顺序。
- [SIFT1M](experiments/sift1m.md)：历史向量近邻参数实验。
- [Flat/HNSW](experiments/flat_hnsw.md)：小规模合成索引成本比较。

报告保留实验日期、配置和解释边界。模型、原始数据和大型结果归档不随仓库分发；
本地原件存在不代表新 clone 已取得全部复现材料。
