# Azusa

Azusa 是一个 Rust HNSW 向量检索与 Python RAG 实验项目，用于比较检索策略并追踪回答证据。
[Kaiji](https://github.com/Gianna-Lynn/Kaiji) 在其上通过 HTTP/Tool Adapter 实现单 Agent 应用。

## 项目重点

- **检索底层**：Rust HNSW 索引、向量记录与 metadata 查询、Axum HTTP 服务，以及 SIFT1M 召回率与延迟实验。
- **RAG 链路**：Dense、BM25、RRF Hybrid、rerank、回答引用与 Search Agent trace。
- **知识库扩展（参考分支）**：manifest、稳定 chunk 身份、增量同步与 metadata filter，配套回归评测与失败案例。

下面的 Rust 最小演示运行于 `main`。Knowledge Base、Filter 等扩展能力的参考实现位于
[`solution/azusa-v2`](https://github.com/Gianna-Lynn/Azusa/tree/solution/azusa-v2)，分支状态见下表。

## 最小体验

需要 Git、Rust stable 工具链和 Cargo；首次构建需要下载 Cargo 依赖：

```powershell
git clone --branch main https://github.com/Gianna-Lynn/Azusa.git
cd Azusa
cargo run --locked --example retrieval_demo
```

演示内置三篇简短文本和三维预设向量，完成建索引、top-1 搜索、按 ID 取回记录，再通过 ID 映射展示原文。
无需 API Key、Python、GPU、模型权重或 HTTP 服务，不读取或覆盖现有 RAG 产物。

预期标准输出：

```text
已索引 3 篇示例文本（预设向量，非模型编码）。
查询：Azusa 的索引重启后会保留吗？
命中：id=1, score=1.000, chunk_id=demo-1
原文：Azusa 的 Rust HNSW 索引保存在内存中，进程重启后需要重建。
已按 ID 取回同一条向量记录与来源信息。
```

向量与查询的对应关系由示例手工设置；该演示只验证索引接口和 ID 映射，不衡量自然语言理解、RAG 回答质量或检索性能。
实现见 [`examples/retrieval_demo.rs`](examples/retrieval_demo.rs)。

## 架构与分支

```text
documents -> KB manifest -> stable chunks -> embeddings
                                      |          |
                                      |          +-> dense / HNSW
                                      +------------> BM25
                                                    |
                                      RRF -> rerank -> filtered evidence
                                                           |
                                                     Kaiji gateway
```

上图描述完整参考实现的链路。manifest 与 chunk/embedding 产物作为持久事实来源，Rust HNSW 是可重建的内存加速层。

| 分支 | 用途 |
|---|---|
| `main` | 开发主线；Knowledge Base、Filter 等部分接口仍以 `NotImplementedError` 标记，Rust 最小演示可运行 |
| `solution/azusa-v2` | Knowledge Base 增量同步、metadata filter、最小 ACL 与 Benchmark Lab 等完整参考实现 |

Python RAG 的模型、数据和安装要求见 [RAG 指南](rag/RAG_README.md)。默认模型配置使用本机路径和 CUDA，
运行前需要按自己的环境核对；`main` 中的 `kb-sync` 等尚未实现的入口不作为开箱即用演示。

## 结果与设计记录

- [SIFT1M 实验](docs/benchmark.md)：已有 100k 调参与 1M sweep 历史摘要，附配置、规模和复测命令；最小演示不代替性能实验。
- [Search Agent 回归基线](rag/docs/stage3_agent_eval.md)：15 条工程回归样例，分别记录检索与引用结果。
- [回答与引用失败案例](rag/docs/badcase_v1.md)：保留错实体、漏答和错误引用，便于追踪限制。
- [HTTP API](docs/http_api.md)、[Knowledge Base 设计](docs/knowledge_base_v1.md)、[失败矩阵](docs/failure_matrix.md)。
- [项目详细指南](docs/README.md)。

以上评测记录均有各自的时间与适用范围，不代表当前提交重新测得的成绩，也不作为公开 benchmark 排名。

## 当前边界

- Rust 服务使用内存索引，暂无 WAL、分布式分片或在线 compaction。
- 参考实现的 metadata filter 采用 over-fetch 后过滤，可能不足 `k` 条，不等价于原生 filtered ANN。
- 小型回归集用于检查工程行为；本地小模型仍可能产生错误答案或遗漏引用。

Azusa（あずさ，梓号）的名字来自 JR 东日本中央本线特急列车；Kaiji（かいじ，甲斐路号）是同线的另一班列车。

开发与学习记录：[课程任务书](rag/docs/azusa_v2_assignment.md) · [要求与测试追踪](rag/docs/azusa_v2_traceability.md)。
