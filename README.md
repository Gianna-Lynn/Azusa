# Azusa

Azusa 是 Rust HNSW 向量检索与 Python RAG 实验项目，用于比较检索策略，并追踪回答使用的证据。
[Kaiji](https://github.com/Gianna-Lynn/Kaiji) 通过 HTTP 网关在其上构建单 Agent 应用。

## 核心能力

- **向量检索**：Rust HNSW、精确扫描基线、向量记录查询与 Axum HTTP 服务。
- **增量知识库**：稳定文档与 chunk 身份、embedding 复用、删除同步和版本清单。
- **检索与回答**：Dense、BM25、RRF Hybrid、rerank，以及带 chunk 引用和 trace 的回答流程。
- **实验与验证**：逐题排名、检索指标、计时边界、实际重排后端和输入/源码身份记录。

## 快速体验

需要 Git 和 Rust stable 工具链；首次构建需要获取 Cargo 依赖。

```powershell
git clone --branch main https://github.com/Gianna-Lynn/Azusa.git
cd Azusa
cargo run --locked --example retrieval_demo
```

演示使用三篇自编短文本与预设向量，完成建索引、top-1 检索和按 ID 取回原文。
无需 Python、模型、GPU 或 API Key；它验证接口与映射，不衡量自然语言检索质量。

Python 的无模型 BM25 smoke、模型配置和完整运行方式见 [RAG 指南](docs/rag.md)。

## 架构

```text
documents -> stable chunks -> embeddings + manifest
                                  |          |
                                  |          +-> BM25
                                  +-> brute / Rust HNSW
                                              |
                                      RRF -> rerank
                                              |
                                      evidence -> answer
                                              |
                                       citations / trace
```

manifest、chunks 和 embeddings 是持久事实来源；Rust HNSW 是可重建的内存加速层。
设计取舍见 [架构说明](docs/architecture.md)，上层对接见 [Kaiji 网关](docs/kaiji_integration.md)。

## 实验与验证

| 范围 | 内容 |
| --- | --- |
| [真实中文文本](docs/experiments/duretrieval.md) | 100,001 段落、2,000 查询；BM25、精确向量、HNSW 与插入顺序对照 |
| [SIFT1M](docs/experiments/sift1m.md) | 历史 100k/1M 参数实验、召回率与查询耗时 |
| [Flat/HNSW](docs/experiments/flat_hnsw.md) | 小规模合成工作负载下的构建、查询与进程内存成本 |
| [工程评测](docs/evaluation.md) | 检索指标、Agent 回归、引用失败案例和可追溯报告 |

这些是各自日期、版本和配置下的历史结果，不代表当前提交重新测得的成绩。

## 当前限制

- Rust 索引在内存中，暂无 WAL 或在线删除；知识库更新后需重启并重建索引。
- 当前分支不提供 metadata filter/ACL，不构成多租户权限边界。
- Search Agent 使用规则规划；模型可能漏答或引用错误，引用 ID 合法不等于证据支持结论。

使用与接口见 [文档导航](docs/README.md)，完整边界见 [限制与失败模式](docs/limitations.md)。
`main` 提供可运行版本，`develop` 用于持续开发，`solution` 保存参考实现。

Azusa（あずさ，梓号）与 Kaiji（かいじ，甲斐路号）的名字来自 JR 东日本中央本线特急列车。
