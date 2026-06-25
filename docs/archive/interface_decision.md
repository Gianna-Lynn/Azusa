# Interface Decision

本文档记录 5.1-5.5 阶段 Azusa 对外接口方案的取舍。

## 决策

当前选择：**Rust HTTP 服务**。

服务入口：

```text
src/bin/azusa_server.rs
```

默认监听地址：

```text
http://127.0.0.1:3000
```

当前最小接口：

```text
GET  /health
POST /insert
POST /search
GET  /record/{id}
```

## 为什么选择 Rust HTTP 服务

Rust HTTP 服务是当前阶段最小、最直接的交付方式：

- 可以直接复用现有 Rust library 和 `HnswIndex`，不需要额外跨语言绑定层。
- 调试路径简单：`cargo run --bin azusa_server` 启动，`curl` 或脚本即可验证闭环。
- 对 Python/RAG 调用友好：调用方只需要发 HTTP JSON 请求，不需要理解 Rust 内部结构。
- 跨语言边界清晰：后续 Python、前端、RAG pipeline 都可以通过同一组 HTTP 接口接入。
- 适合 demo：可以明确展示「插入 -> 搜索 -> 取回记录」这条最小链路。

## 暂不选择的方案

### PyO3

暂不选择 PyO3 作为 5.1-5.5 阶段主线。

原因：

- 需要额外处理 Python wheel、ABI、环境和打包问题。
- 调试边界会从 Rust 内部扩大到 Python/Rust 绑定层。
- 当前目标是证明 Azusa 能作为服务被外部调用，而不是优先做 Python 原生包。

### gRPC

暂不选择 gRPC。

原因：

- 需要引入 proto 定义、代码生成和客户端生成流程。
- 对 demo 来说调试门槛比 HTTP JSON 更高。
- 当前接口还处在最小闭环阶段，HTTP 更容易快速观察请求和响应。

### Python 临时后端

暂不选择 Python 临时后端。

原因：

- 会让核心索引逻辑和服务层分散到两种运行时里。
- 仍然需要设计 Python 与 Rust library 的调用方式。
- 当前已有 Rust library，直接用 Rust HTTP 包一层更短。

## 降级方案

如果 HTTP 服务来不及稳定交付，保留以下独立 artifact 作为降级交付：

- Rust library 中的 `HnswIndex` 插入和搜索能力。
- 现有 smoke test 展示的基础检索路径。
- `sift1m_benchmark` 和 SIFT1M 实验记录，证明索引可构建、可搜索、可评估。
- 文档中明确说明 HTTP 服务未完成，Rust library + demo/benchmark 作为当前阶段成果。

## 当前接口状态

当前 HTTP 服务采用内存索引：

```text
AppState
  Arc<Mutex<HnswIndex>>
```

默认 HNSW 参数：

```text
metric = Cosine
level_sample_probability = 0.5
ef_construction = 8
m = 2
m_max = 4
```

这些参数优先服务 demo 和 smoke test，不代表生产配置。

## 当前限制

- 数据只保存在内存里，服务重启后记录丢失。
- 暂无持久化、删除、更新、批量导入和并发压测。
- 错误处理只覆盖最小场景：重复 id、缺失 record、基础 JSON 解析错误。
- API schema 还没有单独发布 OpenAPI 或 protobuf 定义。
