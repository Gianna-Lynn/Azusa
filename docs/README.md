# Azusa

特急Azusa(あずさ、梓)是JR东日本中央本线(特急)列车, 常见于新宿至松本区段.  东京是我心中的理想国. 用这个列车名来命名我的第一个向量数据库项目, 就显得再好不过了.

Azusa 是一个基于 HNSW 的向量检索 / RAG 实验项目。

当前主线分两层：

- Rust HNSW：内存索引、top-k 检索、metadata 查询、HTTP demo 服务。
- Python RAG：chunk -> embedding -> retrieval -> rerank -> answer -> agent。

## 快速入口

Rust 服务：

```powershell
cargo run --bin azusa_server
```

RAG 子项目：

```powershell
cd rag
python run.py --help
```

SIFT1M benchmark：

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

## 当前能力

- `HnswIndex` 支持插入向量记录、搜索 top-k、返回 id / score、按 id 取 metadata。
- `azusa_server` 提供最小 `health -> insert -> search -> record` HTTP 闭环。
- `sift1m_benchmark` 支持 recall@k / exact@k 复测，并输出 Markdown / CSV。
- `rag/run.py` 统一调度 chunk、embed、search、eval、workflow、agent。

## 文档入口

- RAG 怎么跑：`rag/RAG_README.md`
- HTTP API：`docs/http_api.md`
- Benchmark：`docs/benchmark.md`
- 当前 RAG badcase：`rag/docs/badcase_v1.md`

历史说明、阶段总结、接口取舍和长报告都放在 `docs/archive/` 或 `rag/docs/archive/`。

## 当前限制

- HTTP 服务仍是内存索引，服务重启后记录会丢失。
- 暂无持久化、删除、更新、批量导入和并发压测。
- SIFT1M 原始数据不随仓库提交，benchmark 依赖本机或服务器上的 `data/sift1m/sift/`。
- RAG 子项目仍是实验链路，还不是完整应用。
