# Stage 1 总结

Stage 1 的目标是把 Azusa 从一个可运行的 HNSW 原型，整理成一个可以验证、演示和说明的独立 artifact。本阶段已经完成，不继续扩展核心功能。

## 已完成内容

- Rust HNSW 索引支持向量记录插入、批量插入、top-k 检索和按 id 取回 metadata。
- SIFT1M benchmark 支持官方 ground truth、Recall@k、Exact@k，以及 CSV/Markdown 输出。
- Rust HTTP 服务提供 `health -> insert -> search -> record` 最小闭环。
- Python/RAG 可以通过 HTTP 调用 Azusa，但 RAG、Hybrid、Rerank 和 Agent 属于后续阶段，不计入 Stage 1 验收。

## 交付物

- 项目说明：`docs/AZUSA_README.md`
- HTTP demo：`docs/azusa_server_readme.md`
- HTTP smoke：`scripts/http_smoke_test.ps1`
- 接口决策：`docs/interface_decision.md`
- Benchmark 说明：`docs/benchmark.md`
- 历史性能记录：`docs/archive/perf_logs.md`
- 简历描述：`docs/archive/resume_bullets.md`

## 2026-06-24 本机复验

HTTP smoke：

```powershell
cargo run --bin azusa_server
powershell -ExecutionPolicy Bypass -File .\scripts\http_smoke_test.ps1
```

结果：`GET /health`、`POST /insert`、`POST /search`、`GET /record/1` 和缺失记录的 `404` 检查全部符合预期。

API 与轻量召回测试：

```powershell
cargo test --test hnsw_api_smoke_test
cargo test sift1m_recall_smoke_test -- --nocapture
```

结果：5 个 API smoke 全部通过；SIFT1M recall smoke 通过。

轻量 benchmark：

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

结果：

| base_limit | query_limit | k | ef_search | Recall@5 | Exact@5 | Avg search time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 10 | 5 | 60 | 98.00% | 9/10 | 116.07 us |

这次只做轻量复验，没有重新运行全量 1M benchmark。

## 阶段边界

Stage 1 到此冻结。后续评测收口属于 Stage 2；Search Agent 与 Tool Calling 属于 Stage 3。除非出现明确回归，不再为 Stage 1 修改 HNSW、HTTP 接口或核心数据结构。
