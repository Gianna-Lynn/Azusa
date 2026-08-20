# Kaiji 对接说明

Kaiji 是 Azusa 上层的单 Agent Runtime。Azusa 继续负责检索与证据，不导入 Kaiji，
也不承担 AgentState、Workflow、Checkpoint、Human-in-the-loop、Trace 或 Eval 编排。

## 分支

- `main`：同一契约、测试和启动框架，Gateway 核心逻辑留作 TODO。
- `solution/azusa-v2`：包含 Gateway 与 Azusa v2 的完整可运行答案。

## 稳定接口

```text
POST /search_azusa
{"query": "...", "k": 3, "mode": "bm25"}

GET /chunks/{chunk_id}
```

Python 调用面保持为：

```python
search_azusa(query, k, mode)
get_chunk_by_id(chunk_id)
```

搜索响应包含 `hits` 和实际 `mode`。每个 hit 至少包含 `chunk_id`、`text`，并可包含
`score`、`source`、`page` 和 `metadata`。参数错误、找不到 chunk 和后端失败使用结构化
HTTP 错误，不把 traceback 暴露给 Kaiji。

## 正确答案分支运行方式

```powershell
git switch solution/azusa-v2
Set-Location rag
$py = 'D:\Chuo_Data\conda_envs\ChuoEnv\python.exe'

& $py -m pytest tests/test_kaiji_gateway.py
& $py run_kaiji_gateway.py --port 3010
```

默认加载 `output/artifacts/embedding.jsonl`。`bm25` 不需要 GPU 或 Rust 服务；
`brute` 需要 embedding model；`hnsw` 还需要根目录的 Rust `azusa_server`；
`hybrid`/`hybrid_rerank` 继承现有 Azusa pipeline 的要求。

在另一个终端切到 Kaiji 的正确答案分支，运行跨仓库验证：

```powershell
Set-Location F:\Repositories\Kaiji
git switch solution/kaiji-v0
& $py -m pytest tests/test_azusa_live_integration.py
```

## `main` 学习目标

按 `rag/docs/kaiji_gateway_assignment.md` 的顺序完成：

1. 请求参数与 mode 校验；
2. `search_azusa(query, k, mode)` 对现有检索门面的调用与响应映射；
3. `get_chunk_by_id(chunk_id)` 的稳定查找和 404；
4. HTTP JSON 路由、状态码和结构化错误；
5. Kaiji 真实 HTTP 集成回归。

所有实现必须保持有界输入、明确失败、可离线测试，并且不得复制 Kaiji Runtime 逻辑。
