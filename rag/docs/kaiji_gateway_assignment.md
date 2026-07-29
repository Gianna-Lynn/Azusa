# Kaiji Semantic Gateway 练习

正确答案：`solution/kaiji-semantic-gateway`  
练习分支：`todo/kaiji-semantic-gateway`

环境：

```powershell
Set-Location F:\Repositories\Azusa\rag
$py = 'D:\Chuo_Data\conda_envs\ChuoEnv\python.exe'
```

## TODO 1：请求契约

在 `kaiji_gateway.py` 完成 `validate_search_request`。验证非空 query、`1 <= k <= 20`
和 `SEARCH_MODES`，所有参数错误均为不可重试的 HTTP 400。

```powershell
& $py -m pytest tests/test_kaiji_gateway.py -k validate
```

## TODO 2：检索门面

完成 `KaijiGatewayService.search_azusa`。只调用 Azusa 已有
`search_records_with_sources`，把 `EmbeddingRecord` 映射为稳定 hits；后端异常转为可重试
HTTP 503。

```powershell
& $py -m pytest tests/test_kaiji_gateway.py -k "maps or backend"
```

## TODO 3：证据读取

完成 `get_chunk_by_id`。有效 ID 返回正文与元数据；未知 ID 返回不可重试 HTTP 404。

```powershell
& $py -m pytest tests/test_kaiji_gateway.py -k chunk
```

## TODO 4：HTTP 路由

完成 JSON body 读取、`POST /search_azusa`、`GET /chunks/{chunk_id}`、状态码和结构化错误。

```powershell
& $py -m pytest tests/test_kaiji_gateway.py
```

## TODO 5：Kaiji 真实对接

启动网关后，在 Kaiji 正确答案分支运行 live integration test。解释为什么 `bm25` 可离线
验收，而其他 mode 继承模型、GPU 或 Rust HNSW 服务要求。
