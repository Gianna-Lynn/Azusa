# Azusa HTTP Demo

本文档记录 Azusa 当前的最小 HTTP 闭环：

- `GET /health`
- `POST /insert`
- `POST /search`
- `GET /record/{id}`

当前服务是内存版 HNSW demo，适合验证「插入 -> 搜索 -> 取回记录」链路。

## 使用约束

- 当前 `azusa_server` 只维护一份内存索引，不区分“这是哪一批数据”
- 如果你要切换到一批新的 embedding 数据，请先重启 `azusa_server`
- 否则 Python 侧当前的 `embedding_records` 和 Rust 侧已有索引可能不是同一批，后续按 `id` 回查时就会错位

## 启动服务

在第一个 PowerShell 终端，从仓库根目录启动服务：

```powershell
cargo run --bin azusa_server
```

服务默认监听：

```text
http://127.0.0.1:3000
```

如果终端停在 `Running target\debug\azusa_server.exe`，表示服务正在运行并等待 HTTP 请求。

## 运行 Demo 脚本

在第二个 PowerShell 终端，从仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\http_smoke_test.ps1
```

脚本会依次发送：

```text
health -> insert -> search -> record -> missing record
```

## 接口说明

### GET /health

用途：检查服务是否启动，并返回当前记录数。

命令：

```powershell
curl.exe "http://127.0.0.1:3000/health"
```

预期响应：

```json
{"status":"ok","record_count":0}
```

### POST /insert

用途：插入一条 `VectorRecord`。

脚本中会先把 JSON 写入临时文件，再通过 `curl.exe --data-binary @file` 发送。这样可以避免 PowerShell 传参时破坏 JSON 双引号。

请求体：

```json
{"id":1,"vector":[1.0,0.0,0.0],"metadata":{"doc_id":"demo","chunk_id":"c1","source":"manual","page":1}}
```

预期响应：

```text
HTTP/1.1 201 Created
```

如果服务未重启且重复插入相同 `id`，预期返回：

```text
HTTP/1.1 409 Conflict
```

### POST /search

用途：使用 query vector 搜索近邻。

请求体：

```json
{"vector":[1.0,0.0,0.0],"k":1,"ef_search":16}
```

预期响应：

```json
{"results":[{"id":1,"score":1.0}]}
```

### GET /record/{id}

用途：按 id 取回完整记录。

命令：

```powershell
curl.exe "http://127.0.0.1:3000/record/1"
```

预期响应：

```json
{"id":1,"vector":[1.0,0.0,0.0],"metadata":{"doc_id":"demo","chunk_id":"c1","source":"manual","page":1}}
```

### GET /record/{missing_id}

用途：验证不存在的记录返回 404。

命令：

```powershell
curl.exe -i "http://127.0.0.1:3000/record/999"
```

预期响应：

```text
HTTP/1.1 404 Not Found
```

## 一次成功输出示例

```text
1. GET /health
Expected: JSON with status=ok and current record_count.
{"status":"ok","record_count":0}

2. POST /insert
Expected: HTTP 201 Created. Re-running with the same id returns 409 Conflict.
HTTP/1.1 201 Created

3. POST /search
Expected: JSON with results containing id=1.
{"results":[{"id":1,"score":1.0}]}

4. GET /record/1
Expected: JSON for record id=1, including metadata.
{"id":1,"vector":[1.0,0.0,0.0],"metadata":{"doc_id":"demo","chunk_id":"c1","source":"manual","page":1}}

5. GET /record/999
Expected: HTTP 404 Not Found.
HTTP/1.1 404 Not Found
```

## 当前限制

- 索引只保存在内存中，服务重启后记录会丢失。
- HNSW 参数暂时固定在服务代码中。
- 当前接口只覆盖最小 demo 闭环，还没有持久化、批量导入、删除或更新接口。
- Windows PowerShell 下直接传 inline JSON 给 `curl.exe` 容易出现引号问题，推荐使用 `scripts/http_smoke_test.ps1`。
