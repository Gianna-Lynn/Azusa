# Azusa HTTP Demo

本文档记录 Azusa 当前的最小 HTTP 闭环：

- `GET /health`
- `POST /insert`
- `POST /index/load`
- `POST /search`
- `GET /record/{id}`

当前服务是内存版 HNSW demo，适合验证「插入 -> 搜索 -> 取回记录」链路。

## 使用约束

- 服务维护一份内存索引。Python RAG 用完整、有序 embedding 记录的 SHA-256 指纹绑定快照，覆盖正文、向量、metadata 和顺序；数量相同不代表数据相同。
- `POST /index/load` 只向空索引装入完整快照，校验整批记录后一次发布。已经装载的数据不被替换；更换语料需要重启服务。
- 已绑定快照不能再通过 `/insert` 追加。旧的手工 `insert -> search -> record` 演示仍可用于未绑定快照的索引。
- 指纹是可信本地客户端的数据身份声明，服务端不重新计算原始 Python 记录哈希；它不提供鉴权、签名或数据真实性保证。服务仅用于受信任的本机调用。
- 所有服务重启后数据仍丢失。旧客户端/旧服务缺少指纹协议时，新的 Python 装载器明确报错，不退回仅比较数量。

## 启动服务

在第一个 PowerShell 终端，从仓库根目录启动服务：

```powershell
cargo run --bin azusa_server
```

服务默认监听：

```text
http://127.0.0.1:3000
```

可通过当前进程的 `AZUSA_BIND_ADDR` 指定监听地址；测试使用 `127.0.0.1:0`，由操作系统
分配临时端口，避免触碰已有服务。不需要修改 Windows 用户或系统环境变量。

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

用途：检查服务是否启动，并返回当前记录数、向量维度和可选快照身份。

命令：

```powershell
curl.exe "http://127.0.0.1:3000/health"
```

预期响应：

```json
{"status":"ok","record_count":0,"dimension":null,"dataset_fingerprint":null}
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

### POST /index/load

Python 首次查询前使用的原子装载接口，请求形如：

```json
{"dataset_fingerprint":"64位小写SHA-256字符串","records":[{"id":0,"vector":[1.0,0.0],"metadata":{"chunk_id":"c1"}}]}
```

指纹必须恰好为 64 位小写十六进制。ID 必须按列表顺序从零连续编号，整批维度相同，
向量必须非空且数值安全。空快照允许装载，也会绑定身份。失败不发布部分记录。
相同身份、数量及实际记录的重复装载可复用；其他装载冲突返回 `409 dataset_mismatch`。
响应与 `/health` 结构相同。该路由请求体上限为 64 MiB；其他路由沿用 Axum 默认限制。

### POST /search

用途：使用 query vector 搜索近邻。

请求体：

```json
{"vector":[1.0,0.0,0.0],"k":1,"ef_search":16}
```

上述请求用于手工插入的未绑定索引。查询快照时必须额外携带该快照的
`dataset_fingerprint`；不匹配或遗漏返回 `409 dataset_mismatch`，即使记录数量相同。
`k` 和显式 `ef_search` 必须为正整数；省略 `ef_search` 默认 16，服务端实际使用
`max(ef_search, k)`，避免候选堆容量小于请求数量。返回数量仍受语料规模与 ANN 可达候选限制。

预期响应：

```json
{"results":[{"id":1,"score":1.0}]}
```

### 输入错误与索引保护

- 空向量、维度不匹配、零预算及可能让 f32 距离运算溢出的向量返回 HTTP 400。
- 数字必须有限；为保证乘积、累加和范数运算安全，平方范数不得超过 `f32::MAX / 4`。
  零向量保持原有 Cosine 分数为零的行为。
- JSON/字段类型错误由 Axum 拒绝，返回相应 4xx（例如类型错误 422、请求体过大 413）。
- `/insert`、`/index/load`、`/search` 的 JSON 提取错误及业务错误使用
  `{"error":{"code":"...","message":"..."}}`；非法请求在修改索引前被拒绝。
- 索引锁异常返回 503；错误请求后仍应能执行健康检查和合法查询。
- Python HTTP helper 的单次等待上限为 30 秒。大型快照建图可能超时；超时不等于服务端
  已取消建图，可再次检查 health 并重试相同快照，不能因此换语料覆盖旧索引。

## P0-A 回归验证

从仓库根目录执行 `cargo test --offline --bin azusa_server --test hnsw_api_smoke_test`
和 `cargo build --offline --bin azusa_server`，然后从 `rag/` 执行：

```powershell
python -m pytest tests/test_kaiji_gateway.py tests/test_search_hnsw.py tests/test_hnsw_http_boundary.py -q -p no:cacheprovider -o pythonpath=src --basetemp=.cache/p0a-local-run
```

应看到真实 HTTP 六个用例通过，不能把缺少可执行文件时的 skipped 当作验收。
测试使用预设向量和临时端口，不加载模型、不覆盖 RAG 输出。这里验证协议与证据映射，
不代表自然语言检索质量、性能或 Azusa v2 未完成 TODO 的验收。

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
