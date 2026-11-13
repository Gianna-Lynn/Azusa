# Kaiji 语义网关

Kaiji 是上层单 Agent 应用。Azusa 只提供检索与证据，不导入 Kaiji 的状态、工作流或
checkpoint 模块；网关通过已有检索门面执行查询。

| 接口 | 输入与结果 |
| --- | --- |
| POST /search_azusa | query、k、mode；返回 mode 和 hits |
| GET /chunks/{chunk_id} | 按稳定 chunk_id 返回同一证据正文 |

query 必须是非空字符串，k 为 1 至 20 的整数且不能为 bool，mode 为已支持检索模式。
hits 保留 chunk_id、text、source、page 和定位元数据；score 在没有真实分数时为 null。
来源标记可选，不能因缺失或较短而丢失证据。

错误使用 error.code、error.message、error.retryable。参数错误为不可重试 400，
未知 chunk 为不可重试 404；未预期检索后端失败映射为可重试 503。

## 独立运行

从 rag 目录使用仓库内固定证据集：

```powershell
python run_kaiji_gateway.py --port 3010 --embedding-path tests/fixtures/kaiji_gateway_embedding.jsonl
```

该 fixture 支持 BM25 的 stromules → 1962 → 按 chunk_id 读取相同正文闭环。
它验证 HTTP 与证据契约，不代表真实检索质量。默认不指定路径时读取正常 embedding 产物，
向量和混合模式需要对应的模型与 Rust 服务。

`python -m pytest tests/test_kaiji_gateway.py` 覆盖输入、映射、缺失来源、后端失败和
真实本地 HTTP 路由。上层跨仓库集成测试由 Kaiji 维护；其历史联调不能替代本次执行证据。
