# Azusa v2 分级课程任务书

参考答案：`solution/azusa-v2`  
课程主线：`main`

## 学习目标与非目标

本课程把 Azusa 从检索实验链路推进到可维护知识库，但不要求实现 WAL、在线 HNSW 删除、
分布式分片、OCR 或原生 filtered ANN。持久真相是 manifest/chunks/embeddings 三份 artifact，
Rust HNSW 仍是更新后重启并确定性重建的内存加速层。

## 学习顺序

| TODO | 等级 | 职责 | 窄测试 |
| --- | --- | --- | --- |
| L1a | L1 | 规范化 SHA-256 内容哈希 | `pytest tests/test_knowledge_base.py -k content_digest` |
| L1b | L1 | 稳定 document ID | `pytest tests/test_knowledge_base.py -k stable_document_id` |
| L1c | L1 | MetadataFilter 严格输入协议 | `pytest tests/test_filters.py -k metadata_filter_parser` |
| L1d | L1 | AccessContext 严格输入协议 | `pytest tests/test_filters.py -k access_context_parser` |
| L1e | L1 | binary relevance nDCG@k | `pytest tests/test_eval_metrics.py -k ndcg` |
| L2a | L2 | 内容寻址 chunk 与血缘 | `pytest tests/test_knowledge_base.py -k chunk_identity` |
| L2b | L2 | manifest ChangeSet | `pytest tests/test_knowledge_base.py -k plan_changes` |
| L2c | L2 | 增量 embedding 复用与删除同步 | `pytest tests/test_knowledge_base.py -k sync` |
| L2d | L2 | metadata predicate | `pytest tests/test_filters.py -k metadata_matches` |
| L2e | L2 | tenant/visibility ACL | `pytest tests/test_filters.py -k acl_allows` |
| L3a | L3 | variant × mode × top-k 实验矩阵 | `pytest tests/test_benchmark_lab.py -k run_matrix` |
| L3b | L3 | JSON/CSV/Markdown 可交付报告 | `pytest tests/test_benchmark_lab.py -k write_outputs` |

在 `rag/` 下使用：

```powershell
$py = 'D:\Chuo_Data\conda_envs\ChuoEnv\python.exe'
& $py -m pytest <上表测试> -q -p no:cacheprovider
```

## 分级边界

- L1 提供文件位置、函数签名、数据模型和精确输入输出协议，训练局部边界实现。
- L2 保留公开模型与黑盒契约，训练 ingestion、增量规划、过滤和 artifact 复用的职责组织。
- L3 只规定实验矩阵、失败语义、交付产物与验收边界，训练一次完整实验工程交付。

答案分支应让所有测试通过；课程分支未完成时，每个窄测试必须精确落到对应 TODO。

