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

## L1：边界实现

L1 已提供文件位置、函数签名、数据模型和精确输入输出契约。严格类型校验不能依赖 Python
真值转换：`bool` 不得冒充整数，未知字段不得静默忽略，跨进程身份不得使用 `hash()`。

完成 L1 后，loader/filter/eval 的局部函数应独立变绿；后续 L2/L3 的
`NotImplementedError` 不得遮挡当前反馈。

## L2：职责组织

L2 保留公开模型和黑盒契约，由人组织内部数据流。必须维持：

1. 相同内容 chunk 在无关位置变化后保持 `chunk_id`；重复内容用 occurrence 消歧；
2. embedding 只在 `chunk_id + content_hash` 均匹配时复用；
3. 删除文档后旧 chunk 不再进入当前 artifacts；
4. metadata filter 与 ACL 是两个独立 predicate；未知 visibility fail closed；
5. 三份 artifact 单文件原子写入，不宣称多文件事务。

## L3：实验工程交付

L3 只规定产物与验收边界。实验矩阵必须覆盖多个预构建 chunk variant、多个检索 mode 与
多个 top-k；每个失败单元格都要进入报告，命令最终以非零状态退出，不得只展示成功格。

必须交付：

- `benchmark.json`：完整机器可读 report；
- `benchmark.csv`：固定字段、UTF-8 BOM；
- `benchmark.md`：指标表、P50/P95 口径、失败单元格；
- Hit@k、Recall@k、MRR、nDCG@k、P50/P95；
- 明确说明跨机器 latency 不可直接比较。

## 完成边界

课程完成不能只看一次全量测试。必须同时确认：每个窄测试按顺序变绿、全量测试通过、
`python run.py kb-sync --help` 与 `python run.py benchmark-lab --help` 可用、BM25 smoke 生成
三份报告，并能解释 `restart-and-rebuild-in-memory-hnsw` 的限制。
