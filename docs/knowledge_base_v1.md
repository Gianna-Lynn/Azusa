# Knowledge Base v1

P1 已在 `main` 实现确定性身份、增量同步和空库删除。知识库接口沿用 v1 名称，
新写入的 manifest 为 **schema_version=2**；块和向量 JSONL 仍兼容原有读取入口。
旧答案实现是参考对象，不能替代本轮编码身份与元数据刷新约束。验证入口见
[P1 验证记录](p1-validation.md)。

## 增量与写入边界

- 文档 ID 来自规范化相对路径，版本来自规范化内容 SHA256。块 ID 为文档 ID、块内容
  摘要及同内容出现次数；重复块保持不同 ID，重复文档 ID 直接拒绝。
- 只有块 ID、内容哈希、实际正文和编码身份均一致时才复用向量；当前版本、span、标题、
  section 与 metadata 始终取新块。权限元数据变化不得被旧 embedding 记录覆盖。
- CLI 的编码身份默认包含模型路径和维度。模型文件在原路径内替换、预处理或 revision
  改变时须显式传新的 `--embedding-identity`。Python 回调默认 `caller-managed-v1`，
  由调用方保证同一身份的编码语义不变；这不是模型文件自动认证。
- v1 清单可读，首次迁移到 v2 时重新编码，避免复用没有编码身份的旧向量。v2 清单
  保存两个 JSONL 的 SHA256；文件丢失、改写或混代时拒绝同步，不静默修补。
- 编码数量、向量维度、有限值、重复身份及 JSON 序列化先验证，再替换单个文件，最后
  替换清单。**三份文件不是整体原子事务**，只支持单写者；中途 IO 故障可能留下混代，
  下一次会拒绝，需恢复完整上一代或在新的独立路径重建。不要宣称 WAL 或并发事务支持。
- 删除通过当前产物不再包含旧行实现；没有 Rust 在线删除。同步后须重启并重建 HNSW。
  空集合直接同步可用；CLI 必须显式 `--allow-empty`，缺失目录即便带此参数也拒绝。

## Durable state

`python run.py kb-sync` 默认在 `rag/output/artifacts/` 写三份逐文件替换的产物：

- `knowledge_base_manifest.json`: document versions and active chunk identities;
- `chunks.jsonl`: current active evidence;
- `embedding.jsonl`: current active evidence plus vectors.

Document versions and chunk identities are content-addressed. Unchanged chunks retain their identity and
embedding when unrelated parts of a document change. Removed documents and chunks are omitted from the next
artifact generation, after which the in-memory Rust service must be restarted and rebuilt.

Supported sources are TXT, Markdown, HTML, and PDF text layers. Scanned PDFs/OCR are outside v1.
Optional `azusa_metadata.json` in the source directory maps relative source paths to metadata objects. Use it
to provide `tenant_id`, `department`, `document_type`, `language`, `visibility`, and `tags` without modifying
the source document.

## Filter and ACL semantics

`MetadataFilter` supports tenant, department, document type, language, visibility, and tag matching.
`AccessContext` separately enforces tenant isolation and public/department/private visibility. Unknown
visibility values fail closed.

ANN filtering uses over-fetch then filter. The retrieval trace records pre/post-filter counts so a caller can
observe when the result set contains fewer than the requested `k`.

## Experiment variants

Build each chunk strategy/size/overlap into separate artifact paths with `kb-sync`, then pass each prebuilt
embedding/eval pair to `benchmark-lab`:

```powershell
python run.py benchmark-lab `
  --variant natural-300-30=output/artifacts/embedding.jsonl,data/eval/min_eval.jsonl `
  --modes brute,bm25,hybrid,hybrid_rerank `
  --top-k 3,5,10
```

The runner emits JSON, CSV, and Markdown and fails the command if any matrix cell fails.
