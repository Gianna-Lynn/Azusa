# Knowledge Base

manifest 写入 schema_version=2，兼容读取旧 v1 清单。

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

## 持久产物

`python run.py kb-sync` 默认在 `rag/output/artifacts/` 写三份逐文件替换的产物：

- `knowledge_base_manifest.json`：文档版本、当前 chunk 身份与产物摘要。
- `chunks.jsonl`：当前有效证据。
- `embedding.jsonl`：当前有效证据及其向量。

文档版本与 chunk 身份基于内容生成。无关部分修改后，未变 chunk 可保持身份并复用向量；
删除的文档与 chunk 不进入下一代产物，随后须重启并重建 Rust 内存索引。

支持 TXT、Markdown、HTML 和 PDF 文本层，不处理扫描 PDF 或 OCR。源目录中的可选
`azusa_metadata.json` 将相对路径映射为元数据对象，可提供 tenant_id、department、
document_type、language、visibility、tags 等字段，无须改写正文。保存这些字段不等于
启用权限控制；当前分支不执行 ACL。

## 多组切块实验

使用 kb-sync 将不同切块策略、长度和重叠分别写到独立产物路径，再将预构建的
embedding/eval 对传给 benchmark-lab：

```powershell
python run.py benchmark-lab `
  --variant natural-300-30=output/artifacts/embedding.jsonl,data/eval/min_eval.jsonl `
  --modes brute,bm25,hybrid,hybrid_rerank `
  --top-k 3,5,10
```

矩阵输出 JSON、CSV 和 Markdown；任意单元格失败都会保留失败记录并使命令非零退出。
