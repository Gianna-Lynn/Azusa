# Knowledge Base v1

## Durable state

`python run.py kb-sync` writes three atomic artifacts under `rag/output/artifacts/`:

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
