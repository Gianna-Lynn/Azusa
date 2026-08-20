# Azusa v2 Failure Matrix

| Boundary | Failure | Current control | Residual risk / evidence |
| --- | --- | --- | --- |
| Ingestion | unsupported or blank document | explicit type/blank validation | scanned PDF requires OCR outside v1 |
| Identity | unrelated edit renumbers later chunks | content-addressed chunk IDs | identical repeated chunks use occurrence suffix |
| Update | unchanged text is embedded again | manifest diff and embedding reuse | model/version changes require a deliberate full rebuild |
| Delete | stale evidence remains searchable | removed chunks omitted from durable artifacts | running HNSW must be restarted/rebuilt |
| Artifact write | process stops mid-write | same-directory temporary file and atomic replace | three files are atomic individually, not a multi-file transaction |
| Tenant isolation | cross-tenant evidence leak | ACL tenant equality check | missing tenant metadata is treated as shared/public data |
| Visibility | unknown policy value becomes visible | unknown values fail closed | policy model is intentionally minimal |
| ANN filter | post-filter returns fewer than k | configurable over-fetch and trace counts | not equivalent to native filtered ANN |
| Index version | same record count but different corpus | KB v1 requires restart/rebuild after sync | legacy Rust health endpoint has no generation ID yet |
| Hybrid | one backend fails | failure remains visible to caller | no silent partial-result claim |
| Reranker | model unavailable | explicit warning and rule fallback | fallback quality is not equivalent to model reranking |
| Eval | a mode fails in matrix | failure artifact plus non-zero exit | machine/model cold-start remains environment-dependent |
| Latency | setup cost distorts first query | explicit warm-up in eval | cross-machine latency comparisons remain invalid |

## Static review questions

1. Which artifact is the durable source of truth, and why is HNSW rebuildable?
2. When is an embedding reusable after a document update?
3. Why does post-filter ANN harm recall, and how is that made observable?
4. Why are metadata filtering and authorization separate predicates?
5. What happens if the manifest and embedding artifact come from different generations?
6. Why does Benchmark Lab fail on a partial matrix instead of reporting only successful cells?
7. Which failures are honest v1 boundaries rather than hidden implementation gaps?
