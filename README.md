# Azusa

Azusa is a retrieval and knowledge-base research system with a Rust HNSW index and a Python RAG pipeline.

## What is implemented

- Rust HNSW insert/search/record HTTP service and SIFT1M recall/latency experiments.
- Dense, BM25, RRF hybrid, and hybrid+rerank retrieval.
- Answer citations, Search Agent traces, regression eval, and representative bad cases.
- Knowledge Base v1 with multi-format ingestion, content-addressed chunk identity, document manifests,
  incremental embedding reuse, deletion semantics, metadata filtering, and a minimal tenant/visibility ACL.
- Retrieval Benchmark Lab with Hit@k, Recall@k, MRR, nDCG@k, P50/P95 latency, JSON/CSV/Markdown output,
  and support for comparing multiple prebuilt chunk variants.

## Architecture

```text
documents -> KB manifest -> stable chunks -> embeddings
                                      |          |
                                      |          +-> dense / HNSW
                                      +------------> BM25
                                                    |
                                      RRF -> rerank -> filtered evidence
                                                           |
                                                     Kaiji gateway
```

The durable truth is the manifest plus chunk/embedding artifacts. The Rust HNSW server remains an
in-memory acceleration layer and is deterministically rebuilt after a knowledge-base change.

## Quick start

```powershell
cd rag
python run.py kb-sync
python run.py search --mode bm25 --query-text "your query"
python run.py benchmark-lab --modes bm25,brute --top-k 3,5
```

HNSW and hybrid modes require the Rust service:

```powershell
cargo run --bin azusa_server
```

## Evidence and design notes

- [Detailed project guide](docs/README.md)
- [HTTP API](docs/http_api.md)
- [SIFT1M benchmark](docs/benchmark.md)
- [RAG guide](rag/RAG_README.md)
- [Knowledge Base v1](docs/knowledge_base_v1.md)
- [Failure matrix](docs/failure_matrix.md)

## Honest boundaries

- HNSW is currently in memory; Azusa does not claim WAL, mmap, distributed shards, or online compaction.
- Metadata filtering uses over-fetch then filter in v1; this can return fewer than `k` results and is not
  equivalent to a native filtered ANN index.
- The checked-in eval sets are engineering regression assets, not claims of public-benchmark SOTA.
