# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Azusa is a two-component RAG research project:

1. **Rust HNSW Service** (`src/`, `Cargo.toml`) — an in-memory vector search HTTP server
2. **Python RAG Pipeline** (`rag/`) — chunking → embedding → retrieval → rerank → answer → agent

---

## Commands

### Rust (HNSW Service)

```powershell
# Start the HNSW HTTP server (port 3000) — required for `hnsw` search mode
cargo run --bin azusa_server

# Release build
cargo build --release

# Run SIFT1M benchmark
cargo run --bin sift1m_benchmark
```

### Python RAG Pipeline

All scripts run from inside `rag/` with the `ChuoEnv` conda environment activated. Each step is a standalone runner script:

```powershell
cd rag

# Step 1: Chunk raw documents → output/artifacts/chunks.jsonl
python run_main.py

# Step 2: Embed chunks → output/artifacts/embedding.jsonl
python run_embed.py

# Step 3: Search with a query (--mode: brute/hnsw/hybrid/hybrid_rerank/bm25)
python run_search.py --mode hybrid_rerank

# Step 4: Evaluate all modes on an eval set
python run_eval.py

# Step 5: Full end-to-end workflow (chunk + embed + eval all modes)
python run_workflow.py

# Step 6: Run Search Agent v0 (query classification + rewrite + multi-hop + fusion)
python run_agent.py --mode hybrid_rerank --with-answer --query-text "your query"

# Import HF SQuAD eval set into data/eval/
python run_import_hf_eval.py

# DuRetrieval workflow
python run_du_retrieval_workflow.py
```

**Install Python package (editable, from `rag/`):**
```powershell
pip install -e .
```

---

## Architecture

### Rust Layer (`src/`)

The Rust crate (`azusa`) exposes an HNSW index as an Axum HTTP service:

| Route | Method | Purpose |
|-------|--------|---------|
| `/health` | GET | Status + record count |
| `/insert` | POST | Insert a `VectorRecord` |
| `/search` | POST | top-k ANN search |
| `/record/:id` | GET | Metadata lookup by ID |

Core modules: `index` (HNSW implementation), `distance`, `types`, `legacy`.

### Python Layer (`rag/src/azusa_rag/`)

The pipeline flows through these modules in order:

```
loaders.py → chunker.py → embedder.py → search_*.py → reranker.py → answerer.py → agent.py
```

**Key modules:**
- `config.py` — all paths and model settings; single source of truth
- `schemas.py` — `ChunkRecord`, `EmbeddingRecord` dataclasses used throughout
- `modes.py` — canonical list of search modes and their display order
- `search.py` — unified `search_records()` and `search_records_with_sources()` entry points
- `search_hnsw.py` — delegates to Rust server via HTTP; **requires Rust server running**
- `search_sparse.py` — BM25 via `rank_bm25`
- `search_hybrid.py` — merges dense + sparse candidates, then optionally reranks
- `agent.py` — Search Agent v0: `classify_query` → `build_query_plan` → multi-hop retrieval → `fuse_evidence`

### Search Modes

| Mode | Family | Notes |
|------|--------|-------|
| `brute` | dense-only | Pure vector similarity; the baseline |
| `hnsw` | dense-only | Delegates to Rust server on port 3000 |
| `bm25` | sparse-only | BM25 keyword matching |
| `hybrid` | dense+sparse | Merge strategy: dense-ordered, sparse fills gaps |
| `hybrid_rerank` | dense+sparse+rerank | Best overall; uses bge-reranker-v2-m3 |

### Data & Output Paths (all relative to `rag/`)

| Path | Contents |
|------|----------|
| `data/raw/` | Source documents (`.txt`) |
| `data/eval/min_eval.jsonl` | 10-query self-built regression set |
| `data/eval/hf_squad_sample_eval.jsonl` | 5-query HF SQuAD smoke set |
| `data/queries/` | Ad-hoc query files |
| `data/toy/` | DuRetrieval toy dataset |
| `output/artifacts/` | `chunks.jsonl`, `embedding.jsonl` (generated) |
| `output/eval_runs/` | Per-mode eval JSONs, `workflow_report.md`, `eval_results.md` |
| `output/search_runs/` | Per-query search result JSONs |

### Model Configuration (`config.py`)

Models are stored locally at `D:\Chuo_Data\Models\`:
- **Embedding**: `Qwen3-Embedding-0.6B`, dim=512, CUDA
- **Reranker**: `bge-reranker-v2-m3`, CUDA

Both require CUDA. Change paths in `config.py` if models move.

### Important Dependency: Rust Server for `hnsw` Mode

`hnsw` mode makes HTTP calls to `http://127.0.0.1:3000`. Any eval or search run that includes `hnsw` will fail silently or error if `azusa_server` is not already running. Start it first with `cargo run --bin azusa_server`.
