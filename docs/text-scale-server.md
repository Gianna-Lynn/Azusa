# DuRetrieval full-corpus BM25 deployment

Deployed 2026-11-10 on `ssh E5_server` (host `node12`). This is a new text
retrieval experiment, separate from historical SIFT nearest-neighbor experiments.

Final status: **success**, all 2,000 queries completed. Retrieved and independently
verified locally on 2026-11-11. See [results and archive](text-scale-results-20261110.md).
The launch and progress observations below are retained as historical records.

## Scope

- Full C-MTEB/DuRetrieval corpus: 100,001 passages, 2,000 queries.
- C-MTEB/DuRetrieval-qrels dev: 9,839 judgments. Scores > 0 are binary relevant.
- Dataset revisions are pinned in `scripts/run_text_scale.py`; downloaded bytes
  are hashed in the final report. Public HF mirror is used for transport.
- BM25Okapi 0.2.2 with the tokenizer extracted from the uploaded current
  `search_sparse.py` snapshot. Descending score ties preserve corpus position.
- One index build, ten warmup queries, one sequential pass over all queries.
- Hit/Recall/MRR/binary nDCG at 5, 10, 100; mean and nearest-rank P50/P95 latency.
- Missing gold identities, duplicate corpus/query identities, or queries without
  positive gold fail the run instead of silently reducing the evaluation set.

## Server files

Independent directory: `/mnt/sevenT/zeliny/Experiments/azusa-text-20261110-a/`.
Existing server repository and KandaEnv were not modified. The private venv uses
KandaEnv system packages plus rank-bm25 0.2.2 and pyarrow 18.1.0 installed locally.
No GPU or sudo is used. The worker runs at nice 10 with BLAS thread limits of two.

- `search_sparse.py`: source snapshot, including the existing local annotations.
- `run_text_scale_v2.py`: deployed runner corresponding to the local script.
- `run-002.log`: current execution log; worker PID at launch was 14413.
- `run-002/queries.jsonl`: incremental per-query ranks, gold IDs, metrics, latency.
- `run-002/report.json`: written only after all queries succeed.
- `run-002/failure.json`: written on a caught failure; partial queries retained.
- `run-001*`: first download attempt retained; its blocked process was stopped.

Inspect progress from PowerShell:

```powershell
ssh E5_server 'tail -20 /mnt/sevenT/zeliny/Experiments/azusa-text-20261110-a/run-002.log'
```

Re-run in a NEW output directory from the server experiment directory:

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  venv/bin/python -u run_text_scale_v2.py --source search_sparse.py --output run-003
```

## Interpretation limits

This is BM25 text retrieval only: no dense, HNSW, RRF, reranking, or answer model.
It does not measure concurrent throughput. Single-run timing on a shared server
does not establish a latency SLA or statistical significance.

Index construction includes tokenization and is measured separately. Query latency
includes query tokenization, BM25 scoring, full stable sorting and top-100 ID
selection; it excludes downloading, gold metric calculation, HTTP and generation.
It therefore cannot be directly compared with the old evaluator, which rebuilds
BM25 for every query. The single-character Chinese tokenizer is deliberately
preserved and is not claimed to be an optimized Chinese retrieval baseline.

Self-tests passed locally and on the server for short-ranking metric denominators,
duplicate ranking rejection and tokenizer extraction. Deployment alone is not a
completed quality result: consult report.json or failure.json for final status.

Observed at handoff: all three full splits downloaded and identity checks passed;
index construction took 55,343 ms, warmup finished, and 14/2,000 query observations
had been flushed. A server-side fixture also confirmed NumPy stable descending
sorting matches Azusa's Python score/index ordering, including tied scores.
