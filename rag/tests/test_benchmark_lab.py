from __future__ import annotations

import csv
import json
from pathlib import Path

import run_benchmark_lab as lab


def summary() -> dict:
    return {
        "query_count": 2,
        "avg_hit_at_k": 1.0,
        "avg_recall_at_k": 0.75,
        "avg_mrr": 0.5,
        "avg_ndcg_at_k": 0.625,
        "p50_retrieval_latency_ms": 3.0,
        "p95_retrieval_latency_ms": 7.0,
    }


def test_run_matrix_keeps_successes_and_failures(monkeypatch) -> None:
    load_counts = {"embeddings": 0, "eval": 0}

    def load_embeddings(path: Path) -> list:
        load_counts["embeddings"] += 1
        return [object()]

    def load_cases(path: Path) -> list:
        load_counts["eval"] += 1
        return [{"query": "q"}]

    def run(mode: str, cases: list, records: list, top_k: int) -> dict:
        if mode == "broken":
            raise RuntimeError("controlled failure")
        return summary()

    monkeypatch.setattr(lab, "load_embedding_records", load_embeddings)
    monkeypatch.setattr(lab, "load_eval_cases", load_cases)
    monkeypatch.setattr(lab, "run_eval_for_mode", run)

    report = lab.run_matrix(
        [("natural-300-30", Path("embedding.jsonl"), Path("eval.jsonl"))],
        ["bm25", "broken"],
        [3, 5],
    )

    assert load_counts == {"embeddings": 1, "eval": 1}
    assert len(report["rows"]) == 2
    assert len(report["failures"]) == 2
    assert {row["top_k"] for row in report["rows"]} == {3, 5}
    assert all(item["error"] == "controlled failure" for item in report["failures"])


def test_write_outputs_delivers_all_formats_and_failures(tmp_path: Path) -> None:
    row = {
        "variant": "fixed-256-0",
        "mode": "bm25",
        "top_k": 3,
        "query_count": 2,
        "hit_at_k": 1.0,
        "recall_at_k": 0.75,
        "mrr": 0.5,
        "ndcg_at_k": 0.625,
        "p50_ms": 3.0,
        "p95_ms": 7.0,
    }
    report = {
        "schema_version": 1,
        "rows": [row],
        "failures": [{"variant": "fixed-256-0", "mode": "hnsw", "top_k": 3, "error": "offline"}],
    }

    lab.write_outputs(tmp_path, report)

    assert json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8")) == report
    with (tmp_path / "benchmark.csv").open(encoding="utf-8-sig", newline="") as source:
        assert list(csv.DictReader(source))[0]["variant"] == "fixed-256-0"
    markdown = (tmp_path / "benchmark.md").read_text(encoding="utf-8")
    assert "P95 ms" in markdown
    assert "Failed cells" in markdown
    assert "offline" in markdown
