from __future__ import annotations

import pytest

from run_eval import normalized_discounted_cumulative_gain, summarize_case_results


def test_ndcg_rewards_earlier_relevant_results() -> None:
    relevant = {"a", "b"}
    ideal = normalized_discounted_cumulative_gain(["a", "b", "x"], relevant)
    delayed = normalized_discounted_cumulative_gain(["x", "a", "b"], relevant)
    assert ideal == pytest.approx(1.0)
    assert 0.0 < delayed < ideal


def test_summary_contains_ndcg_and_latency_percentiles() -> None:
    cases = [
        {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, "retrieval_latency_ms": 10.0},
        {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "retrieval_latency_ms": 30.0},
    ]
    summary = summarize_case_results(cases, "bm25", 3)
    assert summary["avg_ndcg_at_k"] == 0.5
    assert summary["p50_retrieval_latency_ms"] == 10.0
    assert summary["p95_retrieval_latency_ms"] == 30.0
