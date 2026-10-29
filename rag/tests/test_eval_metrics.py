"""使用手算排名和负向样例验证评测分母、输入身份与延迟摘要。"""

from __future__ import annotations

import math
import pytest

from run_eval import evaluate_case, normalized_discounted_cumulative_gain, summarize_case_results


def test_ndcg_rewards_earlier_relevant_results() -> None:
    """相同相关证据越靠前，折扣收益越高。"""
    relevant = {"a", "b"}
    ideal = normalized_discounted_cumulative_gain(["a", "b", "x"], relevant)
    delayed = normalized_discounted_cumulative_gain(["x", "a", "b"], relevant)
    assert ideal == pytest.approx(1.0)
    assert 0.0 < delayed < ideal


def test_summary_contains_ndcg_and_latency_percentiles() -> None:
    """两个观测的 nearest-rank P50/P95 应分别取低值和高值。"""
    cases = [
        {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, "retrieval_latency_ms": 10.0},
        {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "retrieval_latency_ms": 30.0},
    ]
    summary = summarize_case_results(cases, "bm25", 3)
    assert summary["avg_ndcg_at_k"] == 0.5
    assert summary["p50_retrieval_latency_ms"] == 10.0
    assert summary["p95_retrieval_latency_ms"] == 30.0


def test_short_ranking_keeps_requested_k_denominator() -> None:
    """少返回一个相关结果不能得到满分；分母由请求 k 而非实际长度决定。"""
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert normalized_discounted_cumulative_gain(["a"], {"a", "b"}, top_k=3) == pytest.approx(expected)
    case = {"query_id": "short", "query": "q", "reference_answer": "a", "expected_citations": ["a"],
            "case_type": "short", "relevant_chunk_ids": ["a", "b"]}
    result = evaluate_case(case, ["a"], 3, 1.0)
    assert result["ndcg_at_k"] == pytest.approx(expected)
    assert result["recall_at_k"] == 0.5


@pytest.mark.parametrize("ranking,gold,k,expected", [
    ([], {"a"}, 3, 0.0),
    (["a"], set(), 3, 0.0),
    (["x", "a"], {"a"}, 2, 1 / math.log2(3)),
    (["x", "a"], {"a"}, 1, 0.0),
    (["a", "b"], {"a", "b"}, 2, 1.0),
])
def test_ndcg_handles_empty_delayed_and_truncated_rankings(ranking, gold, k, expected):
    """固定手算预期值，覆盖空集、非首位命中和 k 截断。"""
    assert normalized_discounted_cumulative_gain(ranking, gold, top_k=k) == pytest.approx(expected)


def test_duplicate_evidence_is_rejected_instead_of_inflating_metrics():
    """相同证据重复两次不得将 nDCG 或 Recall 累加成虚假成绩。"""
    with pytest.raises(ValueError, match="unique"):
        normalized_discounted_cumulative_gain(["a", "a"], {"a"}, top_k=2)
    with pytest.raises(ValueError, match="empty"):
        summarize_case_results([], "bm25", 3)
