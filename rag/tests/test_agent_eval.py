"""验证 Agent 评测样例的数据契约、单题评分和汇总规则。"""

import unittest
from pathlib import Path

from run_agent_eval import (
    evaluate_agent_case,
    evaluate_citations,
    evaluate_retrieved_evidence,
    load_agent_eval_cases,
    summarize_agent_cases,
)


def make_case(
    *,
    relevant_chunk_ids: list[str] | None = None,
    expected_citations: list[str] | None = None,
) -> dict:
    """构造一条最小评测样例，测试只需覆盖关心的字段。"""
    return {
        "case_id": "case_001",
        "category": "single_hop",
        "query": "When was the event?",
        "reference_answer": "1962",
        "expected_query_type": "single_hop",
        "expected_min_hops": 1,
        "relevant_chunk_ids": relevant_chunk_ids or [],
        "expected_citations": expected_citations or [],
    }


def make_result(
    *,
    retrieved_chunk_ids: list[str] | None = None,
    answer_citations: list[str] | None = None,
    with_answer: bool = False,
) -> dict:
    """构造最小 Agent 轨迹，避免单元测试启动真实检索器或语言模型。"""
    result = {
        "query_plan": {"query_type": "single_hop"},
        "hops": [{"hop_index": 1}],
        "fused_evidence": [
            {"chunk_id": chunk_id} for chunk_id in (retrieved_chunk_ids or [])
        ],
    }
    if with_answer:
        result["answer"] = "结论：测试答案"
        result["answer_citations"] = answer_citations or []
    return result


class AgentEvalDatasetTests(unittest.TestCase):
    """验证仓库内 JSONL 样例集仍满足约定的数据结构。"""

    def test_eval_set_contains_fifteen_cases_and_five_categories(self) -> None:
        """当前 Stage 3 基线由 15 题和五类关键场景组成。"""
        eval_path = Path(__file__).parents[1] / "data" / "eval" / "agent_eval.jsonl"

        cases = load_agent_eval_cases(eval_path)
        categories = {case["category"] for case in cases}

        self.assertEqual(len(cases), 15)
        self.assertTrue(
            {
                "single_hop",
                "query_rewrite",
                "multi_hop",
                "insufficient_evidence",
                "citation_badcase",
            }.issubset(categories)
        )


class AgentEvalScoringTests(unittest.TestCase):
    """验证规划、证据和 citation 是彼此独立的评分维度。"""

    def test_evidence_recall_counts_found_gold_chunks(self) -> None:
        # 场景：一道题有两个金标 chunk，但最终证据只找回其中一个。
        # 期望：找回 1/2，因此 evidence recall 为 0.5。
        case = make_case(relevant_chunk_ids=["gold_1", "gold_2"])
        result = make_result(retrieved_chunk_ids=["gold_1", "unrelated"])

        metrics = evaluate_retrieved_evidence(case, result)

        self.assertEqual(metrics["matched_chunk_ids"], ["gold_1"])
        self.assertEqual(metrics["evidence_recall"], 0.5)

    def test_without_answer_citation_metrics_are_not_applicable(self) -> None:
        # 场景：只运行了规划与检索，没有传入 --with-answer。
        # 期望：citation 指标为 None，表示“不适用”，而不是错误地记零分。
        case = make_case(expected_citations=["gold_1"])
        result = make_result(retrieved_chunk_ids=["gold_1"])

        metrics = evaluate_citations(case, result)

        self.assertIsNone(metrics["citation_precision"])
        self.assertIsNone(metrics["citation_recall"])

    def test_insufficient_case_rewards_empty_citations(self) -> None:
        # 场景：语料中没有答案，Agent 也诚实拒答并保持 citation 为空。
        # 期望：没有伪造引用，因此 citation precision 和 recall 都得 1。
        case = make_case()
        result = make_result(with_answer=True, answer_citations=[])

        metrics = evaluate_citations(case, result)

        self.assertEqual(metrics["citation_precision"], 1.0)
        self.assertEqual(metrics["citation_recall"], 1.0)

    def test_insufficient_case_penalizes_invented_citation(self) -> None:
        # 场景：语料中没有答案，但 Agent 仍输出了一个 citation。
        # 期望：这是伪造引用，precision 和 recall 都记为 0。
        case = make_case()
        result = make_result(with_answer=True, answer_citations=["invented"])

        metrics = evaluate_citations(case, result)

        self.assertEqual(metrics["citation_precision"], 0.0)
        self.assertEqual(metrics["citation_recall"], 0.0)

    def test_normal_citation_precision_and_recall_are_calculated_separately(self) -> None:
        # 场景：模型应引用 gold_1 和 gold_2，实际引用了 gold_1 和错误的 wrong。
        # 期望：两个实际引用中一个正确，且两个金标中找回一个，两项均为 0.5。
        case = make_case(expected_citations=["gold_1", "gold_2"])
        result = make_result(
            with_answer=True,
            answer_citations=["gold_1", "wrong"],
        )

        metrics = evaluate_citations(case, result)

        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["citation_recall"], 0.5)

    def test_evaluate_agent_case_keeps_reference_answer_for_manual_review(self) -> None:
        # reference_answer 当前不会自动与模型答案比较，但必须保留在结果中，
        # 这样人可以结合 trace 手动检查错实体、漏答等问题。
        case = make_case(relevant_chunk_ids=["gold_1"])
        result = make_result(retrieved_chunk_ids=["gold_1"])

        evaluated = evaluate_agent_case(case, result, latency_ms=1.0)

        self.assertEqual(evaluated["reference_answer"], "1962")
        self.assertEqual(evaluated["evidence_recall"], 1.0)
        self.assertIsNone(evaluated["citation_precision"])


class AgentEvalSummaryTests(unittest.TestCase):
    """验证多道题的指标如何合并成最终摘要。"""

    def test_summary_ignores_metrics_that_are_not_applicable(self) -> None:
        # 第一题有 evidence recall，第二题是证据不足题，recall 为 None。
        # 汇总时只平均真正有意义的分数，所以结果仍然是 0.5。
        case_results = [
            {
                "query_type_match": True,
                "hop_count_ok": True,
                "evidence_recall": 0.5,
                "citation_precision": None,
                "citation_recall": None,
            },
            {
                "query_type_match": False,
                "hop_count_ok": True,
                "evidence_recall": None,
                "citation_precision": None,
                "citation_recall": None,
            },
        ]

        summary = summarize_agent_cases(case_results, mode="bm25", top_k=5)

        self.assertEqual(summary["query_type_accuracy"], 0.5)
        self.assertEqual(summary["hop_count_pass_rate"], 1.0)
        self.assertEqual(summary["avg_evidence_recall"], 0.5)
        self.assertIsNone(summary["avg_citation_precision"])
        self.assertIsNone(summary["avg_citation_recall"])


if __name__ == "__main__":
    unittest.main()
