"""验证 Search Agent v0 的规划与证据融合纯逻辑。"""

import unittest

from azusa_rag.agent import build_query_plan, fuse_evidence
from azusa_rag.schemas import EmbeddingRecord


def make_record(chunk_id: str) -> EmbeddingRecord:
    """构造供 Agent 融合测试使用的最小 embedding 记录。"""
    return EmbeddingRecord(
        doc_id="doc",
        chunk_id=chunk_id,  # chunk_id 需要通过外部参数指定.
        chunk_index=0,
        source="test.txt",
        embedding=[1.0],
        page=None,
        text=f"text for {chunk_id}",
    )


class QueryPlanTests(unittest.TestCase):
    """覆盖查询清理、单跳分类和显式多跳拆分。"""

    def test_single_hop_query_is_normalized(self) -> None:
        """查询清理: 识别为单跳问题, 压缩空白并移除尾部问号。"""
        plan = build_query_plan("  When   were stromules discovered???  ")

        self.assertEqual(plan.query_type, "single_hop")
        self.assertEqual(plan.rewritten_query, "When were stromules discovered")
        self.assertEqual(plan.retrieval_queries, ["When were stromules discovered"])

    def test_compound_query_is_split_into_hops(self) -> None:
        """显式多跳拆分: 带显式连接词的问题应拆成至少两个检索 hop。"""
        plan = build_query_plan("first fact and second fact")

        self.assertEqual(plan.query_type, "multi_hop")
        self.assertEqual(plan.retrieval_queries, ["first fact", "second fact"])


class EvidenceFusionTests(unittest.TestCase):
    """覆盖跨 hop 去重、重复命中优先和来源记录。"""

    def test_repeated_evidence_ranks_before_single_hit(self) -> None:
        """被两个 hop 同时命中的证据应排在只命中一次的证据之前。"""

        # 模拟两次检索的结果, 其中 common chunk 被两次检索命中, 其他 chunk 只被一次命中.
        common = make_record("common")
        only_first = make_record("only_first")
        only_second = make_record("only_second")

        # 人为构造出两次检索的结果.
        hops = [
            {
                "records": [only_first, common],
                "source_map": [
                    {"source": "bm25 sparse"},
                    {"source": "bm25 sparse"},
                ],
            },
            {
                "records": [common, only_second],
                "source_map": [
                    {"source": "bm25 sparse"},
                    {"source": "bm25 sparse"},
                ],
            },
        ]

        evidence = fuse_evidence(hops, top_k=3)

        self.assertEqual(evidence[0]["chunk_id"], "common")    # 检查 common chunk 是否排在第一位.
        self.assertEqual(evidence[0]["fusion"]["hit_count"], 2)  # 检查 common chunk 的命中次数是否为 2.
        self.assertEqual(evidence[0]["fusion"]["hop_indices"], [1, 2])  # 检查 common chunk 的 hop_indices 是否包含两个 hop.


if __name__ == "__main__":
    unittest.main()
