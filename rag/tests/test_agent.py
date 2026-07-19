"""验证 Search Agent v0 的规划与证据融合纯逻辑。"""

import json
import unittest
from unittest.mock import patch

from azusa_rag.agent import build_query_plan, fuse_evidence, run_search_agent, run_tool_stage
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

class RunToolTests(unittest.TestCase):
    """测试工具调用的正常运行, 测试故障注入"""
    def test_run_tool_stage(self) -> None:
        embedding_records = [make_record("chunk_001")]

        evidence = [
            {
                "rank": 1,
                "doc_id": "doc_001",
                "chunk_id": "chunk_001",
                "source": "source_a",
                "chunk_index": 0,
                "page": 1,
                "preview": "This is a preview of chunk 001...",
                "full_text": "This is the full text of chunk 001.",
                "fusion": {
                    "hit_count": 2,
                    "best_rank": 1,
                    "hop_indices": [1, 2],
                    "retrieval_sources": ["source_a", "source_b"],
                },
            },
        ]

        run_tool_stage_dict = run_tool_stage(evidence=evidence, embedding_records=embedding_records, inject_failure_once=False)


        # 检查：
        # trace 不为空
        # attempts 长度为 1
        # 最后一次 status 为 success
        # context 包含 chunk_id
        # context 包含 正文
        self.assertIsNotNone(run_tool_stage_dict["trace"])
        self.assertEqual(run_tool_stage_dict["trace"].call.tool_name, "get_chunk_by_id")
        self.assertEqual(len(run_tool_stage_dict["trace"].attempts), 1)
        self.assertEqual(run_tool_stage_dict["trace"].attempts[0].status, "success")
        self.assertEqual(run_tool_stage_dict["trace"].used_in_answer, True)
        self.assertIn(evidence[0]["chunk_id"], run_tool_stage_dict["context"])
        self.assertIn("text for chunk_001", run_tool_stage_dict["context"]) # 前者是由这个测试文件构建的embedding record的正文部分. 后者是工具执行完毕以后会给出的上下文.



    def test_inject_failure_once(self) -> None:
        embedding_records = [make_record("chunk_001")]
        evidence = [{"chunk_id": "chunk_001",}]
        run_tool_stage_dict = run_tool_stage(evidence=evidence, embedding_records=embedding_records, inject_failure_once=True)
        trace = run_tool_stage_dict["trace"]

        self.assertEqual(len(run_tool_stage_dict["trace"].attempts), 2)
        self.assertEqual(run_tool_stage_dict["trace"].attempts[0].status, "failed")
        self.assertEqual(run_tool_stage_dict["trace"].attempts[1].status, "success")
        self.assertIn("chunk_001", run_tool_stage_dict["context"])


class SearchAgentIntegrationTests(unittest.TestCase):
    """验证 Search Agent 主流程确实接入工具阶段。"""

    def test_search_agent_persists_tool_trace_and_passes_tool_context(self) -> None:
        record = make_record("chunk_001")
        answer_dict = {
            "answer_text": "结论：测试答案",
            "answer_raw": "答案：测试答案\n引用：chunk_001",
            "evidence_text": "[chunk_001] text for chunk_001",
            "citation_ids": ["chunk_001"],
            "uncertainty": "无",
            "citation_validation": {"status": "valid", "valid_chunk_ids": ["chunk_001"]},
            "context": "[chunk_id=chunk_001]\ntext for chunk_001\n\n[Tool context]\nchunk_001 text for chunk_001",
        }

        with patch("azusa_rag.search.search_records_with_sources") as search_mock:
            search_mock.return_value = ([record], [{"source": "test"}])
            with patch("azusa_rag.answerer.answer_with_citations", return_value=answer_dict) as answer_mock:
                result = run_search_agent(
                    query="What is in chunk_001?",
                    embedding_records=[record],
                    mode="bm25",
                    top_k=1,
                    with_answer=True,
                )

        self.assertEqual(search_mock.call_count, 1)
        answer_mock.assert_called_once()
        self.assertIn("chunk_001", answer_mock.call_args.kwargs["tool_context"])
        self.assertEqual(result["tool_trace"]["call"]["tool_name"], "get_chunk_by_id")
        self.assertEqual(result["tool_trace"]["attempts"][0]["status"], "success")
        self.assertIn("chunk_001", result["tool_context"])
        json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    unittest.main()
