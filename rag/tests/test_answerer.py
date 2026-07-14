"""验证 Answerer 的结构化输出与 citation 安全规则。"""

import unittest

from azusa_rag.answerer import answer_with_citations, normalize_answer_output
from azusa_rag.schemas import EmbeddingRecord


def make_record(chunk_id: str, text: str) -> EmbeddingRecord:
    """构造供 Answerer 单元测试使用的最小 embedding 记录。"""
    return EmbeddingRecord(
        doc_id="doc",
        chunk_id=chunk_id,
        chunk_index=0,
        source="test.txt",
        embedding=[1.0],
        page=None,
        text=text,
    )


class AnswererNormalizationTests(unittest.TestCase):
    """覆盖结构化回答解析、引用过滤和证据不足分支。"""

    def test_valid_structured_answer_keeps_explicit_citation(self) -> None:
        """模型明确给出有效 chunk_id 时应保留该引用。"""
        records = [make_record("chunk_001", "Massachusetts began in 1852.")]
        output = (
            "结论：1852\n"
            "证据：材料明确给出了年份\n"
            "引用来源：chunk_001\n"
            "不确定性：无"
        )

        result = normalize_answer_output(output, records)

        self.assertEqual(result["citation_ids"], ["chunk_001"])
        self.assertEqual(result["citation_validation"]["status"], "valid")
        self.assertIn("结论：1852", result["formatted_text"])  # assertIn 检查子字符串是否在字符串中出现

    def test_missing_citation_does_not_fallback_to_top_one(self) -> None:
        """模型没有给出引用时不得再强制回填 top-1 证据。"""
        records = [make_record("chunk_001", "Massachusetts began in 1852.")]

        result = normalize_answer_output("结论：1852", records)

        self.assertEqual(result["citation_ids"], [])
        self.assertIn("需人工复核", result["uncertainty"])

    def test_unknown_citation_is_filtered_out(self) -> None:
        """不在本次 evidence 中的 chunk_id 应被过滤。"""
        records = [make_record("chunk_001", "Massachusetts began in 1852.")]

        result = normalize_answer_output(
            "结论：1852\n引用来源：unknown_chunk",
            records,
        )

        self.assertEqual(result["citation_ids"], [])

    def test_citation_without_answer_requires_manual_review(self) -> None:
        """只有有效引用而没有答案字段时，应保留证据并要求人工审核。"""
        records = [make_record("chunk_001", "Massachusetts began in 1852.")]

        result = normalize_answer_output("引用：chunk_001", records)

        self.assertEqual(result["answer_text"], "答案缺失")
        self.assertEqual(result["citation_ids"], ["chunk_001"])
        self.assertEqual(result["citation_validation"]["status"], "answer_missing")
        self.assertEqual(result["uncertainty"], "有引用但没有答案，需要人工审核")

    def test_empty_evidence_returns_without_loading_model(self) -> None:
        """没有检索证据时应直接返回结构化拒答，不加载本地模型。"""
        result = answer_with_citations("unknown question", [])

        self.assertEqual(result["citation_ids"], [])
        self.assertIn("结论：证据不足", result["answer_text"])


if __name__ == "__main__":
    unittest.main()
