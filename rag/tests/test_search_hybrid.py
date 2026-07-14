"""验证 RRF 混合检索融合函数的确定性行为。"""

import unittest

from azusa_rag.search_hybrid import rrf_fusion


class RrfFusionTests(unittest.TestCase):
    """
    覆盖 RRF 的共同命中、空输入、截断和同分排序。
    unittest是 Python 内置的单元测试框架，适合小型测试。
    unittest.TestCase 是所有测试类的基类，提供了断言方法和测试运行机制。
    比如说, self.assertEqual(a, b) 会检查 a 和 b 是否相等，如果不相等就会让测试失败。
    """

    def test_common_document_receives_combined_score(self) -> None:
        """多路共同命中的文档应优先于单路命中的文档。"""
        result = rrf_fusion([1, 2], [2, 3], top_k=3)

        self.assertEqual(result[0], 2)

    def test_empty_input_returns_empty_list(self) -> None:
        """没有排行榜输入时应返回空列表。"""
        self.assertEqual(rrf_fusion(top_k=3), [])

    def test_top_k_truncates_result(self) -> None:
        """融合结果数量不得超过 top_k。"""
        self.assertEqual(len(rrf_fusion([1, 2, 3], top_k=2)), 2)

    def test_tie_uses_document_id_for_deterministic_order(self) -> None:
        """RRF 分数相同时应按文档 ID 升序保证结果稳定。"""
        self.assertEqual(rrf_fusion([2], [1], top_k=2), [1, 2])


if __name__ == "__main__":
    unittest.main()
