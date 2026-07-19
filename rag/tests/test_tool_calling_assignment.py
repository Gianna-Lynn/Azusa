"""Stage 3 P3 工具调用验收测试。"""

import unittest

from azusa_rag.schemas import EmbeddingRecord
from azusa_rag.tool_calling import (
    GET_CHUNK_TOOL_NAME,
    ToolCall,
    build_tool_context,
    execute_tool_call,
    execute_with_retry,
    get_chunk_by_id,
    get_tool_definitions,
)


def make_record(chunk_id: str = "chunk_001") -> EmbeddingRecord:
    """构造 P3 测试使用的最小 embedding 记录。"""
    return EmbeddingRecord(
        doc_id="doc_001",
        chunk_id=chunk_id,
        chunk_index=0,
        source="test.txt",
        embedding=[1.0, 0.0],
        page=None,
        text="A small evidence passage.",
    )


class ToolCallingAssignmentTests(unittest.TestCase):
    """验证工具查找、工具分发、重试和回答上下文契约。"""

    def test_catalog_exposes_exactly_one_tool(self) -> None:
        """工具目录应只暴露 get_chunk_by_id。"""
        definitions = get_tool_definitions()

        self.assertEqual([definition.name for definition in definitions], [GET_CHUNK_TOOL_NAME])

    def test_get_chunk_by_id_returns_serializable_record(self) -> None:
        """有效 chunk_id 应返回包含正文和元数据的普通字典。"""
        result = get_chunk_by_id("chunk_001", [make_record()])

        self.assertEqual(result["chunk_id"], "chunk_001")
        self.assertEqual(result["text"], "A small evidence passage.")

    def test_missing_chunk_is_structured_failure(self) -> None:
        """未知 chunk_id 应抛出可被 runtime 记录的明确异常。"""
        with self.assertRaises(Exception):
            get_chunk_by_id("missing", [make_record()])

    def test_execute_tool_call_records_success(self) -> None:
        """工具分发成功时应产生 status=success 的尝试记录。"""
        call = ToolCall("call_001", GET_CHUNK_TOOL_NAME, {"chunk_id": "chunk_001"})
        attempt = execute_tool_call(call, [make_record()], attempt=1)

        self.assertEqual(attempt.status, "success")
        self.assertEqual(attempt.result["chunk_id"], "chunk_001")
        self.assertGreaterEqual(attempt.latency_ms, 0.0)

    def test_unknown_tool_is_structured_failure(self) -> None:
        """未知工具应返回结构化失败，而不是直接抛出异常。"""
        call = ToolCall("call_unknown", "not_exists", {"chunk_id": "chunk_001"})

        attempt = execute_tool_call(call, [make_record()], attempt=1)

        self.assertEqual(attempt.status, "failed")
        self.assertEqual(attempt.error_type, "unknown_tool")
        self.assertIsNone(attempt.result)
        self.assertGreaterEqual(attempt.latency_ms, 0.0)

    def test_missing_argument_is_structured_failure(self) -> None:
        """缺少必需参数时应返回结构化失败。"""
        call = ToolCall("call_missing_argument", GET_CHUNK_TOOL_NAME, {})

        attempt = execute_tool_call(call, [make_record()], attempt=1)

        self.assertEqual(attempt.status, "failed")
        self.assertEqual(attempt.error_type, "missing_argument")
        self.assertIsNone(attempt.result)

    def test_retry_attempts_are_bounded(self) -> None:
        """第一次瞬时失败后最多重试一次，不得无限循环。"""
        calls = {"count": 0}

        def flaky_executor(**_: object) -> dict[str, str]:
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("temporary failure")
            return {"chunk_id": "chunk_001", "text": "ok"}

        call = ToolCall("call_002", GET_CHUNK_TOOL_NAME, {"chunk_id": "chunk_001"})
        trace = execute_with_retry(call, [make_record()], executor=flaky_executor)

        self.assertEqual(len(trace.attempts), 2)
        self.assertEqual(trace.attempts[-1].status, "success")

    def test_retry_stops_after_max_attempts_when_every_attempt_fails(self) -> None:
        """每次都失败时，重试次数不得超过 max_attempts。"""
        calls = {"count": 0}

        def always_fail(**_: object) -> dict[str, str]:
            calls["count"] += 1
            raise TimeoutError("always fails")

        call = ToolCall(
            "call_always_fail",
            GET_CHUNK_TOOL_NAME,
            {"chunk_id": "chunk_001"},
        )
        trace = execute_with_retry(
            call,
            [make_record()],
            max_attempts=2,
            executor=always_fail,
        )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(len(trace.attempts), 2)
        self.assertTrue(all(item.status == "failed" for item in trace.attempts))
        self.assertFalse(trace.used_in_answer)

    def test_retry_stops_after_first_success(self) -> None:
        """第一次成功后不应执行多余的尝试。"""
        calls = {"count": 0}

        def succeeds(**_: object) -> dict[str, str]:
            calls["count"] += 1
            return {"chunk_id": "chunk_001", "text": "ok"}

        call = ToolCall(
            "call_first_success",
            GET_CHUNK_TOOL_NAME,
            {"chunk_id": "chunk_001"},
        )
        trace = execute_with_retry(
            call,
            [make_record()],
            max_attempts=2,
            executor=succeeds,
        )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(trace.attempts), 1)
        self.assertTrue(trace.used_in_answer)

    def test_failed_result_is_not_added_to_answer_context(self) -> None:
        """失败尝试不应进入 Answerer context。"""
        def always_fail(**_: object) -> dict[str, str]:
            raise TimeoutError("failed")

        call = ToolCall(
            "call_failed_context",
            GET_CHUNK_TOOL_NAME,
            {"chunk_id": "chunk_001"},
        )
        trace = execute_with_retry(
            call,
            [make_record()],
            max_attempts=1,
            executor=always_fail,
        )

        context = build_tool_context(trace)

        self.assertEqual(context, "")
        self.assertFalse(trace.used_in_answer)

    def test_tool_result_is_available_to_answer_context(self) -> None:
        """成功工具结果进入 Answerer context 时必须保留 chunk_id。"""
        call = ToolCall("call_003", GET_CHUNK_TOOL_NAME, {"chunk_id": "chunk_001"})
        trace = execute_with_retry(
            call,
            [make_record()],
            executor=lambda **_: {"chunk_id": "chunk_001", "text": "ok"},
        )
        trace.used_in_answer = True

        context = build_tool_context(trace)

        self.assertIn("chunk_001", context)
    
    def test_unknown_tool_is_structured_failure(self):
        """验证execute_tool_call不会直接崩溃, 而是返回结构化失败"""
        call = ToolCall("call_unknown", "invalid_tool_name", {"chunk_id":"chunk_001"}, )
        attempt = execute_tool_call(call, [make_record()], attempt=1)

        self.assertEqual(attempt.status, "failed")
        self.assertEqual(attempt.error_type, "unknown_tool")
        self.assertIsNone(attempt.result)
    
    

if __name__ == "__main__":
    unittest.main()
