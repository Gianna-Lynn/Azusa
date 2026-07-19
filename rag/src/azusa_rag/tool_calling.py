"""
Agent工具调用的实现.
数据流:
evidence
  ↓
ToolCall
  ↓
execute_with_retry
  ↓
execute_tool_call
  ↓
executor
  ↓
ToolAttempt
  ↓
ToolExecutionTrace
  ↓
build_tool_context
  ↓
Answerer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .schemas import EmbeddingRecord


GET_CHUNK_TOOL_NAME = "get_chunk_by_id"

# 这些字符串常量可以放进class ToolAttempt当中的error_type.
ERROR_UNKNOWN_TOOL = "unknown_tool"
ERROR_MISSING_ARGUMENT = "missing_argument"
ERROR_EXECUTION = "execution_error"
ERROR_NOT_FOUND = "tool_not_found"

@dataclass(frozen=True)
class ToolDefinition:
    """描述一个可被 Agent 调用的工具及其参数契约。"""

    name: str
    description: str
    required_arguments: tuple[str, ...]


@dataclass(frozen=True)
class ToolCall:
    """记录一次逻辑工具调用的唯一编号、调用的工具名和传给工具的参数。"""

    call_id: str                    #  call_id="call_001",
    tool_name: str                  #  tool_name="get_chunk_by_id",
    arguments: dict[str, Any]       #  arguments={"chunk_id": "chunk_001"},

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 Agent JSON trace 的普通字典。"""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }


@dataclass
class ToolAttempt:
    """记录工具真正执行了一次实际尝试后的状态、错误和返回结果。
    一次ToolCall可能包含多个ToolAttempt."""

    attempt: int    # 第几次尝试
    status: str     # 成功还是失败
    latency_ms: float = 0.0     # 本次执行的耗时
    error_type: str | None = None   # 失败类型
    error_message: str | None = None    # 失败时的具体错误信息
    result: dict[str, Any] | None = None    # 成功时的返回结果

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 Agent JSON trace 的普通字典。"""
        return {
            "attempt": self.attempt,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "result": self.result,
        }


@dataclass
class ToolExecutionTrace:
    """
    记录一次逻辑调用及其全部执行尝试，方便写入 Agent trace。
    把一个ToolCall和它所有的ToolAttempt放在一起.
    """
    call: ToolCall      # 包含call_id, tool_name, arguments                             
    attempts: list[ToolAttempt] = field(default_factory=list)   # 形如 attempts = [attempt_0 , attempt_1]
    used_in_answer: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入最终 Agent JSON 的完整工具调用履历。"""
        return {
            "call": self.call.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "used_in_answer": self.used_in_answer,
        }


def get_tool_definitions() -> list[ToolDefinition]:
    """返回一个工具列表(工具注册表). 每个列表中。"""
    return [
        ToolDefinition(
            name=GET_CHUNK_TOOL_NAME,
            description="根据 chunk_id 读取一条证据的完整文本和元数据。",
            required_arguments=("chunk_id",),
        )
    ]


def get_chunk_by_id(
    chunk_id: str,
    embedding_records: list[EmbeddingRecord],
) -> dict[str, Any]:
    """根据 chunk_id 读取证据记录并返回可序列化的字典。"""
    for embedding_record in embedding_records:
        if embedding_record.chunk_id == chunk_id:
            return embedding_record.to_dict()

    raise KeyError("Cannot find chunk_id")


def execute_tool_call(
    tool_call: ToolCall,
    embedding_records: list[EmbeddingRecord],
    attempt: int,
    executor: Callable[..., dict[str, Any]] | None = None,  # executor(...) -> dict[str, Any]
) -> ToolAttempt:
    """
    执行"一次"工具尝试, 返回ToolAttempt
    executor是一个"可调用对象", 是实际执行工具工作的函数.
    executor默认调用get_chunk_by_id( chunk_id="chunk_001", embedding_reords=embedding_records, )
    """
    # 传入的ToolCall形如:
    # {
    #     "tool_name": "get_chunk_by_id",
    #     "arguments": {"chunk_id": "chunk_001"}
    # }
    # 预先加载要返回的tool_attempt
    import time
    timer_start = time.perf_counter()

    def finish() -> ToolAttempt:
        tool_attempt.latency_ms = (time.perf_counter() - timer_start) * 1000
        return tool_attempt


    tool_attempt = ToolAttempt(attempt= attempt,status= "failed") # 创建对象时必须提供必需字段. ToolAttempt不能无参数创建. attempt和status没有默认值
    tool_attempt.result = None

    # 获取所有工具的列表, 和本次想调用的工具名, 本次传入的参数.
    tool_definitions_list = get_tool_definitions()
    _tool_name = tool_call.tool_name
    _arguments = tool_call.arguments  # _arguments的类型是dict[str, Any], 形如{"chunk_id": "chunk_001"}
       
    # 工具名称校验
    tool_definition = None
    for definition in tool_definitions_list:
        if tool_call.tool_name == definition.name:
            tool_definition = definition
            break
    if tool_definition is None:
        tool_attempt.error_type = ERROR_UNKNOWN_TOOL
        tool_attempt.error_message = ERROR_UNKNOWN_TOOL

        return finish()
    
    # 工具参数校验
    # 检查: required_arguments 中的每个参数名，是否都存在于 arguments 的键中。
    required_args = tool_definition.required_arguments # 形如 ("chunk_id",)
    for required_arg in required_args:
        if required_arg not in _arguments.keys(): # keys()方法返回的数据类型是dict_keys, 不是list, 不是set, 但是可以迭代.
            tool_attempt.error_type = ERROR_MISSING_ARGUMENT
            tool_attempt.error_message = ERROR_MISSING_ARGUMENT
            return finish()
    
    # 工具名称校验通过, 工具参数校验也通过.
    tool_attempt.error_type = None
    tool_attempt.error_message = None

    # 考虑executor为空的情况, 调用默认的executor
    if executor is None and _tool_name == GET_CHUNK_TOOL_NAME:
        executor = lambda **kwargs: get_chunk_by_id(kwargs["chunk_id"], embedding_records=embedding_records)

    # 尝试调用工具.
    try:
        tool_attempt.result = executor(**_arguments)
        tool_attempt.status = "success"
    except Exception as exc:
        # 捕获异常.
        # 直接写except: 会捕获包括键盘中断在内的所有异常，不利于调试。
        # 可以考虑写 except Exception as exc: 这样可以捕获普通的运行时错误, 并把错误对象保存到变量exc, 后面可以使用str(exc)来获取具体的错误文字.
        tool_attempt.status = "failed"
        tool_attempt.error_type = ERROR_EXECUTION
        tool_attempt.error_message = str(exc)
        tool_attempt.result = None
    return finish()
    # 返回值形如:
    # ToolAttempt(
    #     attempt=0
    #     status="failed"
    #     error_type="unknown_tool",
    # )
    # ToolAttempt(
    #     attempt=0,
    #     status="success",
    #     result={"chunk_id": "chunk_001", "text": "..."},
    # )

    

def execute_with_retry(
    tool_call: ToolCall,
    embedding_records: list[EmbeddingRecord],
    max_attempts: int = 2,
    executor: Callable[..., dict[str, Any]] | None = None,
) -> ToolExecutionTrace:
    """
    调用链的入口, 执行完整的重试流程, 返回ToolExecutionTrace.
    ToolCall最先被传给execute_with_retry, 然后尝试max次tool_attempt, 返回ToolExecutionTrace.
    """
    trace = ToolExecutionTrace(call=tool_call)
    trace.attempts = []
    trace.used_in_answer = False
    for i in range(max_attempts):
        
        # # 考虑函数没有传入自定义executor的情况: 默认执行函数是get_chunk_by_id
        # if executor is None and tool_call.tool_name == GET_CHUNK_TOOL_NAME:
        #     # 如果调用者没有传入自定义 executor，怎样把 get_chunk_by_id 和 embedding_records 绑定成一个可调用对象？
        #     executor = lambda **kwargs: get_chunk_by_id(kwargs["chunk_id"], embedding_records=embedding_records)
        # 由于已经在 execute_tool_call中实现了默认executor, 就不需要在上层函数中单独再次配置默认executor, 避免重复.
        tool_attempt = execute_tool_call(tool_call=tool_call, embedding_records=embedding_records, attempt=i, executor=executor)
        trace.attempts.append(tool_attempt)
        if tool_attempt.status == "success": 
            # 此处不要使用 .result is not None 来判断, 因为失败也可能错误地留下结果, 要优先相信status
            # 比如attempt.status = "failed" , attempt.result = {"partial": "残留数据"}
            trace.used_in_answer = True
            # 当前语义：某次尝试成功后，这个 trace 的结果允许进入回答流程。
            return trace
    return trace


def build_tool_context(tool_trace: ToolExecutionTrace) -> str:
    """
    接收execute_with_retry返回的ToolExecutionTrace,
    把attempt等于"success"的工具结果提取出result, 收集起来转换成 Answerer 可消费的上下文
    """
    
    # tool_trace
    # ├── call
    # │     ├── tool_name
    # │     └── arguments
    # └── attempts
    #         └── 每个 attempt
    #             └── result
    ans = str("")
    for attempt in tool_trace.attempts:
        # tool_trace.used_in_answer 表示整个工具调用(tool_trace)的结果是否被送进了答案的上下文.
        #                           本函数只判断attempt是否成功, 所以不需要判断"这条trace"是否允许进入答案.
        # attempt.status 表示这次尝试是否成功.
        if attempt.status == "success" and attempt.result is not None:
            ans += " "
            ans += str(attempt.result["chunk_id"]) # 右边 形如 "chunk_001"
            ans += " "
            ans += str(attempt.result["text"])   # 右边 是单个字符串.
    return ans
