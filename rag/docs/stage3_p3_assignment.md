# Stage 3 P3 作业：最小 Function Calling Runtime

这是一份已经完成实现并保留作设计记录的工程作业。

任务书 v1.2 对 P3 的要求是：一个工具、一次逻辑调用、一次失败重试、结构化日志，以及工具结果进入最终回答。当前实现已经覆盖这些要求，并保留本文件作为设计与验收记录。

## 目标

实现一个不依赖 MCP、不依赖外部 API 的最小 Function Calling runtime：

```text
query
→ retrieval
→ evidence fusion
→ get_chunk_by_id
→ retry once when appropriate
→ tool result enters answer context
→ citation answer
```

## 已提供的骨架

- `src/azusa_rag/tool_calling.py`
  - `ToolDefinition`
  - `ToolCall`
  - `ToolAttempt`
  - `ToolExecutionTrace`
  - 唯一工具名：`get_chunk_by_id`
  - 工具目录函数
  - 4 个已实现函数
- `tests/test_tool_calling_assignment.py`
  - 工具调用、失败恢复、失败耗尽和上下文相关验收测试
- `agent.py` 已将工具阶段接入 retrieval/evidence/answer 主链路。

## 当前实现核对

### 1. `get_chunk_by_id`

要求：

- 根据 `chunk_id` 找到 `EmbeddingRecord`。
- 返回普通字典，至少包含 `chunk_id`、`doc_id`、`source`、`page`、`text`。
- 找不到时抛出明确异常。
- 返回值必须可以 `json.dumps`。

### 2. `execute_tool_call`

要求：

- 校验工具名称。
- 校验必需参数。
- 调用工具 executor。
- 将成功、参数错误、未知工具和工具内部异常转换成 `ToolAttempt`。
- 不要在这里实现无限重试。

### 3. `execute_with_retry`

要求：

- 默认最多两次尝试。
- 每次尝试都写入 `ToolExecutionTrace.attempts`。
- 成功后立即停止。
- 失败耗尽后返回失败 trace。
- 需要明确决定哪些错误可重试，并在文档中解释。

### 4. `build_tool_context`

要求：

- 只有成功的工具结果可以进入回答上下文。
- 上下文必须保留原始 `chunk_id`。
- 失败工具不能悄悄变成 citation。
- 需要让 `answerer.py` 能够使用这段上下文。

### 5. Agent 接入

在 `agent.py` 中已完成：

- 选择一个 fused evidence 进行工具调用。
- 将可序列化的 `ToolExecutionTrace` 放入最终 JSON trace。
- 将工具上下文传入 Answerer，并记录工具结果是否允许进入回答。

## 故障注入要求

不要用错误的 `chunk_id` 冒充 retry 场景。建议在测试 executor 中模拟一次瞬时异常：

```text
attempt 1: simulated_transient_error
attempt 2: success
```

工具参数在两次尝试间保持不变，这样 retry 语义清楚，也更接近真实网络瞬时失败。

## 验收命令

基础回归：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe -m unittest discover -s tests -v
```

单独运行 P3 测试：

```powershell
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe -m unittest discover -s tests -p "test_tool_calling_assignment.py" -v
```

## 面试准备

完成后应该能回答：

- 为什么只选一个工具？
- 为什么工具按 `chunk_id` 查找？
- 哪些错误可以 retry？
- 为什么最多重试一次？
- 工具结果如何进入 citation pipeline？
- 如何证明最终回答确实使用了工具结果？
- 如果工具阶段失败，系统如何降级？

后续工作重点转为集成回归、Adapter 稳定性和边界维护。 ฅ(•̀ω•́)ฅ
