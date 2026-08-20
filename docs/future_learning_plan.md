# Azusa v2 课程分支实施说明

`solution/azusa-v2` 保留完整答案，`main` 承载课程 TODO。答案提交不进入 `main` 历史，
避免学习者通过上一提交直接取得实现。

课程按下列等级在同一 `main` 快照中组织，不为每个等级长期保留额外分支：

- **L1 — protocol implementation:** retain detailed protocols, function signatures, examples, and tests;
  hollow local boundary implementations such as validation, serialization, and deterministic helpers.
- **L2 — responsibility organization:** retain data models and black-box contracts; hollow orchestration across
  ingestion, incremental planning, filtering, artifact reuse, and error boundaries.
- **L3 — experiment delivery:** provide only the complete requirements, required artifacts, controlled experiment
  matrix, reporting format, and acceptance boundary; the learner delivers the end-to-end experiment project.

实施与验收流程：

1. 冻结并验证 `solution/azusa-v2`；
2. 在临时 work 分支制作 TODO 快照；
3. 逐题验证答案绿灯与课程精确红灯；
4. 只把最终课程快照提交到 `main`；
5. 删除临时与旧语义网关分支，长期只保留 `main` 和 `solution/azusa-v2`。

具体任务、窄测试与完成边界见 `rag/docs/azusa_v2_assignment.md` 和
`rag/docs/azusa_v2_traceability.md`。
