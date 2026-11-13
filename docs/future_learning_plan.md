# 分支与学习工作流

长期只维护三个本地分支：

| 分支 | 职责 | 交付边界 |
| --- | --- | --- |
| main | 对外展示与面试讲述 | 已完成、可解释、验证通过的功能；不包含施工占位 |
| develop | 人的开发与练习 | 包含 main 的成果，继续实现 Filter/ACL 等后续任务 |
| solution | Agent 参考答案 | 包含 develop 的公共改进，补全练习，单独验证 |

当前已完成 V2 的 L1a、L1b、L1e、L2a、L2b、L2c、L3a、L3b；Gateway 和 P3
工具调用也已实现。不要因旧任务书仍在就把这些功能重新挖空。

Filter/ACL 的 L1c、L1d、L2d、L2e 及其接线和验收测试留在 develop；main 不提供
这些 API 或 CLI 参数，也不承诺租户隔离。solution 提供完整参考实现。

## 推进顺序

1. 日常在 develop 阅读、批注、实现与测试；保留已有实验的来源和适用范围。
2. 功能完成后，确认人能解释数据流、设计取舍与失败边界，再按功能提交进入 main。
3. main 上的修复同步到 develop，再同步到 solution；solution 的答案不整体合入 main。
4. develop 恢复 main 尚未提供的练习入口时，只恢复该功能涉及的文件与接线。
5. main 验证实际交付的全部测试；develop 的预期失败必须精确对应剩余练习；
   solution 的全部测试必须通过。不得靠跳过或屏蔽失败测试伪装完成。

三个分支是功能逐层扩展的关系。main 的发布整理可能删除练习脚手架，所以从 develop
晋升功能时应审查具体提交，不能无条件整体合并。

模型、缓存、原始数据和大型实验产物留在本地。任务与窄测试见
`rag/docs/azusa_v2_assignment.md`，状态见 `rag/docs/azusa_v2_traceability.md`。
本次迁移的验证与回退说明见 `docs/branch-reorganization.md`。
