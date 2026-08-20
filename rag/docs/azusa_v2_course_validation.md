# Azusa v2 课程分支验收基线

验收日期：2026-08-20

## 答案分支

`solution/azusa-v2` 在显式使用该 worktree 的 `rag/src` 后：

- Python：65 passed；
- Rust：5 passed；
- Kaiji live integration：1 passed（BM25 固定 fixture）。

## 课程分支

`main` 的模块可编译，`kb-sync --help` 与 `benchmark-lab --help` 可用。未完成 TODO 时，
全量 Python 基线为 35 passed / 30 failed；失败由 Azusa v2 的 12 个训练点和既有 Kaiji
Gateway TODO 产生，不存在 collection/import/fixture error。

Azusa v2 的每条任务书窄测均精确终止于对应的 `TODO AZUSA-V2-L1*`、`L2*` 或 `L3*`。
Kaiji Gateway 的隔离基线为：TODO 1 六项、TODO 2 三项、TODO 3 五项、TODO 4 一项。

本文件记录的是未作答课程态的预期红灯，不应被误解为答案分支回归。
