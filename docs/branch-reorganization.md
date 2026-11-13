# 本地三分支整理

整理日期：2026-11-13。迁移基底是人的提交 `3779d11`，保留源码批注、命名整理与
真实文本实验记录，不重跑或覆盖已有性能实验。

- main：已完成的 Rust 检索、Python RAG、评测、增量 KB、Gateway 与 Agent。
  Filter/ACL 模块、参数、接线及练习测试移到 develop，不提供权限隔离承诺。
- develop：包含 main 的公共改进，保留 V2 的四项 Filter/ACL 练习与验收测试。
- solution：保留旧答案历史，吸收 develop 的改进，提供符合当前契约的 Filter/ACL。

当前仅调整本地。远端分支和 remote-tracking refs 保持原样，远端答案名称仍为
`origin/solution/azusa-v2`；未推送的新名称不代表远端页面已存在。

## 备份与恢复

仓库内被忽略的目录 `local-experiments/branch-reorg-20261113/` 保存：

- `before.bundle`：迁移前所有本地分支、标签和远端跟踪提交的完整历史；已用
  `git bundle verify` 验证。
- `before.json`：原 HEAD、分支引用和干净状态。
- `git-config.before`：原仓库级 Git 配置；不包含在提交中。
- `rollback.ps1`：恢复原四个本地分支及配置，保留现有历史的救援 bundle。

从仓库根目录预览恢复目标：

```powershell
& .\local-experiments\branch-reorg-20261113\rollback.ps1
```

实际恢复时增加 `-Apply`。脚本要求工作区干净、没有额外 worktree，先备份当前分支历史，
然后恢复原 main、原答案分支与两个备份分支。它不修改远端，不删除实验文件。
若以后修改仓库配置，恢复前需另存新配置，因为脚本会还原迁移前的仓库配置。
这些本地备份不会随 clone 复制；保留本目录即可回退本次调整。

## 验证范围

main 的实际验证：Python 全部交付测试 143 passed；Rust 全部测试 16 passed；
`cargo run --locked --example retrieval_demo` 输出符合 README。`search --help` 不再暴露
过滤参数，`kb-sync --help` 可用，Benchmark Lab 的 BM25 smoke 完成 k=1,3 两格并生成
JSON/CSV/Markdown，输出位于 `rag/.cache/branch-reorg-main-smoke/`。

develop 实测：143 passed / 5 failed；JUnit 中五个失败均属于 `test_filters.py`，
分别精确终止于 L1c、L1d、L2d、L2e 的 `NotImplementedError`，没有额外失败或跳过。
四个练习文件逐字恢复自人的提交 `3779d11`，已完成的其余源码保持不变。

solution 实测：170 passed，无失败或跳过。除了原有 148 项契约，还加入 22 项边界
用例，覆盖显式非法假值、非对象载荷、空列表、None 和默认 ACL 不越权。
旧答案的完整历史保留为合并祖先；公共源码采用 develop 的最新版本，评测继续使用
请求 k 的 nDCG 分母，旧 CV 与废弃脚本不恢复。

回退脚本已在独立的本地测试仓库演练：从 before.bundle 恢复历史、模拟分支整理、
执行 rollback.ps1，再逐项比较原四个本地分支的名称与完整提交 ID，全部一致。
演练没有修改真实仓库的分支；本地备份也不会改变远端状态。

develop 实测：143 passed / 5 failed；JUnit 中五个失败均属于 `test_filters.py`，
分别精确终止于 L1c、L1d、L2d、L2e 的 `NotImplementedError`，没有额外失败或跳过。
四个练习文件逐字恢复自人的提交 `3779d11`，已完成的其余源码保持不变。

真实模型质量、GPU 性能和历史大规模实验不属于此次分支调整的复测范围；不把 fixture、
向量替身或历史数字当作新测模型成绩。Python 保留已有 SWIG 弃用警告。
