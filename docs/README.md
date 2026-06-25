# Docs Guide

这份索引用来回答一个简单问题：现在在 Azusa 里，应该先看哪些文档。

## First Read

如果只想快速进入当前项目状态，优先看这几份：

1. `docs/AZUSA_README.md`
2. `rag/RAG_README.md`
3. `rag/docs/eval_results.md`
4. `rag/docs/badcase_v1.md`

它们分别回答：

- Azusa 整体是什么
- RAG 子项目怎么跑；优先使用 `rag/run.py` 这个统一入口
- Stage 2 当前结论是什么
- Stage 2 主要 badcase 在哪里

## Keep vs Archive

### Keep

- `docs/AZUSA_README.md`: 当前项目总览
- `docs/azusa_server_readme.md`: Rust HTTP demo / API 示例
- `docs/benchmark.md`: benchmark 运行方式与结果入口
- `docs/interface_decision.md`: 为什么当前选 HTTP 服务
- `docs/stage1_summary.md`: Stage 1 收口与复验记录
- `rag/RAG_README.md`: RAG 子项目入口
- `rag/docs/eval_results.md`: 当前稳定的 Stage 2 评测说明
- `rag/docs/badcase_v1.md`: 当前稳定的 Stage 2 badcase 汇总

### Archive / Generated

- `docs/archive/`: 历史记录、学习笔记、旧实验留档
- `rag/docs/archive/`: 历史提交说明等非当前主线文档
- `tests/archive/`: 旧测试套件留档
- `rag/output/search_runs/latest/`: 单条 query / agent 默认覆盖的最新运行输出目录；首次运行对应命令时自动创建
- `rag/output/search_runs/archive/`: 旧 search / agent 输出留档
- `rag/output/eval_runs/latest/`: 主 workflow 默认覆盖的最新运行输出目录；首次运行对应命令时自动创建
- `rag/output/eval_runs/archive/`: 旧运行输出留档
- `rag/data/processed/du_retrieval_workflow/latest/`: DuRetrieval workflow 默认覆盖的最新运行输出目录；首次运行对应 workflow 时自动创建
- `rag/data/processed/du_retrieval_workflow/runs/`: DuRetrieval 命名运行副本；首次保留命名副本时自动创建
- `rag/data/import_snapshots/hf_eval/latest/`: HF 样本导入脚本默认覆盖的最新运行输出目录；首次运行导入命令时自动创建
- `rag/data/import_snapshots/hf_eval/runs/`: HF 样本导入脚本命名运行副本；首次保留命名副本时自动创建
- `rag/data/processed/*.json`: 本地样本快照或其他辅助数据，不作为正式文档
- `target/benchmark/*.md`: benchmark 生成物，不作为正式文档

## Practical Reading Order

按目的来读会更省力：

- 想知道项目全貌：`docs/AZUSA_README.md`
- 想跑 RAG：`rag/RAG_README.md`
- 想看当前效果：`rag/docs/eval_results.md`
- 想看失败样例：`rag/docs/badcase_v1.md`
- 想看 Rust benchmark：`docs/benchmark.md` + `docs/archive/perf_logs.md`
- 想看接口设计原因：`docs/interface_decision.md`

## Notes

- `CLAUDE.md` 是 agent 使用的项目级说明，不等于面向人的主文档首页。
- `archive` 里的内容默认不是“当前首先该看”的文档，只在追历史、查决策或翻旧实验时再看。
