# Stage 3 Agent Eval

本文档记录 Search Agent v0 的 P1/P2 样例集、运行方式和 2026-07-13 基线结果。

## 样例集

样例文件：`data/eval/agent_eval.jsonl`

共 15 题：

| 类别 | 数量 | 目的 |
| --- | ---: | --- |
| single-hop | 3 | 验证基础事实查询和检索轨迹 |
| query rewrite | 3 | 验证空白、尾部标点和压缩表达 |
| multi-hop | 3 | 验证显式问题拆分、逐跳检索和证据融合 |
| insufficient evidence | 2 | 验证证据不足时不伪造 citation |
| citation badcase | 4 | 回归 wrong citation、wrong entity 和 incomplete answer |

## 运行命令

仅验证查询规划、检索和 evidence fusion：

```powershell
cd F:\Repositories\Azusa\rag
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent-eval --mode bm25 --top-k 5
```

同时加载本地 0.5B Answerer，验证回答和 citation：

```powershell
D:\Chuo_Data\conda_envs\ChuoEnv\python.exe run.py agent-eval --mode bm25 --top-k 5 --with-answer
```

结果默认写到 `output/eval_runs/latest/agent_eval_<mode>.json`。如果 Windows 占用了目标文件，脚本会回退到 `%TEMP%\azusa_agent_eval\`，并打印实际路径。

## 2026-07-13 BM25 基线

| 指标 | 不生成回答 | 生成回答 |
| --- | ---: | ---: |
| query type accuracy | 1.0000 | 1.0000 |
| hop count pass rate | 1.0000 | 1.0000 |
| average evidence recall | 0.8846 | 0.8846 |
| citation precision | 不适用 | 0.6000 |
| citation recall | 不适用 | 0.5667 |

这些数字是 Stage 3 工程回归基线，不是公开 benchmark。BM25 无法稳定处理跨语言 rewrite，本地 0.5B 模型也仍有错实体、漏答和漏 citation；这些失败会保留在结构化 trace 中，供后续迭代。

## Citation 安全规则

- 只接受当前 evidence 中真实存在、且由模型明确输出的 chunk id。
- 模型未提供有效引用时，`answer_citations` 保持为空，不再回填 top-1。
- 模型只有有效 citation、缺少答案字段时，保留 citation，`citation_validation.status` 为 `answer_missing`，并要求人工审核。
- 证据不足时输出明确的不确定性，不制造表面完整的 citation。
- 对外回答固定为“结论、证据、引用来源、不确定性”四段；证据摘要只从已经确认引用的原文生成。
