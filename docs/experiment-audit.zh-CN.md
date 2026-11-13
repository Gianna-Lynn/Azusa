# 第四阶段：实验数据与性能数字审查

本文件是面试复习入口：先回答六个问题，再按链接核对原始报告和实现。2026-11-13 核对时，
三份 DuRetrieval 原始最终报告均为 success。本文整理已有证据，没有重新执行性能实验。

## 一、先用哪一组实验回答

优先讲 2026-11-11 的真实中文文本向量实验，再用三种插入顺序补充稳定性。

| 实验 | 数据、规模与比较对象 | 结果说明 | 原始报告 |
|---|---|---|---|
| 中文 BM25 基线 | DuRetrieval，100,001 段落、2,000 查询、9,839 标注；BM25Okapi | [BM25 结果](text-scale-results-20261110.md) | [report.json](../local-experiments/du-retrieval-bm25-20261110-run002/run-002/report.json) |
| 精确向量 vs HNSW | 相同文本，BGE-small-zh-v1.5 的 512 维向量；同一 Cosine 度量 | [真实文本向量结果](text-dense-results-20261111.md) | [report.json](../local-experiments/du-retrieval-dense-20261111-run001/run-001/report.json) |
| 插入顺序稳定性 | 相同向量与查询；3 张新图 × 3 档 ef × 2,000 查询 | [稳定性结果](text-hnsw-stability-results-20261111.md) | [report.json](../local-experiments/du-retrieval-stability-20261111-run001/run-001/report.json) |
| 小规模索引决策 | 128/512/2,048 条 32 维均匀合成向量、32 查询；Flat vs HNSW | [P1 决策实验](p1-validation.md) | 文档中记录了本地缓存路径 |
| 历史 SIFT 实验 | 100k/1M SIFT 向量，通常为 100 查询；不同 HNSW 参数 | [benchmark 索引](benchmark.md)、[历史日志](archive/perf_logs.md) | 部分原始目录当前本地缺失，引用时明确是历史留档 |
| 历史文本回归 | min_eval，15 查询、top-k=3；旧融合策略 vs RRF | [CHANGELOG 的 2026-07-10 记录](../CHANGELOG.md) | 属于历史小样本回归，不能替代全量检索实验 |

不同数据、模型、度量、机器或计时范围的数字不能拼成一次实验。DuRetrieval 原始大文件
通过 .gitignore 留在本地；另一台机器只有仓库文档时，需要另外取得这些归档。

## 二、六个问题怎样回答：以真实文本 HNSW 实验为主

### 1. 数据集是什么？

使用 C-MTEB/DuRetrieval 的 corpus、queries，及 C-MTEB/DuRetrieval-qrels 的 dev 标注。
每条 passage 是一个检索单位，本轮没有再用知识库 chunker 切它。
qrels 中 score > 0 的段落作为相关证据，按二值相关性计算指标。

数据集 commit、每个输入文件的行数与 SHA-256 在报告的 `configuration.inputs` 中。
完整语料、查询和标注 Parquet 在 [BM25 归档 run-002](../local-experiments/du-retrieval-bm25-20261110-run002/run-002/)。

### 2. 实验规模是多少？

- 全部 100,001 条中文段落、2,000 条查询、9,839 条相关性标注；没有抽掉难题。
- BGE-small-zh-v1.5 固定 commit，CLS pooling，查询添加检索指令，fp16 推理后保存为归一化 float32，512 维。
- 最大输入 512 token；7,768 段落截断，查询没有截断。模型与项目默认 Qwen embedding 配置不同。
- 单图实验：精确检索一次、HNSW ef=100/200/400 各一次，共 8,000 条查询观测。
- 稳定性实验：三种打乱的插入顺序，各跑三档 ef，共 18,000 条观测；层级种子始终为 42。

完整配置见报告的 `configuration`，编码行数、维度、截断数、缓存哈希见 `encoding`。
这两个实验共用同一批查询，8,000 或 18,000 不能说成不同问题的数量。

### 3. Baseline 是什么？

评估 HNSW 的近似损失与查询加速时，baseline 是对相同向量执行的精确 Cosine 扫描。
两条路径都调用 Azusa 的 `metric_similarity(..., Metric::Cosine)`。
HNSW 参数为 m=16、m_max=32、ef_construction=200，单图插入顺序为原数据顺序。

BM25 是同一任务上的另一种文本检索基线，它不定义向量最近邻金标。其质量可以与 dense
对照，但当前中文单字分词未经优化，且两轮运行时间和实现不同，不据此计算算法加速比。

精确向量排名也不等于语义相关性标注排名，不能把精确检索的证据 Recall 当作理论满分上限。

### 4. Recall@K 如何计算？

先逐题计算，再对 2,000 题取平均。必须区分两个分母：

```text
证据 Recall@K(q) = |返回前 K 个 ID ∩ qrels 中全部相关 ID| / |qrels 中全部相关 ID|
近邻 Recall@K(q) = |HNSW 前 K 个 ID ∩ 精确向量检索前 K 个 ID| / K
```

例子：某题标了 4 个相关段落，前 10 条找回 3 个，证据 Recall@10 是 3/4。
同题 HNSW 前 10 条与精确前 10 条重合 9 个，近邻 Recall@10 是 9/10。
Hit@K 只判断有没有至少一个相关结果，不能把它当 Recall。

相关代码：[score()](../scripts/run_text_dense_scale.py)、同脚本内的 neighbor_recall 计算。
原始记录：[exact-ef100.jsonl](../local-experiments/du-retrieval-dense-20261111-run001/run-001/exact-ef100.jsonl)、
[hnsw-ef200.jsonl](../local-experiments/du-retrieval-dense-20261111-run001/run-001/hnsw-ef200.jsonl)。
它们保留 query_id、原始索引、段落 ID、逐题指标和时间；ID 映射保存在同目录 ids.json。

### 5. 延迟如何测量？

| 数字 | 计时范围 | 明确排除的部分 |
|---|---|---|
| `native_ms` | Rust 内部执行精确扫描或 HNSW，包含 top-k 结果生成 | HTTP、查询编码、回答生成 |
| `http_client_ms` | Python 从发出实验 HTTP 请求到解析响应 | 请求字典及查询向量列表准备、查询编码、生产快照检查、回答生成 |
| `build_ms` | 向 HNSW 插入全部向量并建立图 | 矩阵文件读取、文本编码 |
| `encoding.*.elapsed_s` | 各阶段批量编码与缓存写入流程；不含函数末尾的文件哈希 | 模型下载与加载、图构建、检索 |

计时使用 Rust Instant / Python perf_counter。每配置前十题各先做一次同题未计时请求，
随后记录该题，其余查询顺序执行。统一返回 top-100，再计算 @5/10/100；没有分别测 top-10 请求耗时。

平均值是逐查询时间的算术平均。P95 采用 nearest-rank：排序后取第 ceil(0.95 × N) 项，
N=2,000 时是第 1,900 项。三实例表中的 P95 是各实例 P95 的中位数，不能当作合并后的 P95。

Rust 测量代码见 [text_retrieval_benchmark.rs](../src/bin/text_retrieval_benchmark.rs) 的 search()；
客户端与分位数代码见 [run_text_dense_scale.py](../scripts/run_text_dense_scale.py) 的 run()、distribution()。
批量查询编码约 1.65 秒，不是单查询编码延迟；也不能用它直接拼成自然语言请求 P95。

实验运行于 E5_server，报告记录了 V100 编码设备和依赖版本；未完整保存 CPU 型号、频率
及资源占用轨迹，机器也未隔离。因此查询时间应作为该次共享服务器运行的观测值，跨机器
比较或后续正式复测时还需要补齐这些环境记录。

### 6. 性能提升付出了什么代价？

以下数字来自同一轮单图实验，分别引用报告的 summaries 和 build_ms：

| 项目 | 精确向量 baseline | HNSW ef=200 |
|---|---:|---:|
| 证据 Recall@10 | 78.9552% | 78.8628% |
| Rust 平均查询时间 | 92.8470 ms | 8.5744 ms |
| Rust P95 | 97.0528 ms | 12.0076 ms |
| HTTP 客户端 P95 | 101.7925 ms | 17.4104 ms |

**10.8 倍**来自 92.8470 / 8.5744，比较的是原生平均查询时间。
**0.09 个百分点**来自 (0.789552 − 0.788628) × 100，是证据 Recall@10 的下降。
不能把这个下降描述成近邻漏检率；ef=200 的近邻 Recall@100 是 96.14%。

HNSW 额外承担约 20.51 分钟建图和图邻接结构的内存开销；这轮没有定量测纯索引内存，
因此不能报一个未经测量的内存增量。段落编码约 8.68 分钟是精确向量和 HNSW 共用的准备成本。
图仍为内存结构，生产知识库快照变化后需重启重建。提高 ef 会增加查询计算和候选维护成本。

三种插入顺序补充结果：ef=200 的证据 Recall@10 范围 78.60%–78.82%，实例原生 P95
中位数 11.53 ms；仍有每实例 6–10 题的证据 Recall@10 低于精确检索。它们是另一轮结果，
应单独叙述。详见[稳定性报告](text-hnsw-stability-results-20261111.md)。

## 三、查看数据时打开什么文件

1. 先读[真实文本向量结果](text-dense-results-20261111.md)，掌握数据、配置与主要表格。
2. 打开[原始 report.json](../local-experiments/du-retrieval-dense-20261111-run001/run-001/report.json)，
   只看 `configuration`、`encoding`、`build_ms`、`summaries`。
3. 想核对具体失败题，再看[稳定性案例目录](../local-experiments/du-retrieval-stability-20261111-run001/run-001/order-37/)
   中的 ef-200-cases.json；需要逐题数值时打开相邻 jsonl。
4. 对计算方式有疑问，再读上面链接的 score()、distribution() 和 Rust search()。

归档内的 metrics.csv 便于看表，local-verification.json 记录当时复算与哈希校验结果。
代码继续演进时，以归档中的运行脚本、二进制和源码快照解释历史实验。

## 四、历史数字怎样使用

- SIFT1M 用于说明底层向量索引；百万规模历史实验只用了 100 查询。文档里的 avg_search_time
  是平均时间，不能称为 P95。Exact@K 在当前 harness 中要求前 K 个 ID 连顺序都相同，区别于 Recall。
- 4 月部分原始归档当前本地缺失，100k 子集实验的当时金标处理也需按旧版本继续核对；
  当前 benchmark 的金标回退逻辑不能作为旧运行的证明。历史 SIFT 不冒充当前版本复验。
- P1 合成实验适合说明“小数据未必值得建图”，不能推导真实文本的统一切换规模；其 RSS
  是 worker 整体峰值，包含输入等对象，不能称为纯索引内存。
- 7 月 15 题的 RRF 满分属于小样本回归。旧日志中延迟“无显著变化”的表述缺少统计检验，
  不据此宣称统计显著性。当前尚无同规模 RRF、rerank 或自然语言端到端延迟实验。

## 五、可用于面试的完整回答

> 我在 DuRetrieval 的十万条中文段落和两千条查询上，固定 BGE-small 中文模型生成向量，
> 用相同 Cosine 度量的精确扫描作为 HNSW 对照。ef=200 时，HNSW 的证据 Recall@10
> 从精确检索的约 78.96% 变为 78.86%，原生平均查询耗时从约 92.85 ms 降到 8.57 ms，
> 约快 10.8 倍，代价是约 20.5 分钟建图、额外图内存以及一定近似损失。这些查询使用
> 预编码向量。我又换三种插入顺序重新建图，ef=200 的证据 Recall@10 在 78.60%–78.82%
> 之间。原始排名、计时、模型与源码身份都有本地归档。这些结果限定在固定模型、该数据集
> 和共享服务器顺序执行条件下，尚未验证并发吞吐和完整自然语言请求延迟。
