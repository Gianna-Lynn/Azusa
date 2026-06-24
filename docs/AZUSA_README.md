# Azusa

特急Azusa(あずさ、梓)是JR东日本中央本线(特急)列车, 常见于新宿至松本区段.  东京是我心中的理想国. 用这个列车名来命名我的第一个向量数据库项目, 就显得再好不过了.

## 项目简介

Azusa是一个基于HNSW (Hierarchical Navigable Small World, 分层可导航小世界)的向量数据库雏形.

## 当前功能

目前实现了:

- Rust library 中的 HNSW 索引: 支持插入向量记录, 按 query 检索 top-k 近邻, 返回 id / score, 并可按 id 取回 metadata.
- SIFT1M benchmark: 支持读取 SIFT1M base/query/groundtruth, 构建 HNSW 索引, 搜索并计算 recall@k / exact@k, 输出 Markdown 和 CSV 报告.
- Rust HTTP demo 服务: 提供最小 `health -> insert -> search -> record` 闭环, 适合验证服务化调用路径.
- RAG 子项目: 在 Azusa HNSW 服务之上补了一条最小检索/问答实验链路.

RAG 子项目当前已经支持:

- `raw txt -> chunks.jsonl -> embedding.jsonl`
- `brute / hnsw / bm25 / hybrid / hybrid_rerank` 检索
- 最小检索评测: `Hit@k / Recall@k / MRR`
- 最小 `answer + citation`
- Search Agent v0: query classification / query rewrite / multi-hop retrieval / evidence fusion

相关文档:

- Benchmark 说明见 `docs/benchmark.md`
- RAG 子项目说明见 `rag/README.md`
- HTTP 服务接口说明见 `docs/azusa_server_readme.md`

## HNSW原理

建立一个分层图, 从高层开始向低层搜索. 高层点的分布较为稀疏, 而且点距也很远. 低层点的分布则较为密集. 在高层可以快速跳转寻找最近似的搜索方向, 在低层可以深入到局部仔细搜索.,

### 插入算法 INSERT

算法目标: 把一个新点插入hnsw图.

首先给新插入的点选定层高lc. 新点的层高小于图的最高层高的可能性较大, 以这个作为一般情况.

从全图的最高层开始, 调用层搜索算法逐层向下搜索.在每一层都搜索最好的入口点, 然后等到下一层的时候就从这个入口点开始搜索. 这就像一个不停地坐扶梯下楼的过程.一直搜索到第lc层为止.

从第lc层开始一直到底层搜索. 首先调用层搜索算法找出距离query最近的ef个候选邻居, 记为集合W. 再调用邻居选择算法, 返回一个邻居集合neighbors. 对于neighbors中的所有邻居, 都要和新插入点建立双向连边. 对于neighbors集合中的每一个邻居e, 进行剪枝处理, 如果e持有的邻居数目太多, 超过了最多M_max, 那就重新选择M_max个邻居, 作为e的新邻居列表. 在进入下一层之前, 用集合W作为下一层搜索的入口(在具体的实现中, 利用了集合W中距离新插入点最近的一个元素作为入口).

最后, 如果新点的层高比图的最高层还要高, 那么把新点设置为整个图的入口节点.

### 层搜索算法 SEARCH-LAYER

算法目标: 在给定的一个层上, 求出距离输入q最近的邻居节点, 共ef个.

维护以下数据结构: 已经访问过的节点visited, 候选集合C, 答案集合W. 首先用入口节点来初始化这三个数据结构.

当且仅当候选集合C为空的时候才会退出下面的循环:

取出候选集合C中距离q最近的元素c, 读取答案集合W中距离q最远的元素f. 如果c-q距离 大于f-q距离. 也就是如果候选集合中最有希望的结果, 还不如答案集合中最差的结果, 那就不必再评审候选集合, 直接break退出循环.

如果c-q距离小于等于f-q距离, 也就是候选集合中存在一个有希望加入答案集合的结果c, 检查c在这一层上所有的邻居节点e. 确认e已经被访问过了, 标记visited.

再次检查所有邻居节点. 如果真的有一个节点e,它的距离比答案集合中最远的距离还要近, 那么把这个节点也要放进答案集合W. 如果W满了, 那么就需要做重复元素的清理.

### 邻居选择算法(启发式)

算法目标: 使用启发式方法选出M个邻居(不一定是最近的)

维护以下数据结构: 工作集合W,  答案集合R, 弃置集合Wd. 用候选集合C来初始化工作集合W.

1. 如果需要扩充候选集:

对于在候选集合C中的每个元素, 对于这个元素在第lc层的每一个邻居节点, 如果邻居节点不在工作集合W内,那么加入工作集合W.

1. 当且仅当工作集合W为空或者答案集合R数目达到M个才会退出下面的循环:

从W中取出距离q最近的元素e, 如果e比R中的其他任何元素都要靠近q, 那么把e加入R. 否则, 把e加入弃置集合Wd. (仔细领会, 这个动作天然地避免了e和R中其他的一坨元素混在一起, 就算e和R中这一坨元素到q的距离相等, 也是可以区分开的)

1. 如果需要保留一些已经被剪枝的连边:

从弃置集合Wd中不断挑选距离q最近的元素加入答案集合R, 直到弃置集合Wd为空或者答案集合R的数目达到M个为止.

### KNN搜索算法

目标: 在HNSW图中, 找出对于查询向量q的k个近似最近邻.

维护以下数据结构: 当前的最近邻集合W, 当前的入口搜索点ep

从最高层开始往下搜索: 在每一层调用层搜索算法SEARCH-LAYER, 搜索宽度记为1 (实际上是退化成了贪心搜索),得到当前的最近邻集合W, 从W中选出距离q最近的元素作为下一层的入口ep, 继续向下搜索.

在最底层, 调用层搜索算法SEARCH-LAYER, 搜索宽度记为ef, 得到最终的最近邻集合W

在这个最终的最近邻集合W中选取K个最近邻.

### 一些关键参数

这些参数在Benchmark中有单独的针对性测试.

我对参数的定义还不是很清晰.

**建图部分**用到的参数:

| 参数 | 含义 |
| --- | --- |
| ef_construction | 建图阶段, 为了给新点q找邻居, 要完成一次局部搜索.ef_construction决定这次搜索要返回的邻居数目.
ef_construction越大, 找邻居的时候看得更宽, 建图质量更好, 召回率更高, 但是建图速度变慢, 内存占用和维护成本上升. |
| m | 每一层给新点q最多选择多少个邻居来连边.
例: ef_construction = 100 , m = 16, 意思是先找100个候选近邻, 再从里面挑16个最适合连边的点.这是"建边时的目标数量” |
| m_max | 每个点最多拥有多少个邻居. 
用来限制已有的点的边数膨胀.
这是"图结构的容量上限” |
| level_sample_probablility | 采样概率, 决定新点的新层数的概率分布 |

**搜索部分**用到的参数:

| 参数 | 含义 |
| --- | --- |
| ef_search | 搜索宽度
每当调用一次层搜索算法SEARCH-LAYER, 会返回ef_search这么多个近似最近邻向量
控制的是"查询时保留多少个候选近邻” |
| k | 在进行一次查询的时候返回多少个结果 |
| base_limit | 使用多少个向量建立起总的数据库 |
| query_limit | 进行多少次查询 |

## Benchmark方法

早期的benchmark方法采用的是暴力计算出标准答案, 和HNSW建图-搜索算法得出的答案进行比对, 求解出如下指标: 

| 指标 | 含义 |
| --- | --- |
| exact@k | 每个有效 query 的 top-k 是否完全一致，再统计占比 |
| recall@k | 每个有效 query 的 top-k 重合比例，再取平均 |

后期的benchmark方法利用sift1m的groundtruth参考答案来评分. 

日常推荐使用独立 benchmark binary, 它会验证 HTTP 服务之前的 HNSW 主链路:

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 100000 --query-limit 100 --ks 5 --ef-search 60,100,300 --m 4 --ef-construction 80
```

快速 smoke benchmark:

```powershell
cargo run --release --bin sift1m_benchmark -- --mode hnsw --base-limit 1000 --query-limit 10 --k 5 --ef-search 60 --m 4 --ef-construction 80
```

输出位置:

```text
target/benchmark/sift1m.md
target/benchmark/sift1m.csv
```

更多命令和历史结果见 `docs/benchmark.md`.

## Benchmark结果

当前日常复测优先看 `base_limit=100000, query_limit=100, k=5` 的结果:

| ef_search | total_search_time | avg_search_time | recall@5 | exact@5 |
| --- | --- | --- | --- | --- |
| 60 | 12.9318ms | 129.318us | 87.0% | 62/100 |
| 100 | 22.8890ms | 228.890us | 92.2% | 75/100 |
| 300 | 52.2015ms | 522.015us | 97.0% | 89/100 |

已有全量 1M sweep 记录保留在 `docs/archive/perf_logs.md` 和 `docs/benchmark.md` 中. 摘要如下:

| config | build time | ef_search=60 recall@10 / exact@10 | ef_search=100 recall@10 / exact@10 | ef_search=300 recall@10 / exact@10 |
| --- | --- | --- | --- | --- |
| m8_efc100 | 1251.370706978s | 89.40% / 49/100 | 94.90% / 68/100 | 99.40% / 91/100 |
| m16_efc100 | 3055.649133404s | 97.50% / 76/100 | 99.20% / 89/100 | 99.80% / 94/100 |
| m16_efc200 | 4668.71236162s | 97.80% / 78/100 | 99.20% / 90/100 | 99.90% / 95/100 |

## 当前局限

- HTTP 服务目前是内存索引, 服务重启后记录会丢失.
- 暂无持久化、删除、更新、批量导入和并发压测.
- 当前 HTTP 接口只覆盖最小 demo 闭环, 还没有发布独立 OpenAPI / protobuf schema.
- SIFT1M 原始数据不随仓库提交, benchmark 依赖本机或服务器上的 `data/sift1m/sift/`.
- RAG 子项目仍是实验链路, 重点用于验证检索模式、rerank、answer 和 agent 编排, 还不是完整应用.

## 下一步路线

接下来优先把 Azusa 从"可验证的 HNSW 检索内核"推进到"更像数据库/检索服务"的形态:

- 为 HTTP 服务补持久化、批量导入、删除/更新和更完整的错误处理.
- 为服务接口补 schema 文档和更系统的 smoke / integration test.
- 继续用 SIFT1M benchmark 追踪 HNSW 的质量和性能回归.
- 让 RAG 实验链路继续反向检验 HNSW 服务在真实文本检索场景中的可用性.
