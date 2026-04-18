# Azusa

`Azusa` 是一个用 Rust 编写的向量检索实验项目，目标是把 HNSW 的核心流程一点点搭起来，
再在小规模 SIFT 数据集上验证检索效果。这里更像一个认真打磨的练习场，而不是单纯的玩具仓库。

## 1. 项目目标

- 手搓一个简化版 HNSW
- 先完成插入与搜索主流程
- 再在小规模 SIFT 子集上做 L2 检索验证

## 2. 当前实现内容

### 索引部分
- HNSW 插入
- HNSW 搜索
- fixed-level 插入实验版本
- pruning / 邻边裁剪逻辑

### 距离度量
- 默认 Cosine
- 新增 L2 支持
- metric 可切换

### 数据读取
- fvecs reader 读取 sift_base.fvecs 与 sift_query.fvecs

### 对照基线
- brute-force 精确搜索（exact_top_k 实现）
- 遍历全部向量计算距离，排序取前k个，作为真值基准

## 3. 实验设置

### 数据
- 使用 SIFT1M 文件中的 base/query
- 当前先使用小子集，不是全量 benchmark

### 距离口径
- 使用 L2

### 指标
- **recall@k**：平均每个查询，HNSW 搜索结果的前 k 个中，有 recall@k 个命中了真值结果（不要求顺序，只要 ID 重合即可）
- **exact@k**：在 query_limit 个查询中，有 exact@k 个查询的前 k 个结果与真值完全一致（包括顺序）

### 实验方式
- brute-force 作为小子集真值
- HNSW 搜索结果与 brute-force 对比

## 4. 阶段性结果

小规模 SIFT 子集上的观察：
- base_limit = 100, k = 5 时，ef_search = 30 可恢复完整 top-5
- base_limit = 200, k = 5 时，ef_search = 80 可恢复完整 top-5
- base_limit = 500, k = 5 时，ef_search = 200 可恢复完整 top-5

**结论**：随着 base 规模扩大，恢复完整 top-k 所需的搜索宽度显著增大。

## 5. 对结果的理解

- 小规模下 HNSW 能较稳定逼近 brute-force 结果
- 随着候选规模增大，搜索预算不足会先反映在 recall@k / exact@k 上
- ef_search 对检索质量非常敏感

## 6. 当前局限

- 仍然是小规模子集实验，不是全量 SIFT1M benchmark
- 真值目前由小子集 brute-force 生成
- 尚未接入官方 groundtruth.ivecs
- 尚未做并发与服务化

## 7. 下一步计划

- 接入官方 groundtruth.ivecs
- 扩大 base/query 规模
- 做更正式的 recall benchmark
- 后续推进并发控制与 gRPC 服务化
