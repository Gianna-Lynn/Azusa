//! HNSW 索引的搜索入口与查询相关逻辑。
//!
//! 这个文件只负责“怎么查”，不负责“怎么建图”：
//! - `search_knn` / `search_knn_with_scores` 是对外查询入口；
//! - `search_layer` 是 HNSW 的核心层搜索逻辑；
//! - `greedy_search_at_level` 负责从高层一路贪心下探到更接近查询的位置；
//! - `workspace` 用来复用堆、访问标记和相似度缓存，避免每次查询都重新分配大块内存。
use crate::types::{Id, Vector};

use super::super::hnsw_stats::SearchLayerStats;
use super::super::HnswIndex;

impl HnswIndex {
    /// 搜索的内部通用起点：从入口点逐层贪心下探到第 0 层，并返回该层候选。
    ///
    /// 这是 `search_knn` / `search_knn_with_scores` 共用的内部入口，完整流程为：
    /// 从最高层开始逐层用 `greedy_search_at_level` 找局部最优点，下探到第 0 层后
    /// 再用 `search_layer` 收集候选。
    ///
    /// # Arguments
    /// - `query`: 查询向量；
    /// - `ef_search`: 第 0 层搜索时保留的候选数上限。
    ///
    /// # Returns
    /// 最多 `ef_search` 个按相似度排序的 `(内部节点下标, 分数)`，不在此截取前 k 个；
    /// 截取和转换为对外 `Id` 由上述公开接口完成。如果图里还没有任何节点，
    /// 就直接返回空集合。
    fn search_candidates(&mut self, query: &Vector, ef_search: usize) -> Vec<(usize, f32)> {
        let Some(mut current_entry_index) = self.entry_node_index else {
            return Vec::new();
        };

        for lvl in (1..=self.index_max_level).rev() {
            current_entry_index = self.greedy_search_at_level(query, current_entry_index, lvl);
        }

        let (candidates, _) = self.search_layer(query, current_entry_index, 0, ef_search);
        candidates
    }

    /// 对外查询入口：返回 `(Id, score)` 形式的近似最近邻结果。
    ///
    /// 先走 `search_candidates` 拿到第 0 层候选，截取前 k 个，
    /// 并把内部节点下标换算成对外的记录 `Id`。
    ///
    /// # Arguments
    /// - `query`: 查询向量；
    /// - `k`: 控制最终返回多少个结果；
    /// - `ef_search`: 控制搜索时保留多少候选（内部预算，越大越精确、越慢）。
    ///
    /// # Returns
    /// 按相似度降序的 `(Id, score)` 列表。`score` 的含义取决于索引使用的度量方式：
    /// - Cosine 时越大越相似；
    /// - L2 时这里转换成了“相似度形式”，数值越大越近。
    pub fn search_knn_with_scores(
        &mut self,
        query: &Vector,
        k: usize,
        ef_search: usize,
    ) -> Vec<(Id, f32)> {
        self.search_candidates(query, ef_search)
            .into_iter()
            .take(k)
            .map(|(idx, score)| (self.get_id_by_index(idx), score))
            .collect()
    }

    /// 对外的简化查询接口：只返回 `Id` 列表，不带分数。
    ///
    /// 内部流程和 `search_knn_with_scores` 相同，只是丢弃了 score。
    ///
    /// # Arguments
    /// - `query`: 查询向量；
    /// - `k`: 控制最终返回多少个结果；
    /// - `ef_search`: 控制搜索时保留多少候选。
    ///
    /// # Returns
    /// 按相似度降序的 `Id` 列表。
    pub fn search_knn(&mut self, query: &Vector, k: usize, ef_search: usize) -> Vec<Id> {
        self.search_candidates(query, ef_search)
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.get_id_by_index(idx))
            .collect()
    }

    /// HNSW 查询路径里最核心的函数：在指定层上执行一次层搜索。
    ///
    /// 从入口点出发，用候选堆 + 结果堆做最佳优先扩展：不断弹出最有希望的候选、
    /// 扫描它的邻居，把值得保留的邻居加入两个堆，直到候选不可能再改进结果为止。
    /// visited 标记、相似度缓存和堆都复用 `workspace`，避免每次查询重新分配大块内存。
    ///
    /// # Arguments
    /// - `query`: 查询向量；
    /// - `entry_index`: 当前层的起始入口；
    /// - `level`: 指定搜索哪一层；
    /// - `ef`: 决定要保留多少个候选结果；构建期调用方传入 ef_construction，查询期调用方传入 ef_search。
    ///
    /// # Returns
    /// 二元组，包含：
    /// - `Vec<(usize, f32)>`: 按相似度排序后的候选点 index 和分数；
    /// - `SearchLayerStats`: 用于统计这一轮循环跑了多少步。
    ///
    /// # Panics
    /// `ef` 小于 1 时 panic（内部断言）。
    pub fn search_layer(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
        ef: usize,
    ) -> (Vec<(usize, f32)>, SearchLayerStats) {
        assert!(ef >= 1, "ef至少为1");

        // ---- 第一步：把工作区"逻辑上清空"，并拆出互不冲突的字段借用 ----
        // - visited 用 epoch 标记复用，不需要真的逐项写 false；
        // - candidate_heap / result_heap 直接清空；
        // - similarity cache 也沿用 epoch 复用，避免每次查询重新分配大块内存。
        let mut stats = SearchLayerStats::default();
        self.workspace
            .reset_search_layer_workspace(self.nodes.len());
        self.workspace.reset_similarity_cache(self.nodes.len());
        // nodes 只读、workspace 独占可变——两个不同字段，编译器允许同时借。
        let nodes = &self.nodes;
        let metric = self.metric;
        let workspace = &mut self.workspace;

        // ---- 第二步：用入口点初始化本轮搜索状态 ----
        // 入口点先标记为已访问，并作为初始候选/结果放入堆中。
        workspace.visited_marks[entry_index] = workspace.visited_epoch;
        let entry_similarity =
            Self::get_similarity_from_workspace(nodes, metric, workspace, entry_index, query);
        // candidate_heap 是大根堆，堆顶是当前最优的候选；
        // index 包成 Reverse<usize> 是为了相似度相同时优先扩展 index 小的点，实现稳定输出。
        workspace.candidate_heap.push((
            ordered_float::NotNan::new(entry_similarity).unwrap(),
            std::cmp::Reverse(entry_index),
        ));
        // result_heap 是小根堆，堆顶是当前最差的结果。
        // index 在这里不需要 Reverse, 如果有多个点的相似度分数都最差, 那么优先弹出 index 大的点, 也能保证稳定输出。
        workspace.result_heap.push((
            std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()),
            entry_index,
        ));
        // 目前结果堆里只有入口点，所以最差的结果就是入口点的相似度；
        // 后续候选如果不比这个相似度更好，就没有必要加入堆了。
        let mut worst_result_similarity = entry_similarity;

        // ---- 第三步：主循环（最佳优先扩展） ----
        // 1. 从 candidate_heap 取出当前最有希望扩展的点；
        // 2. 扫描它的邻居；
        // 3. 把没访问过、且值得保留的邻居加入候选集和结果集；
        // 4. 当候选已经不可能改进结果时提前停止。
        while let Some((current_candidate_similarity, std::cmp::Reverse(current_candidate_index))) =
            workspace.candidate_heap.pop()
        {
            stats.loop_iterations += 1;
            // 拆出原始 f32，方便和结果下界比较。
            let current_candidate_similarity = current_candidate_similarity.into_inner();

            // 3.1 提前停止：结果堆填满 ef 后，如果当前最优候选
            // 的相似度仍低于结果堆中最差的相似度，就停止扩展。
            // 点被接纳时会同时加入候选堆和结果堆；结果堆未填满时不会淘汰点，
            // 所以当前候选仍在结果堆中，其相似度不可能低于结果堆的最低分。
            // 填满后，结果堆可能淘汰仍留在候选堆中的点，才可能触发提前停止。
            // 显式检查结果堆是否已满，是为了清楚表达这一标准搜索条件。
            if workspace.result_heap.len() >= ef
                && current_candidate_similarity < worst_result_similarity
            {
                break;
            }

            // 3.2 扫描当前候选在第 level 层的所有邻居，
            // visited 标记保证同一个点不会在这一轮里被重复处理。
            let current_candidate_neighbors =
                Self::get_neighbors_at_lvl_from_parts(nodes, current_candidate_index, level);
            for &neighbor_index in current_candidate_neighbors {
                // 访问过就跳过；否则先标记，避免后续重复访问。
                if workspace.visited_marks[neighbor_index] == workspace.visited_epoch {
                    continue;
                }
                workspace.visited_marks[neighbor_index] = workspace.visited_epoch;

                // 用 workspace 里的 sim_marks / sim_values 缓存相似度计算结果，
                // 同一个点在一轮里只算一次，避免重复计算。
                let neighbor_similarity = Self::get_similarity_from_workspace(
                    nodes,
                    metric,
                    workspace,
                    neighbor_index,
                    query,
                );

                // 3.3 值得保留才入堆：邻居不低于当前结果堆里最差的相似度，
                // 或者结果堆还没有填满 ef 个结果。
                if neighbor_similarity >= worst_result_similarity
                    || workspace.result_heap.len() < ef
                {
                    // 加入候选堆（大根堆，堆顶是当前最优的候选）。
                    workspace.candidate_heap.push((
                        ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                        std::cmp::Reverse(neighbor_index),
                    ));
                    // 加入结果堆（小根堆，堆顶是当前最差的结果）。
                    workspace.result_heap.push((
                        std::cmp::Reverse(ordered_float::NotNan::new(neighbor_similarity).unwrap()),
                        neighbor_index,
                    ));
                    // ef 是结果堆的容量上限，超了就先 pop 掉当前最差的结果，腾出空间。
                    if workspace.result_heap.len() > ef {
                        workspace.result_heap.pop();
                    }
                    // 重新读取结果堆堆顶，更新 worst_result_similarity（当前最差相似度）。
                    worst_result_similarity = workspace
                        .result_heap
                        .peek()
                        .map(|(reverse_score, _)| reverse_score.0.into_inner())
                        .expect("result_heap shouldn't be empty");
                }
            }
        }

        // ---- 第四步：把结果堆倒成有序 Vec ----
        // result_heap 最终保存的是 top-ef 的结果，但它是个堆，不是排序好的 Vec：
        // 连续 pop 按"最差到最好"弹出，reverse 后得到"相似度降序、同分时 index 升序"的稳定输出。
        // result_heap 本轮之后不会再被读取，下一次 search_layer 会 reset/clear 它。
        let mut result_candidates = Vec::with_capacity(workspace.result_heap.len());
        while let Some((reverse_score, result_index)) = workspace.result_heap.pop() {
            result_candidates.push((result_index, reverse_score.0.into_inner()));
        }
        result_candidates.reverse();

        (result_candidates, stats)
    }

    /// 在指定层做贪心导航：只沿着“更近的邻居”往前走，直到局部最优。
    ///
    /// 这是 HNSW 在高层使用的“贪心导航”策略：
    /// - 先看当前点的邻居；
    /// - 如果某个邻居比当前点更接近 query，就跳过去；
    /// - 重复这个过程，直到没有更好的邻居为止。
    ///
    /// # Arguments
    /// - `query`: 查询向量；
    /// - `entry_index`: 本层的起始入口；
    /// - `level`: 在哪一层导航（只在当前层的邻居列表里找）。
    ///
    /// # Returns
    /// 当前层的一个局部最优点（内部节点下标），后续会从这个点开始进入下一层的搜索。
    pub fn greedy_search_at_level(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
    ) -> usize {
        // 这里把 self 拆成几个互不冲突的借用：
        // - nodes: 只读，拿邻居和向量数据；
        // - metric: 拷贝出来；
        // - workspace: 可变借用，用来复用相似度缓存。
        self.workspace.reset_similarity_cache(self.nodes.len());

        let nodes = &self.nodes;
        let metric = self.metric;
        let workspace = &mut self.workspace;

        let mut current_node_index = entry_index;
        // 先把入口点作为当前“最好”的点。
        let mut max_similarity = Self::get_similarity_from_workspace(
            nodes,
            metric,
            workspace,
            current_node_index,
            query,
        );
        let mut max_similarity_index = current_node_index;

        loop {
            // 只看第level层的邻居, 不进入更低层了.
            let current_node_neighbors = &nodes[current_node_index].get_neighbors()[level];

            for &neighbor_index in current_node_neighbors {
                let neighbor_node_similarity = Self::get_similarity_from_workspace(
                    nodes,
                    metric,
                    workspace,
                    neighbor_index,
                    query,
                );
                if neighbor_node_similarity > max_similarity {
                    max_similarity = neighbor_node_similarity;
                    max_similarity_index = neighbor_index;
                }
            }

            if current_node_index == max_similarity_index {
                return max_similarity_index;
            }

            // 发现了更好的点，就继续往那个方向走。
            current_node_index = max_similarity_index;
        }
    }

}
