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
    /// 从当前入口点出发，先逐层贪心下探到第 0 层，再返回第 0 层的候选结果。
    ///
    /// 这是 `search_knn` / `search_knn_with_scores` 的内部通用起点。
    /// 如果图里还没有任何节点，就直接返回空集合。
    fn search_top_k_candidates(&mut self, query: &Vector, ef_search: usize) -> Vec<(usize, f32)> {
        let Some(mut current_entry_index) = self.entry_node_index else {
            return Vec::new();
        };

        for lvl in (1..=self.index_max_level).rev() {
            current_entry_index = self.greedy_search_at_level(query, current_entry_index, lvl);
        }

        let (candidates, _) = self.search_layer(query, current_entry_index, 0, ef_search);
        candidates
    }

    /// 对外返回 `(Id, score)` 形式的近似最近邻结果。
    ///
    /// `k` 控制最终返回多少个结果，`ef_search` 控制搜索时保留多少候选。
    /// `score` 的含义取决于索引使用的度量方式：
    /// - Cosine 时越大越相似；
    /// - L2 时这里转换成了“相似度形式”，数值越大越近。
    pub fn search_knn_with_scores(
        &mut self,
        query: &Vector,
        k: usize,
        ef_search: usize,
    ) -> Vec<(Id, f32)> {
        self.search_top_k_candidates(query, ef_search)
            .into_iter()
            .take(k)
            .map(|(idx, score)| (self.get_id_by_index(idx), score))
            .collect()
    }

    /// 在指定层上执行 HNSW 的层搜索。
    ///
    /// 这是整个查询路径里最核心的函数：
    /// - `entry_index` 是当前层的起始入口；
    /// - `level` 指定搜索哪一层；
    /// - `ef` 决定要保留多少个候选结果。
    ///
    /// 返回值里：
    /// - `Vec<(usize, f32)>` 是按相似度排序后的候选点 index 和分数；
    /// - `SearchLayerStats` 用于统计这一轮循环跑了多少步。
    pub fn search_layer(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
        ef: usize,
    ) -> (Vec<(usize, f32)>, SearchLayerStats) {
        assert!(ef >= 1, "ef至少为1");

        // 每次进入 search_layer 都先把工作区“逻辑上清空”：
        // - visited 用 epoch 标记复用，不需要真的逐项写 false；
        // - candidate_heap / result_heap 直接清空；
        // - similarity cache 也沿用 epoch 复用。
        let mut stats = SearchLayerStats::default();
        self.workspace.reset_search_layer_workspace(self.nodes.len());
        self.workspace.reset_similarity_cache(self.nodes.len());
        let nodes = &self.nodes;
        let metric = self.metric;
        let workspace = &mut self.workspace;

        // 入口点先标记为已访问，并作为初始候选/结果放入堆中。
        workspace.visited_marks[entry_index] = workspace.visited_epoch;

        let entry_similarity =
            Self::get_similarity_from_workspace(nodes, metric, workspace, entry_index, query);
        
        // 用入口点的相似度初始化候选堆, candidate_heap是一个大根堆, 堆顶是当前最优的候选. 
        // 这里把index包成Reverse<usize>是为了在相似度相同的情况下优先扩展index小的点, 以实现稳定输出.
        workspace.candidate_heap.push((
            ordered_float::NotNan::new(entry_similarity).unwrap(),
            std::cmp::Reverse(entry_index),
        ));
        
        // 用入口点的相似度初始化结果堆, result_heap是一个小根堆, 堆顶是当前最差的结果.
        workspace.result_heap.push((
            std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()),
            entry_index,
        ));

        // 目前结果堆里只有入口点, 所以最差的结果就是入口点的相似度. 后续候选如果不比这个相似度更好, 就没有必要加入堆了.
        let mut worst_result_similarity = entry_similarity;

        // 主循环：
        // 1. 从 candidate_heap 取出当前最有希望扩展的点；
        // 2. 扫描它的邻居；
        // 3. 把没访问过、且值得保留的邻居加入候选集和结果集；
        // 4. 当候选已经不可能改进结果时提前停止。
        while let Some((current_candidate_similarity, std::cmp::Reverse(current_candidate_index))) =
            workspace.candidate_heap.pop()
        {
            stats.loop_iterations += 1;

            let current_candidate_similarity = current_candidate_similarity.into_inner();

            // HNSW只有在结果集已经填满ef后, 才能用当前最优候选差于结果集下界来提前停止.
            // 现在的push invariant下len < ef时这个条件本来不可达(被visited判定挡住); 额外显式写出len判断是为了对齐标准语义.
            if workspace.result_heap.len() >= ef
                && current_candidate_similarity < worst_result_similarity
            {
                break;
            }

            // 当前候选所在层的所有邻居都会被看一遍。
            // visited 标记保证同一个点不会在这一轮里被重复处理。
            let current_candidate_neighbors =
                Self::get_neighbors_at_lvl_from_parts(nodes, current_candidate_index, level);
            for &neighbor_index in current_candidate_neighbors {
                
                // 如果这个邻居已经被访问过了, 就跳过.
                if workspace.visited_marks[neighbor_index] == workspace.visited_epoch {
                    continue;
                }

                // 标记这个邻居已经被访问过了, 避免后续重复访问.
                workspace.visited_marks[neighbor_index] = workspace.visited_epoch;

                // 计算邻居的相似度。这里利用了 SearchLayerWorkspace 里的 sim_marks 和 sim_values 来缓存相似度计算结果，避免重复计算。
                let neighbor_similarity = Self::get_similarity_from_workspace(
                    nodes,
                    metric,
                    workspace,
                    neighbor_index,
                    query,
                );

                // 如果这个邻居的相似度足够好, 大于等于当前结果堆里最差的相似度, 
                // 或者结果堆还没有填满 ef 个结果, 就把它加入候选堆和结果堆.
                if neighbor_similarity >= worst_result_similarity
                    || workspace.result_heap.len() < ef
                {

                    // 加入候选堆. candidate_heap是一个大根堆, 堆顶是当前最优的候选.
                    workspace.candidate_heap.push((
                        ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                        std::cmp::Reverse(neighbor_index),
                    ));

                    // 加入结果堆. result_heap是一个小根堆, 堆顶是当前最差的结果.
                    workspace.result_heap.push((
                        std::cmp::Reverse(ordered_float::NotNan::new(neighbor_similarity).unwrap()),
                        neighbor_index,
                    ));

                    // ef 是结果堆的容量上限，如果超过了这个容量，就先 pop 掉当前最差的结果，腾出空间。
                    if workspace.result_heap.len() > ef {
                        workspace.result_heap.pop();
                    }

                    // 取出当前结果堆里最差的相似度，更新 worst_result_similarity。
                    worst_result_similarity = workspace
                        .result_heap
                        .peek()
                        .map(|(reverse_score, _)| reverse_score.0.into_inner())
                        .expect("result_heap shouldn't be empty");
                }
            }
        }

        // result_heap 最终保存的是 top-ef 的结果，但它是个堆，不是排序好的 Vec。
        // 这里把它倒出来并反转，形成“相似度降序、同分时 index 升序”的稳定输出。
        // result_heap本轮之后不会再被读取, 下一次search_layer会reset/clear它.
        // 连续pop会按"最差到最好"弹出; reverse后得到score降序且同分index升序的结果.
        let mut result_candidates = Vec::with_capacity(workspace.result_heap.len());
        while let Some((reverse_score, result_index)) = workspace.result_heap.pop() {
            result_candidates.push((result_index, reverse_score.0.into_inner()));
        }
        result_candidates.reverse();

        (result_candidates, stats)
    }

    /// 从某一层的入口点开始，只沿着“更近的邻居”不断往前走，直到局部最优。
    ///
    /// 这相当于 HNSW 在高层使用的“贪心导航”：
    /// - 先看当前点的邻居；
    /// - 如果某个邻居比当前点更接近 query，就跳过去；
    /// - 重复这个过程，直到没有更好的邻居为止。
    /// - 这个函数的输出是当前层的一个局部最优点，后续会从这个点开始进入下一层的搜索。
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

    /// 对外的简化查询接口，只返回 Id 列表，不带分数。
    pub fn search_knn(&mut self, query: &Vector, k: usize, ef_search: usize) -> Vec<Id> {
        self.search_top_k_candidates(query, ef_search)
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.get_id_by_index(idx))
            .collect()
    }
}
