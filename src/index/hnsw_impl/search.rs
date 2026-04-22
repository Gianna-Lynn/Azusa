use crate::types::{Id, Vector};

use super::super::hnsw_stats::SearchLayerStats;
use super::super::HnswIndex;

impl HnswIndex {
    pub fn search_layer(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
        ef: usize,
    ) -> (Vec<(usize, f32)>, SearchLayerStats) {
        assert!(ef >= 1, "ef至少为1");

        let mut stats = SearchLayerStats::default();
        self.workspace.reset(self.nodes.len());
        let nodes = &self.nodes;
        let metric = self.metric;
        let workspace = &mut self.workspace;

        workspace.visited_marks[entry_index] = workspace.visited_epoch;

        let entry_similarity =
            Self::get_similarity_from_workspace(nodes, metric, workspace, entry_index, query);
        workspace.candidate_heap.push((
            ordered_float::NotNan::new(entry_similarity).unwrap(),
            std::cmp::Reverse(entry_index),
        ));
        workspace.result_heap.push((
            std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()),
            entry_index,
        ));
        let mut worst_result_similarity = entry_similarity;

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

            let current_candidate_neighbors =
                Self::get_neighbors_at_lvl_from_parts(nodes, current_candidate_index, level);
            for &neighbor_index in current_candidate_neighbors {
                if workspace.visited_marks[neighbor_index] == workspace.visited_epoch {
                    continue;
                }

                workspace.visited_marks[neighbor_index] = workspace.visited_epoch;

                let neighbor_similarity = Self::get_similarity_from_workspace(
                    nodes,
                    metric,
                    workspace,
                    neighbor_index,
                    query,
                );

                if neighbor_similarity >= worst_result_similarity
                    || workspace.result_heap.len() < ef
                {
                    workspace.candidate_heap.push((
                        ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                        std::cmp::Reverse(neighbor_index),
                    ));
                    workspace.result_heap.push((
                        std::cmp::Reverse(ordered_float::NotNan::new(neighbor_similarity).unwrap()),
                        neighbor_index,
                    ));
                    if workspace.result_heap.len() > ef {
                        workspace.result_heap.pop();
                    }
                    worst_result_similarity = workspace
                        .result_heap
                        .peek()
                        .map(|(reverse_score, _)| reverse_score.0.into_inner())
                        .expect("result_heap shouldn't be empty");
                }
            }
        }

        // result_heap本轮之后不会再被读取, 下一次search_layer会reset/clear它.
        // 连续pop会按"最差到最好"弹出; reverse后得到score降序且同分index升序的结果.
        let mut result_candidates = Vec::with_capacity(workspace.result_heap.len());
        while let Some((reverse_score, result_index)) = workspace.result_heap.pop() {
            result_candidates.push((result_index, reverse_score.0.into_inner()));
        }
        result_candidates.reverse();

        (result_candidates, stats)
    }

    pub fn greedy_search_at_level(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
    ) -> usize {
        // 把同一个self中的不同字段拆开借用出来, 分别作只读借用, 拷贝, 可变借用, 而且拆开字段后彻底不要回头再碰self.
        self.workspace.reset_similarity_cache(self.nodes.len());

        let nodes = &self.nodes; // 只读借用
        let metric = self.metric; // 拷贝
        let workspace = &mut self.workspace; // 可变借用

        let mut current_node_index = entry_index;
        let mut max_similarity = Self::get_similarity_from_workspace(
            nodes,
            metric,
            workspace,
            current_node_index,
            query,
        );
        let mut max_similarity_index = current_node_index;

        loop {
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

            current_node_index = max_similarity_index;
        }
    }

    pub fn search_knn(&mut self, query: &Vector, k: usize, ef_search: usize) -> Vec<Id> {
        let Some(mut current_entry_index) = self.entry_node_index else {
            return Vec::new();
        };

        for lvl in (1..=self.index_max_level).rev() {
            current_entry_index = self.greedy_search_at_level(query, current_entry_index, lvl);
        }

        let (candidates, _) = self.search_layer(query, current_entry_index, 0, ef_search);

        candidates
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.get_id_by_index(idx))
            .collect()
    }
}
