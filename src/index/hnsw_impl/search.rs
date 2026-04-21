use std::collections::{HashMap, HashSet};

use crate::types::{Id, Vector};

use super::super::hnsw_stats::{HeapAccessMode, SearchLayerStats};
use super::super::HnswIndex;

impl HnswIndex {
    pub fn search_layer(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
        ef: usize,
    ) -> (HashSet<usize>, SearchLayerStats) {
        assert!(ef >= 1, "ef至少为1");

        let mut stats = SearchLayerStats::default();
        self.workspace.reset(self.nodes.len());
        let nodes = &self.nodes;
        let metric = self.metric;
        let workspace = &mut self.workspace;

        let mut candidate_alive_num: usize = 0;

        workspace.visited_marks[entry_index] = workspace.visited_epoch;
        workspace.candidate_alive[entry_index] = true;
        candidate_alive_num += 1;

        let entry_similarity =
            Self::get_similarity_from_workspace(nodes, metric, workspace, entry_index, query);
        workspace.candidate_heap.push((
            ordered_float::NotNan::new(entry_similarity).unwrap(),
            entry_index,
        ));
        workspace.result_heap.push((
            std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()),
            entry_index,
        ));

        while candidate_alive_num > 0 {
            stats.loop_iterations += 1;
            let current_candidate_index = Self::get_heap_extreme_refer_alive_bool_vec(
                &workspace.candidate_alive,
                &mut workspace.candidate_heap,
                HeapAccessMode::Pop,
            )
            .expect("current_candidate_index不存在");

            workspace.candidate_alive[current_candidate_index] = false;
            candidate_alive_num -= 1;

            let worst_result_similarity = workspace
                .result_heap
                .peek()
                .map(|(reverse_score, _)| reverse_score.0.into_inner())
                .expect("result_heap shouldn't be empty");

            let current_candidate_similarity = Self::get_similarity_from_workspace(
                nodes,
                metric,
                workspace,
                current_candidate_index,
                query,
            );

            if current_candidate_similarity < worst_result_similarity {
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

                let worst_result_similarity = workspace
                    .result_heap
                    .peek()
                    .map(|(reverse_score, _)| reverse_score.0.into_inner())
                    .expect("result_heap shouldn't be empty");

                if neighbor_similarity >= worst_result_similarity
                    || workspace.result_heap.len() < ef
                {
                    workspace.candidate_alive[neighbor_index] = true;
                    candidate_alive_num += 1;

                    workspace.candidate_heap.push((
                        ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                        neighbor_index,
                    ));
                    workspace.result_heap.push((
                        std::cmp::Reverse(ordered_float::NotNan::new(neighbor_similarity).unwrap()),
                        neighbor_index,
                    ));
                    if workspace.result_heap.len() > ef {
                        workspace.result_heap.pop();
                    }
                }
            }
        }

        let result_set: HashSet<usize> = workspace
            .result_heap
            .iter()
            .map(|(_, result_index)| *result_index)
            .collect();

        (result_set, stats)
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

        let (candidate_set, _) = self.search_layer(query, current_entry_index, 0, ef_search);

        let mut index_similarity_cache: HashMap<usize, f32> = HashMap::new();
        let mut candidate_sequence: Vec<usize> = candidate_set.into_iter().collect();
        candidate_sequence.sort_by(|idx1, idx2| {
            let score1 = self.get_similarity(&mut index_similarity_cache, *idx1, query);
            let score2 = self.get_similarity(&mut index_similarity_cache, *idx2, query);
            match score2.partial_cmp(&score1).unwrap() {
                std::cmp::Ordering::Equal => self
                    .get_id_by_index(*idx1)
                    .cmp(&self.get_id_by_index(*idx2)),
                other => other,
            }
        });

        candidate_sequence
            .into_iter()
            .take(k)
            .map(|idx| self.get_id_by_index(idx))
            .collect()
    }
}
