use std::cmp::Reverse;
use std::collections::HashSet;
use std::time::Instant;

use rand::Rng;

use crate::{
    distance,
    types::{Id, Metric, Vector},
};

use super::super::{HnswIndex, HnswNode, SearchLayerWorkspace};

impl HnswIndex {
    pub fn sample_level(&mut self) -> usize {
        let p = 0.5;
        let mut level = 0usize;
        while self.rng.gen_bool(p) {
            level += 1;
        }
        level
    }

    pub fn insert(
        &mut self,
        id: Id,
        data: Vector,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        self.timing.calls += 1;
        let node_lvl = self.sample_level();
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, node_lvl);
        let new_index = self.nodes.len();

        if self.is_empty() {
            self.entry_node_index = Some(new_index);
            self.index_max_level = node_lvl;
            self.id_to_index.insert(id, new_index);
            self.nodes.push(new_node);
            return;
        }

        self.id_to_index.insert(id, new_index);
        self.nodes.push(new_node);

        let mut entry_point_index = self
            .entry_node_index
            .expect("in insert, entry_node_index is none");

        if self.index_max_level > node_lvl {
            let t = Instant::now();
            for current_lvl in (node_lvl + 1..=self.index_max_level).rev() {
                entry_point_index =
                    self.greedy_search_at_level(&data, entry_point_index, current_lvl);
            }
            self.timing.greedy_search_ns += t.elapsed().as_nanos();
        }

        let top_connect_lvl = std::cmp::min(self.index_max_level, node_lvl);
        for current_lvl in (0..=top_connect_lvl).rev() {
            let t = Instant::now();
            let (search_results, sl_stats) =
                self.search_layer(&data, entry_point_index, current_lvl, ef_construction);
            self.timing.search_layer_ns += t.elapsed().as_nanos();
            self.timing.search_layer_compute_sim_ns += sl_stats.compute_sim_ns;
            self.timing.search_layer_furthest_scan_ns += sl_stats.furthest_scan_ns;
            self.timing.search_layer_neighbors_fetch_ns += sl_stats.neighbors_fetch_ns;
            self.timing.search_layer_heap_pop_ns += sl_stats.heap_pop_ns;
            self.timing.search_layer_bookkeeping_ns += sl_stats.candidate_bookkeeping_ns;
            self.timing.search_layer_loop_iterations += sl_stats.loop_iterations;

            let t = Instant::now();
            let selected_neighbors = self.select_neighbors_heuristic(
                &search_results,
                max_neighbors_to_select,
                current_lvl,
            );
            self.timing.select_neighbors_ns += t.elapsed().as_nanos();

            let t = Instant::now();
            for &neighbor_index in &selected_neighbors {
                self.add_neighbor_to_node_at_level_by_index(new_index, neighbor_index, current_lvl);
                self.add_neighbor_to_node_at_level_by_index(neighbor_index, new_index, current_lvl);
            }
            self.timing.link_ns += t.elapsed().as_nanos();

            let t = Instant::now();
            for &neighbor_index in &selected_neighbors {
                let neighbor_current_connections: HashSet<usize> = {
                    let neighbor_node = self
                        .get_node_by_index(neighbor_index)
                        .expect("in insert, get_node_by_index cannot handle neighbor_index");
                    neighbor_node.get_neighbors()[current_lvl]
                        .iter()
                        .copied()
                        .collect()
                };

                if neighbor_current_connections.len() > max_neighbors_to_keep {
                    let neighbor_pruned_connections: Vec<usize> = {
                        let nodes = &self.nodes;
                        let metric = self.metric;
                        let workspace = &mut self.workspace;
                        let neighbor_data = nodes[neighbor_index].get_data();

                        Self::select_neighbors_heuristic_from_parts(
                            nodes,
                            metric,
                            workspace,
                            &neighbor_current_connections,
                            neighbor_data,
                            max_neighbors_to_keep,
                            current_lvl,
                            false,
                            true,
                        )
                    };

                    let connections_to_remove: Vec<usize> = {
                        let neighbor_node = self
                            .get_node_by_index(neighbor_index)
                            .expect("in insert, get_node_by_index cannot handle neighbor_index");
                        neighbor_node.neighbors[current_lvl]
                            .iter()
                            .copied()
                            .filter(|connection_index| {
                                !neighbor_pruned_connections.contains(connection_index)
                            })
                            .collect()
                    };

                    let neighbor_node = self
                        .get_mut_node_by_index(neighbor_index)
                        .expect("in insert, get_mut_node_by_index cannot handle neighbor_index");
                    neighbor_node.neighbors[current_lvl] = neighbor_pruned_connections;
                    for removed_neighbor_index in connections_to_remove {
                        self.delete_neighbor_relationship_by_index(
                            neighbor_index,
                            removed_neighbor_index,
                            Some(current_lvl),
                        );
                    }
                }
            }
            self.timing.pruning_ns += t.elapsed().as_nanos();

            entry_point_index = search_results
                .first()
                .map(|(index, _)| *index)
                .expect("search_result_index赋值错误");
        }

        if node_lvl > self.index_max_level {
            self.entry_node_index = Some(new_index);
            self.index_max_level = node_lvl;
        }
    }

    pub fn select_neighbors_simple(
        &mut self,
        candidate_set: &HashSet<usize>,
        query: &Vector,
        max_neighbors_to_select: usize,
    ) -> Vec<usize> {
        Self::select_neighbors_simple_from_parts(
            &self.nodes,
            self.metric,
            &mut self.workspace,
            candidate_set,
            query,
            max_neighbors_to_select,
        )
    }

    pub fn select_neighbors_heuristic(
        &mut self,
        scored_candidates: &[(usize, f32)],
        max_neighbors_to_select: usize,
        level: usize,
    ) -> Vec<usize> {
        Self::select_neighbors_heuristic_from_scored_candidates(
            &self.nodes,
            self.metric,
            &mut self.workspace,
            scored_candidates,
            max_neighbors_to_select,
            level,
            true,
        )
    }

    fn select_neighbors_simple_from_parts(
        nodes: &[HnswNode],
        metric: Metric,
        workspace: &mut SearchLayerWorkspace,
        candidate_set: &HashSet<usize>,
        query: &Vector,
        max_neighbors_to_select: usize,
    ) -> Vec<usize> {
        workspace.reset_similarity_cache(nodes.len());

        let mut index_score_map: Vec<(usize, f32)> = candidate_set
            .iter()
            .map(|candidate_index| {
                (
                    *candidate_index,
                    HnswIndex::get_similarity_from_workspace(
                        nodes,
                        metric,
                        workspace,
                        *candidate_index,
                        query,
                    ),
                )
            })
            .collect();
        index_score_map.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => nodes[a.0].get_id().cmp(nodes[b.0].get_id()),
            other => other,
        });
        let take_count = max_neighbors_to_select.min(index_score_map.len());
        index_score_map
            .iter()
            .take(take_count)
            .map(|(index, _)| *index)
            .collect()
    }

    fn select_neighbors_heuristic_from_parts(
        nodes: &[HnswNode],
        metric: Metric,
        workspace: &mut SearchLayerWorkspace,
        candidate_set: &HashSet<usize>, // 一个index的集合.
        query: &Vector,                 // q
        max_neighbors_to_select: usize, //m :number of neighbors to return
        level: usize,
        _extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> Vec<usize> {
        // 大橘:
        // 这个入口服务旧节点反向边pruning. candidate_set来自某个旧节点当前已有的邻接表,
        // 只有index, 没有现成similarity; 而且这里需要的similarity是相对这个旧节点query重新计算的.
        // 它和search_layer返回的"相对新插入向量的已打分候选"不是同一种分数, 不能复用.
        workspace.reset_select_neighbor_workspace(nodes.len());
        workspace.reset_similarity_cache(nodes.len());

        for candidate_index in candidate_set.iter().copied() {
            if nodes[candidate_index].get_neighbors().len() <= level {
                continue;
            }
            let candidate_similarity = HnswIndex::get_similarity_from_workspace(
                nodes,
                metric,
                workspace,
                candidate_index,
                query,
            );
            workspace.work_heap.push((
                ordered_float::NotNan::new(candidate_similarity).unwrap(),
                Reverse(candidate_index),
            ));
        }

        Self::select_neighbors_heuristic_from_work_heap(
            nodes,
            metric,
            workspace,
            max_neighbors_to_select,
            keep_pruned_connections,
        )
    }

    fn select_neighbors_heuristic_from_work_heap(
        nodes: &[HnswNode],
        metric: Metric,
        workspace: &mut SearchLayerWorkspace,
        max_neighbors_to_select: usize,
        keep_pruned_connections: bool,
    ) -> Vec<usize> {
        let mut result_vec: Vec<(usize, f32)> = vec![]; // R
        while !workspace.work_heap.is_empty() && result_vec.len() < max_neighbors_to_select {
            let nearest_elem_in_work = workspace.work_heap.pop().unwrap();
            // <相似度 ,index>
            let similarity_e_to_q = nearest_elem_in_work.0.into_inner();
            let e_index = nearest_elem_in_work.1 .0;
            let mut flag = true;

            for item in result_vec.iter() {
                let r_index = item.0;
                let similarity_e_to_r = distance::similarity(
                    nodes[e_index].get_data(),
                    nodes[r_index].get_data(),
                    metric,
                );
                if similarity_e_to_q <= similarity_e_to_r {
                    flag = false;
                    break;
                }
            }
            if flag == true {
                result_vec.push((e_index, similarity_e_to_q));
            } else {
                workspace.discard_heap.push((
                    ordered_float::NotNan::new(similarity_e_to_q).unwrap(),
                    Reverse(e_index),
                ));
            }
        }

        if keep_pruned_connections {
            while !workspace.discard_heap.is_empty() && result_vec.len() < max_neighbors_to_select {
                let discard_top = workspace.discard_heap.pop().unwrap(); //while保证不会panic
                let discard_top_similarity = discard_top.0.into_inner();
                let discard_top_index = discard_top.1 .0;
                result_vec.push((discard_top_index, discard_top_similarity))
            }
        }
        let result_index_vec = result_vec
            .iter()
            .map(|(result_index, _)| *result_index)
            .collect();
        result_index_vec
    }

    fn select_neighbors_heuristic_from_scored_candidates(
        nodes: &[HnswNode],
        metric: Metric,
        workspace: &mut SearchLayerWorkspace,
        scored_candidates: &[(usize, f32)],
        max_neighbors_to_select: usize,
        level: usize,
        keep_pruned_connections: bool,
    ) -> Vec<usize> {
        // 大橘:
        // 这个入口服务新节点选邻居. scored_candidates来自search_layer,
        // 已经带有相对新插入向量的similarity, 所以这里只把它们装进work_heap,
        // 避免退回HashSet后再重新计算一遍分数. 真正的heuristic主体在from_work_heap里共用.
        workspace.reset_select_neighbor_workspace(nodes.len());

        for &(candidate_index, candidate_similarity) in scored_candidates {
            if nodes[candidate_index].get_neighbors().len() <= level {
                continue;
            }
            workspace.work_heap.push((
                ordered_float::NotNan::new(candidate_similarity).unwrap(),
                Reverse(candidate_index),
            ));
        }

        Self::select_neighbors_heuristic_from_work_heap(
            nodes,
            metric,
            workspace,
            max_neighbors_to_select,
            keep_pruned_connections,
        )
    }
}
