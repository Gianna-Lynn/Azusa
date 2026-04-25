use std::cmp::Reverse;
use std::collections::HashSet;
use std::time::Instant;

use rand::Rng;

use crate::{
    distance,
    types::{Id, Metadata, Metric, Vector, VectorRecord},
};

use super::super::{HnswIndex, HnswNode, SearchLayerWorkspace};

impl HnswIndex {
    fn upsert_record(&mut self, record: VectorRecord) {
        self.records.insert(record.id, record);
    }

    pub fn sample_level(&mut self) -> usize {
        let mut level = 0usize;
        while self.rng.gen_bool(self.level_sample_probability) {
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
        self.upsert_record(VectorRecord {
            id,
            vector: data.clone(),
            metadata: Metadata::default(),
        });
        self.insert_into_graph(
            id,
            data,
            ef_construction,
            max_neighbors_to_select,
            max_neighbors_to_keep,
        );
    }

    pub fn insert_vector_record(
        &mut self,
        record: VectorRecord,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        let id = record.id;
        let vector = record.vector.clone();
        self.upsert_record(record);
        self.insert_into_graph(
            id,
            vector,
            ef_construction,
            max_neighbors_to_select,
            max_neighbors_to_keep,
        );
    }

    pub fn batch_insert(
        &mut self,
        records: Vec<VectorRecord>,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        for record in records {
            self.insert_vector_record(
                record,
                ef_construction,
                max_neighbors_to_select,
                max_neighbors_to_keep,
            );
        }
    }

    fn insert_into_graph(
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
                &data,
                max_neighbors_to_select,
                current_lvl,
                true,
                true,
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
                let neighbor_current_connections: Vec<usize> = {
                    let neighbor_node = self
                        .get_node_by_index(neighbor_index)
                        .expect("in insert, get_node_by_index cannot handle neighbor_index");
                    neighbor_node.get_neighbors()[current_lvl].clone()
                };

                if neighbor_current_connections.len() > max_neighbors_to_keep {
                    let neighbor_pruned_connections: Vec<usize> = {
                        let neighbor_data = self.nodes[neighbor_index].get_data().clone();
                        let scored_candidates = self.score_and_sort_level_candidates(
                            &neighbor_current_connections,
                            &neighbor_data,
                            current_lvl,
                        );

                        self.select_neighbors_heuristic(
                            &scored_candidates,
                            &neighbor_data,
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

    // 仅 legacy 路径仍在使用这套 simple selection；主线建图已经切到 heuristic selection。
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

    // 大橘:
    // 这是统一的 heuristic 下游入口, 只接受按 similarity 排好序的 scored candidates。
    // 在建图里它会被用在两个阶段:
    // 1. 新节点接入图时: search_layer 已经直接返回 scored candidates, 可以原样喂进来。
    // 2. 旧节点反向边 pruning 时: 先把当前邻接表重新打分并排序成 scored candidates, 再喂进来。
    // 这样下游只需要维护一套 heuristic 主体, 不必同时兼容"有序 scored 输入"和"无序 index 集合"。
    pub fn select_neighbors_heuristic(
        &mut self,
        scored_candidates: &[(usize, f32)],
        query: &Vector,
        max_neighbors_to_select: usize,
        level: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> Vec<usize> {
        self.workspace
            .reset_select_neighbor_workspace(self.nodes.len());
        // search_layer给出的候选已经带有相对query的similarity,
        // 这里只需装入工作堆, 再按heuristic主体筛出最终邻居.
        self.fill_work_queue_from_scored(scored_candidates, level);
        if extend_candidates {
            // extendCandidates: 把当前候选在这一层的一阶邻居也并入工作堆，
            // 用membership marks避免在线性扫描堆里查重。
            self.extend_work_queue_from_scored(scored_candidates, query, level);
        }
        // 新节点选邻居时, 若heuristic丢掉了一些点, 仍允许从discard里补回直到凑满M.
        self.take_selected_neighbors(max_neighbors_to_select, keep_pruned_connections)
    }

    // 把某一层的无序候选 index 集合适配成 heuristic 下游统一使用的 scored ordered sequence。
    fn score_and_sort_level_candidates(
        &mut self,
        candidate_indices: &[usize],
        query: &Vector,
        level: usize,
    ) -> Vec<(usize, f32)> {
        self.workspace.reset_similarity_cache(self.nodes.len());

        let mut scored_candidates = Vec::with_capacity(candidate_indices.len());
        for candidate_index in candidate_indices.iter().copied() {
            if self.nodes[candidate_index].get_neighbors().len() <= level {
                continue;
            }
            let candidate_similarity = HnswIndex::get_similarity_from_workspace(
                &self.nodes,
                self.metric,
                &mut self.workspace,
                candidate_index,
                query,
            );
            scored_candidates.push((candidate_index, candidate_similarity));
        }
        scored_candidates.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
        scored_candidates
    }

    // scored_candidates已经带有分数, 不需要再重新算一次similarity.
    fn fill_work_queue_from_scored(&mut self, scored_candidates: &[(usize, f32)], level: usize) {
        for &(candidate_index, candidate_similarity) in scored_candidates {
            if self.nodes[candidate_index].get_neighbors().len() <= level {
                continue;
            }
            self.workspace.work_heap.push((
                ordered_float::NotNan::new(candidate_similarity).unwrap(),
                Reverse(candidate_index),
            ));
            self.workspace.mark_work_present(candidate_index);
        }
    }

    fn extend_work_queue_from_scored(
        &mut self,
        scored_candidates: &[(usize, f32)],
        query: &Vector,
        level: usize,
    ) {
        for &(candidate_index, _) in scored_candidates {
            if self.nodes[candidate_index].get_neighbors().len() <= level {
                continue;
            }
            let candidate_neighbors = &self.nodes[candidate_index].get_neighbors()[level];
            for &neighbor_index in candidate_neighbors {
                if self.workspace.work_contains(neighbor_index) {
                    continue;
                }
                let neighbor_similarity = HnswIndex::get_similarity_from_workspace(
                    &self.nodes,
                    self.metric,
                    &mut self.workspace,
                    neighbor_index,
                    query,
                );
                self.workspace.work_heap.push((
                    ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                    Reverse(neighbor_index),
                ));
                self.workspace.mark_work_present(neighbor_index);
            }
        }
    }

    fn take_selected_neighbors(
        &mut self,
        max_neighbors_to_select: usize,
        keep_pruned_connections: bool,
    ) -> Vec<usize> {
        let mut result_vec: Vec<(usize, f32)> = vec![]; // R
        while !self.workspace.work_heap.is_empty() && result_vec.len() < max_neighbors_to_select {
            let nearest_elem_in_work = self.workspace.work_heap.pop().unwrap();
            // <相似度 ,index>
            let similarity_e_to_q = nearest_elem_in_work.0.into_inner();
            let e_index = nearest_elem_in_work.1 .0;
            let mut flag = true;

            for item in result_vec.iter() {
                let r_index = item.0;
                let similarity_e_to_r = distance::similarity(
                    self.nodes[e_index].get_data(),
                    self.nodes[r_index].get_data(),
                    self.metric,
                );
                if similarity_e_to_q <= similarity_e_to_r {
                    flag = false;
                    break;
                }
            }
            if flag == true {
                result_vec.push((e_index, similarity_e_to_q));
            } else {
                // discard_heap需要提取最近的, 需要做成大根堆.(Rust默认大根堆)
                // 同similarity, index较小的靠前, 需要reverse.
                self.workspace.discard_heap.push((
                    ordered_float::NotNan::new(similarity_e_to_q).unwrap(),
                    Reverse(e_index),
                ));
            }
        }

        if keep_pruned_connections {
            // 对应论文里的Wd补位阶段: 当diversity heuristic筛得过严时,
            // 允许把之前丢弃的高分候选按顺序补回, 避免结果数不足.
            while !self.workspace.discard_heap.is_empty()
                && result_vec.len() < max_neighbors_to_select
            {
                let discard_top = self.workspace.discard_heap.pop().unwrap(); //while保证不会panic
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
}
