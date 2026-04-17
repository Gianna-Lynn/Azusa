use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::types::{Id, Vector};

use super::{HnswIndex, HnswNode};

impl HnswIndex {
    // 实验: 从指定入口点开始做搜索，方便参数 sweep。
    pub fn search_knn_v1_from_entry_for_experiment(
        &self,
        entrance_id: Option<Id>,
        query: &Vector,
        k: usize,
        ef_search: usize,
    ) -> Vec<Id> {
        let Some(mut current_entry_id) = entrance_id else {
            return Vec::new();
        };

        let current_node = self
            .get_node_by_id(current_entry_id)
            .expect("in search_knn_v1_from_entry_for_experiment, failed to get id with current_entry_id");
        for lvl in (1..=*current_node.get_node_max_level()).rev() {
            current_entry_id = self.greedy_search_at_level(query, current_entry_id, lvl);
        }

        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        let (candidate_set, _) = self.search_layer_v0(query, current_entry_id, 0, ef_search, &mut id_similarity_cache);

        let mut candidate_sequence: Vec<Id> = candidate_set.into_iter().collect();
        candidate_sequence.sort_by(|id1, id2| {
            let score1 = self.get_similarity(&mut id_similarity_cache, id1, query);
            let score2 = self.get_similarity(&mut id_similarity_cache, id2, query);
            match score2.partial_cmp(&score1).unwrap() {
                std::cmp::Ordering::Equal => id1.cmp(id2),
                other => other,
            }
        });

        candidate_sequence.into_iter().take(k).collect()
    }

    // 实验: 固定层高插入，主要供测试和 sweep 使用。
    pub fn insert_v1_fixed_level(
        &mut self,
        id: Id,
        data: Vector,
        ef_construction: usize,
        m: usize,
        m_max: usize,
        level: &usize,
    ) {
        let new_node_level = *level;
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, new_node_level);
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();

        if self.is_empty() {
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
            self.id_to_index.insert(id, self.nodes.len());
            self.nodes.push(new_node);
            return;
        }

        self.id_to_index.insert(id, self.nodes.len());
        self.nodes.push(new_node);

        let mut entry_point_id = self.get_entry_node_id().expect("in insert_v1, entry_node_id is none");
        if self.index_max_level >= new_node_level + 1 {
            for current_level in (new_node_level + 1..=self.index_max_level).rev() {
                entry_point_id = self.greedy_search_at_level(&data, entry_point_id, current_level);
            }
        }

        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        for current_lvl in (0..=min_level).rev() {
            let t = Instant::now();
            let (w_set, sl_stats) = self.search_layer_v0(&data, entry_point_id, current_lvl, ef_construction, &mut id_similarity_cache);
            self.timing.search_layer_ns += t.elapsed().as_nanos();
            self.timing.search_layer_compute_sim_ns += sl_stats.compute_sim_ns;
            self.timing.search_layer_furthest_scan_ns += sl_stats.furthest_scan_ns;
            self.timing.search_layer_neighbors_fetch_ns += sl_stats.neighbors_fetch_ns;
            self.timing.search_layer_heap_pop_ns += sl_stats.heap_pop_ns;
            self.timing.search_layer_bookkeeping_ns += sl_stats.candidate_bookkeeping_ns;
            self.timing.search_layer_loop_iterations += sl_stats.loop_iterations;

            let neighbors_set = self.select_neighbors_simple(&w_set, &data, m, &mut id_similarity_cache);
            for neighbor_node_id in &neighbors_set {
                self.add_neighbor_to_node_at_level(id, *neighbor_node_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_node_id, id, current_lvl);
            }

            for e_id in &neighbors_set {
                let e_conn_set: HashSet<Id> = {
                    let e_node = self
                        .get_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    e_node.get_neighbors()[current_lvl].iter().copied().collect()
                };

                if e_conn_set.len() > m_max {
                    let e_new_conn_vec: Vec<Id> = {
                        let e_node = self
                            .get_node_by_id(*e_id)
                            .expect("in insert_v1, get_node_by_id cannot handle e_id");
                        let mut e_cache: HashMap<Id, f32> = HashMap::new();
                        self.select_neighbors_simple(&e_conn_set, e_node.get_data(), m_max, &mut e_cache)
                    };

                    let e_node = self
                        .get_mut_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    let old_neighbors = e_node.neighbors[current_lvl].clone();
                    let old_set: HashSet<Id> = old_neighbors.iter().cloned().collect();
                    let new_set: HashSet<Id> = e_new_conn_vec.iter().cloned().collect();
                    let to_delete: HashSet<Id> = old_set.difference(&new_set).cloned().collect();
                    e_node.neighbors[current_lvl] = e_new_conn_vec;
                    for neighbor_to_delete_id in to_delete {
                        self.delete_neighbor_relationship_at_level(*e_id, neighbor_to_delete_id, current_lvl);
                    }
                }
            }

            entry_point_id = self
                .get_nearest_v1(&mut id_similarity_cache, &w_set, &data)
                .expect("searchh_result_id赋值错误");
        }

        if new_node_level > self.index_max_level {
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
        }
    }
}
