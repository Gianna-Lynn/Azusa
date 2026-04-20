// Legacy-only helpers.
// 这个文件只服务于旧路径 / 兼容测试，不属于正式 HNSW 实现路径。

use crate::types::{Id, Vector};

use super::{HnswIndex, HnswNode};

use std::collections::{HashMap, HashSet};
use std::time::Instant;

impl HnswIndex {
    // 仅用于旧版参数 sweep / 兼容测试，不属于正式检索路径。
    // 它允许从外部指定入口点开始搜索，主要是为了复现实验条件。
    pub fn search_knn_from_entry_for_legacy(
        &mut self,
        entrance_id: Option<Id>,
        query: &Vector,
        k: usize,
        ef_search: usize,
    ) -> Vec<Id> {
        let Some(mut current_entry_id) = entrance_id else {
            return Vec::new();
        };

        let current_node = self.get_node_by_id(current_entry_id).expect(
            "in search_knn_from_entry_for_legacy, failed to get id with current_entry_id",
        );
        for lvl in (1..=*current_node.get_node_max_level()).rev() {
            current_entry_id = self.greedy_search_at_level(query, current_entry_id, lvl);
        }

        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        let (candidate_set, _) = self.search_layer(
            query,
            current_entry_id,
            0,
            ef_search,
            &mut id_similarity_cache,
        );

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

    // 仅用于旧路径 / 兼容测试。
    // 这个接口允许外部直接指定层高，主要为了复现早期手工构造数据集的条件。
    // 正式插入路径应当使用 `insert`，由索引内部负责随机采样层高。
    pub fn insert_with_fixed_level_for_legacy(
        &mut self,
        id: Id,
        data: Vector,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
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

        let mut entry_point_id = self
            .get_entry_node_id()
            .expect("in insert, entry_node_id is none");
        if self.index_max_level >= new_node_level + 1 {
            for current_level in (new_node_level + 1..=self.index_max_level).rev() {
                entry_point_id = self.greedy_search_at_level(&data, entry_point_id, current_level);
            }
        }

        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        for current_lvl in (0..=min_level).rev() {
            let t = Instant::now();
            let (result_set, sl_stats) = self.search_layer(
                &data,
                entry_point_id,
                current_lvl,
                ef_construction,
                &mut id_similarity_cache,
            );
            self.timing.search_layer_ns += t.elapsed().as_nanos();
            self.timing.search_layer_compute_sim_ns += sl_stats.compute_sim_ns;
            self.timing.search_layer_furthest_scan_ns += sl_stats.furthest_scan_ns;
            self.timing.search_layer_neighbors_fetch_ns += sl_stats.neighbors_fetch_ns;
            self.timing.search_layer_heap_pop_ns += sl_stats.heap_pop_ns;
            self.timing.search_layer_bookkeeping_ns += sl_stats.candidate_bookkeeping_ns;
            self.timing.search_layer_loop_iterations += sl_stats.loop_iterations;

            let neighbors_set =
                self.select_neighbors_simple(
                    &result_set,
                    &data,
                    max_neighbors_to_select,
                    &mut id_similarity_cache,
                );
            for neighbor_node_id in &neighbors_set {
                self.add_neighbor_to_node_at_level(id, *neighbor_node_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_node_id, id, current_lvl);
            }

            for e_id in &neighbors_set {
                let e_conn_set: HashSet<Id> = {
                    let e_node = self
                        .get_node_by_id(*e_id)
                        .expect("in insert, get_node_by_id cannot handle e_id");
                    e_node.get_neighbors()[current_lvl]
                        .iter()
                        .copied()
                        .collect()
                };

                if e_conn_set.len() > max_neighbors_to_keep {
                    let e_new_conn_vec: Vec<Id> = {
                        let e_node = self
                            .get_node_by_id(*e_id)
                            .expect("in insert, get_node_by_id cannot handle e_id");
                        let mut e_cache: HashMap<Id, f32> = HashMap::new();
                        self.select_neighbors_simple(
                            &e_conn_set,
                            e_node.get_data(),
                            max_neighbors_to_keep,
                            &mut e_cache,
                        )
                    };

                    let e_node = self
                        .get_mut_node_by_id(*e_id)
                        .expect("in insert, get_node_by_id cannot handle e_id");
                    let old_neighbors = e_node.neighbors[current_lvl].clone();
                    let old_set: HashSet<Id> = old_neighbors.iter().cloned().collect();
                    let new_set: HashSet<Id> = e_new_conn_vec.iter().cloned().collect();
                    let to_delete: HashSet<Id> = old_set.difference(&new_set).cloned().collect();
                    e_node.neighbors[current_lvl] = e_new_conn_vec;
                    for neighbor_to_delete_id in to_delete {
                        self.delete_neighbor_relationship_at_level(
                            *e_id,
                            neighbor_to_delete_id,
                            current_lvl,
                        );
                    }
                }
            }

            entry_point_id = self
                .get_nearest(&mut id_similarity_cache, &result_set, &data)
                .expect("searchh_result_id赋值错误");
        }

        if new_node_level > self.index_max_level {
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
        }
    }
}
