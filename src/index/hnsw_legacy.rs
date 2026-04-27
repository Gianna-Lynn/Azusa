// Legacy-only helpers.
// 这个文件只服务于旧路径 / 兼容测试，不属于正式 HNSW 实现路径。

use crate::types::{Id, Metadata, Vector, VectorRecord};

use super::{HnswIndex, HnswNode};

use std::collections::HashSet;
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
        let Some(entry_id) = entrance_id else {
            return Vec::new();
        };
        let mut current_entry_index = self
            .get_index_by_id(entry_id)
            .expect("in search_knn_from_entry_for_legacy, failed to map entrance_id to index");

        let current_node = self.get_node_by_index(current_entry_index).expect(
            "in search_knn_from_entry_for_legacy, failed to get node with current_entry_index",
        );
        for lvl in (1..=*current_node.get_node_max_level()).rev() {
            current_entry_index = self.greedy_search_at_level(query, current_entry_index, lvl);
        }

        let (candidates, _) = self.search_layer(query, current_entry_index, 0, ef_search);

        candidates
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.get_id_by_index(idx))
            .collect()
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
        assert!(
            !self.id_to_index.contains_key(&id),
            "duplicate id is not allowed: {}",
            id
        );
        self.records.insert(
            id,
            VectorRecord {
                id,
                vector: data.clone(),
                metadata: Metadata::default(),
            },
        );
        let new_node_level = *level;
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, new_node_level);
        let new_index = self.nodes.len();

        if self.is_empty() {
            self.entry_node_index = Some(new_index);
            self.index_max_level = new_node_level;
            self.id_to_index.insert(id, new_index);
            self.nodes.push(new_node);
            return;
        }

        self.id_to_index.insert(id, new_index);
        self.nodes.push(new_node);

        let mut entry_point_index = self
            .entry_node_index
            .expect("in insert, entry_node_index is none");
        if self.index_max_level >= new_node_level + 1 {
            for current_level in (new_node_level + 1..=self.index_max_level).rev() {
                entry_point_index =
                    self.greedy_search_at_level(&data, entry_point_index, current_level);
            }
        }

        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        for current_lvl in (0..=min_level).rev() {
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

            let result_set: HashSet<usize> = search_results
                .iter()
                .map(|(result_index, _)| *result_index)
                .collect();

            let neighbors_set =
                self.select_neighbors_simple(&result_set, &data, max_neighbors_to_select);
            for &neighbor_index in &neighbors_set {
                self.add_neighbor_to_node_at_level_by_index(new_index, neighbor_index, current_lvl);
                self.add_neighbor_to_node_at_level_by_index(neighbor_index, new_index, current_lvl);
            }

            for &neighbor_index in &neighbors_set {
                let e_conn_set: HashSet<usize> = {
                    let e_node = self
                        .get_node_by_index(neighbor_index)
                        .expect("in insert, get_node_by_index cannot handle neighbor_index");
                    e_node.get_neighbors()[current_lvl]
                        .iter()
                        .copied()
                        .collect()
                };

                if e_conn_set.len() > max_neighbors_to_keep {
                    let e_data = {
                        let e_node = self
                            .get_node_by_index(neighbor_index)
                            .expect("in insert, get_node_by_index cannot handle neighbor_index");
                        e_node.get_data().clone()
                    };

                    let e_new_conn_vec: Vec<usize> = {
                        self.select_neighbors_simple(&e_conn_set, &e_data, max_neighbors_to_keep)
                    };

                    let e_node = self
                        .get_mut_node_by_index(neighbor_index)
                        .expect("in insert, get_mut_node_by_index cannot handle neighbor_index");
                    let old_neighbors = e_node.neighbors[current_lvl].clone();
                    let old_set: HashSet<usize> = old_neighbors.iter().cloned().collect();
                    let new_set: HashSet<usize> = e_new_conn_vec.iter().cloned().collect();
                    let to_delete: HashSet<usize> = old_set.difference(&new_set).cloned().collect();
                    e_node.neighbors[current_lvl] = e_new_conn_vec;
                    for neighbor_to_delete_index in to_delete {
                        self.delete_neighbor_relationship_by_index(
                            neighbor_index,
                            neighbor_to_delete_index,
                            Some(current_lvl),
                        );
                    }
                }
            }

            entry_point_index = search_results
                .first()
                .map(|(index, _)| *index)
                .expect("search_result_index赋值错误");
        }

        if new_node_level > self.index_max_level {
            self.entry_node_index = Some(new_index);
            self.index_max_level = new_node_level;
        }
    }
}
