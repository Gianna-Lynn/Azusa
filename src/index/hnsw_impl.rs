use core::f32;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

use rand::{Rng, SeedableRng};

use crate::distance;
use crate::index::hnsw::SearchLayerWorkspace;
use crate::types::{Id, Metric, Vector};

use super::hnsw_stats::{HeapAccessMode, SearchLayerStats};
use super::{HnswIndex, HnswNode};

// =========================
// 构造 / 基础信息
// =========================
impl HnswIndex {
    pub fn new() -> Self {
        Self::new_with_metric(Metric::Cosine)
    }

    pub fn new_with_metric(metric: Metric) -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_index: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(super::MY_SEED),
            timing: super::hnsw_stats::InsertTimings::default(),
            workspace: SearchLayerWorkspace::new(),
        }
    }

    pub fn print_timings(&self) {
        let total = self.timing.search_layer_ns
            + self.timing.greedy_search_ns
            + self.timing.select_neighbors_ns
            + self.timing.pruning_ns
            + self.timing.link_ns;

        if total == 0 {
            println!("(no timings recorded yet)");
            return;
        }

        println!("\n=== insert timings ({} inserts) ===", self.timing.calls);
        println!(
            "  greedy_search:     {:.2}s ({:.1}%)",
            self.timing.greedy_search_ns as f64 / 1e9,
            self.timing.greedy_search_ns as f64 / total as f64 * 100.0
        );
        println!(
            "  search_layer:      {:.2}s ({:.1}%)",
            self.timing.search_layer_ns as f64 / 1e9,
            self.timing.search_layer_ns as f64 / total as f64 * 100.0
        );
    }
}

// =========================
// 节点与邻接表访问
// =========================
impl HnswIndex {
    pub(crate) fn get_index_by_id(&self, target_id: Id) -> Option<usize> {
        self.id_to_index.get(&target_id).copied()
    }

    pub(crate) fn get_node_by_index(&self, target_index: usize) -> Option<&HnswNode> {
        self.nodes.get(target_index)
    }

    pub(crate) fn get_mut_node_by_index(&mut self, target_index: usize) -> Option<&mut HnswNode> {
        self.nodes.get_mut(target_index)
    }

    pub(crate) fn get_id_by_index(&self, target_index: usize) -> Id {
        self.nodes[target_index].id
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get_nodes(&self) -> &Vec<HnswNode> {
        &self.nodes
    }

    pub fn get_entry_node_id(&self) -> Option<u64> {
        self.entry_node_index.map(|idx| self.get_id_by_index(idx))
    }

    pub fn find_index_of_id_in_neighbors(target_id: Id, target_vector: &[Id]) -> Option<usize> {
        for (idx, &x) in target_vector.iter().enumerate() {
            if x == target_id {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_node_by_id(&self, target_id: Id) -> Option<&HnswNode> {
        self.get_index_by_id(target_id)
            .and_then(|idx| self.get_node_by_index(idx))
    }

    pub fn get_mut_node_by_id(&mut self, target_id: Id) -> Option<&mut HnswNode> {
        self.get_index_by_id(target_id)
            .and_then(|idx| self.get_mut_node_by_index(idx))
    }

    pub(crate) fn add_neighbor_to_node_at_level_by_index(
        &mut self,
        node_index: usize,
        neighbor_index: usize,
        level: usize,
    ) {
        let node = self
            .get_mut_node_by_index(node_index)
            .expect("in add_neighbor_to_node, in get_mut_node_by_index, got None");
        node.add_neighbor(neighbor_index, level);
    }

    pub fn add_neighbor_to_node_at_level(&mut self, node_id: Id, neighbor_id: Id, level: usize) {
        let node_index = self
            .get_index_by_id(node_id)
            .expect("in add_neighbor_to_node_at_level, node_id not found");
        let neighbor_index = self
            .get_index_by_id(neighbor_id)
            .expect("in add_neighbor_to_node_at_level, neighbor_id not found");
        self.add_neighbor_to_node_at_level_by_index(node_index, neighbor_index, level);
    }

    pub(crate) fn delete_neighbor_relationship_by_index(
        &mut self,
        node1_index: usize,
        node2_index: usize,
        level: Option<usize>,
    ) {
        let node1 = self.get_mut_node_by_index(node1_index).expect(
            "In function delete_neighbor_relationship_by_index, get_mut_node_by_index got None",
        );
        match level {
            Some(level) => node1.delete_neighbor_single_direction_at_level(node2_index, level),
            None => node1.delete_neighbor_single_direction_at_all_levels(node2_index),
        }

        let node2 = self.get_mut_node_by_index(node2_index).expect(
            "In function delete_neighbor_relationship_by_index, get_mut_node_by_index got None",
        );
        match level {
            Some(level) => node2.delete_neighbor_single_direction_at_level(node1_index, level),
            None => node2.delete_neighbor_single_direction_at_all_levels(node1_index),
        }
    }

    pub fn delete_neighbor_relationship_at_all_levels(&mut self, node1_id: Id, node2_id: Id) {
        self.delete_neighbor_relationship(node1_id, node2_id, None);
    }

    fn delete_neighbor_relationship(&mut self, node1_id: Id, node2_id: Id, level: Option<usize>) {
        let node1_index = self
            .get_index_by_id(node1_id)
            .expect("In function delete_neighbor_relationship, node1_id not found");
        let node2_index = self
            .get_index_by_id(node2_id)
            .expect("In function delete_neighbor_relationship, node2_id not found");
        self.delete_neighbor_relationship_by_index(node1_index, node2_index, level);
    }

    pub fn delete_neighbor_relationship_at_level(
        &mut self,
        node1_id: Id,
        node2_id: Id,
        level: usize,
    ) {
        self.delete_neighbor_relationship(node1_id, node2_id, Some(level));
    }

    pub fn get_neighbors_at_lvl(&self, id: Id, lvl: usize) -> Vec<Id> {
        let node_to_search = self.get_node_by_id(id).expect("get_node_by_id不存在");
        node_to_search.get_neighbors()[lvl]
            .iter()
            .map(|&neighbor_idx| self.get_id_by_index(neighbor_idx))
            .collect()
    }

    fn get_neighbors_at_lvl_from_parts<'a>(
        nodes: &'a [HnswNode],
        index: usize,
        lvl: usize,
    ) -> &'a [usize] {
        &nodes[index].get_neighbors()[lvl]
    }
}

// =========================
// 相似度 / 堆辅助工具
// =========================
impl HnswIndex {
    pub fn compute_similarity(&self, a: &Vector, b: &Vector) -> f32 {
        distance::similarity(a, b, self.metric)
    }

    pub(super) fn get_nearest(
        &self,
        index_similarity_cache: &mut HashMap<usize, f32>,
        set: &HashSet<usize>,
        query: &Vector,
    ) -> Option<usize> {
        let mut max_similarity = f32::NEG_INFINITY;
        let mut max_elem_index: Option<usize> = None;

        for elem_index in set {
            let temp_similarity = self.get_similarity(index_similarity_cache, *elem_index, query);
            if temp_similarity > max_similarity {
                max_similarity = temp_similarity;
                max_elem_index = Some(*elem_index);
            }
        }

        max_elem_index
    }

    pub fn get_similarity(
        &self,
        index_similarity_cache: &mut HashMap<usize, f32>,
        target_index: usize,
        query: &Vector,
    ) -> f32 {
        if let Some(similarity) = index_similarity_cache.get(&target_index) {
            *similarity
        } else {
            let target_node = self
                .get_node_by_index(target_index)
                .expect("In function get_similarity, failed to get target_node by target_index");
            let new_similarity = self.compute_similarity(target_node.get_data(), query);
            index_similarity_cache.insert(target_index, new_similarity);
            new_similarity
        }
    }

    fn get_similarity_from_parts(
        nodes: &[HnswNode],
        metric: Metric,
        index_similarity_cache: &mut HashMap<usize, f32>,
        target_index: usize,
        query: &Vector,
    ) -> f32 {
        if let Some(similarity) = index_similarity_cache.get(&target_index) {
            *similarity
        } else {
            let new_similarity = distance::similarity(nodes[target_index].get_data(), query, metric);
            index_similarity_cache.insert(target_index, new_similarity);
            new_similarity
        }
    }

    fn get_heap_extreme_refer_alive_bool_vec<T: Ord>(
        alive_bool_vec: &[bool],
        heap: &mut BinaryHeap<(T, usize)>,
        mode: HeapAccessMode,
    ) -> Option<usize> {
        match mode {
            HeapAccessMode::Peek => {
                while let Some(top) = heap.peek() {
                    if alive_bool_vec[top.1] {
                        return Some(top.1);
                    }
                    heap.pop();
                }
                None
            }
            HeapAccessMode::Pop => {
                while let Some(top) = heap.pop() {
                    if !alive_bool_vec[top.1] {
                        continue;
                    }
                    return Some(top.1);
                }
                None
            }
        }
    }
}

// =========================
// 图搜索
// =========================
impl HnswIndex {
    pub fn search_layer(
        &mut self,
        query: &Vector,
        entry_index: usize,
        level: usize,
        ef: usize,
        index_similarity_cache: &mut HashMap<usize, f32>,
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

        let entry_similarity = Self::get_similarity_from_parts(
            nodes,
            metric,
            index_similarity_cache,
            entry_index,
            query,
        );
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

            let current_candidate_similarity = Self::get_similarity_from_parts(
                nodes,
                metric,
                index_similarity_cache,
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

                let neighbor_similarity = Self::get_similarity_from_parts(
                    nodes,
                    metric,
                    index_similarity_cache,
                    neighbor_index,
                    query,
                );

                let worst_result_similarity = workspace
                    .result_heap
                    .peek()
                    .map(|(reverse_score, _)| reverse_score.0.into_inner())
                    .expect("result_heap shouldn't be empty");

                if neighbor_similarity >= worst_result_similarity || workspace.result_heap.len() < ef
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

    pub fn greedy_search_at_level(&self, query: &Vector, entry_index: usize, level: usize) -> usize {
        let mut index_similarity_cache: HashMap<usize, f32> = HashMap::new();
        let mut current_node_index = entry_index;
        let mut max_similarity =
            self.get_similarity(&mut index_similarity_cache, current_node_index, query);
        let mut max_similarity_index = current_node_index;

        loop {
            let current_node = self
                .get_node_by_index(current_node_index)
                .expect("未能根据entry_index检索到entry_node");
            let current_node_neighbors = &current_node.get_neighbors()[level];

            for &neighbor_index in current_node_neighbors {
                let neighbor_node_similarity =
                    self.get_similarity(&mut index_similarity_cache, neighbor_index, query);
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

        let mut index_similarity_cache: HashMap<usize, f32> = HashMap::new();
        let (candidate_set, _) = self.search_layer(
            query,
            current_entry_index,
            0,
            ef_search,
            &mut index_similarity_cache,
        );

        let mut candidate_sequence: Vec<usize> = candidate_set.into_iter().collect();
        candidate_sequence.sort_by(|idx1, idx2| {
            let score1 = self.get_similarity(&mut index_similarity_cache, *idx1, query);
            let score2 = self.get_similarity(&mut index_similarity_cache, *idx2, query);
            match score2.partial_cmp(&score1).unwrap() {
                std::cmp::Ordering::Equal => self.get_id_by_index(*idx1).cmp(&self.get_id_by_index(*idx2)),
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

// =========================
// 构建 / 选择邻居
// =========================
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
        let mut index_similarity_cache: HashMap<usize, f32> = HashMap::new();
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
            let (search_results, sl_stats) = self.search_layer(
                &data,
                entry_point_index,
                current_lvl,
                ef_construction,
                &mut index_similarity_cache,
            );
            self.timing.search_layer_ns += t.elapsed().as_nanos();
            self.timing.search_layer_compute_sim_ns += sl_stats.compute_sim_ns;
            self.timing.search_layer_furthest_scan_ns += sl_stats.furthest_scan_ns;
            self.timing.search_layer_neighbors_fetch_ns += sl_stats.neighbors_fetch_ns;
            self.timing.search_layer_heap_pop_ns += sl_stats.heap_pop_ns;
            self.timing.search_layer_bookkeeping_ns += sl_stats.candidate_bookkeeping_ns;
            self.timing.search_layer_loop_iterations += sl_stats.loop_iterations;

            let t = Instant::now();
            let selected_neighbors = self.select_neighbors_simple(
                &search_results,
                &data,
                max_neighbors_to_select,
                &mut index_similarity_cache,
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
                        let neighbor_node = self
                            .get_node_by_index(neighbor_index)
                            .expect("in insert, get_node_by_index cannot handle neighbor_index");
                        let mut neighbor_similarity_cache: HashMap<usize, f32> = HashMap::new();
                        self.select_neighbors_simple(
                            &neighbor_current_connections,
                            neighbor_node.get_data(),
                            max_neighbors_to_keep,
                            &mut neighbor_similarity_cache,
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

            entry_point_index = self
                .get_nearest(&mut index_similarity_cache, &search_results, &data)
                .expect("search_result_index赋值错误");
        }

        if node_lvl > self.index_max_level {
            self.entry_node_index = Some(new_index);
            self.index_max_level = node_lvl;
        }
    }

    pub fn select_neighbors_simple(
        &self,
        candidate_set: &HashSet<usize>,
        query: &Vector,
        max_neighbors_to_select: usize,
        cache: &mut HashMap<usize, f32>,
    ) -> Vec<usize> {
        let mut index_score_map: Vec<(usize, f32)> = candidate_set
            .iter()
            .map(|candidate_index| {
                (
                    *candidate_index,
                    self.get_similarity(cache, *candidate_index, query),
                )
            })
            .collect();
        index_score_map.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => self.get_id_by_index(a.0).cmp(&self.get_id_by_index(b.0)),
            other => other,
        });
        let take_count = max_neighbors_to_select.min(index_score_map.len());
        index_score_map
            .iter()
            .take(take_count)
            .map(|(index, _)| *index)
            .collect()
    }
}
