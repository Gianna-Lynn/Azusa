use core::f32;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

use rand::{Rng, SeedableRng};

use crate::distance;
use crate::types::{Id, Metric, Vector};

use super::hnsw_stats::{HeapAccessMode, SearchLayerStats};
use super::{HnswIndex, HnswNode};

// =========================
// 构造 / 基础信息
// =========================
impl HnswIndex {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_id: None,
            index_max_level: 0,
            metric: Metric::Cosine,
            id_to_index: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(super::MY_SEED),
            timing: super::hnsw_stats::InsertTimings::default(),
        }
    }

    pub fn new_with_metric(metric: Metric) -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_id: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(super::MY_SEED),
            timing: super::hnsw_stats::InsertTimings::default(),
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
        self.entry_node_id
    }

    pub fn find_index_of_id_in_neighbors(target_id: Id, target_vector: &Vec<Id>) -> Option<usize> {
        for (idx, &x) in target_vector.iter().enumerate() {
            if x == target_id {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_node_by_id(&self, target_id: Id) -> Option<&HnswNode> {
        self.id_to_index.get(&target_id).map(|idx| &self.nodes[*idx])
    }

    pub fn get_mut_node_by_id(&mut self, target_id: Id) -> Option<&mut HnswNode> {
        if let Some(idx) = self.id_to_index.get(&target_id).copied() {
            Some(&mut self.nodes[idx])
        } else {
            None
        }
    }

    pub fn add_neighbor_to_node_at_level(&mut self, node_id: Id, neighbor_id: Id, level: usize) {
        let node = self
            .get_mut_node_by_id(node_id)
            .expect("in add_neighbor_to_node, in get_mut_node_by_id, got None");
        node.add_neighbor(neighbor_id, level);
    }

    pub fn delete_neighbor_relationship_at_all_levels(&mut self, node1_id: Id, node2_id: Id) {
        let node1 = self
            .get_mut_node_by_id(node1_id)
            .expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        node1.delete_neighbor_single_direction_at_all_levels(node2_id);
        let node2 = self
            .get_mut_node_by_id(node2_id)
            .expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        node2.delete_neighbor_single_direction_at_all_levels(node1_id);
    }

    pub fn delete_neighbor_relationship_at_level(
        &mut self,
        node1_id: Id,
        node2_id: Id,
        level: usize,
    ) {
        let node1 = self
            .get_mut_node_by_id(node1_id)
            .expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        node1.delete_neighbor_single_direction_at_level(node2_id, level);
        let node2 = self
            .get_mut_node_by_id(node2_id)
            .expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        node2.delete_neighbor_single_direction_at_level(node1_id, level);
    }

    pub fn get_neighbors_at_lvl(&self, id: Id, lvl: usize) -> HashSet<Id> {
        let node_to_search = self.get_node_by_id(id).expect("get_node_by_id不存在");
        if lvl >= node_to_search.get_neighbors().len() {
            return HashSet::new();
        }
        node_to_search.get_neighbors()[lvl].iter().cloned().collect()
    }
}

// =========================
// 相似度 / 堆辅助工具
// =========================
impl HnswIndex {
    pub fn compute_similarity(&self, a: &Vector, b: &Vector) -> f32 {
        distance::similarity(a, b, self.metric)
    }

    pub fn get_similarity(
        &self,
        id_similarity_cache: &mut HashMap<Id, f32>,
        target_id: &Id,
        query: &Vector,
    ) -> f32 {
        if let Some(similarity) = id_similarity_cache.get(target_id) {
            *similarity
        } else {
            let target_node = self
                .get_node_by_id(*target_id)
                .expect("In function get_similarity, failed to get target_node by target_id");
            let new_similarity = self.compute_similarity(target_node.get_data(), query);
            id_similarity_cache.insert(*target_id, new_similarity);
            new_similarity
        }
    }

    fn get_heap_extreme<T: Ord>(
        &self,
        alive_set: &HashSet<Id>,
        heap: &mut BinaryHeap<(T, Id)>,
        mode: HeapAccessMode,
    ) -> Option<Id> {
        match mode {
            // 返回堆顶，但会跳过那些已经不在 alive_set 中的失效元素。
            HeapAccessMode::PeekPruned => {
                while let Some(top) = heap.peek() {
                    if alive_set.contains(&top.1) {
                        return Some(top.1);
                    }
                    heap.pop();
                }
                None
            }
            // 返回堆顶，同时把失效元素不断弹掉。
            HeapAccessMode::PopAlive => {
                while let Some(top) = heap.pop() {
                    if !alive_set.contains(&top.1) {
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
        &self,
        query: &Vector,
        entry_id: Id,
        level: usize,
        ef: usize,
        id_similarity_cache: &mut HashMap<Id, f32>,
    ) -> (HashSet<Id>, SearchLayerStats) {
        assert!(ef >= 1, "ef至少为1");
        let mut stats = SearchLayerStats::default();
        // 已访问过的点，避免重复扩展。
        let mut visited_set: HashSet<Id> = HashSet::new();
        // 候选点池，负责“下一步先扩展谁”。
        let mut candidate_set: HashSet<Id> = HashSet::new();
        let mut candidate_heap: BinaryHeap<(ordered_float::NotNan<f32>, Id)> = BinaryHeap::new();
        // 当前保留的最好点。
        let mut result_set: HashSet<Id> = HashSet::new();
        let mut result_heap: BinaryHeap<(std::cmp::Reverse<ordered_float::NotNan<f32>>, Id)> =
            BinaryHeap::new();

        // 入口点先进入三套状态。
        visited_set.insert(entry_id);
        candidate_set.insert(entry_id);
        let entry_similarity = self.get_similarity(id_similarity_cache, &entry_id, query);
        candidate_heap.push((ordered_float::NotNan::new(entry_similarity).unwrap(), entry_id));
        result_set.insert(entry_id);
        result_heap.push((
            std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()),
            entry_id,
        ));

        while !candidate_set.is_empty() {
            stats.loop_iterations += 1;
            // 先取出当前最值得继续展开的候选点。
            let current_candidate_id = self
                .get_heap_extreme(&candidate_set, &mut candidate_heap, HeapAccessMode::PopAlive)
                .expect("current_candidate_id不存在");
            candidate_set.remove(&current_candidate_id);

            // 如果候选点已经比结果池里最差的点还差，就可以提前停。
            let mut worst_result_id = self
                .get_heap_extreme(&result_set, &mut result_heap, HeapAccessMode::PeekPruned)
                .expect("worst_result_id不存在");
            let current_candidate_similarity =
                self.get_similarity(id_similarity_cache, &current_candidate_id, query);
            let worst_result_similarity =
                self.get_similarity(id_similarity_cache, &worst_result_id, query);
            if current_candidate_similarity < worst_result_similarity {
                break;
            }

            // 展开当前候选点的邻居。
            let current_candidate_neighbors = self.get_neighbors_at_lvl(current_candidate_id, level);
            for neighbor_id in current_candidate_neighbors {
                if visited_set.contains(&neighbor_id) {
                    continue;
                }
                visited_set.insert(neighbor_id);

                // 看这个邻居值不值得进入候选池 / 结果池。
                worst_result_id = self
                    .get_heap_extreme(&result_set, &mut result_heap, HeapAccessMode::PeekPruned)
                    .expect("worst_result_id不存在");
                let neighbor_similarity = self.get_similarity(id_similarity_cache, &neighbor_id, query);
                let worst_result_similarity =
                    self.get_similarity(id_similarity_cache, &worst_result_id, query);

                if neighbor_similarity >= worst_result_similarity || result_set.len() < ef {
                    candidate_set.insert(neighbor_id);
                    candidate_heap.push((
                        ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                        neighbor_id,
                    ));
                    result_set.insert(neighbor_id);
                    result_heap.push((
                        std::cmp::Reverse(ordered_float::NotNan::new(neighbor_similarity).unwrap()),
                        neighbor_id,
                    ));
                    if result_set.len() > ef {
                        // 超过 ef 后，踢掉结果池里最差的点。
                        if let Some(id_to_remove) =
                            self.get_heap_extreme(&result_set, &mut result_heap, HeapAccessMode::PeekPruned)
                        {
                            result_set.remove(&id_to_remove);
                        }
                    }
                }
            }
        }

        (result_set, stats)
    }

    // 从入口点做单层贪心搜索，找到更好的起点。
    pub fn greedy_search_at_level(&self, query: &Vector, entry_id: Id, level: usize) -> Id {
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        let mut current_node = self
            .get_node_by_id(entry_id)
            .expect("未能根据entry_id检索到entry_node");
        let mut current_node_neighbors = &current_node.get_neighbors()[level];
        let mut max_similarity = self.get_similarity(&mut id_similarity_cache, &entry_id, query);
        let mut max_similarity_id = entry_id;

        loop {
            for neighbor in current_node_neighbors {
                let neighbor_node_similarity =
                    self.get_similarity(&mut id_similarity_cache, neighbor, query);
                if neighbor_node_similarity > max_similarity {
                    max_similarity = neighbor_node_similarity;
                    max_similarity_id = *neighbor;
                }
            }

            if current_node.get_id() == &max_similarity_id {
                return max_similarity_id;
            } else {
                current_node = self
                    .get_node_by_id(max_similarity_id)
                    .expect("未能更新current_node");
                current_node_neighbors = &current_node.get_neighbors()[level];
            }
        }
    }

    pub fn search_knn(&self, query: &Vector, k: usize, ef_search: usize) -> Vec<Id> {
        let Some(mut current_entry_id) = self.entry_node_id else {
            return Vec::new();
        };

        for lvl in (1..=self.index_max_level).rev() {
            current_entry_id = self.greedy_search_at_level(query, current_entry_id, lvl);
        }

        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        let (candidate_set, _) =
            self.search_layer(query, current_entry_id, 0, ef_search, &mut id_similarity_cache);

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

    pub fn insert(&mut self, id: Id, data: Vector, ef_construction: usize, m: usize, m_max: usize) {
        self.timing.calls += 1;
        let new_node_level = self.sample_level();
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
            let t = Instant::now();
            for current_level in (new_node_level + 1..=self.index_max_level).rev() {
                entry_point_id = self.greedy_search_at_level(&data, entry_point_id, current_level);
            }
            self.timing.greedy_search_ns += t.elapsed().as_nanos();
        }

        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        for current_lvl in (0..=min_level).rev() {
            let t = Instant::now();
            let (result_set, sl_stats) =
                self.search_layer(&data, entry_point_id, current_lvl, ef_construction, &mut id_similarity_cache);
            self.timing.search_layer_ns += t.elapsed().as_nanos();
            self.timing.search_layer_compute_sim_ns += sl_stats.compute_sim_ns;
            self.timing.search_layer_furthest_scan_ns += sl_stats.furthest_scan_ns;
            self.timing.search_layer_neighbors_fetch_ns += sl_stats.neighbors_fetch_ns;
            self.timing.search_layer_heap_pop_ns += sl_stats.heap_pop_ns;
            self.timing.search_layer_bookkeeping_ns += sl_stats.candidate_bookkeeping_ns;
            self.timing.search_layer_loop_iterations += sl_stats.loop_iterations;

            let t = Instant::now();
            let neighbors_set = &self.select_neighbors_simple(&result_set, &data, m, &mut id_similarity_cache);
            self.timing.select_neighbors_ns += t.elapsed().as_nanos();

            let t = Instant::now();
            for neighbor_node_id in neighbors_set {
                self.add_neighbor_to_node_at_level(id, *neighbor_node_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_node_id, id, current_lvl);
            }
            self.timing.link_ns += t.elapsed().as_nanos();

            let t = Instant::now();
            for e_id in neighbors_set {
                let e_conn_set: HashSet<u64> = {
                    let e_node = self
                        .get_node_by_id(*e_id)
                        .expect("in insert, get_node_by_id cannot handle e_id");
                    e_node.get_neighbors()[current_lvl].iter().copied().collect()
                };
                if e_conn_set.len() > m_max {
                    let e_new_conn_vec: Vec<u64> = {
                        let e_node = self
                            .get_node_by_id(*e_id)
                            .expect("in insert, get_node_by_id cannot handle e_id");
                        let mut e_cache: HashMap<Id, f32> = HashMap::new();
                        self.select_neighbors_simple(&e_conn_set, e_node.get_data(), m_max, &mut e_cache)
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
                        self.delete_neighbor_relationship_at_level(*e_id, neighbor_to_delete_id, current_lvl);
                    }
                }
            }
            self.timing.pruning_ns += t.elapsed().as_nanos();

            // TODO: 这里可以换成堆实现
            entry_point_id = self
                .get_nearest(&mut id_similarity_cache, &result_set, &data)
                .expect("searchh_result_id赋值错误");
        }

        if new_node_level > self.index_max_level {
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
        }
    }

    // 从候选点里选出前 m 个最相近的邻居。
    pub fn select_neighbors_simple(
        &self,
        candidate_set: &HashSet<Id>,
        query: &Vector,
        m: usize,
        cache: &mut HashMap<Id, f32>,
    ) -> Vec<Id> {
        let mut id_score_map: Vec<(u64, f32)> = candidate_set
            .iter()
            .map(|candidate| (*candidate, self.get_similarity(cache, candidate, query)))
            .collect();
        id_score_map.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
        let take_count = m.min(id_score_map.len());
        id_score_map.iter().take(take_count).map(|(id, _)| *id).collect()
    }
}
