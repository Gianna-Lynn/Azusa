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
            entry_node_id: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(super::MY_SEED),
            timing: super::hnsw_stats::InsertTimings::default(),
            workspace: SearchLayerWorkspace::new()
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
    fn get_index_by_id(&self, target_id: Id) -> Option<usize> {
        self.id_to_index.get(&target_id).copied()
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
        self.get_index_by_id(target_id).map(|idx| &self.nodes[idx])
    }

    pub fn get_mut_node_by_id(&mut self, target_id: Id) -> Option<&mut HnswNode> {
        self.get_index_by_id(target_id)
            .map(|idx| &mut self.nodes[idx])
    }

    pub fn add_neighbor_to_node_at_level(&mut self, node_id: Id, neighbor_id: Id, level: usize) {
        let node = self
            .get_mut_node_by_id(node_id)
            .expect("in add_neighbor_to_node, in get_mut_node_by_id, got None");
        node.add_neighbor(neighbor_id, level);
    }

    pub fn delete_neighbor_relationship_at_all_levels(&mut self, node1_id: Id, node2_id: Id) {
        self.delete_neighbor_relationship(node1_id, node2_id, None);
    }

    fn delete_neighbor_relationship(&mut self, node1_id: Id, node2_id: Id, level: Option<usize>) {
        let node1 = self
            .get_mut_node_by_id(node1_id)
            .expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        match level {
            Some(level) => node1.delete_neighbor_single_direction_at_level(node2_id, level),
            None => node1.delete_neighbor_single_direction_at_all_levels(node2_id),
        }

        let node2 = self
            .get_mut_node_by_id(node2_id)
            .expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        match level {
            Some(level) => node2.delete_neighbor_single_direction_at_level(node1_id, level),
            None => node2.delete_neighbor_single_direction_at_all_levels(node1_id),
        }
    }

    pub fn delete_neighbor_relationship_at_level(
        &mut self,
        node1_id: Id,
        node2_id: Id,
        level: usize,
    ) {
        self.delete_neighbor_relationship(node1_id, node2_id, Some(level));
    }

    pub fn get_neighbors_at_lvl(&self, id: Id, lvl: usize) -> &[Id] {
        let node_to_search = self.get_node_by_id(id).expect("get_node_by_id不存在");
        &node_to_search.get_neighbors()[lvl]
    }

    // 大橘写的: 为了保留 workspace/stamp 设计, 同时避开 search_layer 里对邻居列表的 to_vec 复制.
    fn get_neighbors_at_lvl_from_parts<'a>(
        nodes: &'a [HnswNode],
        id_to_index: &HashMap<Id, usize>,
        id: Id,
        lvl: usize,
    ) -> &'a [Id] {
        let node_idx = id_to_index[&id];
        &nodes[node_idx].get_neighbors()[lvl]
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
        id_similarity_cache: &mut HashMap<Id, f32>,
        set: &HashSet<Id>,
        query: &Vector,
    ) -> Option<Id> {
        let mut max_similarity = f32::NEG_INFINITY;
        let mut max_elem_id: Option<Id> = None;

        for elem_id in set {
            let temp_similarity = self.get_similarity(id_similarity_cache, elem_id, query);
            if temp_similarity > max_similarity {
                max_similarity = temp_similarity;
                max_elem_id = Some(*elem_id);
            }
        }

        max_elem_id
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

    // 大橘写的: 和上面的邻居 helper 配套, 直接基于拆开的字段取相似度, 减少 search_layer 对整个 self 的借用压力.
    fn get_similarity_from_parts(
        nodes: &[HnswNode],
        id_to_index: &HashMap<Id, usize>,
        metric: Metric,
        id_similarity_cache: &mut HashMap<Id, f32>,
        target_id: &Id,
        query: &Vector,
    ) -> f32 {
        if let Some(similarity) = id_similarity_cache.get(target_id) {
            *similarity
        } else {
            let node_idx = id_to_index[target_id];
            let new_similarity = distance::similarity(nodes[node_idx].get_data(), query, metric);
            id_similarity_cache.insert(*target_id, new_similarity);
            new_similarity
        }
    }

    fn get_heap_extreme_refer_alive_bool_vec<T: Ord>(
        id_to_index: &HashMap<Id, usize>,
        alive_bool_vec: &Vec<bool>,
        heap: &mut BinaryHeap<(T, Id)>,
        mode: HeapAccessMode,
    ) -> Option<Id> {
        match mode {
            HeapAccessMode::Peek => {
                while let Some(top) = heap.peek() {
                    if alive_bool_vec[id_to_index[&top.1]] {
                        return Some(top.1);
                    }
                    heap.pop();
                }
                None
            }
            HeapAccessMode::Pop => {
                while let Some(top) = heap.pop() {
                    if !alive_bool_vec[id_to_index[&top.1]] {
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
        entry_id: Id,
        level: usize,
        ef: usize,
        id_similarity_cache: &mut HashMap<Id, f32>,
    ) -> (HashSet<Id>, SearchLayerStats) {
        
        assert!(ef >= 1, "ef至少为1");

        let mut stats = SearchLayerStats::default();
        self.workspace.reset(self.nodes.len());
        let nodes = &self.nodes;
        let id_to_index = &self.id_to_index;
        let metric = self.metric;
        let workspace = &mut self.workspace;



        // 注意:
        // candidate_alive_num 是 candidate_alive 的“计数镜像”.
        // 以后只要新增/删除 candidate 的入口, 必须同时维护这个计数.
        // 否则 while candidate_alive_num > 0 会和真实活性状态脱节.
        let mut candidate_alive_num : usize = 0;
        
        // 入口点先进入三套状态。
        workspace.visited_marks[id_to_index[&entry_id]] = workspace.visited_epoch;
        workspace.candidate_alive[id_to_index[&entry_id]] = true;
        candidate_alive_num += 1;

        let entry_similarity = Self::get_similarity_from_parts(
            nodes,
            id_to_index,
            metric,
            id_similarity_cache,
            &entry_id,
            query,
        );
        workspace.candidate_heap.push((
            ordered_float::NotNan::new(entry_similarity).unwrap(),
            entry_id,
        ));

        // result_heap 维护当前结果集, candidate_heap 维护待扩展候选点.
        workspace.result_heap.push((
            std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()),
            entry_id,
        ));

        while candidate_alive_num > 0 {
            stats.loop_iterations += 1;
            // 先取出当前最值得继续展开的候选点。
            let current_candidate_id = Self::get_heap_extreme_refer_alive_bool_vec(
                    id_to_index,
                    &workspace.candidate_alive,
                    &mut workspace.candidate_heap,
                    HeapAccessMode::Pop,
                )
                .expect("current_candidate_id不存在");

    
            workspace.candidate_alive[id_to_index[&current_candidate_id]] = false;
            candidate_alive_num -= 1;

            // 如果候选点已经比结果池里最差的点还差，就可以提前停。
            let worst_result_similarity = workspace
                .result_heap
                .peek()
                .map(|(reverse_score, _ )| reverse_score.0.into_inner()) //.0的作用是解包 Reverse, into_inner的作用是解包 NotNan, 得到 f32
                .expect("result_heap shouldn't be empty");

            
            let current_candidate_similarity = Self::get_similarity_from_parts(
                nodes,
                id_to_index,
                metric,
                id_similarity_cache,
                &current_candidate_id,
                query,
            );

            if current_candidate_similarity < worst_result_similarity {
                break;
            }

            let current_candidate_neighbors =
                Self::get_neighbors_at_lvl_from_parts(nodes, id_to_index, current_candidate_id, level);
            for &neighbor_id in current_candidate_neighbors {
                

                let idx = id_to_index[&neighbor_id];

                // epoch设计: workspace里有个全局递增的visited_epoch, 
                // visited_marks里记录每个点最后一次被访问的epoch. 
                // 访问时比较两者是否相等即可判断这个点在本轮访问中是否被访问过.
                // 避免了每轮访问前清空visited_marks带来的性能问题.
                if workspace.visited_marks[idx] == workspace.visited_epoch{
                    continue;
                }

                workspace.visited_marks[idx] = workspace.visited_epoch; // visited[映射后的新元素索引] = true


                // 看这个邻居值不值得进入候选池 / 结果池。
                let neighbor_similarity = Self::get_similarity_from_parts(
                    nodes,
                    id_to_index,
                    metric,
                    id_similarity_cache,
                    &neighbor_id,
                    query,
                );

                let worst_result_similarity = workspace
                    .result_heap
                    .peek()
                    .map(|(reverse_score, _ )| reverse_score.0.into_inner()) //.0的作用是解包 Reverse, into_inner的作用是解包 NotNan, 得到 f32
                    .expect("result_heap shouldn't be empty");

                if neighbor_similarity >= worst_result_similarity || workspace.result_heap.len() < ef {

                    workspace.candidate_alive[id_to_index[&neighbor_id]] = true;
                    candidate_alive_num += 1;


                    workspace.candidate_heap.push((
                        ordered_float::NotNan::new(neighbor_similarity).unwrap(),
                        neighbor_id,
                    ));
                    workspace.result_heap.push((
                        std::cmp::Reverse(ordered_float::NotNan::new(neighbor_similarity).unwrap()),
                        neighbor_id,
                    ));
                    if workspace.result_heap.len() > ef {
                        // 超过 ef 后，踢掉结果池里最差的点。
                        workspace.result_heap.pop();
                    }  
                }
            }
        }

        let result_set:HashSet<u64> = workspace.result_heap.iter().map(|( _, result_id)| result_id).copied().collect();

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

    pub fn search_knn(&mut self, query: &Vector, k: usize, ef_search: usize) -> Vec<Id> {
        let Some(mut current_entry_id) = self.entry_node_id else {
            return Vec::new();
        };

        for lvl in (1..=self.index_max_level).rev() {
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
        max_neighbors_to_select: usize,  // 论文m
        max_neighbors_to_keep: usize,    // 论文m_max
    ) {
        self.timing.calls += 1;
        // 伪代码 line 4：随机采样新节点所在的最高层
        let node_lvl = self.sample_level();
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, node_lvl);
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();

        if self.is_empty() {
            self.entry_node_id = Some(id);
            self.index_max_level = node_lvl;
            self.id_to_index.insert(id, self.nodes.len());
            self.nodes.push(new_node);
            return;
        }

        self.id_to_index.insert(id, self.nodes.len());
        self.nodes.push(new_node);

        // 伪代码 line 2-3：取全图入口点，作为逐层下降的起点
        let mut entry_point_id = self
            .get_entry_node_id()
            .expect("in insert, entry_node_id is none");

        // 伪代码 line 5-7：从顶层贪心下降到 node_lvl+1，只找最近邻更新入口点，不建边
        if self.index_max_level >= node_lvl + 1 {
            let t = Instant::now();
            for current_lvl in (node_lvl + 1..=self.index_max_level).rev() {
                entry_point_id = self.greedy_search_at_level(&data, entry_point_id, current_lvl);
            }
            self.timing.greedy_search_ns += t.elapsed().as_nanos();
        }

        // 伪代码 line 8：从 min(全图最高层, 新节点层) 开始往下，逐层建边
        let top_connect_lvl = std::cmp::min(self.index_max_level, node_lvl);
        for current_lvl in (0..=top_connect_lvl).rev() {
            // 伪代码 line 9：在当前层以 efConstruction 为候选池大小做 beam search
            let t = Instant::now();
            let (search_results, sl_stats) = self.search_layer(
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

            // 伪代码 line 10：从搜索结果中选出最近的若干点作为新节点的邻居
            let t = Instant::now();
            let selected_neighbors =
                &self.select_neighbors_simple(
                    &search_results,
                    &data,
                    max_neighbors_to_select,
                    &mut id_similarity_cache,
                );
            self.timing.select_neighbors_ns += t.elapsed().as_nanos();

            // 伪代码 line 11：为新节点与每个选中邻居建双向边
            let t = Instant::now();
            for neighbor_id in selected_neighbors {
                self.add_neighbor_to_node_at_level(id, *neighbor_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_id, id, current_lvl);
            }
            self.timing.link_ns += t.elapsed().as_nanos();

            // 伪代码 line 12-16：检查每个新邻居的出度，超过上限则裁剪
            let t = Instant::now();
            for neighbor_id in selected_neighbors {
                // 伪代码 line 13：读出该邻居在当前层的所有连接（含刚加入的新边）
                let neighbor_current_connections: HashSet<u64> = {
                    let neighbor_node = self
                        .get_node_by_id(*neighbor_id)
                        .expect("in insert, get_node_by_id cannot handle neighbor_id");
                    neighbor_node.get_neighbors()[current_lvl]
                        .iter()
                        .copied()
                        .collect()
                };
                // 伪代码 line 14：若出度超过上限，触发裁剪
                if neighbor_current_connections.len() > max_neighbors_to_keep {
                    // 伪代码 line 15：用 SELECT-NEIGHBORS 从现有连接里保留最优的若干条
                    let neighbor_pruned_connections: Vec<u64> = {
                        let neighbor_node = self
                            .get_node_by_id(*neighbor_id)
                            .expect("in insert, get_node_by_id cannot handle neighbor_id");
                        let mut neighbor_similarity_cache: HashMap<Id, f32> = HashMap::new();
                        self.select_neighbors_simple(
                            &neighbor_current_connections,
                            neighbor_node.get_data(),
                            max_neighbors_to_keep,
                            &mut neighbor_similarity_cache,
                        )
                    };
                    // 伪代码 line 16：将该邻居的连接表替换为裁剪结果，同步删除被裁掉的反向边
                    //
                    // 这里邻接表的规模受 max_neighbors_to_keep 约束，通常很小，所以直接用 Vec::contains
                    // 做小规模线性比对，往往比把旧/新连接都转成 HashSet 再做差集更划算：
                    // 少分配、少哈希，也更 cache-friendly。
                    let connections_to_remove: Vec<Id> = {
                        let neighbor_node = self
                            .get_node_by_id(*neighbor_id)
                            .expect("in insert, get_node_by_id cannot handle neighbor_id");
                        neighbor_node.neighbors[current_lvl]
                            .iter()
                            .copied()
                            .filter(|connection_id| {
                                !neighbor_pruned_connections.contains(connection_id)        
                            })
                            .collect()
                    };
                    let neighbor_node = self
                        .get_mut_node_by_id(*neighbor_id)
                        .expect("in insert, get_node_by_id cannot handle neighbor_id");
                    neighbor_node.neighbors[current_lvl] = neighbor_pruned_connections;
                    for removed_neighbor_id in connections_to_remove {
                        self.delete_neighbor_relationship_at_level(
                            *neighbor_id,
                            removed_neighbor_id,
                            current_lvl,
                        );
                    }
                }
            }
            self.timing.pruning_ns += t.elapsed().as_nanos();

            // 伪代码 line 17：将本层结果中离新节点最近的点设为下一层的入口
            entry_point_id = self
                .get_nearest(&mut id_similarity_cache, &search_results, &data)
                .expect("searchh_result_id赋值错误");
        }

        // 伪代码 line 18-19：若新节点层高超过全图最高层，更新全图入口点
        if node_lvl > self.index_max_level {
            self.entry_node_id = Some(id);
            self.index_max_level = node_lvl;
        }
    }

    // 从候选点里选出前若干个最相近的邻居。
    pub fn select_neighbors_simple(
        &self,
        candidate_set: &HashSet<Id>,
        query: &Vector,
        max_neighbors_to_select: usize,
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
        let take_count = max_neighbors_to_select.min(id_score_map.len());
        id_score_map
            .iter()
            .take(take_count)
            .map(|(id, _)| *id)
            .collect()
    }
}
