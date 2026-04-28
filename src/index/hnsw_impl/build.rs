//! HNSW 索引的插入与建图逻辑。
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
    // 采样层数的函数, 满足 Probability(新节点层数level>=L) = level_sample_probability^L
    pub fn sample_level(&mut self) -> usize {
        let mut level = 0usize;
        while self.rng.gen_bool(self.level_sample_probability) {
            level += 1;
        }
        level
    }

    // 入口函数1: (id, vector)作为入口, metadata先用默认值占位.这是一个简化的调用.
    pub fn insert(
        &mut self,
        id: Id,
        vector: Vector,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        self.insert_record(VectorRecord {
            id,
            vector: vector.clone(),
            metadata: Metadata::default(),
        }, ef_construction, max_neighbors_to_select, max_neighbors_to_keep);
    }

    // 入口函数2: VectorRecord(id, vector, metadata)的入口, 用户要同时提供 metadata.
    pub fn insert_record(
        &mut self,
        record: VectorRecord,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        let id = record.id;
        let vector = record.vector.clone();
        // 用断言保证了每条记录的id都是唯一的.
        assert!(
            !self.id_to_index.contains_key(&id),
            "duplicate id is not allowed: {}",
            id
        );
        // 保存记录
        self.store_record(record);
        // 把点接入图
        self.insert_node_into_graph(
            id,
            vector,
            ef_construction,
            max_neighbors_to_select,
            max_neighbors_to_keep,
        );
    }

    // self.records是按Id保存的原始记录, 这里提供一个store_record方法来统一管理记录的存储.
    fn store_record(&mut self, record: VectorRecord) {
        self.records.insert(record.id, record);
    }

    // 入口函数3: 批量插入, 只是循环调用单条插入的逻辑
    pub fn batch_insert(
        &mut self,
            records: Vec<VectorRecord>,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        for record in records {
            self.insert_record(
                record,
                ef_construction,
                max_neighbors_to_select,
                max_neighbors_to_keep,
            );
        }
    }

    // 真正负责把点接进 HNSW 图
    fn insert_node_into_graph(
        &mut self,
        id: Id,
        vector: Vector,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        // 1) 准备新节点：记录计数、采样层高、构造节点对象。
        self.timing.calls += 1;
        let node_lvl = self.sample_level();
        let data_copy = vector.clone();
        let new_node = HnswNode::new(id, data_copy, node_lvl);
        let new_index = self.nodes.len();

        // 2) 空图特判：第一个点直接成为入口点，不需要走后续搜索和连边流程。
        if self.is_empty() {
            self.entry_node_index = Some(new_index);
            self.index_max_level = node_lvl;
            self.id_to_index.insert(id, new_index);
            self.nodes.push(new_node);
            return;
        }

        // 3) 先把新节点挂进索引表和节点数组，后续搜索和建边会用到它的 index。
        self.id_to_index.insert(id, new_index);
        self.nodes.push(new_node);

        let mut entry_point_index = self
            .entry_node_index
            .expect("in insert, entry_node_index is none");

        // 4) 如果新节点层高更低，先在更高层做贪心下探，找到合适的入口点。
        if self.index_max_level > node_lvl {
            let t = Instant::now();
            for current_lvl in (node_lvl + 1..=self.index_max_level).rev() {
                entry_point_index =
                    self.greedy_search_at_level(&vector, entry_point_index, current_lvl);
            }
            self.timing.greedy_search_ns += t.elapsed().as_nanos();
        }

        // 5) 从新节点实际参与连接的最高层到 0 层逐层执行：
        //    search_layer -> 选邻居 -> 建双向边 -> 必要时剪枝旧邻居。
        let top_connect_lvl = std::cmp::min(self.index_max_level, node_lvl);
        for current_lvl in (0..=top_connect_lvl).rev() {
            // 5.1 先在当前层做局部搜索，拿到候选点和搜索统计信息。
            let t = Instant::now();
            let (search_results, sl_stats) =
                self.search_layer(&vector, entry_point_index, current_lvl, ef_construction);

            // 统计 search_layer 的耗时，并把这一轮的细分统计累加到总表里。
            self.timing
                .add_search_layer_stats(t.elapsed().as_nanos(), &sl_stats);

            // 5.2 基于搜索结果挑选本层最终邻居。
            let t = Instant::now();
            let selected_neighbors = self.select_neighbors_heuristic(
                &search_results,
                &vector,
                max_neighbors_to_select,
                current_lvl,
                true,
                true,
            );
            self.timing.select_neighbors_ns += t.elapsed().as_nanos();

            // 5.3 把新节点和邻居互相连接起来。
            let t = Instant::now();
            for &neighbor_index in &selected_neighbors {
                self.add_neighbor_to_node_at_level_by_index(new_index, neighbor_index, current_lvl);
                self.add_neighbor_to_node_at_level_by_index(neighbor_index, new_index, current_lvl);
            }
            self.timing.link_ns += t.elapsed().as_nanos();

            // 5.4 如果旧邻居太多，就对它们做一次剪枝，保持图的稀疏性和质量。
            let t = Instant::now();
            for &neighbor_index in &selected_neighbors {
                let neighbor_current_connections: Vec<usize> = {
                    let neighbor_node = self
                        .get_node_by_index(neighbor_index)
                        .expect("in insert, get_node_by_index cannot handle neighbor_index");
                    neighbor_node.get_neighbors()[current_lvl].clone()
                };
                // 如果邻居的当前连接数超过了max_neighbors_to_keep, 就对它的连接进行剪枝.
                if neighbor_current_connections.len() > max_neighbors_to_keep {
                    let neighbor_pruned_connections: Vec<usize> = {
                        let neighbor_data = self.nodes[neighbor_index].get_data().clone();
                        let scored_candidates = self.score_and_sort_level_candidates(
                            &neighbor_current_connections,
                            &neighbor_data,
                            current_lvl,
                        );
                        // 这里是在做“旧邻居剪枝”：
                        // 输入的 `scored_candidates` 本来就是这个旧节点当前已经拥有的邻居，
                        // 现在只是想从这些“已存在的边”里重新挑出更值得保留的那一部分。
                        //
                        // 所以这里不能再把二阶邻居扩进来：
                        // - 如果扩展，筛选范围就不再局限于“当前连接”，会变成重新做一次更大范围的选边；
                        // - 剪枝阶段的目标是收缩和整理现有连接，而不是扩大候选集。
                        //
                        // 和上面“新节点接入图”时不同：
                        // - 新节点那一处的候选来自 `search_layer`，只是种子；
                        // - 还会把这些种子的邻居一起并入候选集，帮助新节点找到更好的连接；
                        // - 所以那一处会开启 `extendCandidates = true`，而这里必须是 `false`。
                        self.select_neighbors_heuristic(
                            &scored_candidates,
                            &neighbor_data,
                            max_neighbors_to_keep,
                            current_lvl,
                            false,
                            true,
                        )
                    };

                    // 求解出要删除的连接（即那些没有被剪枝保留的连接）
                    let connections_to_remove: Vec<usize> = {

                        let neighbor_node = self
                            .get_node_by_index(neighbor_index)
                            .expect("in insert, get_node_by_index cannot handle neighbor_index");
                        
                        neighbor_node.neighbors[current_lvl]
                            .iter()
                            .copied()
                            .filter(|connection_index| {
                                !neighbor_pruned_connections.contains(connection_index)  // filter的条件是 "这个连接不在被保留的连接列表里", 即这个连接需要被删除
                            })
                            .collect()
                    };

                    // 替换邻接表, 并从neighbor和被删除连接双方同时删除这些边。
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

            // 5.5 这一层的搜索结果里，最好的点会作为下一层的入口点。
            entry_point_index = search_results
                .first()
                .map(|(index, _)| *index)
                .expect("search_result_index赋值错误");
        }

        // 6) 如果新节点比原有最高层还高，它会成为新的入口点。
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


    // 这是启发式邻居选择算法, 输入必须是按 similarity 排好序的 scored candidates: &[(usize, f32)].
    //
    // 之所以这里不直接收“裸 index 集合”, 是因为 heuristic 主体本身只关心候选的相似度顺序,
    // 不负责重新计算分数或排序。调用方要先把候选整理成“带分数、且已排序”的形状。
    //
    // 在建图里它会被用在两个阶段:
    // 1. 新节点接入图时:
    //    - search_layer 已经直接返回 scored candidates, 可以原样喂进来;
    //    - extend_candidates = true, 会把这些候选的一阶邻居也并入工作堆;
    //    - 目标是从“搜索结果 + 周边局部结构”里挑出更好的新边。
    // 2. 旧节点反向边 pruning 时:
    //    - 先把当前邻接表重新打分并排序成 scored candidates;
    //    - extend_candidates = false, 只在当前已有连接里重新筛边;
    //    - 目标是收缩和整理旧边，而不是扩大候选集。
    //
    // 这样下游就只需要维护一套 heuristic 主体，不必同时兼容“有序 scored 输入”和“无序 index 集合”。
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
            .reset_neighbor_selection_workspace(self.nodes.len());
        // 先把“已经带分数”的候选装入工作堆。
        // 这一步不再计算 similarity，只负责把候选交给 heuristic 主体做筛选。
        self.fill_work_queue_from_scored(scored_candidates, level);
        if extend_candidates {
            // extendCandidates = true 时，把当前候选在这一层的一阶邻居也并入工作堆。
            // 这相当于“先从种子出发，再把周边邻居一起纳入候选池”，能让新节点更容易连到更好的边。
            // membership marks 用来避免在堆里做线性查重。
            self.extend_work_queue_from_scored(scored_candidates, query, level);
        }
        // 从工作堆里执行 heuristic 选择:
        // - 先按 diversity 规则筛掉“太像已有结果”的点;
        // - 如果 keep_pruned_connections = true, 再从 discard 里补回一些高分候选，尽量凑满 M。
        self.take_selected_neighbors(max_neighbors_to_select, keep_pruned_connections)
    }

    // 把某一层的无序候选 index 集合，转换成 heuristic 下游统一使用的“带分数且有序”序列。
    // 这里会对每个候选重新计算 similarity，再按分数降序、同分按 index 升序排序。
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

    // scored_candidates 已经带有分数，所以这里只负责把它们装入工作堆，
    // 不再重新计算 similarity。
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
        // 对每个候选，取出它在当前层的一阶邻居：
        // 如果这个邻居还没进工作堆，就计算它的相似度并补进去。
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
        // 这是 heuristic 的核心筛选阶段:
        // 1. 从 work_heap 里不断取出当前最“近”的候选;
        // 2. 如果它和已选结果太相似，就丢进 discard_heap;
        // 3. 如果它通过了 diversity 检查，就保留下来。
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
            if flag {
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
            // 对应论文里的 Wd 补位阶段:
            // 如果 diversity 规则把结果筛得太少，就从 discard_heap 里按顺序补回一些之前丢掉的高分候选，
            // 这样可以避免最终邻居数不足。
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
