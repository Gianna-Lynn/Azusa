//! HNSW 索引的插入与建图逻辑。
//!
//! 本模块覆盖建图路径：单条/批量插入、层级采样、邻居选择
//! （简单版与启发式版）以及旧节点剪枝。原始记录与图结构分开保存：
//! `records` 存记录本体，`nodes` 存各层的邻接关系。
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
    /// 随机采样新节点的最高层级。
    ///
    /// HNSW 中层级越高节点越少，层级越低节点越密集；
    /// 采样结果决定了新节点会参与哪些层的建图。
    pub fn sample_level(&mut self) -> usize {
        let mut level = 0usize;
        while self.rng.gen_bool(self.level_sample_probability) {
            level += 1;
        }
        level
    }

    /// 插入一条只包含 `id` 和 `vector` 的记录。
    ///
    /// 简化入口：`metadata` 用默认值占位，其余行为与
    /// [`insert_record`](Self::insert_record) 一致。
    ///
    /// # Arguments
    ///
    /// - `id`: 记录 ID，必须尚未被插入过。
    /// - `vector`: 待索引的向量。
    /// - `ef_construction` / `max_neighbors_to_select` / `max_neighbors_to_keep`:
    ///   建图参数，含义见 [`insert_record`](Self::insert_record)。
    ///
    /// # Panics
    ///
    /// 若 `id` 已存在，则断言失败并 panic。
    pub fn insert(
        &mut self,
        id: Id,
        vector: Vector,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        self.insert_record(
            VectorRecord {
                id,
                vector: vector.clone(),
                metadata: Metadata::default(),
            },
            ef_construction,
            max_neighbors_to_select,
            max_neighbors_to_keep,
        );
    }

    /// 插入一条完整的 [`VectorRecord`]。
    ///
    /// 适合上层已经拿到结构化记录的场景。流程固定为三步：
    /// ID 唯一性断言 -> 保存记录本体 -> 把节点接入图结构。
    ///
    /// # Arguments
    ///
    /// - `record`: 待插入的完整记录，`record.id` 必须尚未被插入过。
    /// - `ef_construction`: 建图阶段每层局部搜索的候选队列宽度；
    ///   越大建图质量通常越高，但耗时也更高。
    /// - `max_neighbors_to_select`: 新节点每层可选的邻居数量。
    /// - `max_neighbors_to_keep`: 旧节点每层允许保留的最大边数（度数上限）。
    ///
    /// # Panics
    ///
    /// 若 `record.id` 已存在，则断言失败并 panic。
    pub fn insert_record(
        &mut self,
        record: VectorRecord,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        let id = record.id;
        let vector = record.vector.clone();
        // 1. ID 唯一性断言：同一 ID 不允许重复插入。
        assert!(
            !self.id_to_index.contains_key(&id),
            "duplicate id is not allowed: {}",
            id
        );
        // 2. 保存记录本体。
        self.store_record(record);
        // 3. 把节点接入图结构。
        self.insert_node_into_graph(
            id,
            vector,
            ef_construction,
            max_neighbors_to_select,
            max_neighbors_to_keep,
        );
    }

    /// 保存原始记录本体。
    ///
    /// `records` 负责保存原始记录本体，图结构连接保存在
    /// `nodes` 的邻接表里。单独抽一个方法，便于以后统一调整记录存储策略。
    fn store_record(&mut self, record: VectorRecord) {
        self.records.insert(record.id, record);
    }

    /// 批量插入多条向量记录。
    ///
    /// 本质上只是把单条插入循环执行一遍；
    /// 参数含义与 [`insert_record`](Self::insert_record) 一致。
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

    /// 将新节点接入 HNSW 图结构。
    ///
    /// 流程可粗略分成四件事：
    ///
    /// 1. 给新节点采样层高；
    /// 2. 从图的入口点出发，找到适合新节点的局部入口；
    /// 3. 在每一层做局部搜索并挑选邻居；
    /// 4. 双向连边后，对超出度数上限的旧邻居做剪枝。
    ///
    /// # Arguments
    ///
    /// - `id`: 新节点的 ID，调用前必须尚未注册。
    /// - `vector`: 新节点的向量。
    /// - `ef_construction`: 建图阶段每层局部搜索的候选队列宽度。
    /// - `max_neighbors_to_select`: 新节点每层可选的邻居数量。
    /// - `max_neighbors_to_keep`: 旧节点每层允许保留的最大边数（度数上限）。
    fn insert_node_into_graph(
        &mut self,
        id: Id,
        vector: Vector,
        ef_construction: usize,
        max_neighbors_to_select: usize,
        max_neighbors_to_keep: usize,
    ) {
        // 1. 准备新节点：记录这次插入、采样层高、构造节点对象。
        self.timing.calls += 1;
        let node_lvl = self.sample_level();
        let data_copy = vector.clone();
        let new_node = HnswNode::new(id, data_copy, node_lvl);
        let new_index = self.nodes.len();

        // 2. 空图特判：第一个点直接成为入口点，也是后续所有查询与插入的起点，
        // 不需要走后面的搜索和连边流程。
        if self.is_empty() {
            self.entry_node_index = Some(new_index);
            self.index_max_level = node_lvl;
            self.id_to_index.insert(id, new_index);
            self.nodes.push(new_node);
            return;
        }

        // 3. 把新节点挂进索引表和节点数组。
        // 先挂载再搜索，是因为后续连边和剪枝都要能从 `index` 直接访问这个节点。
        self.id_to_index.insert(id, new_index);
        self.nodes.push(new_node);

        let mut entry_point_index = self
            .entry_node_index
            .expect("in insert, entry_node_index is none");

        // 4. 如果新节点层高更低，先在更高层做贪心下探，找到更接近新节点的入口点，
        // 避免后续低层 `search_layer` 的起点离目标太远。
        if self.index_max_level > node_lvl {
            let t = Instant::now();
            for current_lvl in (node_lvl + 1..=self.index_max_level).rev() {
                entry_point_index =
                    self.greedy_search_at_level(&vector, entry_point_index, current_lvl);
            }
            self.timing.greedy_search_ns += t.elapsed().as_nanos();
        }

        // 5. 从新节点实际参与连接的最高层到 0 层，逐层执行：
        //    局部搜索 -> 选邻居 -> 建双向边 -> 必要时剪枝旧邻居。
        //
        // 循环内“先找候选，再筛边，再连边，再剪旧边”的顺序是固定的：
        // 既要保证新节点连到合适的位置，也要控制旧节点的度数不无限膨胀。
        let top_connect_lvl = std::cmp::min(self.index_max_level, node_lvl);
        for current_lvl in (0..=top_connect_lvl).rev() {
            // 5.1 在当前层做局部搜索，拿到候选点和搜索统计信息。
            // `ef_construction` 越大，这一步看得越宽，建图质量通常越高，但耗时也更高。
            let t = Instant::now();
            let (search_results, sl_stats) =
                self.search_layer(&vector, entry_point_index, current_lvl, ef_construction);

            // 把本轮 `search_layer` 的细分统计累加到总表里。
            self.timing
                .add_search_layer_stats(t.elapsed().as_nanos(), &sl_stats);

            // 5.2 基于搜索结果挑选本层最终邻居。
            // 这里走 heuristic selection：先取 `search_layer` 的结果，
            // 再按“不要太像已有结果”的规则过滤，
            // 最终得到一组更分散、更有信息量的邻居。
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
            // HNSW 的边是双向的：新节点连出去的同时，邻居也要回指新节点。
            let t = Instant::now();
            for &neighbor_index in &selected_neighbors {
                self.add_neighbor_to_node_at_level_by_index(new_index, neighbor_index, current_lvl);
                self.add_neighbor_to_node_at_level_by_index(neighbor_index, new_index, current_lvl);
            }
            self.timing.link_ns += t.elapsed().as_nanos();

            // 5.4 如果旧邻居的边数超过上限，就对它做一次剪枝，保持图的稀疏性和质量。
            // 这一步不是为了“给新点找更多边”，而是为了控制旧点的度数上限。
            let t = Instant::now();
            for &neighbor_index in &selected_neighbors {
                let neighbor_current_connections: Vec<usize> = {
                    let neighbor_node = self
                        .get_node_by_index(neighbor_index)
                        .expect("in insert, get_node_by_index cannot handle neighbor_index");
                    neighbor_node.get_neighbors()[current_lvl].clone()
                };
                // 只有当邻居当前层的边数超过 `max_neighbors_to_keep` 时才剪枝。
                if neighbor_current_connections.len() > max_neighbors_to_keep {
                    let neighbor_pruned_connections: Vec<usize> = {
                        let neighbor_data = self.nodes[neighbor_index].get_data().clone();
                        let scored_candidates = self.score_and_sort_level_candidates(
                            &neighbor_current_connections,
                            &neighbor_data,
                            current_lvl,
                        );
                        // 这是“旧邻居剪枝”：输入的 `scored_candidates` 是当前已连上的边，
                        // 不是新搜索出来的更大候选池，目标是“从现有连接里挑出更值得保留的部分”。
                        //
                        // 因此这里必须把 `extend_candidates` 关掉：
                        // - 新节点接入图时，`search_layer` 的结果只是种子，还会额外扩展周边一阶邻居；
                        // - 旧节点剪枝时，范围必须严格限制在“当前已有连接”内；
                        // - 否则这一步就不再是剪枝，而是重新做一次更大范围的选边。
                        self.select_neighbors_heuristic(
                            &scored_candidates,
                            &neighbor_data,
                            max_neighbors_to_keep,
                            current_lvl,
                            false,
                            true,
                        )
                    };

                    // 求解要删除的连接：当前邻接表里没有被剪枝保留的旧边。
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

                    // 先把邻接表替换成剪枝后的结果，再把被删掉的边从对端同步移除，
                    // 保证边始终是双向对称的。
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

            // 5.5 这一层搜索结果里最好的点作为下一层的入口点，
            // 相当于“沿着新点附近的局部最优方向继续往下走”。
            entry_point_index = search_results
                .first()
                .map(|(index, _)| *index)
                .expect("in insert, search_results should not be empty");
        }

        // 6. 如果新节点比原有最高层还高，它会成为新的入口点。
        if node_lvl > self.index_max_level {
            self.entry_node_index = Some(new_index);
            self.index_max_level = node_lvl;
        }
    }

    /// 按相似度直接选择前 M 个邻居。
    ///
    /// 简单选择：把候选逐个打分、按分数排序、截断为前 `max_neighbors_to_select` 个。
    /// 没有 heuristic 的多样性过滤，因此更容易把彼此过于相近的点都选进来。
    ///
    /// 仅 legacy 路径仍在使用；主线建图已经切到
    /// [`select_neighbors_heuristic`](Self::select_neighbors_heuristic)。
    ///
    /// # Arguments
    ///
    /// - `candidate_set`: 候选节点 index 集合。
    /// - `query`: 用于打分的向量。
    /// - `max_neighbors_to_select`: 最多保留的邻居数。
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

    /// 使用给定节点切片执行简单邻居选择。
    ///
    /// [`select_neighbors_simple`](Self::select_neighbors_simple) 的静态版本：
    /// 打分、排序、截断都在调用方传入的 `nodes` 与 `workspace` 上进行。
    /// 排序规则为相似度降序，同分按 ID 升序，保证结果稳定。
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

    /// 使用启发式规则选择邻居。
    ///
    /// 输入必须是“已经按相似度排序好的候选”。与
    /// [`select_neighbors_simple`](Self::select_neighbors_simple) 不同，
    /// 它不直接取最相似的前 M 个，而是尽量选出“相似度高、但彼此之间不要太像”的一组点，
    /// 让图在局部区域覆盖得更均匀。
    ///
    /// # Arguments
    ///
    /// - `scored_candidates`: 已按相似度降序排列的候选 `(index, score)`。
    /// - `query`: 目标向量。
    /// - `max_neighbors_to_select`: 期望选出的邻居数量。
    /// - `level`: 目标层。
    /// - `extend_candidates`: 是否把种子候选在当前层的一阶邻居也并入候选池。
    ///   插入新节点时为 `true`；对旧节点剪枝时必须为 `false`，
    ///   否则剪枝会退化成一次更大范围的重新选边。
    /// - `keep_pruned_connections`: 若为 `true`，多样性过滤筛掉太多候选时，
    ///   会从 `discard_heap` 补回高分候选；若为 `false`，结果不足就不补位。
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
        // 1. 把已经打过分、且已排好序的候选装入 `work_heap`。
        // 这里不再重复算分，只负责把“种子候选”交给 heuristic 主体。
        self.fill_work_heap_from_scored(scored_candidates, level);
        if extend_candidates {
            // 2. 把这些种子在当前层的“一阶邻居”也并进候选池，
            // 让新节点能看到更完整的局部结构，而不是只盯着 `search_layer` 的种子结果。
            // `membership marks` 用来避免在 `work_heap` 里重复塞同一个 index。
            self.extend_work_heap_from_scored(scored_candidates, query, level);
        }
        // 3. 从 `work_heap` 里执行真正的 heuristic 筛选：
        // 先做多样性过滤，再根据需要从 `discard_heap` 补回一些高分候选。
        self.take_selected_neighbors(max_neighbors_to_select, keep_pruned_connections)
    }

    /// 给候选节点重新打分并排序。
    ///
    /// 把一层里“无序的候选 index 集合”重新打分，整理成按分数排序的序列。
    /// 排序规则：相似度降序，同分按 index 升序。
    ///
    /// 主要服务于旧节点剪枝：剪枝时只看当前已有的连接，不额外扩展候选池。
    fn score_and_sort_level_candidates(
        &mut self,
        candidate_indices: &[usize],
        query: &Vector,
        level: usize,
    ) -> Vec<(usize, f32)> {
        // 逐个计算候选的相似度，最后统一排序。
        self.workspace.reset_similarity_cache(self.nodes.len());

        let mut scored_candidates = Vec::with_capacity(candidate_indices.len());
        for candidate_index in candidate_indices.iter().copied() {
            // 防御性检查：候选在当前层不存在就跳过，
            // 避免 helper 被未来其他路径误喂入不在该层的点。
            if self.nodes[candidate_index].get_neighbors().len() <= level {
                continue;
            }

            // 利用 `SearchLayerWorkspace` 的 `sim_marks` / `sim_values`
            // 缓存当前 epoch 内的相似度，避免重复计算。
            let candidate_similarity = HnswIndex::get_similarity_from_workspace(
                &self.nodes,
                self.metric,
                &mut self.workspace,
                candidate_index,
                query,
            );

            // 把候选 index 与相似度一起收集起来，统一排序。
            scored_candidates.push((candidate_index, candidate_similarity));
        }

        // 排序规则：相似度降序，同分按 index 升序。
        scored_candidates.sort_by(|a, b| match b.1.partial_cmp(&a.1).unwrap() {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });
        scored_candidates
    }

    /// 将已打分候选装入工作堆。
    ///
    /// `scored_candidates` 已经带有分数，所以这里只负责把它们装入
    /// `work_heap`，不再重新计算 similarity。
    fn fill_work_heap_from_scored(&mut self, scored_candidates: &[(usize, f32)], level: usize) {
        for &(candidate_index, candidate_similarity) in scored_candidates {
            // 防御性检查：这些候选应来自当前层的现有邻居，
            // 保留一层保险，避免以后复用这个 helper 时误把不在该层的点塞进工作堆。
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

    /// 将种子候选的一阶邻居扩展进工作堆。
    ///
    /// 对每个种子候选，取出它在当前层的一阶邻居；
    /// 如果某个邻居还没进入 `work_heap`，就计算它的相似度并补进去。
    /// 这一步的效果是把“局部图结构”也纳入候选池，
    /// 而不是只看初始 `search_layer` 的结果。
    fn extend_work_heap_from_scored(
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

    /// 从工作堆中取出最终邻居集合。
    ///
    /// heuristic 的核心筛选阶段：
    ///
    /// - `work_heap` 里放的是“当前还值得考虑”的候选；
    /// - `result_vec` 里放的是“已经选中的邻居”；
    /// - `discard_heap` 里放的是“分数高，但和已选结果太像，所以先放弃”的候选。
    ///
    /// 目标不是简单取最相似的前 M 个，而是让选出的邻居彼此更分散，
    /// 使图的局部连边结构更丰富，搜索时不容易陷入一团相似点里。
    ///
    /// # Arguments
    ///
    /// - `max_neighbors_to_select`: 期望选出的邻居数量上限。
    /// - `keep_pruned_connections`: 若为 `true`，结果不足时从 `discard_heap`
    ///   按分数补回候选（论文里的“从 Wd 补位”）；若为 `false`，不足就不补。
    fn take_selected_neighbors(
        &mut self,
        max_neighbors_to_select: usize,
        keep_pruned_connections: bool,
    ) -> Vec<usize> {
        // 最终选中的邻居列表（论文符号 R）。
        let mut result_vec: Vec<(usize, f32)> = vec![];
        while !self.workspace.work_heap.is_empty() && result_vec.len() < max_neighbors_to_select {
            let nearest_elem_in_work = self.workspace.work_heap.pop().unwrap();
            // `work_heap` 的堆顶是“当前最值得先考虑”的候选。
            let similarity_e_to_q = nearest_elem_in_work.0.into_inner();
            let e_index = nearest_elem_in_work.1 .0;
            let mut flag = true;

            // 多样性检查：如果候选 `e` 和已经选中的任意邻居 `r` 太像，
            // 就不把 `e` 直接塞进结果，避免结果里一堆彼此几乎相同的点。
            for item in result_vec.iter() {
                let r_index = item.0;
                let similarity_e_to_r = distance::metric_similarity(
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
                // 不能进入结果的候选先放进 `discard_heap`，
                // 如果最终凑不满 M，再从这里补回一些高分点。
                self.workspace.discard_heap.push((
                    ordered_float::NotNan::new(similarity_e_to_q).unwrap(),
                    Reverse(e_index),
                ));
            }
        }

        if keep_pruned_connections {
            // 如果多样性过滤把结果筛得太少，就从 `discard_heap` 按顺序补回
            // 之前丢掉的高分候选（论文里的“从 Wd 里补位”），尽量填满 M 个邻居。
            while !self.workspace.discard_heap.is_empty()
                && result_vec.len() < max_neighbors_to_select
            {
                // 循环条件保证 `discard_heap` 非空，这里的 `unwrap` 不会 panic。
                let discard_top = self.workspace.discard_heap.pop().unwrap();
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
