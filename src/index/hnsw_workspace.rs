//! 这个模块提供 SearchLayerWorkspace，用于在 HNSW 的层搜索、贪心搜索和邻居选择过程中
//! 复用临时缓冲区，减少分配和整段清零开销。
//!
//! 当前主要服务三类工作：
//! - 图层搜索 search_layer：复用 visited_marks、candidate_heap、result_heap，
//!   并通过 get_similarity_from_workspace 使用 sim_marks/sim_values 缓存相似度。
//! - 贪心搜索 greedy_search_at_level，以及建图过程中的候选打分/扩展：
//!   复用 sim_marks/sim_values，避免同一轮里重复计算相似度。
//! - 启发式邻居选择 select_neighbors_heuristic 及其 helper：
//!   复用 work_membership_marks、work_heap、discard_heap，并在扩展候选时复用相似度缓存。
//!
//! 通过 epoch 标记数组 visited_marks、sim_marks、work_membership_marks，
//! 以及可复用的 BinaryHeap，该结构避免了每次调用时反复分配或清空大缓冲区。

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ordered_float::NotNan;

// 新建一个struct包装起search_layer中用到的visited数组和堆.避免每次调用search_layer时这些内容都被清空(规模很大)
#[derive(Debug)]
pub struct SearchLayerWorkspace {
    pub visited_epoch: u32,
    pub visited_marks: Vec<u32>, //代替visited_bool, 内含epoch轮次数, 若visited_marks[i] == epoch 表示 第i个点最后一次被访问是在第 #epoch个轮次.

    // Rust的BinaryHeap默认是大根堆, 会先比较similarity, similarity相同再比较index.
    // 如果这里直接用(NotNan<f32>, usize), 同分时会变成index大的候选先出堆.
    // candidate_heap要表达"similarity大优先, 同分index小优先", 所以把index包成Reverse<usize>.
    pub candidate_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>,

    // result_heap用于维护固定容量top-ef结果集, 最常用的操作是peek/pop当前最差结果.
    // 因此只把similarity包成Reverse<NotNan<f32>>, 让堆顶变成"similarity最小/最差"的元素.
    // 这里不把index包成Reverse<usize>: 维护阶段同分时会先pop掉index大的元素, 保留index小的元素;
    // search_layer结束后再连续pop并reverse成输出Vec, 最终就是similarity降序且同分index升序.
    pub result_heap: BinaryHeap<(Reverse<NotNan<f32>>, usize)>,

    // 以下工作区变量服务相似度计算缓存.
    pub sim_epoch: u32,
    pub sim_marks: Vec<u32>, //代替sim_cache, 内含epoch轮次数, 若sim_marks[i] == epoch 表示 sim_values[i]里缓存的相似度是有效的.
    pub sim_values: Vec<f32>, //缓存相似度计算结果, 避免重复计算

    // 以下工作区变量服务邻居选择函数select_neighbors_heuristic
    pub work_membership_epoch: u32,
    pub work_membership_marks: Vec<u32>, // 内含epoch轮次数, 若work_membership_marks[i] == work_membership_epoch表示 第i个点当前在W集合里.
    pub work_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>, // W集合, 按相似度取当前最近的元素
    pub discard_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>,
}

impl SearchLayerWorkspace {
    /// 创建一个空的可复用搜索工作区。
    pub fn new() -> Self {
        Self {
            visited_epoch: 0,
            visited_marks: Vec::new(),
            candidate_heap: BinaryHeap::new(),
            result_heap: BinaryHeap::new(),
            sim_epoch: 0,
            sim_marks: Vec::new(),
            sim_values: Vec::new(),

            work_membership_epoch: 0,
            work_membership_marks: Vec::new(),
            work_heap: BinaryHeap::new(),
            discard_heap: BinaryHeap::new(),
        }
    }

    /// 重置层搜索使用的访问标记和候选堆。
    pub fn reset_search_layer_workspace(&mut self, node_count: usize) {
        self.visited_epoch += 1;

        // resize(new_len, value): 若不够长, 扩容到new_len, 并且扩容的部分初始化value.
        // clear(): 将长度置零, 但不改变capacity. 因此在下一轮访问时如果直接访问下标就会越界.
        if self.visited_marks.len() < node_count {
            //只有当节点数增加时才扩容, 避免每轮都resize
            self.visited_marks.resize(node_count, 0);
        }
        self.candidate_heap.clear();
        self.result_heap.clear();
    }

    /// 重置相似度缓存的 epoch 和容量。
    pub fn reset_similarity_cache(&mut self, node_count: usize) {
        self.sim_epoch += 1;
        if self.sim_marks.len() < node_count {
            self.sim_marks.resize(node_count, 0);
            self.sim_values.resize(node_count, 0.0);
        }
    }

    /// 重置启发式邻居选择使用的工作区。
    pub fn reset_neighbor_selection_workspace(&mut self, node_count: usize) {
        self.work_membership_epoch += 1;
        if self.work_membership_marks.len() < node_count {
            self.work_membership_marks.resize(node_count, 0);
        }
        if self.work_heap.capacity() < node_count {
            // reserve(new_capacity): 如果capacity不足new_capacity, 则扩容到new_capacity;
            // 否则不变. 这里预先reserve到node_count, 避免每次插入都可能触发扩容.
            self.work_heap.reserve(node_count);
        }
        if self.discard_heap.capacity() < node_count {
            self.discard_heap.reserve(node_count);
        }
        self.work_heap.clear();
        self.discard_heap.clear();
    }

    /// 判断节点当前是否已经在工作集合中。
    ///
    /// 在select_neighbors_heuristic里
    /// work_contains: 判断index对应的点当前是否在W集合里. 通过比较work_membership_marks[index]和work_membership_epoch来实现.
    pub fn work_contains(&self, index: usize) -> bool {
        self.work_membership_marks[index] == self.work_membership_epoch
    }

    /// 标记节点已经进入当前工作集合。
    ///
    /// 在select_neighbors_heuristic里
    /// mark_work_present: 将index对应的点标记为当前在W集合里. 通过将work_membership_marks[index]设置为work_membership_epoch来实现.
    pub fn mark_work_present(&mut self, index: usize) {
        self.work_membership_marks[index] = self.work_membership_epoch;
    }
}
