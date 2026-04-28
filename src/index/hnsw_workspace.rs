//! HNSW 搜索和邻居选择共用的可复用工作区。
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
    pub sim_marks: Vec<u32>,  //代替sim_cache, 内含epoch轮次数, 若sim_marks[i] == epoch 表示 sim_values[i]里缓存的相似度是有效的.
    pub sim_values: Vec<f32>, //缓存相似度计算结果, 避免重复计算

    // 以下工作区变量服务邻居选择函数select_neighbors_heuristic
    pub work_membership_epoch: u32,
    pub work_membership_marks: Vec<u32>,                        // 内含epoch轮次数, 若work_membership_marks[i] == work_membership_epoch表示 第i个点当前在W集合里.
    pub work_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>,   // W集合, 按相似度取当前最近的元素
    pub discard_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>,
}

impl SearchLayerWorkspace {
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

    pub fn reset_similarity_cache(&mut self, node_count: usize) {
        self.sim_epoch += 1;
        if self.sim_marks.len() < node_count {
            self.sim_marks.resize(node_count, 0);
            self.sim_values.resize(node_count, 0.0);
        }
    }

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

    // 在select_neighbors_heuristic里
    // work_contains: 判断index对应的点当前是否在W集合里. 通过比较work_membership_marks[index]和work_membership_epoch来实现.
    pub fn work_contains(&self, index: usize) -> bool {
        self.work_membership_marks[index] == self.work_membership_epoch
    }

    // 在select_neighbors_heuristic里
    // mark_work_present: 将index对应的点标记为当前在W集合里. 通过将work_membership_marks[index]设置为work_membership_epoch来实现.
    pub fn mark_work_present(&mut self, index: usize) {
        self.work_membership_marks[index] = self.work_membership_epoch;
    }
}
