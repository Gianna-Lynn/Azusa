use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ordered_float::NotNan;

// 新建一个struct包装起search_layer中用到的visited数组和堆.避免每次调用search_layer时这些内容都被清空(规模很大)
#[derive(Debug)]
pub struct SearchLayerWorkspace {
    pub visited_epoch: u32,
    pub visited_marks: Vec<u32>, //代替visited_bool, 内含epoch轮次数, 若visited_marks[i] == epoch 表示 第i个点最后一次被访问是在第 #epoch个轮次.
    pub candidate_alive: Vec<bool>,
    pub candidate_heap: BinaryHeap<(NotNan<f32>, usize)>,
    pub result_heap: BinaryHeap<(Reverse<NotNan<f32>>, usize)>,

    pub sim_epoch: u32,
    pub sim_marks: Vec<u32>,  //同样的设计, 用于search_layer
    pub sim_values: Vec<f32>, //缓存相似度计算结果, 避免重复计算

    pub select_neighbor_epoch: u32,
    pub work_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>,
    pub discard_heap: BinaryHeap<(NotNan<f32>, Reverse<usize>)>,
}

impl SearchLayerWorkspace {
    pub fn new() -> Self {
        Self {
            visited_epoch: 0,
            visited_marks: Vec::new(),
            candidate_alive: Vec::new(),
            candidate_heap: BinaryHeap::new(),
            result_heap: BinaryHeap::new(),
            sim_epoch: 0,
            sim_marks: Vec::new(),
            sim_values: Vec::new(),

            select_neighbor_epoch: 0,
            work_heap: BinaryHeap::new(),
            discard_heap: BinaryHeap::new(),
        }
    }

    pub fn reset(&mut self, node_count: usize) {
        self.visited_epoch += 1;

        // resize(new_len, value): 若不够长, 扩容到new_len, 并且扩容的部分初始化value.
        // clear(): 将长度置零, 但不改变capacity. 因此在下一轮访问时如果直接访问下标就会越界.
        if self.visited_marks.len() < node_count {
            //只有当节点数增加时才扩容, 避免每轮都resize
            self.visited_marks.resize(node_count, 0);
        }
        if self.candidate_alive.len() < node_count {
            self.candidate_alive.resize(node_count, false);
        }
        self.candidate_heap.clear();
        self.result_heap.clear();

        self.sim_epoch += 1;
        if self.sim_marks.len() < node_count {
            self.sim_marks.resize(node_count, 0);
            self.sim_values.resize(node_count, 0.0);
        }
    }

    pub fn reset_similarity_cache(&mut self, node_count: usize) {
        self.sim_epoch += 1;
        if self.sim_marks.len() < node_count {
            self.sim_marks.resize(node_count, 0);
            self.sim_values.resize(node_count, 0.0);
        }
    }

    pub fn reset_select_neighbor_workspace(&mut self, node_count: usize) {
        self.select_neighbor_epoch += 1;
        if self.work_heap.capacity() < node_count {
            self.work_heap.reserve(node_count);
        }
        if self.discard_heap.capacity() < node_count {
            self.discard_heap.reserve(node_count);
        }
        self.work_heap.clear();
        self.discard_heap.clear();
    }
}
