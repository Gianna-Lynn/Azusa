use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::types::{Id, Metric};

#[path = "hnsw_impl.rs"]
mod hnsw_impl;
#[path = "hnsw_legacy.rs"]
mod hnsw_legacy;
#[path = "hnsw_node.rs"]
mod hnsw_node;
#[path = "hnsw_stats.rs"]
mod hnsw_stats;

pub use hnsw_node::HnswNode;
pub use hnsw_stats::{HeapAccessMode, InsertTimings, SearchLayerStats};

const MY_SEED: u64 = 42;


// 新建一个struct包装起search_layer中用到的visited数组和堆.避免每次调用search_layer时这些内容都被清空(规模很大)
#[derive(Debug)]
pub struct SearchLayerWorkspace {
    pub visited_epoch: u32,
    pub visited_marks: Vec<u32>, //代替visited_bool, 内含epoch轮次数, 若visited_marks[i] == epoch 表示 第i个点最后一次被访问是在第 #epoch个轮次.
    pub candidate_alive: Vec<bool>,
    pub candidate_heap: std::collections::BinaryHeap<(ordered_float::NotNan<f32>, usize)> ,
    pub result_heap: std::collections::BinaryHeap<(std::cmp::Reverse<ordered_float::NotNan<f32>>, usize)>,

    pub sim_epoch: u32,
    pub sim_marks: Vec<u32>, //同样的设计, 用于search_layer
    pub sim_values: Vec<f32>, //缓存相似度计算结果, 避免重复计算
}

impl SearchLayerWorkspace{
    pub fn new() -> Self{
        let visited_epoch: u32 = 0;
        let visited_marks: Vec<u32> = Vec::new();
        let candidate_alive: Vec<bool> = Vec::new();
        let candidate_heap: std::collections::BinaryHeap<(ordered_float::NotNan<f32>, usize)> = std::collections::BinaryHeap::new();
        let result_heap: std::collections::BinaryHeap<(std::cmp::Reverse<ordered_float::NotNan<f32>>, usize)> = std::collections::BinaryHeap::new();
        let sim_epoch: u32 = 0;
        let sim_marks: Vec<u32> = Vec::new();
        let sim_values: Vec<f32> = Vec::new();


        Self { 
            visited_epoch,
            visited_marks,
            candidate_alive,
            candidate_heap, 
            result_heap,
            sim_epoch,
            sim_marks,
            sim_values,
        }
    }

    pub fn reset(&mut self, node_count : usize){
        self.visited_epoch += 1;

        // resize(new_len, value): 若不够长, 扩容到new_len, 并且扩容的部分初始化value. 
        // clear(): 将长度置零, 但不改变capacity. 因此在下一轮访问时如果直接访问下标就会越界.
        if self.visited_marks.len() < node_count{ //只有当节点数增加时才扩容, 避免每轮都resize
            self.visited_marks.resize(node_count, 0);
        }
        if self.candidate_alive.len() < node_count{
            self.candidate_alive.resize(node_count, false);
        }
        self.candidate_heap.clear();
        self.result_heap.clear();

        self.sim_epoch += 1;
        if self.sim_marks.len() < node_count{  
           
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
    
}


#[derive(Debug)]
pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    entry_node_index: Option<usize>,
    index_max_level: usize,
    metric: Metric,
    id_to_index: HashMap<Id, usize>,
    rng: StdRng,
    timing: InsertTimings,
    workspace: SearchLayerWorkspace,
}
