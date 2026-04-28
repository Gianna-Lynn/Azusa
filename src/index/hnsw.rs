//! HNSW 索引的核心状态定义与文件拆分入口。
use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::types::{Id, Metric, VectorRecord};

#[path = "hnsw_impl.rs"]
mod hnsw_impl;
#[path = "hnsw_node.rs"]
mod hnsw_node;
#[path = "hnsw_stats.rs"]
mod hnsw_stats;
#[path = "hnsw_workspace.rs"]
mod hnsw_workspace;

pub use hnsw_node::HnswNode;
pub use hnsw_stats::{InsertTimings, SearchLayerStats};
pub use hnsw_workspace::SearchLayerWorkspace;

const MY_SEED: u64 = 42;

#[derive(Debug)]
pub struct HnswIndex {
    pub(crate) nodes: Vec<HnswNode>,    // nodes: 图节点,用index访问
    pub(crate) records: HashMap<Id, VectorRecord>, //records: 用id访问, 从id映射到原始记录, 用于查询VectorRecord的(id, vector, metadata)等信息
    pub(crate) entry_node_index: Option<usize>,
    pub(crate) index_max_level: usize,  // index_max_level: 当前图的最大层数, 从0开始计数
    metric: Metric,
    pub(crate) id_to_index: HashMap<Id, usize>, 
    rng: StdRng,
    level_sample_probability: f64,
    pub(crate) timing: InsertTimings,
    workspace: SearchLayerWorkspace, 
}
