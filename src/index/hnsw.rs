//! HNSW 索引的核心状态定义与文件拆分入口。
use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::types::{Id, Metric, VectorRecord};

// #[path = "..."] 用于覆盖默认的模块搜索路径，
// 明确告诉编译器该模块的源码文件位于哪个相对路径。常用于将一个大模块拆分
// 成多个文件而不使用子模块目录（mod foo; 会默认查找 foo.rs 或 foo/mod.rs）。
// 这里我们把实现拆到同一目录下的不同文件中并显式指定它们的路径。
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
    pub(crate) nodes: Vec<HnswNode>, // nodes: 图节点,用index访问
    pub(crate) records: HashMap<Id, VectorRecord>, //records: 用id访问, 从id映射到原始记录, 用于查询VectorRecord的(id, vector, metadata)等信息
    pub(crate) entry_node_index: Option<usize>,
    pub(crate) index_max_level: usize, // index_max_level: 当前图的最大层数, 从0开始计数
    metric: Metric,                    // metric: 度量类型, 指示向量之间相似度/距离的计算方式
    pub(crate) id_to_index: HashMap<Id, usize>, // id_to_index: 从外部Id到内部节点索引的映射
    rng: StdRng,                       // rng: 随机数生成器, 用于层级采样和随机化操作
    level_sample_probability: f64,     // level_sample_probability: 采样新节点层级时使用的概率参数
    pub(crate) timing: InsertTimings,  // timing: 插入操作的计时统计
    workspace: SearchLayerWorkspace,   // workspace: 搜索层的临时工作区，复用以减少分配
}
