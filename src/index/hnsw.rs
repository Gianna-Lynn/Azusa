use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::types::{Id, Metric};

#[path = "hnsw_impl.rs"]
mod hnsw_impl;
#[path = "hnsw_legacy_experiment.rs"]
mod hnsw_legacy_experiment;
#[path = "hnsw_node.rs"]
mod hnsw_node;
#[path = "hnsw_stats.rs"]
mod hnsw_stats;

pub use hnsw_node::HnswNode;
pub use hnsw_stats::{HeapAccessMode, InsertTimings, SearchLayerStats};

const MY_SEED: u64 = 42;

#[derive(Debug)]
pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    entry_node_id: Option<Id>,
    index_max_level: usize,
    metric: Metric,
    id_to_index: HashMap<Id, usize>,
    rng: StdRng,
    timing: InsertTimings,
}
