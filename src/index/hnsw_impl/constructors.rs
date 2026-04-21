use std::collections::HashMap;

use rand::SeedableRng;

use crate::types::Metric;

use super::super::{HnswIndex, SearchLayerWorkspace};

impl HnswIndex {
    pub fn new() -> Self {
        Self::new_with_metric(Metric::Cosine)
    }

    pub fn new_with_metric(metric: Metric) -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_index: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(super::super::MY_SEED),
            timing: super::super::hnsw_stats::InsertTimings::default(),
            workspace: SearchLayerWorkspace::new(),
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
