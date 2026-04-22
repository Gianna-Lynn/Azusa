use std::collections::HashMap;

use rand::SeedableRng;

use crate::types::Metric;

use super::super::{HnswIndex, SearchLayerWorkspace};

impl HnswIndex {
    pub fn new(metric: Metric, level_sample_probability: f64) -> Self {
        assert!(
            level_sample_probability > 0.0 && level_sample_probability < 1.0,
            "level_sample_probability must be in (0, 1)"
        );

        Self {
            nodes: Vec::new(),
            entry_node_index: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(super::super::MY_SEED),
            level_sample_probability,
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
