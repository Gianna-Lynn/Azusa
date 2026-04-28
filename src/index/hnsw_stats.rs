//! HNSW 构图和搜索阶段的耗时统计结构。
#[derive(Debug, Default)]
pub struct SearchLayerStats {
    pub loop_iterations: u64,
}

#[derive(Debug, Default)]
pub struct InsertTimings {
    pub greedy_search_ns: u128,
    pub search_layer_ns: u128,
    pub select_neighbors_ns: u128,
    pub pruning_ns: u128,
    pub link_ns: u128,
    pub calls: u64,
    pub search_layer_loop_iterations: u64,
}

impl InsertTimings {
    pub fn add_search_layer_stats(&mut self, elapsed_ns: u128, stats: &SearchLayerStats) {
        self.search_layer_ns += elapsed_ns;
        self.search_layer_loop_iterations += stats.loop_iterations;
    }
}
