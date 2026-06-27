//! HNSW 构图和搜索阶段的耗时统计结构。

// SearchLayerStats 结构体记录了 HNSW 搜索过程中每一层的循环迭代次数等统计数据，供 InsertTimings 汇总使用。
#[derive(Debug, Default)]
pub struct SearchLayerStats {
    pub loop_iterations: u64,
}

// InsertTimings 结构体记录了 HNSW 插入过程中各个阶段的耗时统计数据。
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
    /// 累加一次层搜索的耗时和循环次数统计。
    pub fn add_search_layer_stats(&mut self, elapsed_ns: u128, stats: &SearchLayerStats) {
        self.search_layer_ns += elapsed_ns;
        self.search_layer_loop_iterations += stats.loop_iterations;
    }
}
