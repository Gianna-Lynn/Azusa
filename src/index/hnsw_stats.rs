#[derive(Debug, Default)]
pub struct SearchLayerStats {
    pub compute_sim_ns: u128,
    pub furthest_scan_ns: u128,
    pub neighbors_fetch_ns: u128,
    pub heap_pop_ns: u128,
    pub candidate_bookkeeping_ns: u128,
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
    pub search_layer_compute_sim_ns: u128,
    pub search_layer_furthest_scan_ns: u128,
    pub search_layer_neighbors_fetch_ns: u128,
    pub search_layer_heap_pop_ns: u128,
    pub search_layer_bookkeeping_ns: u128,
    pub search_layer_loop_iterations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapAccessMode {
    PeekPruned,
    PopAlive,
}
