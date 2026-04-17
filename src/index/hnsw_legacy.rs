use std::collections::HashSet;

use crate::types::{Id, Vector};

use super::HnswIndex;

impl HnswIndex {
    // 基线: legacy / v1 helper.
    // 基线: 方法=线性扫描 HashSet
    // 基线: 相似度来源=通过 id_similarity_cache 复用/缓存 get_similarity 结果
    // 基线: 适用场景=已经有“节点到当前 query 的相似度缓存”时，找集合中最近点
    pub fn get_nearest_v1(&self, id_similarity_cache: &mut std::collections::HashMap<Id, f32>, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        let mut max_similarity = f32::NEG_INFINITY;
        let mut max_elem_id: Option<Id> = None;

        for elem_id in set {
            let temp_similarity = self.get_similarity(id_similarity_cache, elem_id, query);
            if temp_similarity > max_similarity {
                max_similarity = temp_similarity;
                max_elem_id = Some(*elem_id);
            }
        }
        max_elem_id
    }

    // 基线: legacy / v1 helper.
    // 基线: 方法=线性扫描 HashSet
    // 基线: 相似度来源=通过 id_similarity_cache 复用/缓存 get_similarity 结果
    // 基线: 适用场景=已经有“节点到当前 query 的相似度缓存”时，找集合中最远点
    pub fn get_furthest_v1(&self, id_similarity_cache: &mut std::collections::HashMap<Id, f32>, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        let mut min_similarity = f32::INFINITY;
        let mut min_elem_id: Option<Id> = None;

        for elem_id in set {
            let temp_similarity = self.get_similarity(id_similarity_cache, elem_id, query);
            if temp_similarity < min_similarity {
                min_similarity = temp_similarity;
                min_elem_id = Some(*elem_id);
            }
        }
        min_elem_id
    }

    // 基线: v0.
    // 基线: 沿着第 0 层做贪心搜索。
    pub fn search_v0(&self, query: &Vector) -> Option<Id> {
        if self.is_empty() {
            return None;
        }

        let mut current_id = self.entry_node_id?;
        loop {
            if let Some(node_to_search) = self.get_node_by_id(current_id) {
                let mut max_similarity = self.compute_similarity(node_to_search.get_data(), query);
                let mut max_idx: usize = 0;
                let mut found_better = false;
                for (idx, neighbor_id) in node_to_search.neighbors[0].iter().enumerate() {
                    if let Some(neighbor_node) = self.get_node_by_id(*neighbor_id) {
                        let temp_similarity = self.compute_similarity(neighbor_node.get_data(), query);
                        if temp_similarity > max_similarity {
                            max_similarity = temp_similarity;
                            max_idx = idx;
                            found_better = true;
                        }
                    }
                }

                if !found_better {
                    return Some(node_to_search.id);
                }
                current_id = node_to_search.neighbors[0][max_idx];
            } else {
                return None;
            }
        }
    }

    // 基线: v0.
    // 基线: 在线性扫描中找最近点。
    pub fn get_nearest_v0(&self, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        let mut max_similarity = f32::NEG_INFINITY;
        let mut max_elem_id: Option<Id> = None;

        for elem_id in set {
            let elem_node = self.get_node_by_id(*elem_id).expect("elem_node不存在");
            let temp_similarity = self.compute_similarity(elem_node.get_data(), query);
            if temp_similarity > max_similarity {
                max_similarity = temp_similarity;
                max_elem_id = Some(*elem_id);
            }
        }
        max_elem_id
    }

    // 基线: v0.
    // 基线: 在线性扫描中找最远点。
    pub fn get_furthest_v0(&self, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        let mut min_similarity = f32::INFINITY;
        let mut min_elem_id: Option<Id> = None;

        for elem_id in set {
            let elem_node = self.get_node_by_id(*elem_id).expect("elem_node不存在");
            let temp_similarity = self.compute_similarity(elem_node.get_data(), query);
            if temp_similarity < min_similarity {
                min_similarity = temp_similarity;
                min_elem_id = Some(*elem_id);
            }
        }
        min_elem_id
    }
}
