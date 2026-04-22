use std::collections::HashMap;

use crate::distance;
use crate::types::{Metric, Vector};

use super::super::{HnswIndex, HnswNode, SearchLayerWorkspace};

impl HnswIndex {
    pub fn compute_similarity(&self, a: &Vector, b: &Vector) -> f32 {
        distance::similarity(a, b, self.metric)
    }

    pub fn get_similarity(
        &self,
        index_similarity_cache: &mut HashMap<usize, f32>,
        target_index: usize,
        query: &Vector,
    ) -> f32 {
        if let Some(similarity) = index_similarity_cache.get(&target_index) {
            *similarity
        } else {
            let target_node = self
                .get_node_by_index(target_index)
                .expect("In function get_similarity, failed to get target_node by target_index");
            let new_similarity = self.compute_similarity(target_node.get_data(), query);
            index_similarity_cache.insert(target_index, new_similarity);
            new_similarity
        }
    }

    // 这个函数的设计初衷是为了在search_layer中替换掉get_similarity_from_parts,
    // 以利用SearchLayerWorkspace中的sim_marks和sim_values来缓存相似度计算结果, 避免重复计算.
    pub(super) fn get_similarity_from_workspace(
        nodes: &[HnswNode],
        metric: Metric,
        workspace: &mut SearchLayerWorkspace,
        target_index: usize,
        query: &Vector,
    ) -> f32 {
        if workspace.sim_marks[target_index] == workspace.sim_epoch {
            workspace.sim_values[target_index]
        } else {
            let target_data = nodes[target_index].get_data();
            let new_similarity = match metric {
                Metric::L2 => distance::l2_similarity_unchecked(target_data, query),
                Metric::Cosine => distance::similarity(target_data, query, metric),
            };
            workspace.sim_marks[target_index] = workspace.sim_epoch;
            workspace.sim_values[target_index] = new_similarity;
            new_similarity
        }
    }
}
