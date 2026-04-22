use crate::distance;
use crate::types::{Metric, Vector};

use super::super::{HnswIndex, HnswNode, SearchLayerWorkspace};

impl HnswIndex {
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
