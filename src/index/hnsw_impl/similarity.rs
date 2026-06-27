//! HNSW 搜索过程中复用的相似度缓存辅助函数。
use crate::distance;
use crate::types::{Metric, Vector};

use super::super::{HnswIndex, HnswNode, SearchLayerWorkspace};

impl HnswIndex {
    /// 从工作区缓存中读取或计算相似度。
    ///
    /// 这个函数的设计初衷是为了在search_layer中替换掉get_similarity_from_parts,
    /// 以利用SearchLayerWorkspace中的sim_marks和sim_values来缓存相似度计算结果, 避免重复计算.
    pub(super) fn get_similarity_from_workspace(
        nodes: &[HnswNode],
        metric: Metric,
        workspace: &mut SearchLayerWorkspace,
        target_index: usize,
        query: &Vector,
    ) -> f32 {
        // 先查 epoch 缓存，缓存命中直接返回
        if workspace.sim_marks[target_index] == workspace.sim_epoch {
            workspace.sim_values[target_index]
        } else {
            // 缓存不命中, 计算相似度并更新缓存
            let target_data = nodes[target_index].get_data();
            let new_similarity = match metric {
                Metric::L2 => {
                    distance::l2_similarity_from_squared_distance_unchecked(target_data, query)
                }
                Metric::Cosine => distance::metric_similarity(target_data, query, metric),
            };
            workspace.sim_marks[target_index] = workspace.sim_epoch;
            workspace.sim_values[target_index] = new_similarity;
            new_similarity
        }
    }
}
