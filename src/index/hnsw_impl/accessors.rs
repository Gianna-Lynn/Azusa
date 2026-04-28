//! HNSW 索引的访问器和基础查询辅助方法。
use crate::types::{Id, Metadata, VectorRecord};

use super::super::{HnswIndex, HnswNode};

impl HnswIndex {
    pub(crate) fn get_index_by_id(&self, target_id: Id) -> Option<usize> {
        self.id_to_index.get(&target_id).copied()
    }

    pub(crate) fn get_node_by_index(&self, target_index: usize) -> Option<&HnswNode> {
        self.nodes.get(target_index)
    }

    pub(crate) fn get_mut_node_by_index(&mut self, target_index: usize) -> Option<&mut HnswNode> {
        self.nodes.get_mut(target_index)
    }

    pub(crate) fn get_id_by_index(&self, target_index: usize) -> Id {
        self.nodes[target_index].id
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get_nodes(&self) -> &Vec<HnswNode> {
        &self.nodes
    }

    pub fn get_entry_node_id(&self) -> Option<u64> {
        self.entry_node_index.map(|idx| self.get_id_by_index(idx))
    }

    pub fn level_sample_probability(&self) -> f64 {
        self.level_sample_probability
    }

    pub fn get_node_by_id(&self, target_id: Id) -> Option<&HnswNode> {
        self.get_index_by_id(target_id)
            .and_then(|idx| self.get_node_by_index(idx))
    }

    pub fn get_record_by_id(&self, target_id: Id) -> Option<&VectorRecord> {
        self.records.get(&target_id)
    }

    pub fn get_metadata_by_id(&self, target_id: Id) -> Option<&Metadata> {
        self.get_record_by_id(target_id)
            .map(|record| &record.metadata)
    }

    pub(crate) fn add_neighbor_to_node_at_level_by_index(
        &mut self,
        node_index: usize,
        neighbor_index: usize,
        level: usize,
    ) {
        let node = self
            .get_mut_node_by_index(node_index)
            .expect("in add_neighbor_to_node, in get_mut_node_by_index, got None");
        node.add_neighbor(neighbor_index, level);
    }

    // 同时删除node1_index和node2_index的邻居关系, 因为HNSW里邻居关系是双向的.
    // level参数是Option类型, 如果是Some(level), 就只删除指定层的邻居关系; 
    // 如果是None, 就删除所有层的邻居关系.
    pub(crate) fn delete_neighbor_relationship_by_index(
        &mut self,
        node1_index: usize,
        node2_index: usize,
        level: Option<usize>,
    ) {
        let node1 = self.get_mut_node_by_index(node1_index).expect(
            "In function delete_neighbor_relationship_by_index, get_mut_node_by_index got None",
        );
        match level {
            Some(level) => node1.delete_neighbor_single_direction_at_level(node2_index, level),
            None => node1.delete_neighbor_single_direction_at_all_levels(node2_index),
        }

        let node2 = self.get_mut_node_by_index(node2_index).expect(
            "In function delete_neighbor_relationship_by_index, get_mut_node_by_index got None",
        );
        match level {
            Some(level) => node2.delete_neighbor_single_direction_at_level(node1_index, level),
            None => node2.delete_neighbor_single_direction_at_all_levels(node1_index),
        }
    }

    // 这个辅助函数配合借用拆分使用: select_neighbors_heuristic里需要同时持有
    // &self.nodes 和 &mut self.workspace, 所以先把nodes单独拆出来，再通过这里读取某层邻居。
    pub(super) fn get_neighbors_at_lvl_from_parts(
        nodes: &[HnswNode],
        index: usize,
        lvl: usize,
    ) -> &[usize] {
        &nodes[index].get_neighbors()[lvl]
    }
}
