use crate::types::Id;

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

    pub fn find_index_of_id_in_neighbors(target_id: Id, target_vector: &[Id]) -> Option<usize> {
        for (idx, &x) in target_vector.iter().enumerate() {
            if x == target_id {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_node_by_id(&self, target_id: Id) -> Option<&HnswNode> {
        self.get_index_by_id(target_id)
            .and_then(|idx| self.get_node_by_index(idx))
    }

    pub fn get_mut_node_by_id(&mut self, target_id: Id) -> Option<&mut HnswNode> {
        self.get_index_by_id(target_id)
            .and_then(|idx| self.get_mut_node_by_index(idx))
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

    pub fn add_neighbor_to_node_at_level(&mut self, node_id: Id, neighbor_id: Id, level: usize) {
        let node_index = self
            .get_index_by_id(node_id)
            .expect("in add_neighbor_to_node_at_level, node_id not found");
        let neighbor_index = self
            .get_index_by_id(neighbor_id)
            .expect("in add_neighbor_to_node_at_level, neighbor_id not found");
        self.add_neighbor_to_node_at_level_by_index(node_index, neighbor_index, level);
    }

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

    pub fn delete_neighbor_relationship_at_all_levels(&mut self, node1_id: Id, node2_id: Id) {
        self.delete_neighbor_relationship(node1_id, node2_id, None);
    }

    fn delete_neighbor_relationship(&mut self, node1_id: Id, node2_id: Id, level: Option<usize>) {
        let node1_index = self
            .get_index_by_id(node1_id)
            .expect("In function delete_neighbor_relationship, node1_id not found");
        let node2_index = self
            .get_index_by_id(node2_id)
            .expect("In function delete_neighbor_relationship, node2_id not found");
        self.delete_neighbor_relationship_by_index(node1_index, node2_index, level);
    }

    pub fn delete_neighbor_relationship_at_level(
        &mut self,
        node1_id: Id,
        node2_id: Id,
        level: usize,
    ) {
        self.delete_neighbor_relationship(node1_id, node2_id, Some(level));
    }

    pub fn get_neighbors_at_lvl(&self, id: Id, lvl: usize) -> Vec<Id> {
        let node_to_search = self.get_node_by_id(id).expect("get_node_by_id不存在");
        node_to_search.get_neighbors()[lvl]
            .iter()
            .map(|&neighbor_idx| self.get_id_by_index(neighbor_idx))
            .collect()
    }

    pub(super) fn get_neighbors_at_lvl_from_parts(
        nodes: &[HnswNode],
        index: usize,
        lvl: usize,
    ) -> &[usize] {
        &nodes[index].get_neighbors()[lvl]
    }
}
