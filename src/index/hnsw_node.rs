use crate::types::{Id, Vector};

#[derive(Debug, Clone)]
pub struct HnswNode {
    pub(crate) id: Id,
    pub(crate) data: Vector,
    pub(crate) node_max_level: usize,
    pub(crate) neighbors: Vec<Vec<Id>>,
}

impl HnswNode {
    pub fn new(id: Id, data: Vector, level: usize) -> Self {
        let mut neighbors = Vec::new();
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }

        Self {
            id,
            data,
            node_max_level: level,
            neighbors,
        }
    }

    pub fn add_neighbor(&mut self, neighbor_id: Id, level_to_add: usize) {
        if level_to_add < self.neighbors.len() {
            self.neighbors[level_to_add].push(neighbor_id);
        }
    }

    pub fn delete_neighbor_single_direction_at_all_levels(&mut self, neighbor_id: Id) {
        for i in 0..=self.node_max_level {
            self.neighbors[i].retain(|&id| id != neighbor_id);
        }
    }

    pub fn delete_neighbor_single_direction_at_level(&mut self, neighbor_id: Id, level: usize) {
        self.neighbors[level].retain(|&id| id != neighbor_id);
    }

    pub fn get_id(&self) -> &Id {
        &self.id
    }

    pub fn get_data(&self) -> &Vector {
        &self.data
    }

    pub fn get_node_max_level(&self) -> &usize {
        &self.node_max_level
    }

    pub fn get_neighbors(&self) -> &Vec<Vec<u64>> {
        &self.neighbors
    }
}
