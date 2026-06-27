//! HNSW 图中单个节点的内部表示。
use crate::types::{Id, Vector};

#[derive(Debug, Clone)]
pub struct HnswNode {
    // 这些字段只开放给 crate 内部的索引实现使用，外部通过 HnswIndex 的接口访问即可。
    pub(crate) id: Id,
    pub(crate) data: Vector,
    pub(crate) node_max_level: usize,
    // 邻接表按层存储，内层保存的是节点 index，而不是 Id。
    pub(crate) neighbors: Vec<Vec<usize>>,
}

impl HnswNode {
    /// 创建一个指定最高层级的 HNSW 节点。
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

    /// 返回节点的外部 Id。
    pub fn get_id(&self) -> &Id {
        &self.id
    }

    /// 返回节点存储的向量。
    pub fn get_data(&self) -> &Vector {
        &self.data
    }

    /// 返回节点参与的最高层级。
    pub fn get_node_max_level(&self) -> &usize {
        &self.node_max_level
    }

    /// 返回按层组织的邻接表。
    pub fn get_neighbors(&self) -> &Vec<Vec<usize>> {
        &self.neighbors
    }

    /// 在指定层添加一个邻居 index。
    pub fn add_neighbor(&mut self, neighbor_index: usize, level_to_add: usize) {
        if level_to_add < self.neighbors.len() {
            // 这里假设上游插入/裁剪逻辑已经保证同层邻居唯一；
            // debug_assert 只是在开发期额外检查这个不变量，帮助尽早发现重复边。
            debug_assert!(
                !self.neighbors[level_to_add].contains(&neighbor_index),
                "duplicate neighbor: node_id={}, neighbor_index={}, level={}",
                self.id,
                neighbor_index,
                level_to_add
            );
            self.neighbors[level_to_add].push(neighbor_index);
        }
    }

    /// 从所有层删除一个单向邻居关系。
    pub fn delete_neighbor_single_direction_at_all_levels(&mut self, neighbor_index: usize) {
        for i in 0..=self.node_max_level {
            // retain 保留满足条件的元素；这里等价于删掉所有等于 neighbor_index 的项。
            self.neighbors[i].retain(|&idx| idx != neighbor_index);
        }
    }

    /// 从指定层删除一个单向邻居关系。
    pub fn delete_neighbor_single_direction_at_level(
        &mut self,
        neighbor_index: usize,
        level: usize,
    ) {
        self.neighbors[level].retain(|&idx| idx != neighbor_index);
    }
}
