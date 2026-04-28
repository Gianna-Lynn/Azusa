//! 简单的内存向量存储，配合 legacy demo 和旧测试使用。
use crate::types::{Id, Metadata, Vector, VectorRecord};
use serde_json;

pub struct InMemoryVectorStore {
    data: Vec<VectorRecord>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn records(&self) -> &[VectorRecord] {
        &self.data
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let json_string = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(path, json_string)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let json_string = std::fs::read_to_string(path)?;
        let data: Vec<VectorRecord> = serde_json::from_str(&json_string)?;
        Ok(Self { data })
    }

    pub fn insert(&mut self, id: Id, vector: Vector) {
        if let Some(rec) = self.data.iter_mut().find(|r| r.id == id) {
            rec.vector = vector;
        } else {
            self.data.push(VectorRecord {
                id,
                vector,
                metadata: Metadata::default(),
            });
        }
    }
}
