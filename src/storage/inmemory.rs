use crate::types::{Id, Record, Vector};
use super::VectorStore;
use serde_json;

/********************************************** Storage **********************************************/

pub struct InMemoryVectorStore {
    data: Vec<Record>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn records(&self) -> &[Record] {
        &self.data
    }

    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let json_string = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(path, json_string)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let json_string = std::fs::read_to_string(path)?;
        let data: Vec<Record> = serde_json::from_str(&json_string)?;
        Ok(Self { data })
    }
}

impl VectorStore for InMemoryVectorStore {
    fn insert(&mut self, id: Id, vector: Vector) {
        if let Some(rec) = self.data.iter_mut().find(|r| r.id == id) {
            rec.vector = vector;
        } else {
            self.data.push(Record { id, vector });
        }
    }
}
