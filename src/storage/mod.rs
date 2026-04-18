pub mod inmemory;
use crate::types::{Id, Vector};
//统一的向量存储接口
pub trait VectorStore {
    // 插入或者更新一条向量
    fn insert(&mut self, id: Id, vector: Vector);

    // 基于相似度搜索前K个
    // fn search(&self, query: &Vector, k: usize) -> Vec<SearchResult>;  //问题：目前对于trait和泛型这一块的知识还不清晰。

    //回答：trait直译为特征，规定了所有VectorStore都必须有的特征, 也就是两个方法签名，insert和search。
    //     在trait当中只写方法签名，不写具体实现, 以分号结束。
    //     VectorStore是一个trait, 是一份合同, 或者说接口说明书.
}
