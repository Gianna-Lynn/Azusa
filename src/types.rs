/********************************************** Types **********************************************/

pub type Id = u64;

// 向量本体（以后可以换成别的结构）
pub type Vector = Vec<f32>;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Metadata {
    pub doc_id: Option<String>,
    pub chunk_id: Option<String>,
    pub source: Option<String>,
    pub page: Option<u32>,
}

// 一条记录：id + 向量 + metadata
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
//Rust中提供的派生宏语法，让编译器自动为这个结构体生成Clone和Debug trait的默认实现，不需要手动编码
pub struct VectorRecord {
    pub id: Id,
    pub vector: Vector,
    #[serde(default)]
    pub metadata: Metadata,
}

pub type Record = VectorRecord;

// 搜索结果：搜索的是哪一个id，匹配程度有多好
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: Id,
    pub score: f32, //比如余弦相似度
                    // SearchResult结构体包含一个id和一个score，
                    // 分别表示搜索结果对应的记录id和相似度得分。
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// PartialEq: 自动生成eq()和ne()方法, 允许使用==和!=
// Eq: 是PartialEq的升级版本, 用在需要"完全相等"(满足反射性对称性传递性)的数据结构里,比如Hashmap
pub enum Metric {
    Cosine,
    L2,
}
