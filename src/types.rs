//! 项目共用的基础类型定义：ID、向量、记录、搜索结果和度量方式。

/// 每条记录的唯一标识符。
///
pub type Id = u64;

/// 一条数据的向量表示。
///
pub type Vector = Vec<f32>;

/// 记录的可选元信息。
///
/// Metadata: 记录的元信息, 比如来源文档、页码等. 目前是可选的, 可以不提供.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Metadata {   // 属于下面的VectorRecord的一部分.
    pub doc_id: Option<String>,
    pub chunk_id: Option<String>,
    pub source: Option<String>,
    pub page: Option<u32>,
}

/// 一条完整的原始向量记录。
///
/// VectorRecord是 "一条完整的原始数据记录", 包括 id + 向量 + metadata
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorRecord {
    pub id: Id,         // 这条数据的唯一标识符(主键)
    pub vector: Vector, // 这条数据的向量表示
    #[serde(default)]
    // #[serde(default)]表示在反序列化时如果缺失metadata字段, 就用Metadata::default()来填充, 避免报错.
    pub metadata: Metadata, // 这条数据的元信息, 比如来源文档、页码等.
}

/// 一条搜索结果。
///
/// SearchResult结构体包含一个id和一个score，分别表示搜索结果对应的记录id和相似度得分。
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: Id,
    pub score: f32, // score: 这个搜索结果的相似度得分, 越高表示越匹配查询
}

/// 向量相似度度量方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    L2,
}
