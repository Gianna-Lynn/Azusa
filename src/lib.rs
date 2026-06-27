//! Azusa crate 根模块，统一对外暴露当前主线索引、距离函数和基础类型。
pub mod distance;
pub mod index;
#[cfg(feature = "legacy")]
pub mod legacy;
pub mod types;
