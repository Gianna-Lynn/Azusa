//! 向量距离与相似度计算工具，供索引和测试复用。
/********************************************** Distance **********************************************/

use crate::types::Metric;

pub fn similarity(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine_similarity(a, b),
        Metric::L2 => squared_euclidean_l2_similarity(a, b),
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // 计算余弦相似度. 相似度值越高, 两个向量的距离越近.
    assert_eq!(a.len(), b.len(), "dimension mismatch"); //猜测：这里的字符串参数应该是assert断言不生效时候，屏幕上会输出的话。

    let mut dot = 0.0f32; //dot: 矢量a和矢量b的点积
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x; // 循环完毕后，norm_a中记录了矢量a中各分量值的平方和。
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt(); //数值上看，denom的值等于a的二范数乘以b的二范数
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

pub fn squared_euclidean_l2_similarity(a: &[f32], b: &[f32]) -> f32 {
    // 距离转相似度：距离越近，值越大
    1.0 / (1.0 + squared_euclidean_l2_distance(a, b))
}

pub fn squared_euclidean_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    // 计算没有开方的欧氏距离. 欧氏距离值越低, 两个向量距离越近.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y) * (x - y);
    }
    //sum.sqrt()
    sum
}

#[inline]
pub(crate) fn l2_squared_unchecked(a: &[f32], b: &[f32]) -> f32 {
    // 这里不再重复做长度检查: 调用方要先保证维度一致, 以减少热路径上的额外开销。
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");

    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[inline]
pub(crate) fn l2_similarity_unchecked(a: &[f32], b: &[f32]) -> f32 {
    1.0 / (1.0 + l2_squared_unchecked(a, b))
}
