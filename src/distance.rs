//! 向量距离与相似度计算工具，供索引和测试复用。

use crate::types::Metric;

/// 按指定度量计算统一的相似度分数。
pub fn metric_similarity(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine_similarity(a, b),
        Metric::L2 => l2_similarity_from_squared_distance(a, b),
    }
}

/// 计算两个向量的余弦相似度。
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

/// 将平方 L2 距离转换为相似度分数。
pub fn l2_similarity_from_squared_distance(a: &[f32], b: &[f32]) -> f32 {
    // 距离转相似度：距离越近，值越大
    1.0 / (1.0 + squared_l2_distance(a, b))
}

/// 计算不取平方根的 L2 距离。
pub fn squared_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    // 计算没有开方的欧氏距离. 欧氏距离值越低, 两个向量距离越近.
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y) * (x - y);
    }
    //sum.sqrt()
    sum
}

// 这里release不再做重复维度检查, 调用方要先保证维度一致, 以减少额外开销。
// debug_assert 只在 debug 构建里生效, 方便开发期尽早发现维度不一致;
// release 构建不会保留这层检查。
// inline: 这个函数会被频繁调用, 标记为inline可以让编译器在调用点展开函数体, 减少函数调用的开销, 提高性能。
#[inline]
/// 在已保证维度一致时计算平方 L2 距离。
pub(crate) fn squared_l2_distance_unchecked(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[inline]
/// 在已保证维度一致时计算 L2 相似度分数。
pub(crate) fn l2_similarity_from_squared_distance_unchecked(a: &[f32], b: &[f32]) -> f32 {
    1.0 / (1.0 + squared_l2_distance_unchecked(a, b))
}
