//! legacy 手工构图测试所用的固定用例数据。
// Legacy handcrafted and synthetic HNSW experiment cases.
// 这些 case 主要服务于早期手搓数据集测试与参数 sweep，不是正式数据集夹具。

#![allow(dead_code)]

pub type NodeData = (u64, Vec<f32>, usize);

#[derive(Debug, Clone)]
pub struct HnswTestCase {
    pub nodes_data: Vec<NodeData>,
    pub query: Vec<f32>,
    pub entry_id: Option<u64>,
    pub expected_result_id: Option<u64>,
    pub level: usize,
    pub k: usize,
    pub ef_search: usize,
}

pub fn two_node_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![(1, vec![1.0, 2.0, 3.0], 3), (2, vec![2.0, 3.0, 4.0], 4)],
        query: vec![1.0, 0.0, 0.0],
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 2,
        ef_search: 2,
    }
}

pub fn three_node_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1.0, 2.0, 3.0], 3),
            (2, vec![2.1, 3.1, 4.1], 4),
            (3, vec![2.0, 3.0, 4.0], 4),
        ],
        query: vec![1.0, 0.0, 0.0],
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 3,
        ef_search: 2,
    }
}

pub fn dense_2d_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0], 2),
            (2, vec![9.0, 1.0], 2),
            (3, vec![8.0, 2.0], 2),
            (4, vec![0.0, 10.0], 2),
            (5, vec![1.0, 9.0], 2),
            (6, vec![2.0, 8.0], 2),
        ],
        query: vec![6.0, 6.0],
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 3,
        ef_search: 3,
    }
}

pub fn flat_2d_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0], 0),
            (2, vec![9.0, 1.0], 0),
            (3, vec![8.0, 2.0], 0),
            (4, vec![0.0, 10.0], 0),
            (5, vec![1.0, 9.0], 0),
            (6, vec![2.0, 8.0], 0),
        ],
        query: vec![6.0, 6.0],
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 3,
        ef_search: 3,
    }
}

pub fn greedy_chain_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0], 0),
            (2, vec![9.0, 1.0], 0),
            (3, vec![8.0, 2.0], 0),
            (4, vec![0.0, 10.0], 0),
            (5, vec![1.0, 9.0], 0),
            (6, vec![2.0, 8.0], 0),
        ],
        query: vec![6.0, 6.0],
        entry_id: Some(1),
        expected_result_id: Some(3),
        level: 0,
        k: 3,
        ef_search: 3,
    }
}

pub fn search_layer_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0], 0),
            (2, vec![9.0, 1.0], 0),
            (3, vec![8.0, 2.0], 0),
            (4, vec![0.0, 10.0], 0),
            (5, vec![1.0, 9.0], 0),
            (6, vec![2.0, 8.0], 0),
        ],
        query: vec![6.0, 6.0],
        entry_id: Some(2),
        expected_result_id: Some(2),
        level: 0,
        k: 3,
        ef_search: 3,
    }
}

pub fn greedy_stop_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0], 0),
            (2, vec![9.0, 1.0], 0),
            (3, vec![8.0, 2.0], 0),
            (4, vec![0.0, 10.0], 0),
            (5, vec![1.0, 9.0], 0),
            (6, vec![2.0, 8.0], 0),
        ],
        query: vec![8.1, 2.1],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 3,
        ef_search: 3,
    }
}

pub fn multilevel_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0], 3),
            (2, vec![8.5, 1.0], 2),
            (3, vec![7.0, 2.0], 1),
            (4, vec![0.0, 10.0], 3),
            (5, vec![1.0, 8.5], 2),
            (6, vec![2.0, 7.0], 1),
            (7, vec![5.0, 5.0], 0),
            (8, vec![4.0, 4.5], 0),
        ],
        query: vec![4.8, 4.9],
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

pub fn single_node_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![(1, vec![3.0, 4.0], 0)],
        query: vec![100.0, -100.0],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 1,
        ef_search: 1,
    }
}

pub fn empty_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![],
        query: vec![0.0, 1.0],
        entry_id: None,
        expected_result_id: None,
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

pub fn unique_ranking_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (5, vec![0.0, 10.0], 0),
            (2, vec![8.0, 1.0], 0),
            (8, vec![4.0, 8.0], 0),
            (1, vec![10.0, 0.0], 0),
            (6, vec![-3.0, 7.0], 0),
            (4, vec![1.0, 9.0], 0),
            (7, vec![7.0, 3.0], 0),
            (3, vec![6.0, 2.0], 0),
        ],
        query: vec![9.2, 0.3],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

pub fn beginning_is_the_best_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![(1, vec![10.0, 0.0], 0), (2, vec![0.0, 10.0], 0)],
        query: vec![9.0, 1.0],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

pub fn identical_vectors_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![5.0, 5.0], 0),
            (2, vec![5.0, 5.0], 0),
            (3, vec![5.0, 5.0], 0),
            (4, vec![10.0, 10.0], 0),
        ],
        query: vec![5.0, 5.0],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 2,
        ef_search: 3,
    }
}

pub fn duplicate_distance_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1.0, 0.0], 0),
            (2, vec![0.0, 1.0], 0),
            (3, vec![0.0, -1.0], 0),
            (4, vec![0.0, 2.0], 0),
        ],
        query: vec![1.0, 0.0],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 3,
        ef_search: 5,
    }
}

pub fn collinear_points_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1.0, 0.0], 0),
            (2, vec![0.95, 0.10], 0),
            (3, vec![0.90, 0.20], 0),
            (4, vec![0.85, 0.30], 0),
            (5, vec![0.80, 0.40], 0),
            (6, vec![0.75, 0.50], 0),
        ],
        query: vec![1.0, 0.0],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 2,
        ef_search: 4,
    }
}

pub fn clustered_distribution_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1.0, 0.0], 0),
            (2, vec![0.99, 0.01], 0),
            (3, vec![0.98, 0.02], 0),
            (4, vec![0.97, 0.03], 0),
            (5, vec![-1.0, 0.0], 0),
            (6, vec![-0.99, -0.01], 0),
            (7, vec![-0.98, -0.02], 0),
            (8, vec![-0.97, -0.03], 0),
        ],
        query: vec![1.0, 0.0],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 3,
        ef_search: 6,
    }
}

pub fn high_dimension_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
            (2, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
            (3, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
            (4, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
            (5, vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0),
        ],
        query: vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        entry_id: Some(1),
        expected_result_id: Some(5),
        level: 0,
        k: 2,
        ef_search: 4,
    }
}

pub fn extreme_values_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1e5, 1e5], 0),
            (2, vec![100001.0, 99999.0], 0),
            (3, vec![1e-5, 2e-5], 0),
            (4, vec![0.0001, 0.0002], 0),
            (5, vec![1e10, 0.0], 0),
        ],
        query: vec![1e5, 1e5],
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 2,
        ef_search: 4,
    }
}

pub fn near_zero_distance_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![1.0, 0.0], 0),
            (2, vec![1.0000001, 0.0000001], 0),
            (3, vec![1.0000002, 0.0000002], 0),
            (4, vec![10.0, 10.0], 0),
        ],
        query: vec![1.0, 0.0],
        entry_id: Some(4),
        expected_result_id: Some(1),
        level: 0,
        k: 2,
        ef_search: 3,
    }
}

pub fn k_larger_than_dataset_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![0.0, 0.0], 0),
            (2, vec![1.0, 0.0], 0),
            (3, vec![0.0, 1.0], 0),
        ],
        query: vec![0.5, 0.5],
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 10,
        ef_search: 10,
    }
}

pub fn tightly_packed_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![0.0, 0.0], 0),
            (2, vec![0.01, 0.0], 0),
            (3, vec![0.0, 0.01], 0),
            (4, vec![0.012, 0.009], 0),
            (5, vec![0.005, 0.005], 0),
            (6, vec![0.007, 0.003], 0),
        ],
        query: vec![0.005, 0.005],
        entry_id: Some(1),
        expected_result_id: Some(5),
        level: 0,
        k: 3,
        ef_search: 5,
    }
}

pub fn extreme_ef_search_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![0.0, 0.0], 0),
            (2, vec![1.0, 0.0], 0),
            (3, vec![0.0, 1.0], 0),
            (4, vec![1.0, 1.0], 0),
        ],
        query: vec![0.5, 0.5],
        entry_id: Some(1),
        expected_result_id: Some(4),
        level: 0,
        k: 2,
        ef_search: 1000,
    }
}

pub fn negative_coordinates_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![-10.0, -11.0], 0),
            (2, vec![-4.0, -4.0], 0),
            (3, vec![0.0, 0.0], 0),
            (4, vec![5.0, 5.0], 0),
            (5, vec![10.0, 9.0], 0),
        ],
        query: vec![-2.5, -2.5],
        entry_id: Some(2),
        expected_result_id: Some(2),
        level: 0,
        k: 2,
        ef_search: 4,
    }
}

pub fn sparse_vectors_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![10.0, 0.0, 0.0, 0.0], 0),
            (2, vec![0.0, 10.0, 0.0, 0.0], 0),
            (3, vec![0.0, 0.0, 10.0, 0.0], 0),
            (4, vec![0.0, 0.0, 0.0, 10.0], 0),
            (5, vec![5.0, 5.0, 0.0, 0.0], 0),
        ],
        query: vec![1.0, 1.0, 0.0, 0.0],
        entry_id: Some(5),
        expected_result_id: Some(5),
        level: 0,
        k: 2,
        ef_search: 4,
    }
}

pub fn bridge_trap_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            // 左团
            (1, vec![10.0, 0.0], 2),
            (2, vec![9.5, 0.5], 1),
            (3, vec![9.0, 1.0], 0),
            // 桥
            (4, vec![5.0, 5.0], 1),
            // 右团
            (5, vec![1.0, 9.0], 0),
            (6, vec![0.5, 9.5], 1),
            (7, vec![0.0, 10.0], 2),
        ],
        query: vec![0.2, 9.8],
        entry_id: Some(1),
        expected_result_id: Some(7),
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

// 弱桥双团 case
// 从左团最高层入口开始，能不能跨过弱桥到右团
// m 小时会不会桥接不牢
// ef_search 小时会不会被左团困住
pub fn weak_bridge_two_clusters_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            // 左团（错误团）
            (1, vec![10.0, 0.0], 3),
            (2, vec![9.8, 0.3], 2),
            (3, vec![9.5, 0.6], 1),
            (4, vec![9.2, 0.9], 0),
            (5, vec![8.9, 1.2], 0),
            // 弱桥
            (6, vec![6.0, 4.0], 1),
            (7, vec![5.0, 5.0], 0),
            // 右团（正确团）
            (8, vec![1.2, 8.8], 0),
            (9, vec![0.9, 9.1], 1),
            (10, vec![0.6, 9.4], 2),
            (11, vec![0.3, 9.7], 2),
            (12, vec![0.0, 10.0], 3),
        ],
        query: vec![0.2, 9.8],
        entry_id: Some(1),
        expected_result_id: Some(12),
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

// 2. 近邻拥挤 case
// 这个不是桥接题，是排名极近题。
// 适合看 exact_hit 会不会先掉。
// 这题想测什么
// top1 也许还能对
// 但 k=5 的 exact ranking 很容易开始抖
// 比较适合看 ef_search 从 4 到 8 有没有改善
pub fn crowded_near_neighbors_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            (1, vec![5.00, 5.00], 2),
            (2, vec![5.05, 4.95], 1),
            (3, vec![4.95, 5.05], 1),
            (4, vec![5.10, 4.90], 0),
            (5, vec![4.90, 5.10], 0),
            (6, vec![5.15, 4.85], 0),
            (7, vec![4.85, 5.15], 0),
            // 一些干扰点
            (8, vec![7.0, 3.0], 1),
            (9, vec![3.0, 7.0], 1),
            (10, vec![8.0, 2.0], 0),
            (11, vec![2.0, 8.0], 0),
        ],
        query: vec![5.02, 4.98],
        entry_id: Some(10),
        expected_result_id: Some(2),
        level: 0,
        k: 5,
        ef_search: 4,
    }
}

// 3. 单桥长链 case
// 这个会更像“先被错误方向拖着走”。
// 这题想测什么
// 坏入口 + 单桥
// 如果图连得不够好，可能一直在左边链条里晃
// 比双团更“线性误导”
pub fn single_bridge_chain_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            // 左侧长链（错误方向）
            (1, vec![10.0, 0.0], 3),
            (2, vec![9.0, 1.0], 2),
            (3, vec![8.0, 2.0], 1),
            (4, vec![7.0, 3.0], 0),
            // 唯一桥点
            (5, vec![5.0, 5.0], 1),
            // 右侧长链（正确方向）
            (6, vec![3.0, 7.0], 0),
            (7, vec![2.0, 8.0], 1),
            (8, vec![1.0, 9.0], 2),
            (9, vec![0.0, 10.0], 3),
        ],
        query: vec![0.1, 9.9],
        entry_id: Some(1),
        expected_result_id: Some(9),
        level: 0,
        k: 3,
        ef_search: 4,
    }
}

// 4. 大一点的稀疏高维桥接 case
// 这个适合看高维下的表现，别总是 2D。
pub fn highdim_bridge_case() -> HnswTestCase {
    HnswTestCase {
        nodes_data: vec![
            // 左团
            (1, vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
            (2, vec![9.5, 0.5, 0.0, 0.0, 0.0, 0.0], 2),
            (3, vec![9.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1),
            // 桥
            (4, vec![5.0, 5.0, 0.0, 0.0, 0.0, 0.0], 1),
            (5, vec![4.0, 6.0, 0.0, 0.0, 0.0, 0.0], 0),
            // 右团
            (6, vec![1.0, 9.0, 0.0, 0.0, 0.0, 0.0], 0),
            (7, vec![0.5, 9.5, 0.0, 0.0, 0.0, 0.0], 1),
            (8, vec![0.0, 10.0, 0.0, 0.0, 0.0, 0.0], 3),
            // 干扰维度点
            (9, vec![0.0, 0.0, 10.0, 0.0, 0.0, 0.0], 1),
            (10, vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0], 1),
            (11, vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0], 0),
            (12, vec![0.0, 0.0, 0.0, 0.0, 0.0, 10.0], 0),
        ],
        query: vec![0.1, 9.9, 0.0, 0.0, 0.0, 0.0],
        entry_id: Some(1),
        expected_result_id: Some(8),
        level: 0,
        k: 4,
        ef_search: 4,
    }
}

// ================================================================
// 第二阶段：中等规模 Synthetic 数据集（Agent 生成）
// 目的：让 m 参数的影响显现
// 特点：点数从 50-150 个，结构可解释，m=2 vs m=4/8 有明显差异
// ================================================================

/// 稀疏多团案例（~50个点）
/// 结构：4个独立的紧团，团之间距离远，m 小时无法跨团
/// 预期行为：m=2 时找到的邻居都在同一团内；m=4+ 时能跨团
pub fn sparse_four_clusters_case() -> HnswTestCase {
    let mut nodes = vec![];
    let mut id_counter = 1u64;

    // 团1：(10, 0) 中心附近 - 12个点
    for i in 0..12 {
        let angle = i as f32 * std::f32::consts::PI / 6.0;
        let x = 10.0 + 0.5 * angle.cos();
        let y = 0.5 * angle.sin();
        nodes.push((id_counter, vec![x, y], 2));
        id_counter += 1;
    }

    // 团2：(0, 10) 中心附近 - 12个点
    for i in 0..12 {
        let angle = i as f32 * std::f32::consts::PI / 6.0;
        let x = 0.5 * angle.cos();
        let y = 10.0 + 0.5 * angle.sin();
        nodes.push((id_counter, vec![x, y], 2));
        id_counter += 1;
    }

    // 团3：(-10, 0) 中心附近 - 12个点
    for i in 0..12 {
        let angle = i as f32 * std::f32::consts::PI / 6.0;
        let x = -10.0 + 0.5 * angle.cos();
        let y = 0.5 * angle.sin();
        nodes.push((id_counter, vec![x, y], 2));
        id_counter += 1;
    }

    // 团4：(0, -10) 中心附近 - 12个点
    for i in 0..12 {
        let angle = i as f32 * std::f32::consts::PI / 6.0;
        let x = 0.5 * angle.cos();
        let y = -10.0 + 0.5 * angle.sin();
        nodes.push((id_counter, vec![x, y], 2));
        id_counter += 1;
    }

    HnswTestCase {
        nodes_data: nodes,
        query: vec![10.3, 0.2], // 查询在团1内
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 8,
        ef_search: 8,
    }
}

/// 高密度单团案例（~80个点）
/// 结构：单个大团，点极其密集，m 小时容易陷入局部最优
/// 预期行为：m=2 时排名抖动；m=4+ 时更稳定
pub fn dense_single_cluster_case() -> HnswTestCase {
    let mut nodes = vec![];
    let mut id_counter = 1u64;

    // 中心在 (5, 5)，半径 2 内均匀分布 ~80 个点
    let radius = 2.0;
    let center_x = 5.0;
    let center_y = 5.0;

    for i in 0..80 {
        let angle = (i as f32) * 2.0 * std::f32::consts::PI / 80.0;
        let r = (i as f32 / 80.0) * radius;
        let x = center_x + r * angle.cos();
        let y = center_y + r * angle.sin();
        let level = if i < 10 {
            2
        } else if i < 30 {
            1
        } else {
            0
        };
        nodes.push((id_counter, vec![x, y], level));
        id_counter += 1;
    }

    HnswTestCase {
        nodes_data: nodes,
        query: vec![5.05, 5.05], // 查询在中心附近
        entry_id: Some(1),
        expected_result_id: Some(1),
        level: 0,
        k: 10,
        ef_search: 8,
    }
}

/// 金字塔型分层案例（~100个点）
/// 结构：4层，每层密度递减，m 小时易被下层点吸引
/// 预期行为：m=2 时可能漏掉上层点；m=4+ 时能正确维护跨层连接
pub fn pyramid_hierarchy_case() -> HnswTestCase {
    let mut nodes = vec![];
    let mut id_counter = 1u64;

    // 第0层（底部）：稀疏，8个点，离中心远
    for i in 0..8 {
        let angle = i as f32 * std::f32::consts::PI / 4.0;
        nodes.push((id_counter, vec![15.0 * angle.cos(), 15.0 * angle.sin()], 0));
        id_counter += 1;
    }

    // 第1层：16个点，距离适中
    for i in 0..16 {
        let angle = i as f32 * std::f32::consts::PI / 8.0;
        nodes.push((id_counter, vec![10.0 * angle.cos(), 10.0 * angle.sin()], 1));
        id_counter += 1;
    }

    // 第2层：32个点，靠近中心
    for i in 0..32 {
        let angle = i as f32 * std::f32::consts::PI / 16.0;
        nodes.push((id_counter, vec![5.0 * angle.cos(), 5.0 * angle.sin()], 1));
        id_counter += 1;
    }

    // 第3层（中心）：44个点，极其紧密
    for i in 0..44 {
        let angle = i as f32 * std::f32::consts::PI / 22.0;
        let r = (i as f32 / 44.0) * 1.0;
        nodes.push((id_counter, vec![r * angle.cos(), r * angle.sin()], 2));
        id_counter += 1;
    }

    HnswTestCase {
        nodes_data: nodes,
        query: vec![0.1, 0.1], // 查询在最中心
        entry_id: Some(1),     // 从底层开始查询
        expected_result_id: None,
        level: 0,
        k: 12,
        ef_search: 8,
    }
}

/// 水平片状分布案例（~90个点）
/// 结构：5条平行的密集"线条"，m 小时无法跨线切换
/// 预期行为：m=2 时被困在一条线上；m=4+ 时能跨线切换
pub fn layered_strips_case() -> HnswTestCase {
    let mut nodes = vec![];
    let mut id_counter = 1u64;

    // 5条平行线，每条 18 个点
    for strip_idx in 0..5 {
        let y = (strip_idx as f32 - 2.0) * 4.0; // y: -8, -4, 0, 4, 8

        for i in 0..18 {
            let x = (i as f32 - 8.5) * 1.0;
            let level = if i == 9 {
                2
            } else if i >= 4 && i <= 14 {
                1
            } else {
                0
            };
            nodes.push((id_counter, vec![x, y], level));
            id_counter += 1;
        }
    }

    HnswTestCase {
        nodes_data: nodes,
        query: vec![0.0, 0.1], // 查询在中间线条
        entry_id: Some(10),    // 从底层开始
        expected_result_id: None,
        level: 0,
        k: 10,
        ef_search: 8,
    }
}

/// 放射状星形案例（~72个点）
/// 结构：中心 1 个点，周围 8 条射线，每条 8 个点
/// 预期行为：m 小时只能沿射线走；m 大时能在射线间切换
pub fn radial_star_case() -> HnswTestCase {
    let mut nodes = vec![];
    let mut id_counter = 1u64;

    // 中心点
    nodes.push((id_counter, vec![0.0, 0.0], 3));
    id_counter += 1;

    // 8条射线，每条 8 个点
    for ray_idx in 0..8 {
        let angle = ray_idx as f32 * std::f32::consts::PI / 4.0;

        for dist_idx in 1..=8 {
            let r = dist_idx as f32 * 1.5;
            let x = r * angle.cos();
            let y = r * angle.sin();
            let level = if dist_idx <= 2 {
                2
            } else if dist_idx <= 5 {
                1
            } else {
                0
            };
            nodes.push((id_counter, vec![x, y], level));
            id_counter += 1;
        }
    }

    HnswTestCase {
        nodes_data: nodes,
        query: vec![1.5, 1.5], // 查询在左下射线附近
        entry_id: Some(1),     // 从中心开始
        expected_result_id: None,
        level: 0,
        k: 8,
        ef_search: 8,
    }
}

// ================================================================
// 第二阶段结束
// ================================================================

// 团 A → 桥 1 → 团 B → 桥 2 → 团 C
// A 是起点附近的错误大团
// C 是 query 所在的正确大团
// B 是中间过渡团
// 两个桥都很关键
// 只要有一处桥不够牢，搜索就容易被困在前面
pub fn multi_cluster_chain_case() -> HnswTestCase {
    let mut nodes = vec![];
    let mut id_counter = 1u64;

    let a_x = 0;
    let a_y = 0;

    // 团A: 16个点, 中心(0,0)
    for i in 0..16 {
        let angle = i as f32 * std::f32::consts::PI / 8.0;
        let lvl = if i == 0 {
            3
        } else if (1..=4).contains(&i) {
            2
        } else if (5..=9).contains(&i) {
            1
        } else {
            0
        };
        let r = if i <= 7 { 0.8 } else { 1.3 };
        nodes.push((
            id_counter,
            vec![a_x as f32 + r * angle.cos(), a_y as f32 + r * angle.sin()],
            lvl,
        ));
        id_counter += 1;
    }

    let b_x = 12;
    let b_y = 0;
    // 团B: 16个点, 中心(12,0)
    for i in 0..7 {
        let angle = i as f32 * std::f32::consts::PI / 4.0;
        let lvl = if i == 0 {
            3
        } else if (1..=4).contains(&i) {
            2
        } else if (5..=9).contains(&i) {
            1
        } else {
            0
        };
        let r = if i <= 7 { 0.8 } else { 1.3 };
        nodes.push((
            id_counter,
            vec![b_x as f32 + r * angle.cos(), b_y as f32 + r * angle.sin()],
            lvl,
        ));
        id_counter += 1;
    }

    let c_x = 24;
    let c_y = 0;
    // 团C: 16个点, 中心(24,0)
    for i in 0..16 {
        let angle = i as f32 * std::f32::consts::PI / 8.0;
        let lvl = if i == 0 {
            3
        } else if (1..=4).contains(&i) {
            2
        } else if (5..=9).contains(&i) {
            1
        } else {
            0
        };
        let r = if i <= 7 { 0.8 } else { 1.3 };
        nodes.push((
            id_counter,
            vec![c_x as f32 + r * angle.cos(), c_y as f32 + r * angle.sin()],
            lvl,
        ));
        id_counter += 1;
    }

    //桥1
    nodes.push((id_counter, vec![6.0, 0.2], 0));
    id_counter += 1;
    // nodes.push((id_counter, vec![6.0, -0.2] ,1));
    // id_counter += 1;
    //桥2
    nodes.push((id_counter, vec![18.0, 0.2], 0));
    // nodes.push((id_counter, vec![18.0, -0.2] ,1));
    // id_counter += 1;

    HnswTestCase {
        nodes_data: nodes,
        query: vec![24.8, 0.1], //更偏向于C团
        entry_id: Some(1),
        expected_result_id: None,
        level: 0,
        k: 8,
        ef_search: 16,
    }
}
