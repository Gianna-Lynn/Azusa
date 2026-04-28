//! legacy HNSW 手工构图用例测试。
// Legacy handcrafted HNSW tests.
// 这份文件主要验证早期手搓数据集在 legacy 实验构图路径下的行为。

#[path = "legacy_handcrafted_cases.rs"]
mod legacy_handcrafted_cases;
use legacy_handcrafted_cases::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use azusa::{index::hnsw::HnswIndex, types::Id};

/// id: u64, data: Vec<f32>, lvl: usize
/// 构图 helper
fn build_index(nodes_data: Vec<(u64, Vec<f32>, usize)>) -> HnswIndex {
    let mut index = HnswIndex::new(TEST_METRIC, 0.5);
    let ef_construction = 10;
    let m = 2;
    let m_max = 4;
    for (id, data, level) in nodes_data.iter() {
        index.insert_with_fixed_level_for_legacy(
            *id,
            data.clone(),
            ef_construction,
            m,
            m_max,
            level,
        );
    }
    index
}

fn expected_search_knn_ids(case: &HnswTestCase) -> Vec<Id> {
    let mut scored_ids: Vec<(Id, f32)> = case
        .nodes_data
        .iter()
        .map(|(id, data, _)| {
            (
                *id,
                azusa::distance::cosine_similarity(&case.query, data),
            )
        })
        .collect();

    scored_ids.sort_by(|(left_id, left_score), (right_id, right_score)| {
        match right_score.partial_cmp(left_score).unwrap() {
            std::cmp::Ordering::Equal => left_id.cmp(right_id),
            other => other,
        }
    });

    scored_ids
        .into_iter()
        .take(case.k)
        .map(|(id, _)| id)
        .collect()
}

fn assert_search_knn_case(case_name: &str, case: HnswTestCase) {
    let expected_result_ids = expected_search_knn_ids(&case);
    let HnswTestCase {
        nodes_data,
        query,
        k,
        ef_search,
        ..
    } = case;
    let index = build_index(nodes_data);
    let result_ids = index.search_knn(&query, k, ef_search);

    assert_eq!(
        result_ids, expected_result_ids,
        "{}: expected {:?}, got {:?}",
        case_name, expected_result_ids, result_ids
    );
}

fn run_search_knn_case(case_name: &str, case: HnswTestCase) -> Result<(), String> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        assert_search_knn_case(case_name, case);
    }));

    match result {
        Ok(()) => Ok(()),
        Err(payload) => {
            if let Some(message) = payload.downcast_ref::<String>() {
                Err(message.clone())
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                Err((*message).to_string())
            } else {
                Err(format!(
                    "{}: test panicked with a non-string payload",
                    case_name
                ))
            }
        }
    }
}

#[test]
fn test_search_knn() {
    test_search_knn_empty_case();
    test_search_knn_single_node_case();
    test_search_knn_unique_ranking_case();
}

#[test]
fn test_search_knn_empty_case() {
    // 空图应返回空结果
    let case = empty_case();
    let index = build_index(case.nodes_data);
    let test_query = case.query;

    println!("Test C get_entry_node_id is:");
    println!("{:#?}", index.get_entry_node_id());
    let result_vector = index.search_knn(&test_query, 3, 3);
    assert!(result_vector.is_empty(), "空图应返回空结果");

    println!("Test C result_vector is:");
    println!("{:#?}", result_vector);
}

#[test]
fn test_search_knn_single_node_case() {
    // 单节点只应返回它自己
    let case = single_node_case();
    let index = build_index(case.nodes_data);
    let test_query = case.query;
    let result_vector = index.search_knn(&test_query, 3, 3);
    assert_eq!(result_vector.len(), 1);
    let expected_result_id = case
        .expected_result_id
        .expect("single_node_case should define expected_result_id");
    assert_eq!(result_vector[0], expected_result_id);

    println!("Test D result_vector is:");
    println!("{:#?}", result_vector);
}

#[test]
fn test_search_knn_unique_ranking_case() {
    // 结果数量和排序都要对
    let case = unique_ranking_case();
    let index = build_index(case.nodes_data);
    let test_query = case.query;
    let result_vector = index.search_knn(&test_query, 3, 3);
    assert!(result_vector.len() <= case.k, "结果数量不应超过 k");
    let expected_result_id = case
        .expected_result_id
        .expect("unique_ranking_case should define expected_result_id");
    assert_eq!(result_vector[0], expected_result_id);
    assert!(
        result_vector.windows(2).all(|pair| {
            let left_node = index.get_node_by_id(pair[0]).expect("left node not found");
            let right_node = index.get_node_by_id(pair[1]).expect("right node not found");
            let left_score =
                azusa::distance::cosine_similarity(&test_query, left_node.get_data());
            let right_score =
                azusa::distance::cosine_similarity(&test_query, right_node.get_data());

            left_score >= right_score
        }),
        "result_vector should be sorted by similarity"
    );

    println!("Test E/F result_vector is:");
    println!("{:#?}", result_vector);
}

#[test]
fn test_search_knn_exact_cases() {
    let cases = [
        ("empty_case", empty_case as fn() -> HnswTestCase),
        ("single_node_case", single_node_case as fn() -> HnswTestCase),
        (
            "unique_ranking_case",
            unique_ranking_case as fn() -> HnswTestCase,
        ),
        ("multilevel_case", multilevel_case as fn() -> HnswTestCase),
        (
            "identical_vectors_case",
            identical_vectors_case as fn() -> HnswTestCase,
        ),
        (
            "duplicate_distance_case",
            duplicate_distance_case as fn() -> HnswTestCase,
        ),
        (
            "collinear_points_case",
            collinear_points_case as fn() -> HnswTestCase,
        ),
        (
            "clustered_distribution_case",
            clustered_distribution_case as fn() -> HnswTestCase,
        ),
        (
            "high_dimension_case",
            high_dimension_case as fn() -> HnswTestCase,
        ),
        (
            "extreme_values_case",
            extreme_values_case as fn() -> HnswTestCase,
        ),
        (
            "near_zero_distance_case",
            near_zero_distance_case as fn() -> HnswTestCase,
        ),
        (
            "k_larger_than_dataset_case",
            k_larger_than_dataset_case as fn() -> HnswTestCase,
        ),
        (
            "tightly_packed_case",
            tightly_packed_case as fn() -> HnswTestCase,
        ),
        (
            "extreme_ef_search_case",
            extreme_ef_search_case as fn() -> HnswTestCase,
        ),
        (
            "negative_coordinates_case",
            negative_coordinates_case as fn() -> HnswTestCase,
        ),
        (
            "sparse_vectors_case",
            sparse_vectors_case as fn() -> HnswTestCase,
        ),
    ];

    let mut failures = Vec::new();

    for (case_name, case_fn) in cases {
        println!("running search_knn case: {}", case_name);
        if let Err(message) = run_search_knn_case(case_name, case_fn()) {
            eprintln!("case failed: {}", message);
            failures.push(message);
        }
    }

    assert!(
        failures.is_empty(),
        "search_knn exact cases failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn test_search_knn_multilevel_case() {
    assert_search_knn_case("multilevel_case", multilevel_case());
}

#[test]
fn test_insert_single_node_minimal() {
    // 单节点插入
    let mut index = HnswIndex::new(TEST_METRIC, 0.5);
    assert!(index.is_empty(), "初始图应为空");

    let id = 1u64;
    let data = vec![1.0, 2.0, 3.0];
    let ef = 10;
    let m = 2;
    let m_max = 4;

    index.insert(id, data, ef, m, m_max);

    assert_eq!(index.len(), 1, "插入一个点后，图大小应为1");

    assert!(index.get_entry_node_id().is_some(), "插入后应存在入口点");
    let entry_id = index.get_entry_node_id().expect("入口点不应为空");
    assert_eq!(entry_id, id, "入口点应为新插入的节点");

    assert!(
        index.get_node_by_id(id).is_some(),
        "新插入的节点应能被检索到"
    );

    println!("Test 1 passed: 单节点插入成功");
}

#[test]
fn test_insert_two_nodes_minimal() {
    // 两个节点至少要在某一层互相可达
    let mut index = HnswIndex::new(TEST_METRIC, 0.5);

    let id1 = 1u64;
    let data1 = vec![1.0, 0.0, 0.0];
    let id2 = 2u64;
    let data2 = vec![1.1, 0.1, 0.0];

    let ef = 10;
    let m = 2;
    let m_max = 4;

    index.insert(id1, data1, ef, m, m_max);
    assert_eq!(index.len(), 1, "第一次插入后大小应为1");

    index.insert(id2, data2, ef, m, m_max);
    assert_eq!(index.len(), 2, "第二次插入后大小应为2");

    let node1 = index.get_node_by_id(id1).expect("节点1不存在");
    let node2 = index.get_node_by_id(id2).expect("节点2不存在");

    let mut found_bidirectional = false;
    let max_level = std::cmp::max(*node1.get_node_max_level(), *node2.get_node_max_level());
    let empty_vec: Vec<u64> = Vec::new();

    for lvl in 0..=max_level {
        let neighbors1 = if lvl < node1.get_neighbors().len() {
            &node1.get_neighbors()[lvl]
        } else {
            &empty_vec
        };
        let neighbors2 = if lvl < node2.get_neighbors().len() {
            &node2.get_neighbors()[lvl]
        } else {
            &empty_vec
        };

        if neighbors1.contains(&id2) && neighbors2.contains(&id1) {
            found_bidirectional = true;
            println!("在第{}层找到双向连边", lvl);
            break;
        }
    }

    assert!(
        found_bidirectional,
        "两个节点之间至少应该在某一层有双向连边"
    );

    println!("Test 2 passed: 两个节点插入成功，双向连边存在");
}

#[test]
fn test_insert_multiple_nodes_connected() {
    // 多个节点插入后，图里不应出现孤点
    let mut index = HnswIndex::new(TEST_METRIC, 0.5);

    let nodes_data = vec![
        (1u64, vec![1.0, 0.0, 0.0]),
        (2u64, vec![1.0, 1.0, 0.0]),
        (3u64, vec![0.0, 1.0, 0.0]),
        (4u64, vec![0.0, 0.0, 1.0]),
    ];

    let ef = 10;
    let m = 2;
    let m_max = 4;

    for (id, data) in nodes_data.iter() {
        index.insert(*id, data.clone(), ef, m, m_max);
    }

    for (id, _) in nodes_data.iter() {
        assert!(index.get_node_by_id(*id).is_some(), "节点{} 应该存在", id);
    }

    for (id, _) in nodes_data.iter() {
        let node = index.get_node_by_id(*id).expect("节点不存在");
        let neighbors = node.get_neighbors();

        let has_neighbors = neighbors.iter().any(|layer| !layer.is_empty());
        assert!(has_neighbors, "节点{} 应该至少有一个邻边（不应该悬空）", id);
    }

    println!("Test 3 passed: 多个节点插入成功，没有悬空节点");
}

#[test]
fn test_tricky_identical_vectors() {
    let case = identical_vectors_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.len(),
        2,
        "identical_vectors should return exactly k results"
    );
    assert!(
        result_vector.iter().all(|id| *id <= 3),
        "identical_vectors should only return the tied best nodes"
    );
}

#[test]
fn test_tricky_duplicate_distance() {
    let case = duplicate_distance_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.as_slice(),
        &[1, 2, 3],
        "duplicate_distance should keep all equally ranked nodes in ID order"
    );
}

#[test]
fn test_tricky_collinear_points() {
    let case = collinear_points_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.as_slice(),
        &[1, 2],
        "collinear_points should rank the nearest directions first"
    );
}

#[test]
fn test_tricky_clustered_distribution() {
    let case = clustered_distribution_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.as_slice(),
        &[1, 2, 3],
        "clustered_distribution should stay inside the nearest cluster"
    );
}

#[test]
fn test_tricky_high_dimension() {
    let case = high_dimension_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "high_dimension should return the exact best match first"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}

#[test]
fn test_tricky_extreme_values() {
    let case = extreme_values_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "extreme_values should return the exact best match first"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}

#[test]
fn test_tricky_near_zero_distance() {
    let case = near_zero_distance_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "near_zero_distance should return the closest node first"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}

#[test]
fn test_tricky_k_larger_than_dataset() {
    let case = k_larger_than_dataset_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.len(),
        3,
        "k larger than the dataset should still return all available nodes"
    );
}

#[test]
fn test_tricky_tightly_packed() {
    let case = tightly_packed_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "tightly_packed should return the densest center first"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}

#[test]
fn test_tricky_extreme_ef_search() {
    let case = extreme_ef_search_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "extreme_ef_search should still find the best node"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}

#[test]
fn test_tricky_negative_coordinates() {
    let case = negative_coordinates_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "negative_coordinates should return the correct nearest node first"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}

#[test]
fn test_tricky_sparse_vectors() {
    let case = sparse_vectors_case();
    let index = build_index(case.nodes_data);
    let result_vector = index.search_knn(&case.query, case.k, case.ef_search);
    assert_eq!(
        result_vector.first().copied(),
        case.expected_result_id,
        "sparse_vectors should return the sparse center first"
    );
    assert!(
        result_vector.len() <= case.k,
        "result size should not exceed k"
    );
}
