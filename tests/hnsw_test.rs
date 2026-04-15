mod datasets;

use datasets::*;
use std::{collections::HashSet, u64, usize, vec};
use toy_vector_db::{
    index::hnsw::{HnswIndex, HnswNode},
    types::Id,
};

///id: u64, data: Vec<f32>, lvl: usize
fn build_index_v0(nodes_data: Vec<(u64, Vec<f32>, usize)>) -> HnswIndex {
    let mut index = HnswIndex::new();

    // let nodes_data=[
    //     (1, vec![10.0, 0.0], 0),
    //     (2, vec![9.0, 1.0], 0),
    //     (3, vec![8.0, 2.0], 0),
    //     (4, vec![0.0, 10.0], 0),
    //     (5, vec![1.0 , 9.0], 0),
    //     (6, vec![2.0, 8.0], 0),
    // ];
    for (id, data, lvl) in nodes_data.iter() {
        index.insert(HnswNode::new(*id, data.to_vec(), *lvl));
    }
    return index;
}

#[test]
fn test_insert_two_nodes() {
    // 1. 创建 index
    let mut index = HnswIndex::new();
    // 2. 创建两个 node
    let node_1 = HnswNode::new(1, vec![1.0, 2.0, 3.0], 3);
    let node_2 = HnswNode::new(2, vec![2.0, 3.0, 4.0], 4);
    // 3. 插入
    index.insert(node_1);
    index.insert(node_2);

    
    // 4. 检查结果
    //println!("{:?}", index.get_nodes());
    assert_eq!(index.len(), 2, "index.len() is not 2");
    // 先拿到节点 → 再拿到某层邻居表 → 再检查里面是否包含某个 id
    let test_node_1 = index.get_node_by_id(1).expect("test_node_1 does not exist");
    assert!(
        test_node_1.get_neighbors()[0].contains(&2),
        "test_node_1 cannot view 2"
    );

    let test_node_2 = index.get_node_by_id(2).expect("test_node_2 does not exist");
    assert!(
        test_node_2.get_neighbors()[0].contains(&1),
        "test_node_2 cannot view 1"
    );
}

#[test]
fn test_insert_three_nodes() {
    // 1. 创建 index
    let mut index = HnswIndex::new();
    // 2. 创建三个 node
    let node_1 = HnswNode::new(1, vec![1.0, 2.0, 3.0], 3);
    let node_2 = HnswNode::new(2, vec![2.1, 3.1, 4.1], 4);
    let node_3 = HnswNode::new(3, vec![2.0, 3.0, 4.0], 4);
    // 3. 插入
    index.insert(node_1);
    index.insert(node_2);
    index.insert(node_3);
    // 4. 检查结果
    println!("{:#?}", index.get_nodes());

    assert_eq!(index.len(), 3, "index.len() is not 3");
    // 先拿到节点 → 再拿到某层邻居表 → 再检查里面是否包含某个 id
    let test_lvl = 1;
    let test_node_1 = index.get_node_by_id(1).expect("test_node_1 does not exist");
    assert!(
        test_node_1.get_neighbors()[test_lvl].contains(&2),
        "test_node_1 cannot view 2"
    );

    let test_node_2 = index.get_node_by_id(2).expect("test_node_2 does not exist");
    assert!(
        test_node_2.get_neighbors()[test_lvl].contains(&1),
        "test_node_2 cannot view 1"
    );
    assert!(
        test_node_2.get_neighbors()[test_lvl].contains(&3),
        "test_node_2 cannot view 3"
    );

    let test_node_3 = index.get_node_by_id(3).expect("test_node_3 does not exist");
    assert!(
        test_node_3.get_neighbors()[test_lvl].contains(&2),
        "test_node_3 cannot view 2"
    );
}

#[test]
fn test_search_v0() {
    // 1. 创建 index
    let mut index = HnswIndex::new();

    // 2. 创建node并插入.
    let nodes_data = [
        (1, vec![10.0, 0.0], 2),
        (2, vec![9.0, 1.0], 2),
        (3, vec![8.0, 2.0], 2),
        (4, vec![0.0, 10.0], 2),
        (5, vec![1.0, 9.0], 2),
        (6, vec![2.0, 8.0], 2),
    ];
    for (id, data, lvl) in nodes_data.iter() {
        index.insert(HnswNode::new(*id, data.to_vec(), *lvl));
    }

    let test_query = vec![6.0, 6.0];

    // 4. 检查结果(之前的部分)
    println!("{:#?}", index.get_nodes());

    // assert_eq!(index.len(), 3, "index.len() is not 3");
    // // 先拿到节点 → 再拿到某层邻居表 → 再检查里面是否包含某个 id
    // let test_lvl = 1;
    // let test_node_1 = index.get_node_by_id(1).expect("test_node_1 does not exist");
    // assert!(test_node_1.get_neighbors()[test_lvl].contains(&2), "test_node_1 cannot view 2");

    // let test_node_2 = index.get_node_by_id(2).expect("test_node_2 does not exist");
    // assert!(test_node_2.get_neighbors()[test_lvl].contains(&1), "test_node_2 cannot view 1");
    // assert!(test_node_2.get_neighbors()[test_lvl].contains(&3), "test_node_2 cannot view 3");

    // let test_node_3 = index.get_node_by_id(3).expect("test_node_3 does not exist");
    // assert!(test_node_3.get_neighbors()[test_lvl].contains(&2), "test_node_3 cannot view 2");

    // 5.检查search_v0的搜索结果.
    if let Some(result_id) = index.search_v0(&test_query) {
        if let Some(result_node) = index.get_node_by_id(result_id) {
            println!("The result is:");
            println!("{:#?}", result_node.get_id());
            println!("{:#?}", result_node.get_data());
        }
    }
}



#[test]
fn test_nearest_and_furthest() {
    // 1. 创建 index
    let mut index = HnswIndex::new();
    // 1.1 创建set
    let mut set: HashSet<Id> = HashSet::new();

    // 2. 创建node并插入.
    let nodes_data = [
        (1, vec![10.0, 0.0], 2),
        (2, vec![9.0, 1.0], 2),
        (3, vec![8.0, 2.0], 2),
        (4, vec![0.0, 10.0], 2),
        (5, vec![1.0, 9.0], 2),
        (6, vec![2.0, 8.0], 2),
    ];
    for (id, data, lvl) in nodes_data.iter() {
        index.insert(HnswNode::new(*id, data.to_vec(), *lvl));
        set.insert(*id);
    }

    let test_query = vec![6.0, 6.0];

    // 5.检查搜索结果.
    if let Some(result_id) = index.get_nearest_v0(&set, &test_query) {
        assert!(result_id == 3 || result_id == 6);
        if let Some(result_node) = index.get_node_by_id(result_id) {
            println!("The nearest result is:");
            println!("{:#?}", result_node.get_id());
            println!("{:#?}", result_node.get_data());
        }
    } else {
        panic!("get_nearest方法返回None")
    }

    if let Some(result_id) = index.get_furthest_v0(&set, &test_query) {
        assert!(result_id == 1 || result_id == 4);
        if let Some(result_node) = index.get_node_by_id(result_id) {
            println!("The furthest result is:");
            println!("{:#?}", result_node.get_id());
            println!("{:#?}", result_node.get_data());
        }
    } else {
        panic!("get_furthest方法返回None")
    }
}

#[test]
fn test_search_layer_v0() {
    // 1. 创建 index
    let mut index = HnswIndex::new();
    // 1.1 创建set
    let mut set: HashSet<Id> = HashSet::new();

    // 2. 创建node并插入.
    let nodes_data = [
        (1, vec![10.0, 0.0], 0),
        (2, vec![9.0, 1.0], 0),
        (3, vec![8.0, 2.0], 0),
        (4, vec![0.0, 10.0], 0),
        (5, vec![1.0, 9.0], 0),
        (6, vec![2.0, 8.0], 0),
    ];
    for (id, data, lvl) in nodes_data.iter() {
        index.insert(HnswNode::new(*id, data.to_vec(), *lvl));
        set.insert(*id);
    }

    let test_query = vec![6.0, 6.0];
    let result_set = index.search_layer_v0(&test_query, 2, 0, 3);
    let mut result_vec: Vec<_> = result_set.iter().cloned().collect();
    result_vec.sort();

    println!("{:#?}", result_vec);

    assert!(!result_set.is_empty());
    assert!(result_set.len() <= 3);
}

#[test]
fn test_greedy_search_at_level_old() {
    //let nodes_data = generate_nodes_data();

    //Test A
    //构造一个简单单层图，使得从某个入口点出发可以沿邻居不断接近查询点。
    //检查返回值是否为最终停止节点。
    let nodes_data = vec![
        (1, vec![10.0, 0.0], 0),
        (2, vec![9.0, 1.0], 0),
        (3, vec![8.0, 2.0], 0),
        (4, vec![0.0, 10.0], 0),
        (5, vec![1.0, 9.0], 0),
        (6, vec![2.0, 8.0], 0),
    ];
    let index = build_index_v0(nodes_data);
    let test_query = vec![6.0, 6.0];
    let result_id = index.greedy_search_at_level(&test_query, 1, 0);
    println!("Test A result_id is:");
    println!("{:#?}", result_id);

    //TestB
    // 构造一个节点在某层没有任何更优邻居的情形。
    // 检查函数是否直接返回入口点本身。
    let nodes_data = vec![
        (1, vec![10.0, 0.0], 0),
        (2, vec![9.0, 1.0], 0),
        (3, vec![8.0, 2.0], 0),
        (4, vec![0.0, 10.0], 0),
        (5, vec![1.0, 9.0], 0),
        (6, vec![2.0, 8.0], 0),
    ];
    let index = build_index_v0(nodes_data);
    let test_query = vec![8.1, 2.1];
    let result_id = index.greedy_search_at_level(&test_query, 1, 0);
    println!("Test B result_id is:");
    println!("{:#?}", result_id);
}

#[test]
fn test_greedy_search_at_level_greedy_chain_case() {
    //let nodes_data = generate_nodes_data();

    //Test A
    //构造一个简单单层图，使得从某个入口点出发可以沿邻居不断接近查询点。
    //检查返回值是否为最终停止节点。

    let case = greedy_chain_case();
    let index = build_index_v0(case.nodes_data);
    let test_query = case.query;
    
    let result_id = index.greedy_search_at_level(&test_query, 1, 0);
    let expected_result_id = case
        .expected_result_id
        .expect("greedy_chain_case should define expected_result_id");
    assert_eq!(result_id, expected_result_id, 
        "test_greedy_search_at_level, greedy_chain_case: expected {}, got {}", expected_result_id, result_id);
    println!("Test A result_id is:");
    println!("{:#?}", result_id);

}

#[test]
fn test_greedy_search_at_level_beginning_is_the_best_case() {
    //TestB
    // 构造一个节点在某层没有任何更优邻居的情形。
    // 检查函数是否直接返回入口点本身。

    let case = beginning_is_the_best_case();
    let index = build_index_v0(case.nodes_data);
    let test_query = case.query;
    let result_id = index.greedy_search_at_level(&test_query, 1, 0);
    let expected_result_id = case
        .expected_result_id
        .expect("beginning_is_the_best_case should define expected_result_id");
    assert_eq!(result_id, expected_result_id,
        "test_greedy_search_at_level,beginning_is_the_best_case: expected {}, got {}", expected_result_id, result_id);
    
    println!("Test B result_id is:");
    println!("{:#?}", result_id);
}


#[test]
fn test_search_knn_v1_old() {
    // 空图
    let nodes_data: Vec<(u64, Vec<f32>, usize)> = vec![];
    let index = build_index_v0(nodes_data);
    let test_query = vec![8.1, 2.1];
    println!("Test C get_entry_node_id is:");
    println!("{:#?}", index.get_entry_node_id());
    let result_vector = index.search_knn_v1(&test_query, 3, 3);
    println!("Test C result_vector is:");
    println!("{:#?}", result_vector);

    // 单节点
    let nodes_data: Vec<(u64, Vec<f32>, usize)> = vec![(1, vec![10.0, 0.0], 0)];
    let index = build_index_v0(nodes_data);
    let result_vector = index.search_knn_v1(&test_query, 3, 3);
    println!("Test D result_vector is:");
    println!("{:#?}", result_vector);

    // 结果数量与排序
    let nodes_data: Vec<(u64, Vec<f32>, usize)> = vec![
        (1, vec![10.0, 0.0], 3),
        (2, vec![9.0, 1.0], 1),
        (3, vec![8.0, 2.0], 4),
        (4, vec![0.0, 10.0], 3),
        (5, vec![1.0, 9.0], 2),
        (6, vec![2.0, 8.0], 0),
    ];
    let index = build_index_v0(nodes_data);
    let result_vector = index.search_knn_v1(&test_query, 3, 3);
    println!("Test E/F result_vector is:");
    println!("{:#?}", result_vector);
}

