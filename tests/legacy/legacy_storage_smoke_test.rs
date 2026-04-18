// Legacy storage smoke tests.
// 这份文件保留最早期的存储/flat search 烟雾测试，不属于当前主要关注的 SIFT1M 验证路径。

// #[test]
// fn it_runs() {
//     // 这里先不调真实逻辑，只要测试框架能跑就行
//     assert_eq!(2 + 2, 4);
// }

use toy_vector_db::search::flat::search_flat;
use toy_vector_db::storage::inmemory::InMemoryVectorStore;
use toy_vector_db::storage::VectorStore;

const LEGACY_STORAGE_FIXTURE_PATH: &str = "tests/legacy/fixtures/test_vectors.json";

#[test]
fn simple_insert_and_search() {
    let mut store = InMemoryVectorStore::new();

    store.insert(1, vec![1.0, 0.0]);
    store.insert(2, vec![0.0, 1.0]);
    store.insert(3, vec![1.0, 1.0]);

    let query = vec![1.0, 0.0];
    let results = search_flat(&store, &query, 2);

    assert_eq!(results.len(), 2);
    let ids: Vec<u64> = results.iter().map(|r| r.id).collect(); //r是闭包参数,类型是迭代器产生的类型.
                                                                //编译器的推断逻辑:
                                                                // map 需要一个闭包，闭包的参数类型由迭代器决定
                                                                // 迭代器产生 &SearchResult
                                                                // 所以 r 必定是 &SearchResult
                                                                // 编译器自动推断 r: &SearchResult ✅

    assert!(ids.contains(&1));
}

#[test]
fn test_save_and_load() {
    let mut store = InMemoryVectorStore::new();

    //插入数据
    store.insert(1, vec![1.0, 0.0]);
    store.insert(2, vec![0.0, 1.0]);
    store.insert(3, vec![1.0, 1.0]);

    //保存到文件
    store
        .save(LEGACY_STORAGE_FIXTURE_PATH)
        .expect("保存失败");

    //从文件加载
    // store.load 为什么这里不能这样写?
    let loaded_store =
        InMemoryVectorStore::load(LEGACY_STORAGE_FIXTURE_PATH).expect("加载失败");

    //验证数据是否正确
    let results = search_flat(&loaded_store, &vec![1.0, 0.0], 10);
    assert_eq!(results.len(), 3, "应该有3条记录");

    println!("Save and Load test passed!");
}
