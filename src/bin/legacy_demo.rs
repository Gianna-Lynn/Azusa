//! 最小化的 legacy 演示二进制，用于验证旧版 flat search 路径。
use azusa::{
    legacy::{search::flat::search_flat, storage::inmemory::InMemoryVectorStore},
};

fn main() {
    println!("========================================");
    println!("      Azusa Legacy Demo (booting up)    ");
    println!("========================================");

    let mut store = InMemoryVectorStore::new();
    store.insert(1, vec![1.0, 2.0, 3.0, 4.0]);
    store.insert(2, vec![2.0, 3.0, 4.0, 5.0]);
    store.insert(3, vec![3.0, 4.0, 5.0, 6.0]);
    let result = search_flat(&store, &vec![2.0, 3.0, 4.0, 5.0], 2);
    println!("Flat search result: {:?}", result)
}
