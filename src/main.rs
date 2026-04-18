use azusa::{
    search,
    storage::{inmemory::InMemoryVectorStore, VectorStore},
};

fn main() {
    println!("========================================");
    println!("           Azusa -v0.1  (booting up)    ");
    println!("========================================");

    let mut store = InMemoryVectorStore::new();
    store.insert(1, vec![1.0, 2.0, 3.0, 4.0]);
    store.insert(2, vec![2.0, 3.0, 4.0, 5.0]);
    store.insert(3, vec![3.0, 4.0, 5.0, 6.0]);
    let result = search::flat::search_flat(&store, &vec![2.0, 3.0, 4.0, 5.0], 2);
    print!("{:?}", result)
}
