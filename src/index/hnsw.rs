use core::f32;
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::{StdRng};
use std::collections::{HashMap, HashSet, BinaryHeap};
use crate::distance;
use crate::types::{Id, Metric, Vector};

#[path = "hnsw_legacy.rs"]
mod hnsw_legacy;
#[path = "hnsw_experiment.rs"]
mod hnsw_experiment;

const MY_SEED : u64 = 42;

// 统计: 插入和搜索过程中的时间开销, 方便定位性能热点.
#[derive(Debug, Default)]
pub struct SearchLayerStats {
    pub compute_sim_ns: u128,
    pub furthest_scan_ns: u128,
    pub neighbors_fetch_ns: u128,
    pub heap_pop_ns: u128,
    pub candidate_bookkeeping_ns: u128,
    pub loop_iterations: u64,
}

#[derive(Debug, Default)]
pub struct InsertTimings {
    pub greedy_search_ns: u128,
    pub search_layer_ns: u128,
    pub select_neighbors_ns: u128,
    pub pruning_ns: u128,
    pub link_ns: u128,
    pub calls: u64,
    pub search_layer_compute_sim_ns: u128,
    pub search_layer_furthest_scan_ns: u128,
    pub search_layer_neighbors_fetch_ns: u128,
    pub search_layer_heap_pop_ns: u128,
    pub search_layer_bookkeeping_ns: u128,
    pub search_layer_loop_iterations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapAccessMode {
    PeekPruned,
    PopAlive,
}

// 结构: HNSW 主索引结构, 负责保存图、入口点、层高和统计信息.
#[derive(Debug)]
pub struct HnswIndex {
    nodes: Vec<HnswNode>, //保存当前图中所有节点
    //entry_point: HnswNode, //因为一开始有可能是空的,没有接入点,是None.
    entry_node_id: Option<Id>, //保存入口节点Id
    //max_level: u64
    index_max_level: usize, //保存图中最高层数
    metric: Metric,
    id_to_index: HashMap<Id, usize>, //负责给出这个 id 对应在 nodes 里的第几个位置
    rng: StdRng,
    timing: InsertTimings,
}
// 结构: 图中的单个节点, 保存 id、向量本体、层高和每层邻居表.
#[derive(Debug, Clone)]
pub struct HnswNode {
    id: Id,
    //data: Vec<Record>, //不能这么写.一个HNSW节点只对应一条向量记录,也就是一个Record, 不要再建立一个Vec<Record>
    data: Vector,
    //level: u64,
    node_max_level: usize, // level表示该节点可到达的最高层编号.例如,如果level=3, 则该节点在0,1,2,3层都有邻居.如果level=0, 则该节点只有在第0层有邻居.
    //neighbors: Vec<Id> // 不能这么写,因为HNSW是分层图.
    neighbors: Vec<Vec<Id>>, //neighbors[i]：第 i 层的邻居
}

impl HnswNode {
    // 避坑: 初始化节点时, 需要按层创建空邻居表, 避免后续插边越界.
    pub fn new(id: Id, data: Vector, level: usize) -> Self {
        // self.neighbors = Vec::new();
        // for (int i = 0; i < self.level; i++){
        //     new_vec = Vec::new();
        //     self.neighbors.push(new_vec);
        // }
        let mut neighbors = Vec::new();
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }

        Self {
            id,
            data,
            node_max_level: level,
            neighbors,
        }
    }

    // 连边: 给某个节点的某一层添加一个邻居 id, 用于局部连边.
    pub fn add_neighbor(&mut self, neighbor_id: Id, level_to_add: usize) {
        //self.neighbors[neighbor_lvl].append(neighbor_id); //append是拿来“把一个 Vec 里的所有元素接到另一个 Vec 后面”的。它要的不是一个单独的 Id，而是一个 Vec<Id>。
        if level_to_add < self.neighbors.len() {
            self.neighbors[level_to_add].push(neighbor_id);
        }
    }

    // 剪枝: 从当前节点的所有层里删除某个邻居, 用于剪枝时的单向清理.
    pub fn delete_neighbor_single_direction_at_all_levels(&mut self, neighbor_id: Id){
        // pub struct HnswNode {
        //     id: Id,
        //     //data: Vec<Record>, //不能这么写.一个HNSW节点只对应一条向量记录,也就是一个Record, 不要再建立一个Vec<Record>
        //     data: Vector,
        //     //level: u64,
        //     node_max_level: usize, // level表示该节点可到达的最高层编号.例如,如果level=3, 则该节点在0,1,2,3层都有邻居.如果level=0, 则该节点只有在第0层有邻居.
        //     //neighbors: Vec<Id> // 不能这么写,因为HNSW是分层图.
        //     neighbors: Vec<Vec<Id>>, //neighbors[i]：第 i 层的邻居
        // }
        for i in 0..=self.node_max_level{
            // find neighbor_id in neighbors[i] and delete it
            // 单方面剪枝.
            self.neighbors[i].retain(|&id| id != neighbor_id);
            // retain()保留id不等于neighbor_id的元素.
        }
    }

    // 剪枝: 从当前节点的指定层删除某个邻居, 用于局部剪枝.
    pub fn delete_neighbor_single_direction_at_level(&mut self, neighbor_id: Id, level: usize){
        self.neighbors[level].retain(|&id| id != neighbor_id);
    }

    pub fn get_id(&self) -> &Id {
        &self.id
    }

    pub fn get_data(&self) -> &Vector {
        &self.data
    }

    pub fn get_node_max_level(&self) -> &usize{
        &self.node_max_level

    }

    pub fn get_neighbors(&self) -> &Vec<Vec<u64>> {
        &self.neighbors
    }

}

impl HnswIndex {
    // 初始化: 创建默认索引实例, 使用固定随机种子确保构图可复现.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_id: None,
            index_max_level: 0,
            metric: Metric::Cosine, //默认使用Cosine
            id_to_index: HashMap::new(),
            rng : StdRng::seed_from_u64(MY_SEED),
            timing: InsertTimings::default(),
        }
    }

    // 初始化: 创建指定 metric 的索引实例, 方便在 Cosine 和 L2 之间切换.
    pub fn new_with_metric(metric: Metric) -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_id: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng : StdRng::seed_from_u64(MY_SEED),
            timing: InsertTimings::default(),
        }
    }

    // 统计: 打印插入阶段的计时拆分, 用来判断瓶颈落在哪一步.
    pub fn print_timings(&self) {
        let total = self.timing.search_layer_ns
            + self.timing.greedy_search_ns
            + self.timing.select_neighbors_ns
            + self.timing.pruning_ns
            + self.timing.link_ns;

        if total == 0 {
            println!("(no timings recorded yet)");
            return;
        }

        println!("\n=== insert_v1 timings ({} inserts) ===", self.timing.calls);
        println!("  greedy_search:     {:.2}s ({:.1}%)",
            self.timing.greedy_search_ns as f64 / 1e9,
            self.timing.greedy_search_ns as f64 / total as f64 * 100.0);
        println!("  search_layer_v0:   {:.2}s ({:.1}%)",
            self.timing.search_layer_ns as f64 / 1e9,
            self.timing.search_layer_ns as f64 / total as f64 * 100.0);
        if self.timing.search_layer_loop_iterations > 0 {
            let search_layer_total = self.timing.search_layer_ns as f64;
            let avg_sim_time_per_iter = self.timing.search_layer_compute_sim_ns as f64 / self.timing.search_layer_loop_iterations as f64;
            println!("    ├─ compute_sim:   {:.2}s ({:.1}% of search_layer)",
                self.timing.search_layer_compute_sim_ns as f64 / 1e9,
                self.timing.search_layer_compute_sim_ns as f64 / search_layer_total * 100.0);
            println!("    ├─ furthest_scan:  {:.2}s ({:.1}% of search_layer)",
                self.timing.search_layer_furthest_scan_ns as f64 / 1e9,
                self.timing.search_layer_furthest_scan_ns as f64 / search_layer_total * 100.0);
            println!("    ├─ neighbors_get:  {:.2}s ({:.1}% of search_layer)",
                self.timing.search_layer_neighbors_fetch_ns as f64 / 1e9,
                self.timing.search_layer_neighbors_fetch_ns as f64 / search_layer_total * 100.0);
            println!("    ├─ heap_pop:       {:.2}s ({:.1}% of search_layer)",
                self.timing.search_layer_heap_pop_ns as f64 / 1e9,
                self.timing.search_layer_heap_pop_ns as f64 / search_layer_total * 100.0);
            println!("    ├─ bookkeeping:    {:.2}s ({:.1}% of search_layer)",
                self.timing.search_layer_bookkeeping_ns as f64 / 1e9,
                self.timing.search_layer_bookkeeping_ns as f64 / search_layer_total * 100.0);
            println!("    └─ iterations:   {} ({:.3}ms/iter)",
                self.timing.search_layer_loop_iterations,
                avg_sim_time_per_iter / 1e6);
        }
        println!("  select_neighbors:  {:.2}s ({:.1}%)",
            self.timing.select_neighbors_ns as f64 / 1e9,
            self.timing.select_neighbors_ns as f64 / total as f64 * 100.0);
        println!("  pruning:           {:.2}s ({:.1}%)",
            self.timing.pruning_ns as f64 / 1e9,
            self.timing.pruning_ns as f64 / total as f64 * 100.0);
        println!("  link (add edges):  {:.2}s ({:.1}%)",
            self.timing.link_ns as f64 / 1e9,
            self.timing.link_ns as f64 / total as f64 * 100.0);
        println!("  total:             {:.2}s\n", total as f64 / 1e9);
    }

    // 统一: 统一相似度计算入口, 保证索引内部所有比较都走同一 metric.
    pub fn compute_similarity(&self, a: &Vector, b:&Vector)->f32{
        distance::similarity(a, b, self.metric)
    }

    pub fn len(&self) -> usize {
        // return nodes.len(); // 不可以这么写,因为这个方法是面向当前对象的,必须有self
        return self.nodes.len();
    }

    pub fn is_empty(&self) -> bool {
        // return nodes.empty();
        return self.nodes.is_empty();
    }

    // 基线: 早期插入逻辑, 现在更多是历史/对照用途, 保留供参考.
    pub fn insert(&mut self, mut node_to_insert: HnswNode) {
        if self.is_empty() {
            self.entry_node_id = Some(node_to_insert.id); // 不清楚为什么要加这个Some()
                                                          //entry_point的类型是Option<Id>,表示没有值None, 要么有值Some(id)
                                                          // //node.neighbors = Empty;             //没有Empty这个写法,一开始, node也不是mut node,默认不能修改它的字段.
                                                          // //node.neighbors = Vec::new();          //空的邻居表应该是空的vector

            // node.neighbors = vec![Vec::new(); node.level];  //创建一个长度为node.level的Vec<Vec<Id>>,每个元素都是一个空的Vec<Id>
            //                                                 //neighbors.len()==level+1
            // // vec![value; count] 是 Rust 中创建一个包含 count 个 value 的 Vec
            // //这一部分删除了,因为在HnswNode::new()中已经创建好了neighbors了. 这里不需要再创建一次了.

            self.index_max_level = node_to_insert.node_max_level;
        }
        // self.nodes.append(node);    //Vec::append需要另一个Vec. 这不是"把单个元素放进Vector"的方法.
        else {
            //else分支: 如果图非空
            let mut max_similarity: f32 = f32::NEG_INFINITY;
            let mut max_idx: usize = 0;
            for (idx, old_node) in self.nodes.iter().enumerate() {
                let temp_similarity =
                    self.compute_similarity(old_node.get_data(), node_to_insert.get_data());
                if temp_similarity > max_similarity {
                    max_similarity = temp_similarity;
                    max_idx = idx;
                }
                //max_similarity = temp_similarity.max(max_similarity);   //ご注意ください:这里max的用法用于比较两个f32
            } //endfor: 得到了和node_to_insert最近的一个已有节点在self.nodes中的下标max_idx
              // self.nodes[max_idx].add_neighbor(node_to_insert.id,node_to_insert.node_max_level);
              // node_to_insert.add_neighbor( self.nodes[max_idx].id,  self.get_nodes()[max_idx].node_max_level);

            //如果新来的节点层次更高, 要修改整个图的最高层
            if self.index_max_level < node_to_insert.node_max_level {
                self.index_max_level = node_to_insert.node_max_level;
                self.entry_node_id = Some(node_to_insert.id);
            }

            let lvls = std::cmp::min(
                self.nodes[max_idx].node_max_level,
                node_to_insert.node_max_level,
            );

            for i in 0..=lvls {
                self.nodes[max_idx].add_neighbor(node_to_insert.id, i);
                node_to_insert.add_neighbor(self.nodes[max_idx].id, i);
            }
        }
        self.nodes.push(node_to_insert);
    }

    pub fn find_index_of_id_in_neighbors(target_id: Id, target_vector: &Vec<Id>) -> Option<usize> {
        //根据某个节点的id寻找节点在一个邻居数组中的下标

        /*
        let mut idx: u64 = 0;
        for x in target_vector.iter(){
            if  *x == target_id {
                return idx as u64;
            }
            idx += 1;
        }
        return idx as u64;
        */

        //利用enumerate可以同时获取元素的值和下标,不需要手动维护一个idx变量了.
        //返回的序列形如((0, &x0), (1, &x1), (2, &x2), ...), 其中0,1,2是下标, &x0, &x1, &x2是元素的引用.
        for (idx, &x) in target_vector.iter().enumerate() {
            if x == target_id {
                return Some(idx);
            }
        }
        None
    }

    //而“按 id 找节点”是在做：
    //在 self.nodes: Vec<HnswNode> 里，找到 node.id == target_id 的那个节点
    pub fn get_node_by_id(&self, target_id: Id) -> Option<&HnswNode> {
        // for x in self.nodes.iter() {
        //     if x.id == target_id {
        //         return Some(x);
        //     }
        // }
        // None

        // 复习: self.id_to_index[&target_id] 直接这么写, 意思是“我认定这里一定有，没找到就直接炸。”
        //       self.id_to_index.get(&target_id), 意思是“帮我查一下，有就给我，没有就返回 None。”
        if let Some(idx) = self.id_to_index.get(&target_id){
            Some(&self.nodes[*idx])
        } else{
            None
        }

    }

    //“按 id 找到节点然后修改它”。
    pub fn get_mut_node_by_id(&mut self, target_id: Id) -> Option<&mut HnswNode> {
        //因为未来想要要修改节点,所以需要&mut self,返回值也是&mut HnswNode
        //想要返回一个可变引用,就必须使用iter_mut()方法来遍历self.nodes,而不是iter()方法.
        //iter_mut()会返回一个可变引用的迭代器,这样我们就可以修改迭代器中的元素了.
        //find()方法会返回一个Option<&mut HnswNode>,如果找到了满足条件的元素,
        //就返回Some(&mut HnswNode),否则返回None.
        // self.nodes.iter_mut().find(|x| x.id == target_id)

        if let Some(idx) = self.id_to_index.get(&target_id){
            Some(&mut self.nodes[*idx])
        } else{
            None
        }

    }

    //把编号为neighbor_id的节点添加到node_id的第level层的邻居Vec中
    pub fn add_neighbor_to_node_at_level(&mut self, node_id:Id, neighbor_id:Id, level: usize){
        // let neighbor_node = self.get_node_by_id(neighbor_id)
        //     .expect("in add_neighbor_to_node, in get_mut_node_by_id, got None");
        // //下面这地方只需要值, 不需要引用, 所以都加上*解引用, 只要值.
        // let neighbor_node_id_value = *neighbor_node.get_id();
        // 改进: 实际上根本不需要解引用, 因为neighbor_node_id_value就是函数参数中的neighbor_node, 本来就有了, 白绕一大圈
        
        let node = self.get_mut_node_by_id(node_id)
            .expect("in add_neighbor_to_node, in get_mut_node_by_id, got None");
        node.add_neighbor(neighbor_id, level);
    }

    pub fn get_nodes(&self) -> &Vec<HnswNode> {
        &self.nodes
    }

    pub fn get_entry_node_id(&self) -> Option<u64> {
        self.entry_node_id
    }

    // 堆顶: 堆顶访问工具, 用于在搜索层里快速取“当前最优/最差”节点并跳过失效项。
    // - 方法: 基于 BinaryHeap 的堆顶访问，而不是线性扫描整个集合
    // - 数据来源: 依赖外部维护好的 heap + alive_set
    // - mode:
    //   - PeekPruned : 查看当前有效堆顶，并顺手清理失效堆顶
    //   - PopAlive   : 弹出并返回当前有效堆顶，同时跳过失效元素
    //
    // Ord 约束表示 T 必须能进行稳定的大小比较，这样 BinaryHeap 才能正确维护堆顶顺序。
    fn get_heap_extreme<T: Ord>(
        &self,
        alive_set: &HashSet<Id>,
        heap: &mut BinaryHeap<(T, Id)>,
        mode: HeapAccessMode,
        ) -> Option<Id> {

        match mode {
            HeapAccessMode::PeekPruned => {
                while let Some(top) = heap.peek() {
                    if alive_set.contains(&top.1) {
                        return Some(top.1);
                    }
                    heap.pop();
                }
                None
            }
            HeapAccessMode::PopAlive => {
                while let Some(top) = heap.pop() {
                    if !alive_set.contains(&top.1){
                        continue;
                    }
                    return Some(top.1);
                }
                None
            }
        }
    }

    // 邻居: 读取某个节点在指定层的邻居集合, 供搜索与插边复用.
    pub fn get_neighbors_at_lvl(&self, id: Id, lvl: usize) -> HashSet<Id> {
        //ご注意ください:如果你用的是一个不带有&self的静态方法,方法不能访问self.nodes,不能访问索引内部的图结构,只能依靠传入的参数工作.
        //因此,必须访问self.
        //这样一来就可以用self.get_neighbors_at_lvl(id, lvl), 而不是更麻烦的HnswIndex::get_neighbors_at_lvl(&self, id, lvl)
        //unimplemented!()
        // 避坑: index_max_level仅指代整张图最高能有多少层.而单个节点拥有的层数可能小于最大层数.
        //                 所以这里if条件不应该是if lvl > self.index_max_level
        let node_to_search = self.get_node_by_id(id).expect("get_node_by_id不存在");

        //工程问题: 现在node_max_level并没有明确的更新机制,暂时不要用这个字段行事.
        // if lvl > node_to_search.node_max_level{
        //还有一个小问题,Rust的下标从0开始, 所以这个地方应该是>=而不是>
        //例如,get_neighbors().len()==3的时候,合法下标只有0,1,2,并不包括3.
        if lvl >= node_to_search.get_neighbors().len() {
            //return None
            //语法问题:返回一个空集合,不要返回None,而是返回HashSet::new()
            return HashSet::new();
        }
        let neighbors_set: HashSet<Id> = node_to_search.get_neighbors()[lvl]
            .iter()
            .cloned()
            .collect();
        neighbors_set
    }

    // 缓存: 输入一个cache数组<向量Id, 向量和query的相似度>, 如果目标向量的相似度值在cache中,直接返回,否则手动计算,更新cache
    pub fn get_similarity(&self, id_similarity_cache: &mut HashMap<Id, f32>, target_id: &Id, query: &Vector)->f32{
        if let Some(similarity) = id_similarity_cache.get(target_id){
            //cache命中
            *similarity
        }
        else{
            //cache不命中
            let target_node = self.get_node_by_id(*target_id).expect("In function get_similarity, failed to get target_node by target_id");
            let new_similarity = self.compute_similarity(target_node.get_data(), query);
            id_similarity_cache.insert(*target_id, new_similarity);
            new_similarity
        }
    }

    // 搜索: 输出在level这一层, 距离查询向量query最近的一批向量, 总共ef个.
    pub fn search_layer_v0(
        &self,
        query: &Vector,
        entry_id: Id,
        level: usize,
        ef: usize,
        id_similarity_cache: &mut HashMap<Id, f32>,
    ) -> (HashSet<Id>, SearchLayerStats) {
        assert!(ef >= 1, "ef至少为1");

        let mut stats = SearchLayerStats::default();

        // 避坑: 已经访问过的节点集合v
        let mut v_set: HashSet<Id> = HashSet::new();
        // 避坑: 候选集合C (大地图)
        let mut c_set: HashSet<Id> = HashSet::new();
        let mut c_heap: BinaryHeap<(ordered_float::NotNan<f32>, Id)> = BinaryHeap::new();
        // 避坑: 目前找到的最近点集合W
        let mut w_set: HashSet<Id> = HashSet::new();
        let mut w_heap: BinaryHeap<(std::cmp::Reverse<ordered_float::NotNan<f32>>, Id)> = BinaryHeap::new();

        //ご注意ください:初始化时要把entry_id放进这三个集合,否则一开始集合为空,后面的逻辑都不会执行.
        v_set.insert(entry_id);
        c_set.insert(entry_id);

        // 初始化 c_heap：计算 entry_id 的相似度并推入堆
        let entry_similarity = self.get_similarity(id_similarity_cache, &entry_id, query);
        c_heap.push((ordered_float::NotNan::new(entry_similarity).unwrap(), entry_id));

        w_set.insert(entry_id);
        w_heap.push((std::cmp::Reverse(ordered_float::NotNan::new(entry_similarity).unwrap()), entry_id));

        while !c_set.is_empty() {
            stats.loop_iterations += 1;
            // 避坑: get方法是查看, 要完成"拿出来+删掉", 就需要一个額外的删除语句: .remove
            let t = Instant::now();
            let c_id = self
                .get_heap_extreme(&c_set, &mut c_heap, HeapAccessMode::PopAlive)
                .expect("c_id不存在");
            stats.heap_pop_ns += t.elapsed().as_nanos();
            let t = Instant::now();
            c_set.remove(&c_id);
            stats.candidate_bookkeeping_ns += t.elapsed().as_nanos();
            // 避坑: get方法是查看, 要完成"拿出来读取", 而不是删掉.
            let t = Instant::now();
            let mut f_id = self.get_heap_extreme(&w_set, &mut w_heap, HeapAccessMode::PeekPruned).expect("f_id不存在");
            stats.furthest_scan_ns += t.elapsed().as_nanos();
            // 避坑: similarity的值: 两个向量距离越近, similarity的值越大.不要写反了.
            let t = Instant::now();
            let c_similarity = self.get_similarity(id_similarity_cache, &c_id, query);
            let f_similarity = self.get_similarity(id_similarity_cache, &f_id, query);
            let should_break = c_similarity < f_similarity;
            stats.compute_sim_ns += t.elapsed().as_nanos();
            if should_break {
                break;
            }

            let t = Instant::now();
            let c_neighbors_set_at_lvl = self.get_neighbors_at_lvl(c_id, level);
            stats.neighbors_fetch_ns += t.elapsed().as_nanos();
            // 避坑: 此处c_neighbors_set_at_lvl是Option<HashSet<u64>>.
            // 如果直接接上for e_id in c_neighbors_set_at_lvl开始迭代Option<HashSet<u64>>,
            // 那么最多循环1次(当Option的类型是Some),因为Option最多只有1个元素.
            // 两种修改方法,一是match语句,二是.expect("msg")
            for e_id in c_neighbors_set_at_lvl {
                if v_set.contains(&e_id) {
                    continue;
                }
                v_set.insert(e_id);

                let t = Instant::now();
                f_id = self.get_heap_extreme(&w_set, &mut w_heap, HeapAccessMode::PeekPruned).expect("f_id不存在");
                stats.furthest_scan_ns += t.elapsed().as_nanos();
        
                // 避坑: similarity的值: 两个向量距离越近, similarity的值越大.不要写反了.
                let t = Instant::now();
                let e_similarity = self.get_similarity(id_similarity_cache, &e_id, query);
                let f_similarity = self.get_similarity(id_similarity_cache, &f_id, query);
                let should_add = e_similarity >= f_similarity || w_set.len() < ef;
                stats.compute_sim_ns += t.elapsed().as_nanos();
                if should_add {
                    let t = Instant::now();
                    c_set.insert(e_id);
                    // 计算相似度并推入 c_heap / w_heap
                    c_heap.push((ordered_float::NotNan::new(e_similarity).unwrap(), e_id));
                    w_set.insert(e_id);
                    w_heap.push((std::cmp::Reverse(ordered_float::NotNan::new(e_similarity).unwrap()), e_id));
                    stats.candidate_bookkeeping_ns += t.elapsed().as_nanos();
                    if w_set.len() > ef {
                        let t = Instant::now();
                        let id_to_remove_op = self.get_heap_extreme(&w_set, &mut w_heap, HeapAccessMode::PeekPruned);
                        stats.furthest_scan_ns += t.elapsed().as_nanos();
                        if let Some(id_to_remove) = id_to_remove_op {
                            let t = Instant::now();
                            w_set.remove(&id_to_remove);
                            stats.candidate_bookkeeping_ns += t.elapsed().as_nanos();
                        }
                    }
                } //endif
            } //endfor
        }
        (w_set, stats)
    }

    // 搜索: 在单层里做贪心下降, 给上层搜索提供更接近 query 的入口点.
    pub fn greedy_search_at_level(&self, query: &Vector, entry_id: Id, level: usize) -> Id {
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        
        //在该层经过贪心搜索后得到的最终节点 id. 区分于search_layer_v0(返回的是一个set)
        let mut current_node = self
            .get_node_by_id(entry_id)
            .expect("未能根据entry_id检索到entry_node");
        let mut current_node_neighbors = &current_node.get_neighbors()[level];
        let mut max_similarity = self.get_similarity(&mut id_similarity_cache, &entry_id, query);
        let mut max_similarity_id = entry_id;
        loop {
            for neighbor in current_node_neighbors {
                let neighbor_node = self
                    .get_node_by_id(*neighbor)
                    .expect("未能检索到neighbor_node");
                let neighbor_node_similarity =
                    self.get_similarity(&mut id_similarity_cache, neighbor, query);
                if neighbor_node_similarity > max_similarity {
                    max_similarity = neighbor_node_similarity;
                    max_similarity_id = *neighbor_node.get_id();
                }
            }
            if current_node.get_id() == &max_similarity_id {
                return max_similarity_id;
            } else {
                current_node = self
                    .get_node_by_id(max_similarity_id)
                    .expect("未能更新current_node");
                current_node_neighbors = &current_node.get_neighbors()[level];
            }
        }
    }


    // 主线: 正式搜索入口, 先逐层贪心, 再在第0层取 top-k.
    pub fn search_knn_v1(&self, query: &Vector, k: usize, ef_search: usize) -> Vec<Id> {
        
        //不要直接用expect来panic, 而是结束程序, 返回一个空集合, 表示没有搜索到.
        let Some(mut current_entry_id) = self.entry_node_id else {
            return Vec::new();
        };

        //从第index_max_level层开始一直向下搜索到第1层, 使用朴素贪心搜索
        for lvl in (1..=self.index_max_level).rev() {
            current_entry_id = self.greedy_search_at_level(query, current_entry_id, lvl);
        }
        //第0层使用带候选集的精细搜索
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        let (candidate_set, _) = self.search_layer_v0(query, current_entry_id, 0, ef_search, &mut id_similarity_cache);
        let mut candidate_sequence: Vec<Id> = candidate_set.into_iter().collect();
        //into_iter消费了candidate_set, 以后就不能用了.



        /******************************** 排序: 将候选集按相似度降序, 再按 id 稳定打平 *******************************************/
        
        candidate_sequence.sort_by(|id1, id2| {
            let score1 = self.get_similarity(&mut id_similarity_cache, id1, query);
            let score2 = self.get_similarity(&mut id_similarity_cache, id2, query);
            // partial_cmp 只做一件事：比较两个数，返回 Less、Equal、Greater。
            match score2.partial_cmp(&score1).unwrap() {
                // partical_cmp语句返回: score2 (  > / < / = ) score1
                // 这会导致高分排在更小下标,也就是降序
                // sort_by 约定：比较器返回 Less，就表示第一个参数要排在第二个参数前面；
                // 返回 Greater，就表示第一个参数要排在第二个参数后面。
                
                // partical_cmp 返回类型不是 Ordering，而是 Option<Ordering>。
                // unwrap 是 Option 的方法, 把 Option<Ordering> 取出成 Ordering
                // Some(order) -> 返回 order, None -> 直接 panic
                std::cmp::Ordering::Equal => id1.cmp(id2),
                other => other,
            }
        });

        if candidate_sequence.len() < k {
            return candidate_sequence;
        } else {
            return candidate_sequence[..k].to_vec();
        }
    }


    // 采样: 采样新节点的层高, 让高层更稀少、低层更多.
    pub fn sample_level(&mut self) -> usize {
        //返回一个非负整数，表示新插入节点的最高层编号。
        //返回值应满足“高层更稀少、低层更多”的分布趋势.
        //连续随机抛硬币，直到失败为止, 或用某种几何分布近似方法
        let p = 0.5;
        let mut level = 0usize;

        //以p的概率返回true. 大多数节点会停在0或者1层.
        while self.rng.gen_bool(p) {
            level += 1;
        }
        level
    }

    // 主线: 插入新节点的核心逻辑, 先不考虑并发和持久化等问题, 只关注算法本身.
    // 参数说明:
    // - id: 新节点的唯一标识符
    // - data: 新节点的向量数据
    // - ef_construction: 在构建过程中搜索层时使用的参数，控制搜索的宽度
    // - m: 每层连接的最大邻居数. 在插入新节点时, 每层最多连接m个邻居.
    // - m_max: 每层允许的最大邻居数. 在插入新节点时, 如果某个节点的邻居数超过m_max, 就需要进行邻居选择和替换.
    pub fn insert_v1(&mut self, id: Id, data: Vector, ef_construction: usize, m: usize, m_max: usize) {
        self.timing.calls += 1;
        //************************* line 4 *******************************//
        // 主线: Phase 1: Create new node.
        // 采样: 为该节点采样层高
        let new_node_level = self.sample_level();
        // 创建节点
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, new_node_level);
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        // 主线: Case 1:
        // 初始化: 若图为空：
        if self.is_empty(){
            // 设置 entry_node_id
            self.entry_node_id = Some(id);
            // 更新 index_max_level
            self.index_max_level = new_node_level;
            // 插入新节点时更新hashmap
            // 注意先更新hashmap然后再push, 否则需要改成self.nodes.len()-1
            self.id_to_index.insert(id, self.nodes.len());
            // 插入图中
            self.nodes.push(new_node);
            //结束
            return;
        }

        // 主线: Case 2: 图非空。

        // 主线: Phase 2: Top-down greedy routing.
        // - 从当前 entry_node_id 出发。
        // - 从 index_max_level 一路下降到 new_node_level + 1。
        // - 在这些层上调用 greedy_search_at_level(...)
        // - 得到一个更接近新节点向量的入口点。

        
        // 避坑: 因为前面的代码实在太多, 导致遗漏了"新节点进图"的这个动作.
        self.id_to_index.insert(id, self.nodes.len());
        self.nodes.push(new_node); // 所有权分析: push方法接收的是无引用的参数, 获得了所有权.
   
        // 避坑: 新节点在此时 push，虽然邻接表还未完全建立。
        // 但由于新节点仅在 [0, new_node_level] 层出现，
        // 而 Phase 2 只在更高层运行，所以不会被意外访问。
        // Phase 3 中 search_layer_v0 虽然会看到新节点，但只用于"找邻居"而非搜索结果。
        // TODO: 可优化: 换一个方法, 不提前 push，改用临时邻居表 Phase 3 完成后，一次性完成插入.

        // 主线: *************************************** line 5, line 6, line 7  *************************************************//
        // 主线: 这是一个类似于坐电梯下楼的逻辑.在这一层楼上寻找离目的地最近的电梯,然后坐那个电梯下到下一层楼, 再在下一层楼上寻找离目的地最近的电梯, 以此类推, 直到坐到目的地所在的楼层为止.
        let mut entry_point_id = self.get_entry_node_id().expect("in insert_v1, entry_node_id is none");

        // 避坑: greedy_search是在“新节点根本不存在的更高层”上完成的. 如果新节点的层数大于等于旧图最高层, 那么这个assert断言就会失败.
        //                 正确的做法应该是使用条件逻辑, 分为 "新节点之上有更高层"和 "新节点之上没有更高层"两种情况处理.
        //        盲信伪代码: 伪代码 line 5 的原文是从 旧图最高层开始向下遍历到新节点所在层的上一层, 但是没考虑到"新节点之上没有更高层"这种情况.
        //assert!(self.index_max_level >= new_node_level + 1, "in insert_v1, expected self.index_max_level >= new_node_level + 1, got < ");
        

        if self.index_max_level >= new_node_level + 1{
            // 避坑: if这个条件不等式的成立保证了下面的for循环正常运行.
            // 混淆: 伪代码line7中的ep已经不再是"整个Hnsw图的入口点", 而是一个"中间变量".
            // 处理: 因为我们已经有"返回最近的唯一答案"greedy_search_at_level函数,所以line 6和line 7被一起执行.
            // 处理: 伪代码中的ep, 实际上是下面这个for循环中的entry_point_id.
            // 处理: 每一层, entry_point_id都会被greedy_search_at_level更新为"在当前层上, 离新节点更近的那个节点", 这个节点在下一层上作为新的入口点继续被greedy_search_at_level处理.

            // 避坑: rust中, a..b要求a < b, a..=b要求a <= b. 而.rev()也不会允许a > b的情况.
            // 写法: for current_level in (self.index_max_level .. new_node_level + 1).rev(){  // 不满足 a < b, 错误的写法.
            let t = Instant::now();
            for current_level in (new_node_level + 1 ..= self.index_max_level).rev(){ //满足 a < b. 不加=号是左闭右开, 加=号是全闭区间. 这里需要加=号, 因为current_level需要从index_max_level一直遍历到new_node_level+1, 包含两端.
                // 所有权: 由于new_node在前面的"新节点进图"中, 所有权已经转移, 被消费掉了, 所以在这里调用new_node.get_data()就是非法的.
                // 解决: 解决方案就是在push之前, 就把get_data()的结果保存下来, 供这里调用.
                // 说明: 但是由于get_data()返回的是data: Vector,也就是insert_v1的参数data,所以这里可以直接用&data代替.
                entry_point_id = self.greedy_search_at_level(&data,entry_point_id, current_level);


                // 备选: 以下是严格按照伪代码来写的另一种思路, 先调用search_layer_v0得到一个集合, 再从集合中选出一个最近的节点作为下一层的入口点.
                // 但是由于我们已经有了greedy_search_at_level这个函数, 直接调用它就可以了, 不需要再调用search_layer_v0了.
                // 可以看到在这里ef被设置成了1, 也就是只搜索一个最近的节点(粗搜)
                // 不同于后面在新节点所在层上搜索时的ef大于1, 以便得到一个更大的候选集合.(精搜)
                // let w_set = self.search_layer_v0(&data, entry_point_id, current_level, 1);
                // entry_point_id = self.get_nearest(&w_set, &data).expect("entry_point_id not found");
            }
            self.timing.greedy_search_ns += t.elapsed().as_nanos();

        }

        
        /**************************************** 主线: line 8  **************************************************/
        /********************************** 主线: 在会出现新节点的层上建边 ****************************************/


        // 主线: Phase 3: Layer-by-layer neighbor search.
        
        // 主线: - 对每一层 level = min(index_max_level, new_node_level), ..., 0。
        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        
        // 避坑: 对for循环范围声明不清楚. current_lvl需要从min_level开始一直反向遍历到0, 并且包含min_level和0.
        // 写法: for current_lvl in (0 .. min_level).rev(){    //这样的写法是从min_level_-1开始遍历到0
        for current_lvl in (0..=min_level).rev(){ //这样的写法是从min_level开始遍历到0
            
            /**************************************** 主线: line 9 获得候选集合 W_set **************************************************/
            // 主线: 在会出现新节点的层上, 需要扩大搜索范围, ef不再只是1.
            // 主线: 不同于前面在更高层上搜索时的粗搜, 这里需要在新节点所在层上进行精搜, 以便得到一个更大的候选集合.因此ef的值应该大于1.
            let t = Instant::now();
            let (w_set, sl_stats) = self.search_layer_v0(&data, entry_point_id, current_lvl, ef_construction,&mut id_similarity_cache);
            self.timing.search_layer_ns += t.elapsed().as_nanos();
            self.timing.search_layer_compute_sim_ns += sl_stats.compute_sim_ns;
            self.timing.search_layer_furthest_scan_ns += sl_stats.furthest_scan_ns;
            self.timing.search_layer_neighbors_fetch_ns += sl_stats.neighbors_fetch_ns;
            self.timing.search_layer_heap_pop_ns += sl_stats.heap_pop_ns;
            self.timing.search_layer_bookkeeping_ns += sl_stats.candidate_bookkeeping_ns;
            self.timing.search_layer_loop_iterations += sl_stats.loop_iterations;

            /**************************************** 主线: line 10 从候选中选出最多 M 个邻居 **************************************************/
            // 优化: 可引入启发式
            // simple方法按距离排序, 选出前M个. 但是如果候选集合中有很多距离相近的节点, 可能会导致选出的邻居过于集中, 从而影响搜索效率.
            // 因此需要引入一些多样性启发式，避免所有边都指向一团特别近但很拥挤的点.
            let t = Instant::now();
            let neighbors_set = &self.select_neighbors_simple(&w_set,&data,m, &mut id_similarity_cache);
            self.timing.select_neighbors_ns += t.elapsed().as_nanos();

            /**************************************** 主线: line 11 将这些邻居与新节点双向连边 **************************************************/
            // 主线: 双向连边保证图的互相可达性.
            let t = Instant::now();
            for neighbor_node_id in neighbors_set{
                // 避坑: 遗漏导致的后果: 如果new_node没有及时进图的话,在图中通过id来查找node的动作就会失败,整个for循环都无法正常执行.
                self.add_neighbor_to_node_at_level(id, *neighbor_node_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_node_id, id, current_lvl);
            }
            self.timing.link_ns += t.elapsed().as_nanos();

            /************************************** 剪枝: 下面的line 12-16处理的都是剪枝逻辑,为方便理解可整块跳过 **************************************************/
            /**************************************** 剪枝: line 12 对刚刚连上的每个老邻居 e 逐个检查 **************************************************/
            // 剪枝: 对于每一个被选为邻居的节点e, 如果它的邻居数超过了m_max, 就需要进行邻居选择和替换.
            // 剪枝: 这一步是为了控制图的稀疏度, 避免某些节点的邻居过多导致搜索效率下降.

            // 避坑: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // 这个for内部的引用体系堪称灾难, 字面意义上的堪称灾难.

            let t = Instant::now();
            for e_id in neighbors_set{

                // let mut e_node = self.get_mut_node_by_id(*e_id)
                //     .expect("in insert_v1, get_node_by_id cannot handle e_id");
                // let e_conn = &e_node.get_neighbors()[current_lvl];
                // let e_conn_set: HashSet<u64> = e_conn.iter().take(m).copied().collect();

                /**************************************** 剪枝: line 13 取出 e 在这一层的所有邻居集合 eConn **************************************************/
                let e_conn_set: HashSet<u64> = {
                    let e_node = self.get_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    e_node.get_neighbors()[current_lvl]
                        .iter()
                        // .take(m) // 完全是会错意了.m是用在line10的, 不是用在line13的. 这里需要取出e在这一层的所有邻居
                        .copied()
                        .collect()
                };
                
                /**************************************** 剪枝: line 14  **************************************************/          
                /*********** 剪枝: 在第current_level 这一层上, 如果点 e 的邻居数目超过Mmax (一个点在一层上最多可拥有的邻居数)，那就得删一些边 */
                if e_conn_set.len() > m_max {
                    // let e_new_conn_set = &self.select_neighbors_simple(&e_conn_set,e_node.get_data(), m_max);
                    // 优化: 这里返回的是一个有序Vec
                    
                    /***************************** 剪枝: line 15 重新从 e 当前所有邻居里，筛出一个更好的、大小不超过 Mmax 的集合 e_new_conn **************************************************/
                
                    let e_new_conn_vec: Vec<u64> = {
                        let e_node = self.get_node_by_id(*e_id)
                            .expect("in insert_v1, get_node_by_id cannot handle e_id");
                        // 缓存: 这里必须新建 e_cache, 不能复用外层的 id_similarity_cache.
                        // 原因: id_similarity_cache 里存的是 similarity(某节点, &data),
                        //       也就是"某节点 和 新插入节点" 之间的相似度.
                        // 但是 select_neighbors_simple 在这里的 query 是 e_node.get_data(),
                        //       也就是"某节点 和 老节点e" 之间的相似度, 是完全不同的一张表.
                        // 如果直接传 id_similarity_cache, get_similarity 会在里面查到 "和新节点的相似度"
                        //       当成 "和e的相似度" 直接用, 导致剪枝结果完全错误.
                        // 因此每个 e_id 的剪枝都需要一个以 e_node 为 query 基准的独立空 cache.
                        let mut e_cache: HashMap<Id, f32> = HashMap::new();
                        self.select_neighbors_simple(&e_conn_set, e_node.get_data(), m_max, &mut e_cache)
                    };
                    // 剪枝: 此处对应 line 16, 但目前不是很会写.

                    let e_node = self.get_mut_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    
                    /***************************** 剪枝: line 16 更新 e 的邻居表 **************************************************/
                
                    // 优化: e_node.neighbors[current_lvl] = e_new_conn_set.iter().copied().collect();
                    // 优化: 利用vec直接赋值
                    // e_node.neighbors[current_lvl] = e_new_conn_vec;

                    // 剪枝: e_new_conn_vec 是新的邻居列表
                    // e_node.neighbors[current_lvl] 是旧的邻居列表
                    // 我们过去的做法是直接用新的覆盖旧的, 但是这样会导致单方面的修改.
                    // 那么如果要改成利用函数pub fn delete_neighbor_relationship(&mut self, node1_id: Id, node2_id:Id){进行双向删除, 要怎么做?
                    // 怎样才能知道哪些元素在旧的邻居列表当中包含, 但是不在新的邻居列表中包含?
                    // 答案: 用集合差集找出需要删除的元素
                    let old_neighbors = e_node.neighbors[current_lvl].clone();
                    let old_set: HashSet<Id> = old_neighbors.iter().cloned().collect();
                    let new_set: HashSet<Id> = e_new_conn_vec.iter().cloned().collect();
                    // 计算差集
                    let to_delete: HashSet<Id> = old_set.difference(&new_set).cloned().collect();
                    // 避坑: 不清楚这里.cloned().collect();的用法f目标是什么

                    // 更新e_node的邻居表.
                    e_node.neighbors[current_lvl] = e_new_conn_vec;

                    // 对于每个被删除的邻居, 进行双向删除
                    for neighbor_to_delete_id in to_delete{
                        self.delete_neighbor_relationship_at_level(*e_id, neighbor_to_delete_id, current_lvl);
                    }
                }
            }//endfor
            self.timing.pruning_ns += t.elapsed().as_nanos();
            /**************************************** 剪枝: 上面的line 12-16处理的都是剪枝逻辑,为方便理解可整块跳过 **************************************************/



            /**************************************** 主线: line 17 为下一层更新入口点 **************************************************/

            // 避坑: 理解有误.line 17的ep是一个局部变量, 不再代表"整个hnsw图的入口点"
            // self.entry_node_id = w_set.iter().next().copied();

            // 主线: 调用get_nearest函数来从w_set中选出一个最近的节点作为下一层的入口点.
            entry_point_id = self.get_nearest_v1(&mut id_similarity_cache, &w_set, &data).expect("searchh_result_id赋值错误");
        }



        // 主线: Phase 4: Update entry point if needed.
        /**************************************** 主线: line 18  **************************************************/
        /************************* 主线: 如果新点 q 可以到达的最高层 l 比当前图最高层 L 还高  */
                
        if new_node_level > self.index_max_level{

            /************************* 主线: line 19 那就把 q 升级为新的入口点，并把图的最高层更新为 l  **************************************************/
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
        }
    }

    // 优化: 实现一个简化版本的邻居选择函数, 从候选集合中取前 m 个.
    // 优化: cadidates: 候选集合, m:返回的邻居数目,如果不足m个,就全部返回.
    pub fn select_neighbors_simple(&self, candidate_set: &HashSet<Id>, query: &Vector, m: usize, cache: &mut HashMap<Id, f32>) -> Vec<Id>{
        let mut id_score_map: Vec<(u64, f32)>= candidate_set
            .iter()
            .map( |candidate|{
                let pair = (*candidate, self.get_similarity(cache, candidate, query));
                pair
            })
            .collect();
        id_score_map.sort_by(|a, b|
            match b.1.partial_cmp(&a.1).unwrap(){       //"反过来比", 大的在前面
                std::cmp::Ordering::Equal => a.0.cmp(&b.0), //"正着比", 小的在前面
                other=> other,//如果不是equal, 那就原封不动地返回partial_cmpd的结果.
                                        //这里的前后两个other都是变量名.你可以写 x => x
            }
        );   
        let take_count = m.min(id_score_map.len());
        return id_score_map.iter().take(take_count).map(|(id, _)| *id).collect();
    }
 
    pub fn delete_neighbor_relationship_at_all_levels(&mut self, node1_id: Id, node2_id:Id){
        let node1 = self.get_mut_node_by_id(node1_id).expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        
        node1.delete_neighbor_single_direction_at_all_levels(node2_id);
        
        let node2 = self.get_mut_node_by_id(node2_id).expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        // 为了符合rust的所有权要求, 中间两句话不能调换.

        node2.delete_neighbor_single_direction_at_all_levels(node1_id);
    }

    pub fn delete_neighbor_relationship_at_level(&mut self, node1_id: Id, node2_id:Id, level: usize){
        let node1 = self.get_mut_node_by_id(node1_id).expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        
        node1.delete_neighbor_single_direction_at_level(node2_id, level);
        
        let node2 = self.get_mut_node_by_id(node2_id).expect("In function delete_neighbor_relationship, get_mut_node_by_id got None");
        // 为了符合rust的所有权要求, 中间两句话不能调换.

        node2.delete_neighbor_single_direction_at_level(node1_id, level);
    }



}

