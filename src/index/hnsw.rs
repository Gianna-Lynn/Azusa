use core::f32;
use rand::{Rng, SeedableRng};
use rand::rngs::{StdRng};
use std::collections::{HashMap, HashSet};
use std::f32::{INFINITY};

use crate::distance;
use crate::types::{Id, Metric, Vector};

const MY_SEED : u64 = 42;

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
}
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

    //给某个节点的某一层添加一个邻居 id."我自己这一层加一个邻居"
    pub fn add_neighbor(&mut self, neighbor_id: Id, level_to_add: usize) {
        //self.neighbors[neighbor_lvl].append(neighbor_id); //append是拿来“把一个 Vec 里的所有元素接到另一个 Vec 后面”的。它要的不是一个单独的 Id，而是一个 Vec<Id>。
        if level_to_add < self.neighbors.len() {
            self.neighbors[level_to_add].push(neighbor_id);
        }
    }

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
    // 关联函数: 不依赖具体对象,不需要self参数
    // 实例方法: 依赖对象当前的状态,必须有&self参数
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_id: None,
            index_max_level: 0,
            metric: Metric::Cosine, //默认使用Cosine
            id_to_index: HashMap::new(),
            rng : StdRng::seed_from_u64(MY_SEED),
        }
    }

    pub fn new_with_metric(metric: Metric) -> Self {
        Self {
            nodes: Vec::new(),
            entry_node_id: None,
            index_max_level: 0,
            metric,
            id_to_index: HashMap::new(),
            rng : StdRng::seed_from_u64(MY_SEED),
        }
    }

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

    pub fn search_v0(&self, query: &Vector) -> Option<Id> {
        if self.is_empty() {
            return None;
        }
        //如果图非空
        //let mut search_id_opt = self.entry_node_id;

        //改进逻辑,省去loop内部的match search_id_opt的分支.直接利用current_id.
        let mut current_id = if let Some(id) = self.entry_node_id {
            id
        } else {
            return None;
        };
        loop {
            // match search_id_opt{
            //     Some(search_id) =>{
            if let Some(node_to_search) = self.get_node_by_id(current_id) {
                let mut max_similarity: f32 =
                    self.compute_similarity(node_to_search.get_data(), query);
                let mut max_idx: usize = 0;
                let mut flag = 0;
                for (idx, neighbor_id) in node_to_search.neighbors[0].iter().enumerate() {
                    if let Some(neighbor_node) = self.get_node_by_id(*neighbor_id) {
                        let temp_similarity =
                            self.compute_similarity(neighbor_node.get_data(), query);
                        if temp_similarity > max_similarity {
                            max_similarity = temp_similarity;
                            max_idx = idx;
                            flag = 1;
                        }
                    }
                    //max_similarity = temp_similarity.max(max_similarity);   //ご注意ください:这里max的用法用于比较两个f32
                } //endfor: 得到了和query向量最近的一个已有节点在neighbors[0]中的下标max_idx
                if flag == 0 {
                    //利用node_to_search
                    //没有任何邻居比当前节点更好.
                    return Some(node_to_search.id);
                } else {
                    //利用neighbors[0][max_idx], 这个邻居比当前节点更好.
                    let next_id = node_to_search.neighbors[0][max_idx]; //等号右边是一个id.
                                                                        // let next_node_opt = self.get_node_by_id(next_id);
                    current_id = next_id;
                }
            } else {
                return None;
            }
        }
    }

    pub fn get_nearest_v1(&self, id_similarity_cache: &mut HashMap<Id, f32>, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        //ご注意ください:如果函数签名中返回值仅仅写Id, 那么假如集合HashSet为空,就不会有合法的返回值.
        //需要改成返回Option<Id>
        //unimplemented!();
        let mut max_similarity = -INFINITY;
        let mut max_elem_id: Option<Id> = None;

        for elem_id in set {

            let temp_similarity = self.get_similarity(id_similarity_cache, elem_id, query);

            if temp_similarity > max_similarity {
                max_similarity = temp_similarity;
                max_elem_id = Some(*elem_id);
            } //endif
        } //endfor
        max_elem_id
    }

    pub fn get_furthest_v1(&self,  id_similarity_cache: &mut HashMap<Id, f32>, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        let mut min_similarity = INFINITY;
        let mut min_elem_id: Option<Id> = None;

        for elem_id in set {
           let temp_similarity = self.get_similarity(id_similarity_cache, elem_id, query);
            if temp_similarity < min_similarity {
                min_similarity = temp_similarity;
                min_elem_id = Some(*elem_id);
            } //endif
        } //endfor
        min_elem_id
    }

    pub fn get_nearest_v0(&self, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        //ご注意ください:如果函数签名中返回值仅仅写Id, 那么假如集合HashSet为空,就不会有合法的返回值.
        //需要改成返回Option<Id>
        //unimplemented!();
        let mut max_similarity = -INFINITY;
        let mut max_elem_id: Option<Id> = None;

        for elem_id in set {
            let elem_node = self.get_node_by_id(*elem_id).expect("elem_node不存在");
            let temp_similarity = self.compute_similarity(elem_node.get_data(), query);
            if temp_similarity > max_similarity {
                max_similarity = temp_similarity;
                max_elem_id = Some(*elem_id);
            } //endif
        } //endfor
        max_elem_id
    }

    pub fn get_furthest_v0(&self, set: &HashSet<Id>, query: &Vector) -> Option<Id> {
        let mut min_similarity = INFINITY;
        let mut min_elem_id: Option<Id> = None;

        for elem_id in set {
            let elem_node = self.get_node_by_id(*elem_id).expect("elem_node不存在");
            let temp_similarity = self.compute_similarity(elem_node.get_data(), query);
            if temp_similarity < min_similarity {
                min_similarity = temp_similarity;
                min_elem_id = Some(*elem_id);
            } //endif
        } //endfor
        min_elem_id
    }


    pub fn get_neighbors_at_lvl(&self, id: Id, lvl: usize) -> HashSet<Id> {
        //ご注意ください:如果你用的是一个不带有&self的静态方法,方法不能访问self.nodes,不能访问索引内部的图结构,只能依靠传入的参数工作.
        //因此,必须访问self.
        //这样一来就可以用self.get_neighbors_at_lvl(id, lvl), 而不是更麻烦的HnswIndex::get_neighbors_at_lvl(&self, id, lvl)
        //unimplemented!()
        //数据段职能理解错误:index_max_level仅指代整张图最高能有多少层.而单个节点拥有的层数可能小于最大层数.
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

    //输入一个cache数组<向量Id, 向量和query的相似度>, 如果目标向量的相似度值在cache中,直接返回,否则手动计算,更新cache
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

    // 输出在level这一层, 距离查询向量query最近的一批向量, 总共ef个.
    pub fn search_layer_v0(
        &self,
        query: &Vector,
        entry_id: Id,
        level: usize,
        ef: usize,
    ) -> HashSet<Id> {
        assert!(ef >= 1, "ef至少为1");

        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();

        //已经访问过的节点集合v
        let mut v_set: HashSet<Id> = HashSet::new();
        //候选集合C (大地图)
        let mut c_set: HashSet<Id> = HashSet::new();
        //目前找到的最近点集合W
        let mut w_set: HashSet<Id> = HashSet::new();

        //ご注意ください:初始化时要把entry_id放进这三个集合,否则一开始集合为空,后面的逻辑都不会执行.
        v_set.insert(entry_id);
        c_set.insert(entry_id);
        w_set.insert(entry_id);

        while !c_set.is_empty() {
            //ご注意ください:get方法是查看, 要完成"拿出来+删掉", 就需要一个额外的删除语句: .remove
            let c_id = self.get_nearest_v1(&mut id_similarity_cache, &c_set, query).expect("c_id不存在");
            c_set.remove(&c_id);
            //ご注意ください:get方法是查看, 要完成"拿出来读取", 而不是删掉.
            let mut f_id = self.get_furthest_v1(&mut id_similarity_cache, &w_set, query).expect("f_id不存在");
            // 逻辑:similarity的值: 两个向量距离越近, similarity的值越大.不要写反了.
            if self.get_similarity(&mut id_similarity_cache, &c_id, query)
                < self.get_similarity(&mut id_similarity_cache, &f_id, query)
            {
                break;
            }

            let c_neighbors_set_at_lvl = self.get_neighbors_at_lvl(c_id, level);
            // 语法:此处c_neighbors_set_at_lvl是Option<HashSet<u64>>.
            // 如果直接接上for e_id in c_neighbors_set_at_lvl开始迭代Option<HashSet<u64>>,
            // 那么最多循环1次(当Option的类型是Some),因为Option最多只有1个元素.
            // 两种修改方法,一是match语句,二是.expect("msg")
            for e_id in c_neighbors_set_at_lvl {
                if v_set.contains(&e_id) {
                    continue;
                }
                v_set.insert(e_id);

                f_id = self.get_furthest_v1(&mut id_similarity_cache, &w_set, query).expect("f_id不存在");
        
                // 逻辑:similarity的值: 两个向量距离越近, similarity的值越大.不要写反了.
                if self.get_similarity(&mut id_similarity_cache, &e_id, query)
                    >= self.get_similarity(&mut id_similarity_cache, &f_id, query)
                    || w_set.len() < ef
                {
                    c_set.insert(e_id);
                    w_set.insert(e_id);
                    if w_set.len() > ef {
                        let id_to_remove_op = self.get_furthest_v1(&mut id_similarity_cache, &w_set, query);
                        if let Some(id_to_remove) = id_to_remove_op {
                            w_set.remove(&id_to_remove);
                        }
                    }
                } //endif
            } //endfor
        }
        w_set
    }

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


    //输入: ef_search是第0层的搜索宽度.
    //返回: 按与 query 的相似度从高到低排序后的 top-k 节点 id
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
        let candidate_set = self.search_layer_v0(query, current_entry_id, 0, ef_search);
        let mut candidate_sequence: Vec<Id> = candidate_set.into_iter().collect();
        //into_iter消费了candidate_set, 以后就不能用了.



        /******************************** 排序逻辑 *******************************************/
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        
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


    // 基于上面的search_knn_v1, 把搜索入口节点改成由参数传入.
    pub fn search_knn_v1_from_entry_for_experiment(
        &self,
        entrance_id: Option<Id>,
        query: &Vector,
        k: usize,
        ef_search: usize,) -> Vec<Id> {
        let Some(mut current_entry_id) = entrance_id else {
            return Vec::new();
        };
        let current_node = self.get_node_by_id(current_entry_id)
            .expect("in search_knn_v1_from_entry_for_experiment, failed to get id with current_entry_id");
        for lvl in (1..=*current_node.get_node_max_level()).rev() {
            current_entry_id = self.greedy_search_at_level(query, current_entry_id, lvl);
        }
        let candidate_set = self.search_layer_v0(query, current_entry_id, 0, ef_search);
        let mut id_similarity_cache: HashMap<Id, f32> = HashMap::new();
        let mut candidate_sequence: Vec<Id> = candidate_set.into_iter().collect();
        candidate_sequence.sort_by(|id1, id2| {
            let score1 = self.get_similarity(&mut id_similarity_cache, id1, query);
            let score2 = self.get_similarity(&mut id_similarity_cache, id2, query);
            match score2.partial_cmp(&score1).unwrap() {
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

    // 插入新节点的核心逻辑, 先不考虑并发和持久化等问题, 只关注算法本身.
    // 参数说明:
    // - id: 新节点的唯一标识符
    // - data: 新节点的向量数据
    // - ef_construction: 在构建过程中搜索层时使用的参数，控制搜索的宽度
    // - m: 每层连接的最大邻居数. 在插入新节点时, 每层最多连接m个邻居.
    // - m_max: 每层允许的最大邻居数. 在插入新节点时, 如果某个节点的邻居数超过m_max, 就需要进行邻居选择和替换.
    pub fn insert_v1(&mut self, id: Id, data: Vector, ef_construction: usize, m: usize, m_max: usize) {
        
        //************************* line 4 *******************************//
        // Phase 1: Create new node.
        // 为该节点采样层高
        let new_node_level = self.sample_level();
        // 创建节点
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, new_node_level);

        // Case 1: 
        // 若图为空：
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

        // Case 2: 图非空。

        // Phase 2: Top-down greedy routing.
        // - 从当前 entry_node_id 出发。
        // - 从 index_max_level 一路下降到 new_node_level + 1。
        // - 在这些层上调用 greedy_search_at_level(...)
        // - 得到一个更接近新节点向量的入口点。

        
        // debug: 遗漏: 因为前面的代码实在太多, 导致遗漏了"新节点进图"的这个动作.
        self.id_to_index.insert(id, self.nodes.len());
        self.nodes.push(new_node); // 所有权分析: push方法接收的是无引用的参数, 获得了所有权.
   
        // IMPORTANT: 新节点在此时 push，虽然邻接表还未完全建立。
        // 但由于新节点仅在 [0, new_node_level] 层出现，
        // 而 Phase 2 只在更高层运行，所以不会被意外访问。
        // Phase 3 中 search_layer_v0 虽然会看到新节点，但只用于"找邻居"而非搜索结果。
        // TODO: 可优化: 换一个方法, 不提前 push，改用临时邻居表 Phase 3 完成后，一次性完成插入.

        // *************************************** line 5, line 6, line 7  *************************************************//
        // 这是一个类似于坐电梯下楼的逻辑.在这一层楼上寻找离目的地最近的电梯,然后坐那个电梯下到下一层楼, 再在下一层楼上寻找离目的地最近的电梯, 以此类推, 直到坐到目的地所在的楼层为止.
        let mut entry_point_id = self.get_entry_node_id().expect("in insert_v1, entry_node_id is none");

        // debug: 理解有误: greedy_search是在“新节点根本不存在的更高层”上完成的. 如果新节点的层数大于等于旧图最高层, 那么这个assert断言就会失败.
        //                 正确的做法应该是使用条件逻辑, 分为 "新节点之上有更高层"和 "新节点之上没有更高层"两种情况处理.
        //        盲信伪代码: 伪代码 line 5 的原文是从 旧图最高层开始向下遍历到新节点所在层的上一层, 但是没考虑到"新节点之上没有更高层"这种情况.
        //assert!(self.index_max_level >= new_node_level + 1, "in insert_v1, expected self.index_max_level >= new_node_level + 1, got < ");
        

        if self.index_max_level >= new_node_level + 1{
            // if这个条件不等式的成立保证了下面的for循环正常运行.
            // debug: 混淆点位于伪代码(line 6, line 7)
            // 伪代码line7中的ep已经不再是"整个Hnsw图的入口点", 而是一个"中间变量".
            // 因为我们已经有"返回最近的唯一答案"greedy_search_at_level函数,所以line 6和line 7被一起执行.
            // 伪代码中的ep, 实际上是下面这个for循环中的entry_point_id. 
            // 每一层, entry_point_id都会被greedy_search_at_level更新为"在当前层上, 离新节点更近的那个节点", 这个节点在下一层上作为新的入口点继续被greedy_search_at_level处理.
            
            // debug: rust中, a..b要求a < b, a..=b要求a <= b. 而.rev()也不会允许a > b的情况. 
            // for current_level in (self.index_max_level .. new_node_level + 1).rev(){  // 不满足 a < b, 错误的写法.
            for current_level in (new_node_level + 1 ..= self.index_max_level).rev(){ //满足 a < b. 不加=号是左闭右开, 加=号是全闭区间. 这里需要加=号, 因为current_level需要从index_max_level一直遍历到new_node_level+1, 包含两端.
                // 所有权分析: 由于new_node在前面的"新节点进图"中, 所有权已经转移, 被消费掉了, 所以在这里调用new_node.get_data()就是非法的.
                //            解决方案就是在push之前, 就把get_data()的结果保存下来, 供这里调用.
                //            但是由于get_data()返回的是data: Vector,也就是insert_v1的参数data,所以这里可以直接用&data代替.
                entry_point_id = self.greedy_search_at_level(&data,entry_point_id, current_level);
            
                
                // 以下是严格按照伪代码来写的另一种思路, 先调用search_layer_v0得到一个集合, 再从集合中选出一个最近的节点作为下一层的入口点. 
                // 但是由于我们已经有了greedy_search_at_level这个函数, 直接调用它就可以了, 不需要再调用search_layer_v0了.
                // 可以看到在这里ef被设置成了1, 也就是只搜索一个最近的节点(粗搜)
                // 不同于后面在新节点所在层上搜索时的ef大于1, 以便得到一个更大的候选集合.(精搜)
                // let w_set = self.search_layer_v0(&data, entry_point_id, current_level, 1);
                // entry_point_id = self.get_nearest(&w_set, &data).expect("entry_point_id not found");
            }

        }

        
        /**************************************** line 8  **************************************************/
        /********************************** 在会出现新节点的层上建边 ****************************************/


        // Phase 3: Layer-by-layer neighbor search. 
        
        // - 对每一层 level = min(index_max_level, new_node_level), ..., 0。
        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        
        // debug: 对for循环范围声明不清楚. current_lvl需要从min_level开始一直反向遍历到0, 并且包含min_level和0.
        // for current_lvl in (0 .. min_level).rev(){    //这样的写法是从min_level_-1开始遍历到0
        for current_lvl in (0..=min_level).rev(){ //这样的写法是从min_level开始遍历到0
            
            /**************************************** line 9 获得候选集合 W_set **************************************************/
            // 在会出现新节点的层上, 需要扩大搜索范围, ef不再只是1.
            // 不同于前面在更高层上搜索时的粗搜, 这里需要在新节点所在层上进行精搜, 以便得到一个更大的候选集合.因此ef的值应该大于1.
            let w_set = self.search_layer_v0(&data, entry_point_id, current_lvl, ef_construction);
            
            /**************************************** line 10 从候选中选出最多 M 个邻居 **************************************************/
            // TODO: 可优化: 引入启发式
            // simple方法按距离排序, 选出前M个. 但是如果候选集合中有很多距离相近的节点, 可能会导致选出的邻居过于集中, 从而影响搜索效率. 
            // 因此需要引入一些多样性启发式，避免所有边都指向一团特别近但很拥挤的点.
            let neighbors_set = &self.select_neighbors_simple(&w_set,&data,m);
            
            /**************************************** line 11 将这些邻居与新节点双向连边 **************************************************/
            // 双向连边保证图的互相可达性.
            for neighbor_node_id in neighbors_set{
                // debug: 遗漏导致的后果: 如果new_node没有及时进图的话,在图中通过id来查找node的动作就会失败,整个for循环都无法正常执行.
                self.add_neighbor_to_node_at_level(id, *neighbor_node_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_node_id, id, current_lvl);
            }

            /************************************** 下面的line 12-16处理的都是剪枝逻辑,为方便理解可整块跳过 **************************************************/
            /**************************************** line 12 对刚刚连上的每个老邻居 e 逐个检查 **************************************************/
            // 对于每一个被选为邻居的节点e, 如果它的邻居数超过了m_max, 就需要进行邻居选择和替换. 
            // 这一步是为了控制图的稀疏度, 避免某些节点的邻居过多导致搜索效率下降.
            
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // 这个for内部的引用体系堪称灾难, 字面意义上的堪称灾难.

            
            
            for e_id in neighbors_set{

                // let mut e_node = self.get_mut_node_by_id(*e_id)
                //     .expect("in insert_v1, get_node_by_id cannot handle e_id");
                // let e_conn = &e_node.get_neighbors()[current_lvl];
                // let e_conn_set: HashSet<u64> = e_conn.iter().take(m).copied().collect();

                /**************************************** line 13 取出 e 在这一层的所有邻居集合 eConn **************************************************/
                let e_conn_set: HashSet<u64> = {
                    let e_node = self.get_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    e_node.get_neighbors()[current_lvl]
                        .iter()
                        // .take(m) // 完全是会错意了.m是用在line10的, 不是用在line13的. 这里需要取出e在这一层的所有邻居
                        .copied()
                        .collect()
                };
                
                /**************************************** line 14  **************************************************/          
                /*********** 在第current_level 这一层上, 如果点 e 的邻居数目超过Mmax (一个点在一层上最多可拥有的邻居数)，那就得删一些边 */
                if e_conn_set.len() > m_max {
                    // let e_new_conn_set = &self.select_neighbors_simple(&e_conn_set,e_node.get_data(), m_max);
                    // 改进: 这里返回的是一个有序Vec
                    
                    /***************************** line 15 重新从 e 当前所有邻居里，筛出一个更好的、大小不超过 Mmax 的集合 e_new_conn **************************************************/
                
                    let e_new_conn_vec: Vec<u64> = {
                        let e_node = self.get_node_by_id(*e_id)
                            .expect("in insert_v1, get_node_by_id cannot handle e_id");

                        self.select_neighbors_simple(&e_conn_set,e_node.get_data(), m_max)
                    };
                    //此处对应 line 16, 但目前不是很会写.

                    let e_node = self.get_mut_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    
                    /***************************** line 16 更新 e 的邻居表 **************************************************/
                
                    //e_node.neighbors[current_lvl] = e_new_conn_set.iter().copied().collect();
                    // 改进: 利用vec直接赋值
                    // e_node.neighbors[current_lvl] = e_new_conn_vec;

                    // e_new_conn_vec 是新的邻居列表
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
                    // 不清楚这里.cloned().collect();的用法f目标是什么

                    // 更新e_node的邻居表.
                    e_node.neighbors[current_lvl] = e_new_conn_vec;

                    // 对于每个被删除的邻居, 进行双向删除
                    for neighbor_to_delete_id in to_delete{
                        self.delete_neighbor_relationship_at_level(*e_id, neighbor_to_delete_id, current_lvl);
                    }
                }
            }//endfor
            /**************************************** 上面的line 12-16处理的都是剪枝逻辑,为方便理解可整块跳过 **************************************************/
            


            /**************************************** line 17 为下一层更新入口点 **************************************************/

            // debug: 理解有误.line 17的ep是一个局部变量, 不再代表"整个hnsw图的入口点"
            // self.entry_node_id = w_set.iter().next().copied();
            
            // 调用get_nearest函数来从w_set中选出一个最近的节点作为下一层的入口点.
            entry_point_id = self.get_nearest_v0(&w_set, &data).expect("searchh_result_id赋值错误");
        }
       


        // Phase 4: Update entry point if needed.
        /**************************************** line 18  **************************************************/
        /************************* 如果新点 q 可以到达的最高层 l 比当前图最高层 L 还高  */
                
        if new_node_level > self.index_max_level{

            /*************************  line 19 那就把 q 升级为新的入口点，并把图的最高层更新为 l  **************************************************/
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
        }
    }

    // 基于insert_v1实现的测试专用函数, 人为指定这一点位于哪一层, 不使用随机采样层高的逻辑.
    // 插入新节点的核心逻辑, 先不考虑并发和持久化等问题, 只关注算法本身.
    // 参数说明:
    // - id: 新节点的唯一标识符
    // - data: 新节点的向量数据
    // - ef_construction: 在构建过程中搜索层时使用的参数，控制搜索的宽度
    // - m: 每层连接的最大邻居数. 在插入新节点时, 每层最多连接m个邻居.
    // - m_max: 每层允许的最大邻居数. 在插入新节点时, 如果某个节点的邻居数超过m_max, 就需要进行邻居选择和替换.
    // - level: 人为故意指定这一点位于哪一层.
    pub fn insert_v1_fixed_level(&mut self, id: Id, data: Vector, ef_construction: usize, m: usize, m_max: usize, level: &usize) {
        let new_node_level = *level;
        let data_copy = data.clone();
        let new_node = HnswNode::new(id, data_copy, new_node_level);
        if self.is_empty() {
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
            self.id_to_index.insert(id, self.nodes.len());
            self.nodes.push(new_node);
            return;
        }
        self.id_to_index.insert(id, self.nodes.len());
        self.nodes.push(new_node);
        let mut entry_point_id = self.get_entry_node_id().expect("in insert_v1, entry_node_id is none");
        if self.index_max_level >= new_node_level + 1 {
            for current_level in (new_node_level + 1..=self.index_max_level).rev() {
                entry_point_id = self.greedy_search_at_level(&data, entry_point_id, current_level);
            }
        }
        let min_level = std::cmp::min(self.index_max_level, new_node_level);
        for current_lvl in (0..=min_level).rev() {
            let w_set = self.search_layer_v0(&data, entry_point_id, current_lvl, ef_construction);
            let neighbors_set = &self.select_neighbors_simple(&w_set, &data, m);
            for neighbor_node_id in neighbors_set {
                self.add_neighbor_to_node_at_level(id, *neighbor_node_id, current_lvl);
                self.add_neighbor_to_node_at_level(*neighbor_node_id, id, current_lvl);
            }
            for e_id in neighbors_set{
                let e_conn_set: HashSet<u64> = {
                    let e_node = self.get_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    e_node.get_neighbors()[current_lvl]
                        .iter()
                        .copied()
                        .collect()
                };
                if e_conn_set.len() > m_max {
                    let e_new_conn_vec: Vec<u64> = {
                        let e_node = self.get_node_by_id(*e_id)
                            .expect("in insert_v1, get_node_by_id cannot handle e_id");
                        self.select_neighbors_simple(&e_conn_set,e_node.get_data(), m_max)
                    };
                    let e_node = self.get_mut_node_by_id(*e_id)
                        .expect("in insert_v1, get_node_by_id cannot handle e_id");
                    let old_neighbors = e_node.neighbors[current_lvl].clone();
                    let old_set: HashSet<Id> = old_neighbors.iter().cloned().collect();
                    let new_set: HashSet<Id> = e_new_conn_vec.iter().cloned().collect();
                    let to_delete: HashSet<Id> = old_set.difference(&new_set).cloned().collect();
                    e_node.neighbors[current_lvl] = e_new_conn_vec;
                    for neighbor_to_delete_id in to_delete{
                        self.delete_neighbor_relationship_at_level(*e_id, neighbor_to_delete_id, current_lvl);
                    }
                }
            }//endfor
            entry_point_id = self.get_nearest_v0(&w_set, &data).expect("searchh_result_id赋值错误");
        }
        if new_node_level > self.index_max_level {
            self.entry_node_id = Some(id);
            self.index_max_level = new_node_level;
        }
    }



    // 实现一个简化版本的邻居选择函数
    // cadidates: 候选集合, m:返回的邻居数目,如果不足m个,就全部返回.
    pub fn select_neighbors_simple(&self, candidate_set: &HashSet<Id>, query: &Vector, m: usize) -> Vec<Id>{
        let mut id_score_map: Vec<(u64, f32)>= candidate_set
            .iter()
            .map( |candidate|{
                let candidate_node = self.get_node_by_id(*candidate).expect("In function select_neighbors_simple, ");
                let pair = (*candidate, self.compute_similarity(query, candidate_node.get_data()));
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

