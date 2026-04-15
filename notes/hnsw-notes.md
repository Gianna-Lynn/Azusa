# 区分Vec, vec!, Vector
## Vec
Vec是Rust标准库内的动态数组类型名.Vec是一种容器类型.下面的说法都是正确的:
Vec<f32>;
Vec<Id>;
Vec<Record>;
Vec<(f32, Id)>; # 这是一个包含f32和Id的元组的Vec
分别表示一个包含f32类型元素的Vec,一个包含Id类型元素的Vec,一个包含Record类型元素的Vec,一个包含(f32, Id)类型元素的Vec.
## vec!
vec!是Rust中的一个宏,可以快速创建一个Vec.它的语法是vec![元素1, 元素2, ...].例如:
let v = vec![1, 2, 3];
## Vector
Vector通常不是Rust中的一个具体类型,而是我们自己用type alias定义的一个类型别名.例如:
type Vector = Vec<f32>;
这里我们定义了一个名为Vector的类型别名,它实际上是Vec<f32>.
data:Vector;
等价于
data:Vec<f32>;



# 区分iter(), iter_mut(), into_iter()
let v = vec![1, 2, 3];


## 方式 1: iter() - 借用，不动容器
for &x in v.iter() {
    println!("{}", x);
}
println!("{:?}", v);  // v 还在，可以用

## 方式 2: iter_mut() - 可变借用，不动容器
for x in v.iter_mut() {
    *x += 1;
}
println!("{:?}", v);  // v 还在，可以用

## 方式 3: into_iter() - 移动所有权.v被吃掉了，不能再用
for x in v.into_iter() {
    println!("{}", x);
}
// println!("{:?}", v);  // 错！v 已经被吃掉了,不能再用


## HNSW 搜索过程示例

### 初始
- v = {A}
- C = [A8]
- W = [A8]

### 第1轮
- pop A8, expand A -> {B6, C7}
- v = {A,B,C}
- C = [B6, C7]
- W = [B6, C7, A8]

### 第2轮
- pop B6, expand B -> {A, D3, E4}
- A 已访问
- 加入 D3, 踢掉 A8
- 加入 E4, 踢掉 C7
- v = {A,B,C,D,E}
- C = [D3, E4, C7]
- W = [D3, E4, B6]

### 第3轮
- pop D3, expand D -> {B}
- B 已访问
- C = [E4, C7]
- W = [D3, E4, B6]

### 第4轮
- pop E4, expand E -> {B}
- B 已访问
- C = [C7]
- W = [D3, E4, B6]

### 第5轮
- c = C7
- f = B6
- 7 > 6
- break

### 结果
- 返回 W = [D3, E4, B6]