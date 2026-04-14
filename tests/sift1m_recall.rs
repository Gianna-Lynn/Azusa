
pub fn load_sift_bytes(path: &str) -> Result<Vec<u8>, std::io::Error>{

    let result: Vec<u8> = std::fs::read(path)?;
    Ok(result)
}


// 注意一个很重要的小点: read_u32 读出来的是 u32，
// 如果读出来的数后面要拿来做循环次数，大概率会需要：把它转成 usize
pub fn read_u32(input: &mut &[u8]) -> u32{
    // 需要从语法角度完全理解&mut &[u8]:
    // 外层的&mut  : 一个可变引用, 可以理解为这是一个可以修改的读指针.
    // 内层的&[u8] : 一个不可变的字节切片. 只能读, 不能改. 
    // 合起来就是: 拿着一个可变引用, 指向一个不可变的字节切片.
    //            字节切片的内容不能改, 但是指向字节切片的指针本身可以改.


    let (int_bytes, rest) = input.split_at(size_of::<u32>());
    //把input切走一个u32大小的内容(也就是第0个u32),分给int_bytes, 同时把指向第1个u32的指针交给rest.
    // TODO: 可优化: 此处默认了假设, 可能有漏洞: 输入串里一定还有4个字节.正式版时使用Result
    
    
    *input = rest;
    //把指向rest的指针交给input(似乎是这样.)
    //可以理解为把指针往后挪.
    u32::from_le_bytes(int_bytes.try_into().unwrap()) //不太理解try_into和unwarp的用法.
    //from_le_bytes需要一个固定长度的[u8; 4]数组.(可以查看函数的定义.)
    //split_at返回的int_bytes的类型是&[u8](可以查看函数的定义.)
    //因此需要利用try_into()将&[u8]转换成为[u8; 4]
    // TODO: try_into()的具体逻辑还需要了解一下.
    //unwarp()是一种用来处理Some的逻辑.可以把外层的Some直接拿掉,而不需要except.这需要你 "认定这里一定可以成功, 否则就直接panic."

    
    //函数最终返回一个u32类型. 这个u32是input的第一个字节的内容.
}
//基于上面的read_u32, 改写成read_f32
pub fn read_f32(input: &mut &[u8]) -> f32{
    let(float_bytes, rest) = input.split_at(size_of::<u32>());
    *input = rest;
    f32::from_le_bytes(float_bytes.try_into().unwrap())
}

// TODO: 可优化: 使用这个函数的时候, 要检查取出的每一条向量是不是和dim一致
pub fn read_one_fvec(input: &mut &[u8]) -> Vec<f32>{
    // 第一步
    // 先读一个维度 dim
    let dim = read_u32(input) as usize; 
    // println!("In function read_one_vec, dim is {}", dim);
    // 最好令dim的数据类型为usize
    // 在 Rust 里，这些和“长度 / 下标 / 容量”相关的东西，主类型就是 usize。
    // 比如你可能会遇到这样的例子: if vec_f32.len() != dim, 左边的类型就是usize.
    //                           some_vec[dim], dim也要求是usize, 不是u32

    // 第二步
    // 建一个空的 Vec<f32>
    let mut vec_f32:Vec<f32> = Vec::new();
    // 第三步
    // 循环 dim 次，每次读一个 f32 push 进去
    for _ in 0..dim{
        // TODO: [URGENT][SOLVED]这里早停的逻辑语法不正确
        // debug: match不是Option, 所以不会有None可以match. 
        // match input.clone() {
        //     None => break;
        //     Other =>continue;
        // }
        // debug: 真正想要检查的, 是"剩余字节够不够再读一个f32", 可以直接判断剩余切片的长度.
        if input.len() < size_of::<f32>(){
            // TODO: 可优化: 更稳的做法以后可以是：要么返回 Result, 要么外层检查 vec.len() == dim
            panic!("In function read_one_vec, got a vector without enough dim");
        }
        let tmp = read_f32(input);
        vec_f32.push(tmp);
    }

    // 第四步
    // 返回这一条向量
    vec_f32
}

pub fn read_fvecs(bytes: &[u8], limit: usize) -> Vec<Vec<f32>>{
    let mut input = bytes;   
    
    //形式上, let mut input = &mut bytes的类型推导结果确实可以让input变成 &mut &[u8],
    //但是这里的bytes是不可变引用, 你不能写 mut bytes.这样写就不正确.

    // 下面这两句话效果是完全相同的:
    // let mut input: &[u8] = &bytes;   // &bytes 是 &&[u8], Rust 自动解引用转换为 &[u8]
    // let mut input: &[u8] = bytes;    // bytes直接就是 &[u8]

    // 这里暂存读取出来的向量列表，每条向量的元素类型是 f32。
    let mut res: Vec<Vec<f32>> =  Vec::new(); 
    // TODO: 可优化:使用std::io::Cursor自动管理指针的位置, 读完以后位置自动向前.
    for _ in 0..limit{
        // 问题: read_one_vec也有 "剩余字节不够读满dim个f32时, 提前break, 返回一个长度可能小于dim的向量"
        //       那为什么还要检查if input.len() < size_of::<f32>(){ 
        if input.len() < size_of::<f32>(){ 
            panic!("In function read_fvecs, got a vector without enough dim");
        }
        let tmp = read_one_fvec(&mut input);
        if !tmp.is_empty(){
            res.push(tmp); 
        }
    }
    res
}

pub fn read_one_uvec(input: &mut &[u8]) -> Vec<u32>{
    let dim = read_u32(input) as usize; 
    let mut vec_u32:Vec<u32> = Vec::new();
    for _ in 0..dim{
        if input.len() < size_of::<u32>(){
            panic!("In function read_one_vec, got a vector without enough dim");
        }
        let tmp = read_u32(input);
        vec_u32.push(tmp);
    }
    vec_u32
}

pub fn read_uvecs(bytes: &[u8], limit: usize) -> Vec<Vec<u32>>{
    let mut input = bytes;
    let mut res: Vec<Vec<u32>> =  Vec::new(); 
    for _ in 0..limit{
        if input.len() < size_of::<f32>(){ 
            panic!("In function read_ivecs, got a vector without enough dim");
        }
        let tmp = read_one_uvec(&mut input);
        if !tmp.is_empty(){
            res.push(tmp); 
        }
    }
    res
}