//! SIFT1M 实验共用的读数、评估和辅助工具。
// Shared byte-reading helpers for SIFT1M experiment tests.
// 这个文件提供测试辅助工具，本身不是测试入口。

pub fn load_sift_bytes(path: &str) -> Result<Vec<u8>, std::io::Error> {
    // 从磁盘读取 SIFT1M 原始二进制文件，供实验和 smoke test 复用。
    let result: Vec<u8> = std::fs::read(path)?;
    Ok(result)
}

// 从字节流中读取一个小端序 u32，并把读取游标往后移动 4 字节。
pub fn read_u32(input: &mut &[u8]) -> u32 {
    let (int_bytes, rest) = input.split_at(size_of::<u32>());
    *input = rest;
    u32::from_le_bytes(int_bytes.try_into().unwrap())
}

// 从字节流中读取一个小端序 f32，并把读取游标往后移动 4 字节。
pub fn read_f32(input: &mut &[u8]) -> f32 {
    let (float_bytes, rest) = input.split_at(size_of::<f32>());
    *input = rest;
    f32::from_le_bytes(float_bytes.try_into().unwrap())
}

// 读取一条 fvecs 向量：先读维度，再顺序读出对应数量的 f32。
pub fn read_one_fvec(input: &mut &[u8]) -> Vec<f32> {
    let dim = read_u32(input) as usize;
    let mut vec_f32: Vec<f32> = Vec::new();
    for _ in 0..dim {
        if input.len() < size_of::<f32>() {
            panic!("In function read_one_vec, got a vector without enough dim");
        }
        let tmp = read_f32(input);
        vec_f32.push(tmp);
    }
    vec_f32
}

// 从 fvecs 字节流中最多读取 limit 条向量，遇到截断数据时提前停止。
pub fn read_fvecs(bytes: &[u8], limit: usize) -> Vec<Vec<f32>> {
    let mut input = bytes;
    let mut res: Vec<Vec<f32>> = Vec::new();
    for _ in 0..limit {
        // 这里只检查“是否还有维度头”。
        // 真正的维度完整性由 read_one_fvec 内部继续保证，避免读到半条向量时越界。
        if input.len() < size_of::<f32>() {
            panic!("In function read_fvecs, got a vector without enough dim");
        }
        let tmp = read_one_fvec(&mut input);
        if !tmp.is_empty() {
            res.push(tmp);
        }
    }
    res
}

// 读取一条 uvecs/ivecs 向量：先读维度，再顺序读出对应数量的 u32。
pub fn read_one_uvec(input: &mut &[u8]) -> Vec<u32> {
    let dim = read_u32(input) as usize;
    let mut vec_u32: Vec<u32> = Vec::new();
    for _ in 0..dim {
        if input.len() < size_of::<u32>() {
            panic!("In function read_one_vec, got a vector without enough dim");
        }
        let tmp = read_u32(input);
        vec_u32.push(tmp);
    }
    vec_u32
}

// 从 uvecs/ivecs 字节流中最多读取 limit 条向量，遇到截断数据时提前停止。
pub fn read_uvecs(bytes: &[u8], limit: usize) -> Vec<Vec<u32>> {
    let mut input = bytes;
    let mut res: Vec<Vec<u32>> = Vec::new();
    for _ in 0..limit {
        // 同样只先确认还有维度头；后续由 read_one_uvec 保证每个元素都能完整读出。
        if input.len() < size_of::<u32>() {
            panic!("In function read_uvecs, got a vector without enough dim");
        }
        let tmp = read_one_uvec(&mut input);
        if !tmp.is_empty() {
            res.push(tmp);
        }
    }
    res
}
