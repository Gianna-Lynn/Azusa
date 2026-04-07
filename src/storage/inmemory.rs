use crate::types::{Id, Vector, Record};
use super::VectorStore;  //在这里,Super表示上一级模块,也就是storage模块. VectorStore是storage模块中定义的trait. 这里的use是为了在当前文件中使用这个trait.
use serde_json; //引入serde_json库,用于序列化和反序列化JSON数据. 这个库需要在Cargo.toml中添加依赖: serde_json = "1.0"。


/********************************************** Storage **********************************************/


pub struct InMemoryVectorStore{ //定义了一个struct结构体.
    data: Vec<Record>,
    // fn save(&self, path: &str) -> Result<(), Error>,  // 错误:在struct定义中,不能直接写函数签名.struct只能包含字段(数据),不能包含方法.
}

impl InMemoryVectorStore{       //第一个impl: 直接为结构体添加方法.
    pub fn new() -> Self{       //猜测：这里是基于上面的定义，增加了一个具体的实现：如何通过new函数新建一个InMemoryVectorStore对象。
                                //回答：Rust中struct定义数据结构，而impl块定义它的行为（函数/方法）。implement=实现。
        Self{data: Vec::new()}  //Self是InMemoryVectorStore的别名,等价于InMemoryVectorStore{data: Vec::new()}
                                //这类似于面向对象编程中的构造函数,提供了一种创建InMemoryVectorStore实例的方式.
                                //调用InMemoryVectorStore::new()就会得到一个data字段是空的InMemoryVectorStore对象.
    }

    pub fn records(&self) -> &[Record]{
        //&self表示以只读引用的方法借用当前的实例,可以通过self访问结构体定义的唯一字段:self.data,只读.如果要修改,需要用&mut self
        &self.data
        //&[Record]表示返回一个Record类型的切片(slice)的引用.切片是Rust中一种动态大小的视图,可以看作是对Vec<Record>的一部分或者全部的引用. 这里返回的是整个Vec<Record>的切片.
    }

    //函数功能:把当前内存中所有Record写到path指定的文件中
    pub fn save(&self, path: &str) -> Result<(), std::io::Error>{
        //&self表示以只读引用的方法借用当前的实例,可以通过self访问结构体定义的唯一字段:self.data,只读.如果要修改,需要用&mut self
        //path: &str表示一个名为path的参数,变量类型是&str,也就是字符串切片.使用&str而不是String,因为前者更加通用高效.
        //Result<(), Error>是一个枚举类型,成功时返回 Ok(()).内层的()表示Unit Type(单元类型),等价于C语言的void.失败时候返回Err(Error).参见RustBook 9.2
        
        // 废稿
        // let raw_data = new(InMemoryVectorStore);
        // let serialized_data = serde_json::to_string(&raw_data).unwrap();
        // println!("{}", serialized_data);
        // std::fs::write(path, contents);

        //第一步:序列化self.data
        let json_string = serde_json::to_string_pretty(&self.data)?;
        //serde_json是一个第三方库,功能是把Rust的数据结构转换成JSON格式的字符串. 
        //to_string_pretty方法会生成格式化的JSON字符串,更易读.
        //问号表示"如果成功就继续,若果失败就从当前函数跳出并返回错误"
        
        //第二步:写入文件
        std::fs::write(path, json_string)?;

        //第三步:返回成功
        Ok(())
    
    }

    //函数功能:从path指定的文件中读出所有Record,写入到一个新的InMemoryVectorStore中.
    pub fn load(path: &str) -> Result<Self, std::io::Error>{
        //第一步:从文件读取内容
        let json_string = std::fs::read_to_string(path)?;
        //第二步:反序列化,构造一个新的Vec<Record>
        let data: Vec<Record> = serde_json::from_str(&json_string)?;

        Ok(Self{data})
    }
}

//猜测：泛型实际上说的是模板和具体实现这两件事。有点像C++的template。
//回答：二者非常相似。编译器会根据实际使用的类型，比如i32或者f64，自动生成多份具体的代码。

impl VectorStore for InMemoryVectorStore{   //第二个impl: 为结构体实现trait.
    //人话就是, 要让 InMemoryVectorStore 这个类型遵守 VectorStore 这份契约.
    //VectorStore是一个trait/接口/合同/契约, 规定了所有VectorStore都必须有的特征,
    // 也就是两个方法签名，insert和search。
    fn insert(&mut self, id: Id, vector: Vector) {
        // VectorStore的定义中包含了insert和search两个函数的声明，但是没有给出具体实现。这里应该就是为了补充实现。

        // 暂时先占位，待实现
        //unimplemented!("insert is not implemented yet"); //猜测：这里的unimplemented!是一个既定的宏或者用法，括号当中的是参数，用于输出。
                                                           //回答：这里确实是Rust标准库提供的一个宏。如果运行到这一行代码，程序会直接panic，然后打印这句话。
        
        // 如果已经存在，那么就覆盖；如果不存在，那么添加新记录。
        if let Some(rec) = self.data.iter_mut().find(|r| r.id == id){
            //问题：看不懂这个if条件。后面的两条竖线是表示绝对值的意思吗？
            //回答：这里的if let是Rust中的一种语法糖，用于简化对Option类型的匹配。
            //它的意思是：如果self.data.iter_mut().find(|r| r.id == id)返回Some(rec)，
            //就把rec绑定到这个变量上，并执行if块内的代码；
            //把rec绑定到这个变量上的意思是：如果find方法找到了一个满足条件的记录，
            //就把它赋值给rec变量，供if块内使用。
            //如果返回None，就跳过if块，继续执行后面的代码。

            // |r|是闭包的参数,两竖线是闭包的语法,不是绝对值,用来定义匿名函数.
            //          可以把闭包理解成一个超小型的函数吗?
            //          回答: 是的，闭包就是一个匿名函数，可以直接在需要函数的地方定义和使用。
            //          它可以捕获周围作用域的变量，非常灵活。
            // r.id == id 是闭包的主体,用来判断当前迭代到的Record的id是否等于传入的id.
            // 返回值是一个布尔值,如果为true,就说明找到了匹配的记录, 
            // find方法就会返回Some(rec),否则继续迭代直到结束,如果没有找到任何匹配的记录,就返回None.
            
            rec.vector = vector;
        }
        else{
            self.data.push(Record { id, vector });
            //猜测：此处应该是为了添加一条全新的记录。
        }

    }
    
}

