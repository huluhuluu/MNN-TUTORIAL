# MNN 介绍

MNN框架作为一个高性能的深度学习推理引擎，核心设计围绕着几个关键的抽象类展开。其中，Varp，Expr，Op等类是整个框架的基石，它们不仅定义了数据的表示方式，还构建了计算图的基本结构。本文将深入分析这些核心类的设计理念、实现细节以及它们之间的相互关系。

## 1. MNN核心类


MNN 中的核心类主要包括：

| 类名 | 职责 |
|------|------|
| `VARP` | 智能指针包装的 Variable，表示表达式中的变量节点 |
| `Variable` | 表达式图中的变量节点，持有张量数据或计算信息 |
| `Expr` | 表达式边，表示一个计算操作 以及 输入节点和输出节点 |
| `Tensor` | 张量数据容器，存储实际的多维数据 |
| `Op` | 算子描述符，定义计算的类型和参数、模型权重等 |
| `Pipeline` | 计算流水线，TODO: |
| `Session` | 执行会话，推理数据的持有者, TODO |
| `Interpreter` | 解释器, 模型数据的持有者 |
| `Executor` | 执行器， TODO |
| `ExecutorScope` | 执行器作用域， TODO |

### 1.1 类之间关系

- **逻辑依赖关系： **VARP和Expr是MNN计算图格式的核心，分别表示计算图的节点和边，Op类是整个计算图的核心，表示计算图的边Expr的计算操作。关系图如下：

	```
	VARP (Variable Ptr)
	    ↓ 指向
	Variable
	    ↓ 包含
	Expr ──→ Op (操作描述)
	    ↓
	Tensor (存储数据)
	```

	每个边Expr都有属性`std::vector<VARP> mInputs;`指示输入来自的节点;
	每个VARP节点有属性`EXPRP mFrom; // typedef std::shared_ptr<Expr> EXPRP;`指示输入的边，同时VARP的节点的Tensor数据也存放在入边mFrom中，数据读取和写入在```Variable::readMap```和```Variable::writeMap```中，需要转换成对应数据格式的指针进行读取/写入。一个简单的计算图可以如下：

	```
	Expr (mFrom) -> Tensor x(data)        Expr (mFrom) -> Tensor y(data) 
	         	│									    │
	         	│									    │
	         VARP x                    				  VARP y
	            ↓                        	  		    ↓
	            └────────────────────┬──────────────────┘
	                                 │
	                				 │ Expr -------> Op(Add)
		                             │   └─--------> Tensor z
		                             ↓
		             			  VARP z
	```

- **执行依赖关系： **MNN的计算图有两种模式，**Defe**r(延迟计算)模式或**Eager**(立即计算)模式：Defer模式下，调用表达式相关API不直接计算，而是搭建模型，在需要获取输出值时才执行；Eager模式下，直接进行计算，对应地无法搭建模型。
	下面以Eager模式为例，梳理MNN中表达式计算顺序，代码分析见 TODO:

	```
	用户定义VARP计算代码 // 如：VARP x = _Input({2, 3});VARP y = _Input({2, 3});VARP z = _Add(x, y);
	    ↓
	varp->readmap()  // 触发计算
	    ↓
	ExecutorScope::Current()->computeInfo() //计算节点信息  ─→ SizeComputer::computeOutputSize() // 动态计算中间节点/输出的形状
	    ↓
	ExecutorScope::Current()->makeCache() // 计算缓存, 可复用, 按数据依赖顺序准备中间节点的Tensor 以及Session会话的计算信息
	    ↓
	Executor::ComputeCache::compute() // 按数据依赖顺序执行计算缓存中: 算子的resize和执行。 算子执行会从计算缓存的mSession进入下一步
	    ↓
	Session::run() // 这里继续进入Session::mPipelines的执行
	    ↓
	Pipeline::execute() // 这里执行execution->onExecute 进入算子的后端执行
	    ↓
	后端执行计算	// 计算结果存在tensor中 最后从readmap读取为指针
	    ↓
	获得指针
	```

	这当中主要配置的信息是在`ExecutorScope::Current()->makeCache()`中配置Session，后续的执行大多依赖Session的数据

### 1.2 VARP类

`VARP`本质是 `Variable` 的智能指针包装类，利用Variable指针的地址重载了比较运算符，核心代码如下

```cpp
// express/Expr.cpp
class MNN_PUBLIC VARP {
public:
    // 构造函数
    VARP();
    VARP(std::shared_ptr<Variable> c);
    VARP(Variable* c);
    
    // 获取内部 Variable 变量
    Variable* get() const;
    
    // 重载运算符
    bool operator==(const VARP& var) const = 0;
    bool operator<(const VARP& var) const = 0;
    bool operator<=(const VARP& var) const = 0;
	
    // ... 其它代码  

    // 类型标记
    enum InputType {
        INPUT = 0,      // 输入变量
        CONSTANT = 1,   // 常量变量
        TRAINABLE = 2   // 可训练变量
    };
    // 固定变量类型
    bool fix(InputType type) const;
    // 设置数据格式
    void setOrder(Dimensionformat format); // enum Dimensionformat { NHWC, NC4HW4, NCHW };
    
private:
    std::shared_ptr<Variable> mContent;
};
```

#### 1.2.1 Variable 类

描述算子图的一个节点，主要刻画了对节点和数据的部分操作，核心代码如下

```cpp
// express/Expr.cpp
class MNN_PUBLIC Variable {
public:
    // 获取节点名称
    const std::string& name() const;
    void setName(const std::string& name);
    
    // 获取数据信息
    const Info* getInfo();
	struct Info {
        Dimensionformat order;  // 数据格式（NCHW/NHWC/NC4HW4）
        INTS dim;              // 维度
        halide_type_t type;    // 数据类型
        size_t size;           // 元素总数
        void syncSize();       // 同步大小
    };
    
    // 设置和获取(输入边 以及 数据是“输入边”的第几个输出Tensor)
	void setExpr(EXPRP expr, int index);
    std::pair<EXPRP, int> expr() const;
    
    // 获取 Tensor
    const Tensor* getTensor() const;
    
    // 调整大小
    bool resize(INTS dims);
    
    // 读取/保存节点到文件
	static std::vector<VARP> load(const char* fileName);
    static void save(const std::vector<VARP>& vars, NetT* dest);
    
    // 获得数据的指针映射
    template <typename T>
    const T* readMap();
    template <typename T>
    T* writeMap();
};
```

####  1.2.2 readMap详解

接下来详细解释readMap是怎么执行的，TODO



### 1.3 Expr类

Expr表示计算图的边，核心属性包括边的输入(VARP) 输出(Tensor) 以及计算算子(Op)，**这里的输出是Tensor格式，被输出节点VARP通过边引用**，核心代码如下

```cpp
// express/Expr.cpp
class MNN_PUBLIC Expr {
public:
    struct Inside;
    enum MemoryType {
        COPY,
        MOVE,
        REF
    };
    // 多种构造函数 使用各种信息构造Expr类
    static EXPRP create(Tensor* tensor, bool own = false);
    static EXPRP create(Variable::Info&& info, const void* ptr, VARP::InputType type, MemoryType copy = COPY);
    static EXPRP create(const OpT* op, std::vector<VARP> inputs, int outputSize = 1);
    static EXPRP create(std::shared_ptr<BufferStorage> extra, std::vector<VARP>&& inputs, int outputSize = 1);
    static EXPRP create(std::unique_ptr<OpT>&& op, std::vector<VARP> inputs, int outputSize = 1);
    
    // 获取/设置节点信息
    const Op* get() const;
    const std::vector<VARP>& inputs() const = 0;
    int outputSize() const = 0;
    void setName(const std::string& name);
    const std::string& name() const = 0;
    const std::string& outputName(int index);
    static void replace(EXPRP oldExpr, EXPRP newExpr);
    VARP::InputType inputType() const;
    
    // 判断当前操作依赖的输入节点 以及输入节点往前依赖的所有节点的信息是否正确, 过程中会推导节点的shape
    bool requireInfo();
    
    // 遍历，过程中会调用用户传入的 回调函数
    void visitOutputs(const std::function<bool(EXPRP, int)>& visit);
    static void visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after);
   

private:
    const Op* mOp;				// 表示的计算算子
    std::vector<VARP> mInputs;	// 输入节点
	
    // 节点信息
    VARP::InputType mType;
    std::string mName;
    std::vector<std::string> mOutputNames;
    // 其中inside记录了很多更多有关边的信息
    std::shared_ptr<Inside> mInside = nullptr;
    struct Expr::Inside {
		// 构造 析构函数
        Inside(int outputSize);
        Inside(Tensor* tensor, bool own = false);
        ~ Inside();
        
        std::vector<Variable::Info> mOutputInfos;	// 输出tensor的信息 与tensor一一对应
        std::vector<Tensor*> mOutputTensors;	// 边对应的输出tensor	
        
        std::shared_ptr<Executor::ComputeCache> mCache; // 计算cache缓存 在1.1 类之间关系提到 通过cache 
        int mCacheOffset = 0;	// 用来索引tensor
        
        // 标记位
        Executor::Requirement mReq;	
        bool mInfoDirty = true;
        bool mContentDirty = true;
        bool mOwnTensor = true;
        
        Tensor* mHostTensor = nullptr;			// 设备内存（GPU/NPU）中的tensor
        std::shared_ptr<Backend> mHoldBackend;	// 持有数据的后端
    };
};
```

#### 1.3.1 requireInfo 详解

### 1.4 Tensor类

张量数据类，包括数据指针，数据格式，数据维度等信息，

#### 1.4.1 数据格式

MNN支持常见的数据格式，其中N C H W分别表示 批次大小 通道数 高度 宽度，在大模型推理中输入shape是TODO 

```cpp
// 维度类型
    enum DimensionType {
        TENSORFLOW,  // TensorFlow 格式：NHWC
        CAFFE,       // Caffe 格式：NCHW
        CAFFE_C4     // Caffe 格式：NC4HW4（4通道对齐）
    };
```



#### 1.4.2 数据类型



#### 1.4.3  核心接口

核心代码如下

```cpp

```

### 1.5 Op类

Op 是算子的描述类，定义了神经网络中各种操作的类型和参数。MNN 使用 FlatBuffers 序列化格式来表示 Op， 好处是TODO

核心接口:



### 1.6 Session类

