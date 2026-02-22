# MNN 介绍

MNN框架作为一个高性能的深度学习推理引擎，核心设计围绕着几个关键的抽象类展开。其中，Varp，Expr，Op等类是整个框架的基石，它们不仅定义了数据的表示方式，还构建了计算图的基本结构。本文将深入分析这些核心类的设计理念、实现细节以及它们之间的相互关系。

- [MNN 介绍](#mnn-介绍)
  - [1. MNN核心类](#1-mnn核心类)
    - [1.1 类之间关系](#11-类之间关系)
    - [1.2 VARP类](#12-varp类)
      - [1.2.1 Variable 类](#121-variable-类)
      - [1.2.2 readMap详解](#122-readmap详解)
    - [1.3 Expr类](#13-expr类)
      - [1.3.1 requireInfo 详解](#131-requireinfo-详解)
    - [1.4 Tensor类](#14-tensor类)
      - [1.4.1 数据格式](#141-数据格式)
        - [1.4.2 底层数据格式](#142-底层数据格式)
      - [1.4.2 核心接口](#142-核心接口)
    - [1.5 Op类](#15-op类)
      - [1.5.1 GEMM转卷积算子的理解](#151-gemm转卷积算子的理解)
    - [1.6 Pipeline类](#16-pipeline类)
    - [1.7 Session类](#17-session类)
    - [1.8 Executor \& ExecutorScope类](#18-executor--executorscope类)

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
	下面以Eager模式为例，梳理MNN中表达式计算顺序，更具体的代码分析见[readmap详解](#122-readmap详解):

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

`VARP`本质是 `Variable` 的智能指针包装类，利用Variable指针的地址重载了比较运算符，核心代码如下：

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

描述算子图的一个节点，主要刻画了对节点和数据的部分操作，核心代码如下：

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

Expr表示计算图的边，核心属性包括边的输入(VARP) 输出(Tensor) 以及计算算子(Op)，**这里的输出是Tensor格式，被输出节点VARP通过边引用**，核心代码如下: 

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
TODO:
### 1.4 Tensor类

张量数据类，包括数据指针，数据格式，数据维度等信息，所有属性存储在下面两个结构体对象中，

```cpp
// include/MNN/Tensor.hpp
class MNN_PUBLIC Tensor{
// 其它代码
private:
    halide_buffer_t mBuffer;
    struct InsideDescribe* mDescribe;
};
```



#### 1.4.1 数据格式

MNN支持常见的数据格式，其中N C H W分别表示 批次大小 通道数 高度 宽度。这个格式是对图片格式的兼容，在大模型推理中输入embedding的shape是(batch_size, seq_len, hidden_size)，依次是输入、序列长度和隐藏层大小，不需要太关注这个NCHW，（大模型推理过程中的shape变化可以参考[这篇介绍](https://github.com/huluhuluu/Transformers-Code-View/blob/main/blog/module_construction.md)）。

```cpp
// include/MNN/Tensor.hpp
// 维度类型
    enum DimensionType {
        TENSORFLOW,  // TensorFlow 格式：NHWC
        CAFFE,       // Caffe 格式：NCHW
        CAFFE_C4     // Caffe 格式：NC4HW4（4通道对齐）
    };
```

值得一提的是 **MNN默认只支持1条数据输入**，输入embedding维度是(seq_len, hidden_size)，但是除了Attention算子的大部分算子都做了batch维度的适配，对MNN中大模型推理感兴趣可以看[这篇介绍](https://github.com/huluhuluu/MNN-TUTORIAL/blob/main/blog/llm-infer.md)。

**我尝试在MNN框架上做了Chunk prefill，把不同输入请求合并在seq_len维度上，并在Attention算子中展开:[传送门](https://github.com/huluhuluu/MNN_INFER/tree/feature/batch)**



##### 1.4.2 底层数据格式

从更底层出发，这个数据格式信息由下面结构体存储，对应`Tensor`类的`halide_buffer_t mBuffer`属性，其中存储了数据的指针、数据类型、数据维度等信息，核心代码如下：

```cpp
// include/MNN/HalideRuntime.h
/**
 * The raw representation of an image passed around by generated
 * Halide code. It includes some stuff to track whether the image is
 * not actually in main memory, but instead on a device (like a
 * GPU). For a more convenient C++ wrapper, use Halide::Buffer<T>. */
typedef struct halide_buffer_t {
    /** A device-handle for e.g. GPU memory used to back this buffer. */
    uint64_t device; // 设备句柄

    /** The interface used to interpret the above handle. */
    const struct halide_device_interface_t *device_interface; // 接口指针

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */
    uint8_t* host; // 指针 指向数据

    /** flags with various meanings. */
    uint64_t flags;	

    /** The type of each buffer element. */
    struct halide_type_t type;	// 数据类型

    /** The dimensionality of the buffer. */
    int32_t dimensions;	// 数据维度，如[batch_size, seq_len, hidden_size]大小的Tensor数据就是2维

    /** The shape of the buffer. Halide does not own this array - you
     * must manage the memory for it yourself. */
    halide_dimension_t *dim; // 数据各维度的数值, 如如[batch_size, seq_len, hidden_size]大小的Tensor数据 dim[0]就表示batch_size维度的信息

    /** Pads the buffer up to a multiple of 8 bytes */
    void *padding; // 用来对齐内存
} halide_buffer_t;
```

其中的数据类型`halide_type_t type`通过数据占比特数和数据的性质 判断数据的类型，例如`code = halide_type_float`并且`bits = 16`表示半精度浮点数。核心代码如下：

```cpp
// include/MNN/HalideRuntime.h
/** A runtime tag for a type in the halide type system. Can be ints,
 * unsigned ints, or floats of various bit-widths (the 'bits'
 * field). Can also be vectors of the same (by setting the 'lanes'
 * field to something larger than one). This struct should be
 * exactly 32-bits in size. */
struct halide_type_t {
    /** The basic type code: signed integer, unsigned integer, or floating point. */
    
    // 这个code表示数据的性质 见本代码块最下方结构体，例如 halide_type_int = 0, 表示 signed integers 有符号整型
#ifndef _MSC_VER
    HALIDE_ATTRIBUTE_ALIGN(1) halide_type_code_t code; // halide_type_code_t
#else
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t code; // halide_type_code_t
#endif

    /** The number of bits of precision of a single scalar value of this type. */
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t bits;	// 数据占用比特数

    /** How many elements in a vector. This is 1 for scalar types. */
    HALIDE_ATTRIBUTE_ALIGN(2) uint16_t lanes; // 一次处理的数据宽度 用于SIMD

    // 构造函数
#ifdef __cplusplus
    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    HALIDE_ALWAYS_INLINE halide_type_t(halide_type_code_t code, uint8_t bits, uint16_t lanes = 1)
        : code(code), bits(bits), lanes(lanes) {
    }
    /** Default constructor is required e.g. to declare halide_trace_event
     * instances. */
    HALIDE_ALWAYS_INLINE halide_type_t() : code((halide_type_code_t)0), bits(0), lanes(0) {}

    // 重载了对比函数
    /** Compare two types for equality. */
    HALIDE_ALWAYS_INLINE bool operator==(const halide_type_t &other) const {
        return (code == other.code &&
                bits == other.bits &&
                lanes == other.lanes);
    }
    HALIDE_ALWAYS_INLINE bool operator!=(const halide_type_t &other) const {
        return !(*this == other);
    }
    
	// 单个数据占据的内存字节数, 按8bit向上对齐
    /** Size in bytes for a single element, even if width is not 1, of this type. */
    HALIDE_ALWAYS_INLINE int bytes() const { return (bits + 7) / 8; }
#endif
};


typedef enum halide_type_code_t
{
    halide_type_int = 0,   //!< signed integers
    halide_type_uint = 1,  //!< unsigned integers
    halide_type_float = 2, //!< IEEE floating point numbers
    halide_type_handle = 3, //!< opaque pointer type (void *)
    halide_type_bfloat = 4  //!< floating point numbers in the bfloat format
} halide_type_code_t;

```

这里数据维度信息`halide_dimension_t *dim` 包含该维度的元素个数(`extend`)和在该维度移动一步时内存地址的偏移量信息(`stride`，通常在`TensorUtils::setLinearLayout`中利用`extend`计算`stride`)。核心代码如下：
```cpp
// include/MNN/HalideRuntime.h
typedef struct halide_dimension_t { 
    // extend: 该维度的元素个数
    // stride: 在该维度移动一步时内存地址的偏移量信息
    int32_t min, extent, stride;

    // Per-dimension flags. None are defined yet (This is reserved for future use).
    uint32_t flags;

#ifdef __cplusplus
    HALIDE_ALWAYS_INLINE halide_dimension_t() : min(0), extent(0), stride(0), flags(0) {}
    HALIDE_ALWAYS_INLINE halide_dimension_t(int32_t m, int32_t e, int32_t s, uint32_t f = 0) :
        min(m), extent(e), stride(s), flags(f) {}

    HALIDE_ALWAYS_INLINE bool operator==(const halide_dimension_t &other) const {
        return (min == other.min) &&
            (extent == other.extent) &&
            (stride == other.stride) &&
            (flags == other.flags);
    }

    HALIDE_ALWAYS_INLINE bool operator!=(const halide_dimension_t &other) const {
        return !(*this == other);
    }
#endif
} halide_dimension_t;
```

#### 1.4.2 核心接口
Tensor类的核心接口包括设置/获取数据信息、把数据映射到执行设备、调整大小等，[MNN文档](https://mnn-docs.readthedocs.io/en/latest/cpp/Tensor.html)有详细介绍, 需要用到时可以自己查看，比较常用的调试接口是打印数据和打印形状，这里打印数据中有根据数据底层的bits和code信息自动转换成对应数据类型的指针进行打印的转化。
```cpp
// include/MNN/Tensor.hpp
class MNN_PUBLIC Tensor {
public:
    /**
        * @brief print tensor data. for DEBUG use only.
        */
    void print() const;

    /**
        *@brief print tensor shape
        */
    void printShape() const;
}
```

### 1.5 Op类

Op 是算子的描述类，定义了神经网络中各种操作的类型和参数。MNN 使用内存高效的[FlatBuffers](https://github.com/google/flatbuffers)库来序列化/反序列化来表示Op等信息，核心结构体是`OpT`，在需要修改算子信息时通常使用该结构体，通常代码中使用的就是该结构体，核心代码如下：
```cpp
// schema/current/MNN_generated.h
   struct OpT : public flatbuffers::NativeTable {
     std::vector<int32_t> inputIndexes;      // 输入张量索引列表
     OpParameterUnion main;                   // 算子参数（联合体）
     std::string name;                        // 算子名称
     std::vector<int32_t> outputIndexes;     // 输出张量索引列表
     OpType type;                             // 算子类型枚举
     MNN_DATA_FORMAT defaultDimentionFormat; // 默认数据格式（NHWC/NCHW等）
     std::string externalPath;               // 外部权重路径
   };
```
这里`OpType` 是一个枚举类型，用于表示不同的算子类型，例如 `Conv2D`, `Add`, `Relu` 等。MNN 中定义了多个算子类型，每个类型对应一个具体的算子实现，部分代码如下：
```cpp
// schema/current/MNN_generated.h
enum OpType {
    OpType_AbsVal = 0,
    OpType_QuantizedAdd = 1,
    OpType_ArgMax = 2,
    OpType_AsString = 3,
    OpType_InstanceNorm = 4,
    // ...
};
```
继续往底层是`flatbuffers::Table`的数据接口，提供了获取输入索引、算子类型等信息的接口，可以通过`unpack`操作转换为上层的`OpT`结构体，主要在读取、解析、构建算子的`Execution`时用到`Op`结构体，核心代码如下：
```cpp
struct Op FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef OpT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return OpTypeTable();
  }
  // 获取输入索引列表
  const flatbuffers::Vector<int32_t> *inputIndexes() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(4);
  }
  // 获取参数类型
  OpParameter main_type() const {
    return static_cast<OpParameter>(GetField<uint8_t>(6, 0));
  }
  // 获取指针
  const void *main() const {
    return GetPointer<const void *>(8);
  }
  
  // 通过一堆main_as_XXX函数根据算子类型转换成对应的参数结构体指针，例如main_as_ArgMax算子就转换成ArgMax结构体指针
  template<typename T> const T *main_as() const;
  const QuantizedAdd *main_as_QuantizedAdd() const {
    return main_type() == OpParameter_QuantizedAdd ? static_cast<const QuantizedAdd *>(main()) : nullptr;
  }
  const ArgMax *main_as_ArgMax() const {
    return main_type() == OpParameter_ArgMax ? static_cast<const ArgMax *>(main()) : nullptr;
  }
  // ... 其它代码

  // 通过FlatBuffers进行序列化和反序列化
  OpT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(OpT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Op> Pack(flatbuffers::FlatBufferBuilder &_fbb, const OpT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};
```
接下来各个算子都有一个自己的参数解析类，通过判断`main_type()`的参数类型来转换成对应的参数结构体指针，例如`main_as_ArgMax()`函数会判断参数类型是否是`ArgMax`，如果是就转换成`ArgMax`结构体指针，这里`OpParameter`是一个枚举类型，用于表示不同的算子参数类型，例如 `QuantizedAdd`, `ArgMax`, `InstanceNorm` 等，部分代码如下：
```cpp
// schema/current/MNN_generated.h
enum OpParameter {
  OpParameter_NONE = 0,
  OpParameter_QuantizedAdd = 1,
  OpParameter_ArgMax = 2,
  // ... 其它代码 
};
```
以`ArgMax`为例，算子可以通过`main_as_XXX`转换成对应算子**读取**时使用的底层Flatbuffer结构体，可以继续通过`UnPack`函数转换成**修改/写入时用到**的`ArgMaxT`结构体，核心代码如下：
```cpp
// schema/current/CaffeOp_generated.h
struct ArgMax FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ArgMaxT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ArgMaxTypeTable();
  }
  // 获取各种算子的参数
  int32_t outMaxVal() const {
    return GetField<int32_t>(4, 0);
  }
  int32_t topK() const {
    return GetField<int32_t>(6, 0);
  }
  int32_t axis() const {
    return GetField<int32_t>(8, 0);
  }
  int32_t softmaxThreshold() const {
    return GetField<int32_t>(10, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, 4) &&
           VerifyField<int32_t>(verifier, 6) &&
           VerifyField<int32_t>(verifier, 8) &&
           VerifyField<int32_t>(verifier, 10) &&
           verifier.EndTable();
  }
  // 序列化和反序列化
  ArgMaxT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ArgMaxT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<ArgMax> Pack(flatbuffers::FlatBufferBuilder &_fbb, const ArgMaxT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};
```
反序列化后的`ArgMaxT`结构体就是我们平时上层使用的算子参数结构体，属性包含了算子参数的具体数值，例如`outMaxVal`, `topK`, `axis`, `softmaxThreshold`等，核心代码如下：
```cpp
struct ArgMaxT : public flatbuffers::NativeTable {
  typedef ArgMax TableType;
  int32_t outMaxVal;
  int32_t topK;
  int32_t axis;
  int32_t softmaxThreshold;
  ArgMaxT()
      : outMaxVal(0),
        topK(0),
        axis(0),
        softmaxThreshold(0) {
  }
};
```
从`Op`类再往上层就是后端算子的具体实现，通过继承`Execution`类实现，该类解释可见[后端介绍](./introduce-backend.md)，例如：
```cpp
// source/backend/cpu/CPUArgMax.hpp
class CPUArgMax : public Execution {
public:
    enum ArgMinOrMax {
        ARGMIN,
        ARGMAX
    };
    CPUArgMax(Backend *backend, ArgMinOrMax mode, int topk, int outMaxVal, int softmaxThreshold, int axis);
    virtual ~CPUArgMax() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mInputBuffer;
    Tensor mOutputBuffer;
    int mTopk;
    int mOutMaxVal;
    int mSoftmaxThreshold;
    int mAxis;
    int mNum;
    int mDim;
    int mKeyExtent;
    bool mFromNHWC;
    ArgMinOrMax mMode;
};
```
在构造不同后端的算子实现时，会把`Op`结构体中的参数传入到成员变量中，例如：`topK`, `outMaxVal`, `softmaxThreshold`, `axis`等，后续在执行时就可以直接使用这些参数进行计算了，核心代码如下：
```cpp
// source/backend/cpu/CPUArgMax.cpp
class CPUArgMaxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto argMax = op->main_as_ArgMax();
        if (op->type() == OpType_ArgMin) {
            return new CPUArgMax(backend, CPUArgMax::ArgMinOrMax::ARGMIN,
                    argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), argMax->axis());
        } else {
            return new CPUArgMax(backend, CPUArgMax::ArgMinOrMax::ARGMAX,
                    argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), argMax->axis());
        }
    }
};
```
**总结一下：读取算子时通常用`Op`结构体，修改/写入时用`OpT`结构体。阅读代码时可以查看带T的结构体名称，例如`ArgMaxT`、`ConvolutionT`，便于阅读。**

MNN中会把常见的线性层转换为卷积`Convolution`算子，常用算子`UnaryOp`和`BinaryOp`分别表示一元和二元算子。

**常用接口**：可以使用`EnumNameXXX`方式获取OpType、OpParameter等枚举类型的字符串名称，便于调试，例如：
```cpp
// schema/current/MNN_generated.h
inline const char *EnumNameOpType(OpType e);
inline const char *EnumNameOpParameter(OpParameter e);

// schema/current/TensorflowOp_generated.h
inline const char *EnumNameBinaryOpOperation(BinaryOpOperation e);
```
这里的获取都是从一个静态数组中取值实现的，例如二元操作的各个名称存储在静态数组中，并且通过`EnumNameBinaryOpOperation`函数根据枚举值获取对应的名称字符串，核心代码如下：
```cpp
// schema/current/TensorflowOp_generated.h
inline const char * const *EnumNamesBinaryOpOperation() {
  static const char * const names[] = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MAX_TEMP",
    "MIN_TEMP",
    "POW",
    "REALDIV",
    "MINIMUM",
    "MAXIMUM",
    "GREATER",
    "GREATER_EQUAL",
    "LESS",
    "FLOORDIV",
    "SquaredDifference",
    "EQUAL",
    "LESS_EQUAL",
    "FLOORMOD",
    "",
    "MOD",
    "ATAN2",
    "LOGICALOR",
    "NOTEQUAL",
    "BITWISE_AND",
    "BITWISE_OR",
    "BITWISE_XOR",
    "LOGICALXOR",
    "LEFTSHIFT",
    "RIGHTSHIFT",
    nullptr
  };
  return names;
}
inline const char *EnumNameBinaryOpOperation(BinaryOpOperation e) {
  if (e < BinaryOpOperation_ADD || e > BinaryOpOperation_RIGHTSHIFT) return "";
  const size_t index = static_cast<int>(e);
  return EnumNamesBinaryOpOperation()[index];
}
```
#### 1.5.1 GEMM转卷积算子的理解
MNN中会把常见的线性层转换为$1\times1$的卷积`Convolution`算子，

TODO: 

### 1.6 Pipeline类

TODO: 

### 1.7 Session类

TODO: 

### 1.8 Executor & ExecutorScope类