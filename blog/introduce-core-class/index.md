---
title: "MNN 核心类介绍"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-05-24T23:00:00+08:00
draft: false
description: "深入分析MNN框架的核心类设计理念、实现细节以及它们之间的相互关系"
slug: "introduce-core-class"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# MNN 核心类介绍

`MNN` 框架作为一个高性能的深度学习推理引擎，核心设计围绕着几个关键的抽象类展开。其中，`VARP`、`Expr`、`Op` 等类是整个框架的基石，它们不仅定义了数据的表示方式，还构建了计算图的基本结构。本文按当前仓库代码，直接看这些核心类的职责、文件位置和相互关系。

- [MNN 核心类介绍](#mnn-核心类介绍)
  - [1. MNN核心类](#1-mnn核心类)
    - [1.1 类之间关系](#11-类之间关系)
    - [1.1.1 逻辑依赖关系](#111-逻辑依赖关系)
    - [1.1.2 执行依赖关系](#112-执行依赖关系)
    - [1.2 VARP类](#12-varp类)
      - [1.2.1 Variable 类](#121-variable-类)
      - [1.2.2 readMap详解](#122-readmap详解)
        - [1.2.2.1 第一步：判断是不是叶子节点](#1221-第一步判断是不是叶子节点)
        - [1.2.2.2 第二步：通过 `requireInfo()` 补齐图上的 shape 信息](#1222-第二步通过-requireinfo-补齐图上的-shape-信息)
        - [1.2.2.3 第三步：通过 `makeCache()` 把子图收敛成 `ComputeCache`](#1223-第三步通过-makecache-把子图收敛成-computecache)
        - [1.2.2.4 第四步：`compute()` 触发 `resize` 和 `run`](#1224-第四步compute-触发-resize-和-run)
        - [1.2.2.5 第五步：`mapOutput()` 把结果映射回 host](#1225-第五步mapoutput-把结果映射回-host)
        - [1.2.2.6 `readMap()` 的整体流程](#1226-readmap-的整体流程)
    - [1.3 Expr类](#13-expr类)
      - [1.3.1 requireInfo 详解](#131-requireinfo-详解)
    - [1.4 Tensor类](#14-tensor类)
      - [1.4.1 数据格式](#141-数据格式)
      - [1.4.2 底层数据格式](#142-底层数据格式)
      - [1.4.3 核心接口](#143-核心接口)
    - [1.5 Op类](#15-op类)
      - [1.5.1 `OpT`结构体](#151-opt结构体)
      - [1.5.2 `Op`结构体](#152-op结构体)
      - [1.5.3 其他细节](#153-其他细节)
      - [1.5.4 GEMM转卷积算子的理解](#154-gemm转卷积算子的理解)
    - [1.6 Pipeline类](#16-pipeline类)
    - [1.7 Session类](#17-session类)
    - [1.8 Executor \& ExecutorScope类](#18-executor--executorscope类)

## 1. MNN核心类

MNN 中的核心类主要包括：

| 类名 | 职责 | 文件位置 |
|------|------|------|
| `VARP` | 智能指针包装的 `Variable`，表示表达式中的变量节点 | `include/MNN/expr/Expr.hpp` |
| `Variable` | 表达式图中的变量节点，持有张量数据或计算信息 | `include/MNN/expr/Expr.hpp` |
| `Expr` | 表达式边，表示一个计算操作以及输入节点和输出节点 | `include/MNN/expr/Expr.hpp`、`express/Expr.cpp` |
| `Tensor` | 张量数据容器，存储实际的多维数据 | `include/MNN/Tensor.hpp` |
| `Op` | 算子描述符，定义计算的类型和参数、模型权重等 | `schema/current/MNN_generated.h` |
| `Runtime` | 运行时抽象层，负责创建具体 `Backend` 并管理线程、内存与运行时资源 | `source/core/Backend.hpp` |
| `Backend` | 执行层后端抽象基类，负责创建 `Execution` 并管理张量内存与设备资源 | `source/core/Backend.hpp` |
| `Execution` | 单个算子的执行抽象基类，负责 `resize` 和 `execute` 两阶段计算 | `source/core/Execution.hpp` |
| `Pipeline` | 一条执行流水线，负责 `encode`、`allocMemory` 和 `execute` | `source/core/Pipeline.hpp` |
| `Session` | 一次推理会话，持有 `Runtime`、`Pipeline` 和输入输出 `Tensor` | `source/core/Session.hpp`、`source/core/Session.cpp` |
| `Interpreter` | 模型数据持有者，也是创建 `Session` 的入口 | `include/MNN/Interpreter.hpp` |
| `Executor` | `Express` 层的执行协调器，负责 shape 推导、缓存和临时 `Session` | `include/MNN/expr/Executor.hpp`、`express/Executor.cpp` |
| `ExecutorScope` | `Executor` 的线程局部作用域，用来切换当前执行上下文 | `include/MNN/expr/ExecutorScope.hpp`、`express/ExecutorScope.cpp` |

其中 `Runtime`、`Backend`、`Execution` 这三个类属于执行层的核心抽象。这里只先给出职责总览，具体的分层关系、调用顺序和关键接口可结合[后端介绍](../introduce-backend/index.md)的第 2 节一起看。

### 1.1 类之间关系

### 1.1.1 逻辑依赖关系
`VARP` 和 `Expr` 是 MNN 计算图格式的核心，分别表示计算图的节点和边，`Op` 类负责描述 `Expr` 对应的计算操作。关系图如下：

```
VARP (Variable Ptr)
    ↓ 指向
Variable
    ↓ 包含
   Expr ───────────────→ Op (操作描述)
    ↓包含输出张量
Tensor (存储数据)
```

其中：
- 每个 `Expr` 边都有 `std::vector<VARP> mInputs;` 记录输入节点，`const Op* mOp;` 代表当前边的计算操作；
- 同时 `Expr` 还持有一个 `std::shared_ptr<Inside> mInside;`，其中保存了当前边的输出 `Tensor`、shape 信息等；
- 每个 `VARP` 节点内部持有一个 `std::shared_ptr<Variable>`；
- `Variable` 继续朝输入方向指向 `std::shared_ptr<Expr> mFrom` 输入边，从而一步一步构成计算图；
- 叶节点就是用 `VARP` 指向输入边 `Expr`，这个输入边没有 `mOp` 和 `mInputs`，只持有一个 `Tensor` 数据。

一个简单的计算图`VARP z = _Add(x,  y)`可以表示如下：

```text
Expr (mFrom) -> Tensor x(data)      Expr (mFrom) -> Tensor y(data)
            │                                     │
            │                                     │
            VARP x                               VARP y
            ↓                                     ↓
            └──────────────┬──────────────────────┘
                           │
                           ▼
         Tensor z ←────── Expr ───────→ Op(Add)
                           │
                           ↓
                       VARP z
```

### 1.1.2 执行依赖关系 
MNN 的计算图有两种模式，`Defer`（延迟计算）模式或 `Eager`（立即计算）模式：`Defer` 模式下，调用表达式相关 API 不会直接计算，而是先搭建模型，在需要获取输出值时才执行（如 `Variable::readMap()` 或 `Variable::writeMap()` 接口）；`Eager` 模式下会直接进行计算，对应地无法搭建模型。
下面以 Eager 模式为例，梳理 MNN 中表达式计算顺序，更具体的代码分析见[readmap详解](#122-readmap详解):

```text
用户定义 `VARP` 计算代码 // 如：`VARP x = _Input({2,  3}); VARP y = _Input({2,  3}); VARP z = _Add(x,  y);`
    ↓
`varp->readMap()`  // 触发延迟计算接口
    ↓
//计算节点信息
`ExecutorScope::Current()->computeInfo()`   ─→ `SizeComputer::computeOutputSize()` // 动态计算中间节点/输出的形状
    ↓
`ExecutorScope::Current()->makeCache()` // 计算缓存，可复用，按数据依赖顺序准备中间节点的 `Tensor` 以及 `Session` 会话的计算信息
    ↓
`Executor::ComputeCache::compute()` // 按数据依赖顺序执行计算缓存中的 `resize` 和执行，算子执行会从计算缓存的 `mSession` 进入下一步
    ↓
`Session::run()` // 这里继续进入 `Session::mPipelines` 的执行
    ↓
`Pipeline::execute()` // 这里执行 `execution->onExecute` 进入算子的后端执行
    ↓
`Execution::onExecute()`	// 按计算图依次执行各算子的后端计算，计算结果存在 `Tensor` 中，最后从 `readMap` 读取为指针
    ↓
获得数据指针
```

这当中主要配置的信息是在 `ExecutorScope::Current()->makeCache()` 中写入的 `Session`，后续执行都依赖这份缓存。

### 1.2 VARP类

`VARP` 本质是 `Variable` 的智能指针包装类。当前实现位于 `include/MNN/expr/Expr.hpp`，比较运算符直接比较内部指针。

```cpp
// include/MNN/expr/Expr.hpp
class MNN_PUBLIC VARP {
public:
    // 绑定 / 解绑底层 Variable
    VARP();
    VARP(std::shared_ptr<Variable> c);
    VARP(Variable* c);

    // 直接取内部的 Variable 指针
    Variable* get() const;

    enum InputType {
        INPUT = 0,
        CONSTANT = 1,
        TRAINABLE = 2,
    };

    // 标记当前 VARP 的输入类型和维度格式
    bool fix(InputType type) const;
    void setOrder(Dimensionformat format);
    
private:
    std::shared_ptr<Variable> mContent;
};
```

#### 1.2.1 Variable 类

`Variable` 描述算子图里的一个变量节点，既提供图结构接口，也提供数据访问接口，部分核心代码如下：

```cpp
// include/MNN/expr/Expr.hpp
class MNN_PUBLIC Variable {
public:
    // 节点命名，便于调试和导出
    const std::string& name() const;
    void setName(const std::string& name);

    struct Info {
        Dimensionformat order = NHWC;
        INTS dim;
        halide_type_t type;
        size_t size;
        void syncSize();
    };

    // 读取当前节点的 shape / dtype / format 信息
    const Info* getInfo();
    // 建立 Variable 到 Expr 的反向引用
    void setExpr(EXPRP expr,  int index);
    std::pair<EXPRP,  int> expr() const;
    // 返回当前节点关联的 Tensor
    const Tensor* getTensor() const;
    // 仅调整逻辑 shape，不直接触发算子执行
    bool resize(INTS dims);

    // 序列化 / 反序列化相关入口
    static std::vector<VARP> load(const char* fileName);
    static void save(const std::vector<VARP>& vars,  NetT* dest);

    // 直接把底层 Tensor 映射成可读 / 可写指针
    template <typename T>
    const T* readMap();
    template <typename T>
    T* writeMap();
private:
    // 核心类属性 表示入边
    EXPRP mFrom;
    int mFromIndex;
};
```

####  1.2.2 readMap详解

`Variable::readMap<T>()` 只是模板包装，真正逻辑在 `express/Expr.cpp` 的 `Variable::readInternal()`。这条调用链基本就是 `Express` 图执行的主入口之一，顺着它往下看，能把 `shape` 推导、缓存构建、`Session` 执行和结果回传这几步串起来。

```cpp
// express/Expr.cpp
void* Variable::readInternal(bool forShape) {
    if (nullptr == mFrom->get()) {
        // 输入 / 常量 / 可训练参数，直接走已有 Tensor
        return mFrom->inside()->mOutputTensors[0]->buffer().host;
    }
    // 普通 Expr，先补齐 shape 信息
    if (!mFrom->requireInfo()) {
        return nullptr;
    }
    // 没有 cache 时，先把当前子图收敛成一个临时 Session
    auto cache = mFrom->inside()->mCache;
    if (nullptr == cache) {
        ExecutorScope::Current()->makeCache({mFrom}, forShape);
        cache = mFrom->inside()->mCache;
    }
    if (nullptr == cache || NO_ERROR != cache->compute()) {
        return nullptr;
    }
    return cache->mapOutput(mFrom->mInside->mCacheOffset + mFromIndex,
                            mFrom->mInside->mOutputTensors[mFromIndex]);
}
```

这一段如果按代码路径展开，可以分成下面几步。

#####  1.2.2.1 第一步：判断是不是叶子节点

```cpp
if (nullptr == mFrom->get()) {
    // 输入 / 常量 / 可训练参数，直接走已有 Tensor
    return mFrom->inside()->mOutputTensors[0]->buffer().host;
}
```

这里的 `mFrom->get()` 实际对应当前 `Expr` 持有的 `Op`。如果它是空，说明当前 `Variable` 对应的不是普通计算算子，而是输入、常量或者可训练参数这类叶子节点。这种情况下不需要再走调度和执行流程，直接返回已有 `Tensor` 的 host 指针即可。

实际代码里这里还有一层额外处理：如果底层数据在其它设备后端上，或者带量化属性，会先构造一个 host `Tensor`，调用 `copyToHostTensor()` 把数据同步回 CPU，再返回 host 地址。也就是说，`readMap()` 对调用方暴露出来的始终是“可直接读的 CPU 指针”。

#####  1.2.2.2 第二步：通过 `requireInfo()` 补齐图上的 shape 信息

普通算子节点会先进入：

```cpp
auto res = mFrom->requireInfo();
if (false == res) {
    return nullptr;
}
```

`Expr::requireInfo()` 的核心作用不是执行计算，而是确保当前节点以及依赖输入的 `shape / dtype / format` 都已经准备好。对应代码在 `express/Expr.cpp`：

```cpp
bool Expr::requireInfo() {
    if (!mInside->mInfoDirty) {
        return true;
    }
    if (nullptr == mOp) {
        return !HasUnknownDim(mInside->mOutputInfos[0].dim);
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        auto inputInfo = mInputs[i]->getInfo();
        if (nullptr == inputInfo) {
            mValid = false;
            return false;
        }
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        if (mInside->mReq.shapeNeedContent[i]) {
            auto ptr = mInputs[i]->readInternal(true);
            if (nullptr == ptr) {
                return false;
            }
        }
    }
    auto res = ExecutorScope::Current()->computeInfo(this);
    if (NO_ERROR == res) {
        mInside->mInfoDirty = false;
    }
    return NO_ERROR == res;
}
```

这里有两个重点：

- `requireInfo()` 会递归调用输入节点的 `getInfo()`，因此会沿着当前 `Expr` 一直向前，把依赖链上的 shape 信息逐层补齐；
- 如果某些算子的 shape 推导依赖输入内容而不只是维度，比如部分 `shape` 相关算子，那么它会通过 `readInternal(true)` 先把输入内容读出来。

真正做 shape 推导的是 `Executor::computeInfo()`：

```cpp
// express/Executor.cpp
ErrorCode Executor::computeInfo(Expr* expr) {
    auto op = expr->get();
    std::vector<Tensor*> inputTensors(expr->inputs().size());
    for (int i = 0; i < inputTensors.size(); ++i) {
        auto inputExpr = expr->inputs()[i]->expr();
        inputTensors[i] = inputExpr.first->inside()->mOutputTensors[inputExpr.second];
    }
    bool res = SizeComputer::computeOutputSize(op, inputTensors, expr->inside()->mOutputTensors);
    if (!res) {
        return COMPUTE_SIZE_ERROR;
    }
    for (int i = 0; i < expr->outputSize(); ++i) {
        auto tensor = expr->inside()->mOutputTensors[i];
        TensorUtils::setLinearLayout(tensor);
        auto shape = expr->outputInfo(i);
        Utils::copyTensorToInfo(shape, tensor);
    }
    return NO_ERROR;
}
```

这一层最终会进入 `SizeComputer::computeOutputSize()`，也就是前面讲过的 shape 工厂系统。执行完成后，当前 `Expr` 的输出 `Tensor` 和 `Variable::Info` 会同步更新，后面才能继续构建执行缓存。

#####  1.2.2.3 第三步：通过 `makeCache()` 把子图收敛成 `ComputeCache`

shape 信息准备好之后，`readInternal()` 会检查当前节点有没有现成的执行缓存：

```cpp
auto cache = mFrom->inside()->mCache;
if (nullptr == cache) {
    ExecutorScope::Current()->makeCache({mFrom}, forShape);
    cache = mFrom->inside()->mCache;
}
```

这里的 `makeCache()` 会把“当前输出节点往前依赖到的整段子图”收敛成一个临时 `Session`，对应 `express/Executor.cpp` 里的 `Executor::_makeCache()`。这个过程主要做四件事：

1. 从目标 `Expr` 开始反向 DFS，找到所有真正参与本次输出计算的依赖节点；
2. 按依赖顺序把每个 `Expr` 改写成 `Schedule::OpCacheInfo`，整理出输入输出 `Tensor`；
3. 生成 `Schedule::ScheduleInfo`，并给目标输出记录 `mCacheOffset`；
4. 用这份 `ScheduleInfo` 构造一个临时 `Session`，保存到 `ComputeCache::mSession`。

核心代码如下：

```cpp
// express/Executor.cpp
void Executor::_makeCache(const std::vector<EXPRP>& expr, bool forceCPU) {
    Schedule::ScheduleInfo scheduleInfo;
    scheduleInfo.pipelineInfo.resize(1);
    auto& pipeline = scheduleInfo.pipelineInfo[0].second;
    // 反向遍历 Expr 子图，构造 pipeline 里的 OpCacheInfo
    while (!dfsStack.empty()) {
        // ...
        Schedule::OpCacheInfo opInfo;
        opInfo.op = expr->get();
        opInfo.inputs.resize(inputs.size());
        opInfo.outputs.resize(expr->outputSize());
        // ...
        pipeline.emplace_back(std::move(opInfo));
    }
    std::shared_ptr<ComputeCache> cahce(new ComputeCache);
    for (auto& iter : dstExpr) {
        iter.first->inside()->mCacheOffset = iter.second;
        iter.first->inside()->mCache = cahce;
    }
    cahce->mSession.reset(new Session(std::move(scheduleInfo), group, std::move(rt)));
}
```

所以这里的 `ComputeCache` 可以理解成“某一段 `Expr` 子图对应的可执行快照”，里面最重要的成员就是 `mSession`。

#####  1.2.2.4 第四步：`compute()` 触发 `resize` 和 `run`

缓存建好以后，`readInternal()` 会调用：

```cpp
if (NO_ERROR != cache->compute()) {
    return nullptr;
}
```

`ComputeCache::compute()` 在 `express/Utils.cpp` 里，核心逻辑如下：

```cpp
ErrorCode Executor::ComputeCache::compute() {
    while (!dfsStack.empty()) {
        auto cache = dfsStack.top();
        if (cache->mShapeDirty) {
            auto code = cache->resize();
            if (NO_ERROR != code) {
                return code;
            }
        }
        if (!cache->mContentDirty) {
            // 已有可复用结果，跳过执行
            continue;
        }
        // 输入依赖都满足后，真正执行 Session
        code = cache->mSession->run();
        if (NO_ERROR != code) {
            return code;
        }
        cache->mContentDirty = false;
    }
    return NO_ERROR;
}
```

这里可以看到两层状态：

- `mShapeDirty` 控制是否需要重新 `resize`；
- `mContentDirty` 控制是否需要真正重新执行算子。

其中 `resize()` 最终会走到：

```cpp
ErrorCode Executor::ComputeCache::resizeImpl() {
    mShapeDirty = false;
    mSession->setNeedResize();
    mSession->resize();
    mContentDirty = true;
    return NO_ERROR;
}
```

也就是先把临时 `Session` 标记为需要重新调整，再执行 `Session::resize()`。这一层会继续进入 `Pipeline`，为每个算子创建 `Execution`、分配内存并调用 `Execution::onResize()`。

而真正执行计算时，`ComputeCache::compute()` 会调用 `Session::run()`：

```cpp
// source/core/Session.cpp
ErrorCode Session::run() const {
    for (auto& iter : mPipelines) {
        auto error = iter->execute();
        if (NO_ERROR != error) {
            return error;
        }
    }
    return NO_ERROR;
}
```

继续往下就是 `Pipeline::execute()`，再到每个算子的 `Execution::onExecute()`。也就是说，`readMap()` 表面上是“读一个变量”，底下实际上已经把整段子图执行完了。

#####  1.2.2.5 第五步：`mapOutput()` 把结果映射回 host

执行完成后，`readInternal()` 最后一步是：

```cpp
return cache->mapOutput(mFrom->mInside->mCacheOffset + mFromIndex,
                        mFrom->mInside->mOutputTensors[mFromIndex]);
```

这里的 `mCacheOffset + mFromIndex` 用来定位当前输出在临时 `Session` 里的目标 `Tensor`。`mapOutput()` 对应代码如下：

```cpp
// express/Utils.cpp
void* Executor::ComputeCache::mapOutput(int offset, Tensor* dest) {
    auto tensor = mSession->getTensor(offset);
    if (0 == tensor->deviceId() && TensorUtils::getDescribe(tensor)->quantAttr.get() == nullptr) {
        auto ptr = tensor->host<void>();
        dest->buffer().host = (uint8_t*)ptr;
        return ptr;
    }
    Utils::allocMemoryForHostTensor(dest);
    tensor->copyToHostTensor(dest);
    return dest->host<void>();
}
```

这里分成两种情况：

- 如果结果本身就在 host 内存上，直接返回底层指针；
- 如果结果在其它后端设备上，例如 GPU，或者需要量化相关处理，就先复制回 host `Tensor`，再返回 host 指针。

#####  1.2.2.6 `readMap()` 的整体流程

把上面的步骤合起来，`readMap()` 的完整调用路径大致如下：

```text
`Variable::readMap<T>()`
    ↓
`Variable::readInternal(false)`
    ↓
判断是否为叶子节点
    ↓
`Expr::requireInfo()`
    ↓
`Executor::computeInfo()`
    ↓
`SizeComputer::computeOutputSize()`
    ↓
`Executor::makeCache()`
    ↓
`Executor::_makeCache()`
    ↓
构造 `ComputeCache` + 临时 `Session`
    ↓
`ComputeCache::compute()`
    ↓
`ComputeCache::resize()` / `Session::resize()`
    ↓
`Session::run()`
    ↓
`Pipeline::execute()`
    ↓
`Execution::onExecute()`
    ↓
`ComputeCache::mapOutput()`
    ↓
返回 host 可读指针
```

所以从接口语义上看，`readMap()` 是“取结果地址”；但从实际执行过程看，它也是 `Express` 图从延迟状态切换到真正求值状态的入口。这也是为什么在 `Defer` 模式下，很多图上的计算直到 `readMap()`、`writeMap()`、`getInfo()` 这类接口被调用时才真正发生。

### 1.3 Expr类

`Expr` 表示计算图中的一条边，核心属性就是输入 `VARP`、输出 `Tensor` 和计算算子 `Op`。当前公开声明在 `include/MNN/expr/Expr.hpp`。

```cpp
// include/MNN/expr/Expr.hpp
class MNN_PUBLIC Expr {
public:
    struct Inside;
    enum MemoryType {
        COPY,
        MOVE,
        REF
    };
    // 用不同来源构造 Expr
    static EXPRP create(Tensor* tensor,  bool own = false);
    static EXPRP create(Variable::Info&& info,  const void* ptr,  VARP::InputType type,  MemoryType copy = COPY);
    static EXPRP create(const OpT* op,  std::vector<VARP> inputs,  int outputSize = 1);
    static EXPRP create(std::shared_ptr<BufferStorage> extra,  std::vector<VARP>&& inputs,  int outputSize = 1);
    static EXPRP create(std::unique_ptr<OpT>&& op,  std::vector<VARP> inputs,  int outputSize = 1);

    // 当前边关联的算子和输入
    const Op* get() const;
    const std::vector<VARP>& inputs() const;
    int outputSize() const;
    // 当前边的名字和输出名
    void setName(const std::string& name);
    const std::string& name() const;
    const std::string& outputName(int index);
    // 替换旧边，供图改写使用
    static void replace(EXPRP oldExpr,  EXPRP newExpr);
    VARP::InputType inputType() const;
    // 递归补齐 shape 信息
    bool requireInfo();
    // 遍历输出引用
    void visitOutputs(const std::function<bool(EXPRP,  int)>& visit);
    static void visit(EXPRP expr,  const std::function<bool(EXPRP)>& before,  const std::function<bool(EXPRP)>& after);

private:
    const Op* mOp; // 核心属性 表示边的操作 如果为空表示当前边的输出VARP节点是常量节点
    std::vector<VARP> mInputs; // 核心属性 表示边的输入节点
    VARP::InputType mType;
    std::string mName;
    std::vector<std::string> mOutputNames;
    std::shared_ptr<Inside> mInside = nullptr; // 包括当前边的输出信息
};
```
其中边有关输出信息都存在 `Inside` 类中，其内部状态定义在 `express/Utils.hpp`:

```cpp
// express/Utils.hpp
struct Expr::Inside {
    // 按输出个数初始化内部状态
    Inside(int outputSize);
    // 直接用已有 Tensor 构造 Expr 的内部表示
    Inside(Tensor* tensor, bool own = false);
    // 释放输出 Tensor / Host Tensor 等内部资源
    ~ Inside();
    std::vector<Variable::Info> mOutputInfos;  // 每个输出对应的 shape / dtype / format 描述
    std::vector<Tensor*> mOutputTensors;       // 每个输出真正绑定的 Tensor 指针
    Executor::Requirement mReq;                // 当前 Expr 对输入内容和 shape 的依赖要求
    std::shared_ptr<Executor::ComputeCache> mCache; // 当前 Expr 关联的执行缓存，里面会持有临时 Session
    int mCacheOffset = 0;                      // 当前 Expr 的第一个输出在 cache Tensor 列表中的偏移
    bool mInfoDirty = true;                    // 输出信息是否失效，失效时需要重新做 shape 推导
    bool mContentDirty = true;                 // 输出内容是否失效，失效时需要重新执行算子
    bool mOwnTensor = true;                    // 当前 Inside 是否负责释放 mOutputTensors
    Tensor* mHostTensor = nullptr;             // 设备 Tensor 读回 Host 时使用的临时拷贝 Tensor
    std::shared_ptr<Backend> mHoldBackend;     // 持有输出数据的后端，避免底层资源被提前释放
};
```

#### 1.3.1 requireInfo 详解
`Expr::requireInfo()` 的职责是把当前 `Expr` 和所有输入 `Expr` 的 shape、dtype、format 信息补齐，必要时会读取部分输入内容。

```cpp
// express/Expr.cpp
bool Expr::requireInfo() {
    // 已经计算过就直接返回
    if (!mInside->mInfoDirty) {
        return true;
    }
    // 节点本身已经失效，直接失败
    if (!mValid) {
        return false;
    }
    // 没有算子时，只检查输出维度是否还包含未知值
    if (nullptr == mOp) {
        return !HasUnknownDim(mInside->mOutputInfos[0].dim);
    }
    // 先把所有输入节点的信息补齐
    for (int i = 0; i < mInputs.size(); ++i) {
        auto inputInfo = mInputs[i]->getInfo();
        if (nullptr == inputInfo) {
            mValid = false;
            return false;
        }
    }
    // 某些算子的 shape 推导需要真实输入内容
    for (int i = 0; i < mInputs.size(); ++i) {
        if (mInside->mReq.shapeNeedContent[i]) {
            // 某些算子的 shape 推导依赖输入内容
            if (nullptr == mInputs[i]->readInternal(true)) {
                return false;
            }
        }
    }
    // 交给 Executor 做真正的 shape / info 推导
    auto res = ExecutorScope::Current()->computeInfo(this);
    if (NO_ERROR == res) {
        mInside->mInfoDirty = false;
    } else {
        mValid = false;
    }
    return NO_ERROR == res;
}
```

这里有两个点需要特别注意：

- `mInside->mReq.shapeNeedContent` 不是固定为真，它来自 `Executor::getRequirement()`，只有少数算子会在 shape 阶段要求真实输入内容；
- 真正做 shape 推导的是 `Executor::computeInfo()`，里面会继续调用 `SizeComputer::computeOutputSize()`。
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

MNN支持下面列出的常见的数据格式，其中N C H W分别表示 批次大小 通道数 高度 宽度。这个格式是对图片格式的兼容，在大模型推理中输入`embedding`的shape是(`batch_size`,  `seq_len`,  `hidden_size`)，依次是输入、序列长度和隐藏层大小(大模型推理过程中的shape变化可以参考[这篇介绍](https://github.com/huluhuluu/Transformers-Code-View/blob/main/blog/module_construction.md))。

```cpp
// include/MNN/Tensor.hpp
// 维度类型
    enum DimensionType {
        TENSORFLOW,   // TensorFlow 格式：NHWC
        CAFFE,        // Caffe 格式：NCHW
        CAFFE_C4     // Caffe 格式：NC4HW4（4通道对齐）
    };
```

在 LLM 场景里, `MNN-LLM`默认把输入整理成 `[seq_len, 1, hidden_size]` 或 `[1, seq_len, hidden_size]`维度, 具体流程可以参考[这篇介绍](../llm-infer/index.md)。

**我尝试在MNN框架上做了Chunk prefill，把不同输入请求合并在seq_len维度上，并在Attention算子中展开:[传送门](https://github.com/huluhuluu/MNN_INFER/tree/feature/batch)**

#### 1.4.2 底层数据格式

从更底层出发，`Tensor`类的数据格式信息由`halide_buffer_t`结构体存储，对应`Tensor::halide_buffer_t mBuffer`属性，其中存储了数据的指针、数据类型、数据维度等信息，核心代码如下：

```cpp
// include/MNN/HalideRuntime.h
/**
 * The raw representation of an image passed around by generated
 * Halide code. It includes some stuff to track whether the image is
 * not actually in main memory,  but instead on a device (like a
 * GPU). For a more convenient C++ wrapper,  use Halide::Buffer<T>. */
typedef struct halide_buffer_t {
    /** A device-handle for e.g. GPU memory used to back this buffer. */
    uint64_t device; // 设备句柄

    /** The interface used to interpret the above handle. */
    const struct halide_device_interface_t *device_interface; // 接口指针

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system,  this is the address of the min
     * coordinates (defined below). */
    uint8_t* host; // 指针 指向数据

    /** flags with various meanings. */
    uint64_t flags;	

    /** The type of each buffer element. */
    struct halide_type_t type;	// 数据类型

    /** The dimensionality of the buffer. */
    int32_t dimensions;	// 数据维度，如 [batch_size, seq_len, hidden_size] 就是 3 维

    /** The shape of the buffer. Halide does not own this array - you
     * must manage the memory for it yourself. */
    halide_dimension_t *dim; // 数据各维度的数值,  如如[batch_size,  seq_len,  hidden_size]大小的Tensor数据 dim[0]就表示batch_size维度的信息

    /** Pads the buffer up to a multiple of 8 bytes */
    void *padding; // 用来对齐内存
} halide_buffer_t;
```

其中的数据类型`halide_type_t type`通过数据占比特数和数据的性质 判断数据的类型，例如`code = halide_type_float`并且`bits = 16`表示半精度浮点数。核心代码如下：

```cpp
// include/MNN/HalideRuntime.h
/** A runtime tag for a type in the halide type system. Can be ints, 
 * unsigned ints,  or floats of various bit-widths (the 'bits'
 * field). Can also be vectors of the same (by setting the 'lanes'
 * field to something larger than one). This struct should be
 * exactly 32-bits in size. */
struct halide_type_t {
    /** The basic type code: signed integer,  unsigned integer,  or floating point. */
    
    // 这个code表示数据的性质 见本代码块最下方结构体，例如 halide_type_int = 0,  表示 signed integers 有符号整型
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
    HALIDE_ALWAYS_INLINE halide_type_t(halide_type_code_t code,  uint8_t bits,  uint16_t lanes = 1)
        : code(code),  bits(bits),  lanes(lanes) {
    }
    /** Default constructor is required e.g. to declare halide_trace_event
     * instances. */
    HALIDE_ALWAYS_INLINE halide_type_t() : code((halide_type_code_t)0),  bits(0),  lanes(0) {}

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
    
	// 单个数据占据的内存字节数,  按8bit向上对齐
    /** Size in bytes for a single element,  even if width is not 1,  of this type. */
    HALIDE_ALWAYS_INLINE int bytes() const { return (bits + 7) / 8; }
#endif
};

typedef enum halide_type_code_t
{
    halide_type_int = 0,    //!< signed integers
    halide_type_uint = 1,   //!< unsigned integers
    halide_type_float = 2,  //!< IEEE floating point numbers
    halide_type_handle = 3,  //!< opaque pointer type (void *)
    halide_type_bfloat = 4  //!< floating point numbers in the bfloat format
} halide_type_code_t;

```

`halide_buffer_t`结构体记录的数据维度信息 `halide_dimension_t *dim` 包含该维度的元素个数(`extent`)和在该维度移动一步时内存地址的偏移量信息(`stride`，通常在 `TensorUtils::setLinearLayout` 中计算)。核心代码如下：
```cpp
// include/MNN/HalideRuntime.h
typedef struct halide_dimension_t { 
    // extend: 该维度的元素个数
    // stride: 在该维度移动一步时内存地址的偏移量信息
    int32_t min,  extent,  stride;

    // Per-dimension flags. None are defined yet (This is reserved for future use).
    uint32_t flags;

#ifdef __cplusplus
    HALIDE_ALWAYS_INLINE halide_dimension_t() : min(0),  extent(0),  stride(0),  flags(0) {}
    HALIDE_ALWAYS_INLINE halide_dimension_t(int32_t m,  int32_t e,  int32_t s,  uint32_t f = 0) :
        min(m),  extent(e),  stride(s),  flags(f) {}

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

#### 1.4.3 核心接口
Tensor类的核心接口包括设置/获取数据信息、把数据映射到执行设备、调整大小等，[MNN文档](https://mnn-docs.readthedocs.io/en/latest/cpp/Tensor.html)有详细介绍,  需要用到时可以自己查看，比较常用的调试接口是打印数据和打印形状，这里打印数据中有根据数据底层的bits和code信息自动转换成对应数据类型的指针进行打印的转化。
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

这一层的核心包括 `Op` 和 `OpT` 两个结构体: 

- `Op` 是 `FlatBuffers` 映射出来的**只读视图**，主要用于运行时读取算子类型、参数和输入输出索引；
- `OpT` 是 `FlatBuffers` 对应的 **NativeTable** 结构，主要用于修改、构造和重新打包算子信息。
- 两者之间可以通过 `UnPack()` / `Pack()`接口切换，其中`Op` 适合“读”，`OpT` 适合“改”。
- 具体的算子则包括`ArgMax`、`QuantizedAdd`、`InstanceNorm`等，每个算子都有对应的参数结构体，例如 `ArgMaxT`、`QuantizedAddT`、`InstanceNormT`等。

从宏观角度看，`Op` 和 `OpT` 的关系如下：
```text
构造阶段（NativeTable，可修改）

OpT
├── type = OpType_ArgMax                  # 算子类型
├── inputIndexes / outputIndexes / name  # 输入输出索引和算子名
└── main : OpParameterUnion
    ├── type  = OpParameter_ArgMax        # 参数类型标签
    └── value -------> ArgMaxT            # 指向具体参数结构体
                         ├── axis
                         ├── topK
                         ├── outMaxVal
                         └── softmaxThreshold


模型二进制转换过程（FlatBuffers）

OpT
  └── Pack / CreateOp(...)                # 把 NativeTable 打包成 FlatBuffers 二进制
        ↓
   FlatBuffer binary                      # 模型中的实际存储形式
        ↓
flatbuffers::GetRoot<Op>(...)             # 从二进制上取得 Op 只读视图
        ↓
       Op
       ├── type()
       ├── inputIndexes()
       └── main_as_ArgMax()               # 按参数类型取出 ArgMax 只读视图
              ↓
            ArgMax
              └── UnPack()                # 如需修改，再转回 NativeTable
                    ↓
                  ArgMaxT
```

其中：

- `OpT` 和 `ArgMaxT` 都是 `NativeTable` 风格的可编辑结构体，适合在内存里构造、修改和重写参数；
- `flatbuffers::GetRoot<Op>(...)` 拿到的 `Op` 不是重新分配出来的新对象，而是对模型二进制 buffer 的只读映射，因此 `ArgMax` 也是只读参数视图；
- 只有继续调用 `UnPack()`，才会把 `Op` 或 `ArgMax` 重新转换成可修改的 `OpT`、`ArgMaxT`。

#### 1.5.1 `OpT`结构体
`Op` 是算子的描述类，定义了神经网络中各种操作的类型和参数。`MNN` 使用内存高效的 [FlatBuffers](https://github.com/google/flatbuffers) 库来序列化/反序列化 `Op` 等信息。其底层通过 `OpT`结构体存储各参数属性，核心代码如下：
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
这里 `OpType` 是一个枚举类型，用于表示不同的算子类型，例如 `Conv2D`、`Add`、`Relu` 等。`MNN` 中定义了多个算子类型，每个类型对应一个具体的算子实现，部分代码如下：
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

`OpT`结构体的`main`成员是一个联合体（`OpParameterUnion`），根据算子类型的不同存储不同的参数结构体，例如 `QuantizedAdd` 算子会存储一个 `QuantizedAddT` 结构体，`ArgMax` 算子会存储一个 `ArgMaxT` 结构体，对应的算子结构体通过`OpParameterUnion`的方法进行**参数值指针 `value`**的类型转换，例如`AsQuantizedAdd()`,`AsArgMax()`等。
```cpp
// schema/current/MNN_generated.h
struct OpParameterUnion {
  OpParameter type; // 参数类型
  void *value;      // 参数值，指向具体的参数结构体，如 QuantizedAddT、ArgMaxT 等

  // 根据参数类型转换成对应的参数结构体指针
  QuantizedAddT *AsQuantizedAdd() {
    return type == OpParameter_QuantizedAdd ?
      reinterpret_cast<QuantizedAddT *>(value) : nullptr;
  }
  const QuantizedAddT *AsQuantizedAdd() const {
    return type == OpParameter_QuantizedAdd ?
      reinterpret_cast<const QuantizedAddT *>(value) : nullptr;
  }
  ArgMaxT *AsArgMax() {
    return type == OpParameter_ArgMax ?
      reinterpret_cast<ArgMaxT *>(value) : nullptr;
  }
  const ArgMaxT *AsArgMax() const {
    return type == OpParameter_ArgMax ?
      reinterpret_cast<const ArgMaxT *>(value) : nullptr;
  }
```
`OpParameterUnion`中出现的`OpParameter type`是枚举类型，其包括不同的算子参数类型，例如 `QuantizedAdd`、`ArgMax`、`InstanceNorm` 等，部分枚举如下：
```cpp
// schema/current/MNN_generated.h
enum OpParameter {
  OpParameter_NONE = 0, 
  OpParameter_QuantizedAdd = 1, 
  OpParameter_ArgMax = 2, 
  // ... 其它代码 
};
```
也就是一个算子通过`OpType`来区分算子类型，通过`OpParameter`来区分参数类型，二者共同决定了算子具体的计算逻辑和参数结构，例如`ArgMax`算子的构建主要是确定类型以及参数：
```cpp
// express/MathOp.cpp
VARP _ArgMax(VARP input, int axis) {
    input = _checkNC4HW4(input);
    std::unique_ptr<OpT> op(new OpT);
    // 设置算子类型和参数
    op->main.type                         = OpParameter_ArgMax;
    op->type                              = OpType_ArgMax;
    // 构造参数结构体并设置参数值
    op->main.value                        = new ArgMaxT;
    op->main.AsArgMax()->axis = axis;
    op->main.AsArgMax()->outMaxVal = 0;
    op->main.AsArgMax()->topK = 0;
    op->main.AsArgMax()->softmaxThreshold = 0;
    return (Variable::create(Expr::create(std::move(op), {input})));
}
```
相对应的 `ArgMaxT` 结构体可以通过 `OpParameterUnion::AsArgMax()` 获得，其包含 `outMaxVal`、`topK`、`axis`、`softmaxThreshold` 等参数，核心代码如下：
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

`OpT` 不能直接强转成 `Op`，要先 `Pack` 成 `FlatBuffers`，才能通过`GetRoot<Op>(uoffset_t i)` 接口取出来。例如，`Module`类加载模型时：
```cpp
// express/module/Module.cpp:461
auto op = net->oplists()->GetAs<Op>(i);
```

#### 1.5.2 `Op`结构体
`Op` 结构体是模型二进制上的只读视图，主要在读取、解析、构建算子的 `Execution` 时，部分核心代码如下：
```cpp
// schema/current/MNN_generated.h
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
    return static_cast<OpParameter>(GetField<uint8_t>(6,  0));
  }
  // 获取指针
  const void *main() const {
    return GetPointer<const void *>(8);
  }
  
  // 通过一组 main_as_XXX 函数，根据参数类型转换成对应的参数结构体
  template<typename T> const T *main_as() const;
  const QuantizedAdd *main_as_QuantizedAdd() const {
    return main_type() == OpParameter_QuantizedAdd ? static_cast<const QuantizedAdd *>(main()) : nullptr;
  }
  const ArgMax *main_as_ArgMax() const {
    return main_type() == OpParameter_ArgMax ? static_cast<const ArgMax *>(main()) : nullptr;
  }
  // ... 其它代码

  // 通过 FlatBuffers 在 Op 和 OpT 之间切换
  OpT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(OpT *_o,  const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Op> Pack(flatbuffers::FlatBufferBuilder &_fbb,  const OpT* _o,  const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};
```
在`Op`结构体中，算子有一个自己的参数解析方法: 通过`main_type()` 获取参数类型(`OpParameter` 枚举类型), 并通过一组 `main_as_XXX` 函数来把`Op`转换成具体算子的结构体，例如，通过 `main_as_ArgMax()` 转换成 `ArgMax` 结构体指针。
`ArgMax`结构体核心代码如下，是`flatbuffers`生成的解析函数，可以解析对应参数的值：
```cpp
// schema/current/CaffeOp_generated.h
struct ArgMax FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ArgMaxT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ArgMaxTypeTable();
  }
  // 获取各种算子的参数
  int32_t outMaxVal() const {
    return GetField<int32_t>(4,  0);
  }
  int32_t topK() const {
    return GetField<int32_t>(6,  0);
  }
  int32_t axis() const {
    return GetField<int32_t>(8,  0);
  }
  int32_t softmaxThreshold() const {
    return GetField<int32_t>(10,  0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier,  4) &&
           VerifyField<int32_t>(verifier,  6) &&
           VerifyField<int32_t>(verifier,  8) &&
           VerifyField<int32_t>(verifier,  10) &&
           verifier.EndTable();
  }
  // `UnPack`接口可以把反序列化成包含了算子参数的各种属性
  ArgMaxT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ArgMaxT *_o,  const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<ArgMax> Pack(flatbuffers::FlatBufferBuilder &_fbb,  const ArgMaxT* _o,  const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};
```

因此，执行层最常见的访问路径通常是 `Op -> main_as_ArgMax() -> ArgMax -> 读取字段`。例如`source/backend/cpu/CPUArgMax.cpp`中构建`Execution`时就通过`Op`结构体获取到`ArgMax`参数结构体，然后通过`FlatBuffers`生成的访问接口获取参数，核心代码如下：
```cpp
// source/backend/cpu/CPUArgMax.cpp
class CPUArgMaxCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        // 通过 Op 结构体获取 ArgMax 参数结构体
        auto argMax = op->main_as_ArgMax();
        if (op->type() == OpType_ArgMin) {
            return new CPUArgMax(backend, CPUArgMax::ArgMinOrMax::ARGMIN,
                    argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), argMax->axis());
            // 上面通过topK()、outMaxVal()、softmaxThreshold()、axis()等接口获取算子的参数值
        } else {
            return new CPUArgMax(backend, CPUArgMax::ArgMinOrMax::ARGMAX,
                    argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), argMax->axis());
        }
    }
};
```

#### 1.5.3 其他细节
**总结一下：读取算子时通常用 `Op` 结构体，修改/写入时通常用 `OpT` 结构体。阅读代码时可以优先留意带 `T` 的结构体名称，例如 `ArgMaxT`、`ConvolutionT`，它们一般更接近可编辑的上层表示。**

`MNN` 中会把常见的线性层转换为卷积 `Convolution` 算子，常用算子 `UnaryOp` 和 `BinaryOp` 分别表示一元和二元算子。

**常用接口**：可以使用 `EnumNameXXX` 方式获取 `OpType`、`OpParameter` 等枚举类型的字符串名称，便于调试，例如：
```cpp
// schema/current/MNN_generated.h
inline const char *EnumNameOpType(OpType e);
inline const char *EnumNameOpParameter(OpParameter e);

// schema/current/TensorflowOp_generated.h
inline const char *EnumNameBinaryOpOperation(BinaryOpOperation e);
```
这里的获取都是从一个静态数组中取值实现的，例如二元操作的各个名称存储在静态数组中，并且通过 `EnumNameBinaryOpOperation()` 根据枚举值获取对应的名称字符串，代码如下：
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
#### 1.5.4 GEMM转卷积算子的理解
这一节先不从 `MNN` 代码入口讲，而是先把“卷积”和“矩阵乘法”之间的等价关系讲清楚。因为后面看到 `InnerProduct`、`Convolution`、`MatMul` 之间互相改写时，本质上都是在利用同一套线性代数关系。

#####  1.5.4.1 二维卷积

最常见的二维卷积输入可以写成：

- 输入：`[N, C_in, H, W]`
- 卷积核：`[C_out, C_in, K_h, K_w]`
- 输出：`[N, C_out, H_out, W_out]`

对于输出上的某一个位置 `(n, c_out, h, w)`，它的值本质上就是“输入局部窗口”和“卷积核权重”的点积：

```text
output[n, c_out, h, w]
= sum_{c_in, kh, kw}
  input[n, c_in, h + kh, w + kw] * weight[c_out, c_in, kh, kw]
```

也就是说，二维卷积虽然写成了四维张量操作，但对单个输出位置来看，本质仍然是一次向量点积。区别只在于这个点积会在所有空间位置上重复滑动执行。

#####  1.5.4.2 三维卷积

三维卷积只是把二维卷积再多加一个深度维度。输入和卷积核通常写成：

- 输入：`[N, C_in, D, H, W]`
- 卷积核：`[C_out, C_in, K_d, K_h, K_w]`
- 输出：`[N, C_out, D_out, H_out, W_out]`

对应某个输出位置 `(n, c_out, d, h, w)`，计算公式会变成：

```text
output[n, c_out, d, h, w]
= sum_{c_in, kd, kh, kw}
  input[n, c_in, d + kd, h + kh, w + kw] *
  weight[c_out, c_in, kd, kh, kw]
```

从这个角度看，三维卷积和二维卷积没有本质区别，都是“取一个局部块，然后和一组权重做点积”；只是局部块的维度从平面窗口变成了体积窗口。

#####  1.5.4.3 `im2col`

卷积虽然可以直接按窗口滑动实现，但工程上更常见的做法是先把输入展开，再把卷积改写成一次大的矩阵乘法，这个展开过程通常就叫 `im2col`。

以二维卷积为例，`im2col` 的思路是：

1. 对每一个输出空间位置，取出对应的输入感受野；
2. 把这个局部窗口拉平成一行；
3. 把所有输出位置对应的局部窗口按行堆起来，形成一个大矩阵。

如果卷积核大小是 `K_h x K_w`，输入通道数是 `C_in`，那么每个局部窗口拉平后长度就是：

```text
C_in * K_h * K_w
```

如果总共有

```text
H_out * W_out
```

个输出位置，那么 `im2col` 后的输入矩阵可以理解成：

```text
[H_out * W_out, C_in * K_h * K_w]
```

而卷积核本身也可以拉平成另一个矩阵：

```text
[C_out, C_in * K_h * K_w]
```

这样卷积就被改写成了一个标准的矩阵乘法：

```text
[H_out * W_out, C_in * K_h * K_w]
            ×
[C_in * K_h * K_w, C_out]
            =
[H_out * W_out, C_out]
```

最后再把结果 reshape 回 `[C_out, H_out, W_out]` 即可。

#####  1.5.4.4 为什么 `GEMM` 可以等价成卷积

`GEMM` 一般指通用矩阵乘法，即：

```text
C = A × B
```

如果把矩阵乘法写到神经网络里，可以把它理解成“每一行输入向量”和“每一个输出通道权重向量”做一次点积。这和卷积在单个位置上的计算本质是一样的。

最典型的情况是全连接层，也就是 `InnerProduct`：

- 输入通常先被展平成一维向量；
- 每个输出神经元对应一组权重；
- 每个输出值都是输入向量和权重向量的点积。

如果把这个过程改写成卷积，其实就是一个特殊的卷积：

- 当输入已经是 `1 x 1` 的空间尺寸时，它等价于 `1 x 1` 卷积；
- 当输入先展平成 `C_in * H * W` 的一维向量时，也可以把它看成卷积核覆盖整个输入空间的一次卷积，即 `kernel = [H, W]` 的卷积；
- 从结果上看，输出通道数就对应全连接层的输出维度。

所以“`GEMM` 等价成卷积”并不是额外发明了一种计算，而是把同样的线性变换换了一种张量组织方式：

- 用矩阵视角看，它是 `A × B`；
- 用卷积视角看，它是“局部块展开 + 权重点积”；
- 用实现视角看，它们最后都可以落到同一类高效的 `GEMM` kernel 上。

#####  1.5.4.5 为什么框架里经常互相改写

理解了上面的关系后，再看框架里的算子改写就比较自然了。很多时候并不是“卷积”和“矩阵乘法”各做各的，而是根据后端更擅长的实现方式，在几种等价表示之间切换：

- 卷积可以通过 `im2col + GEMM` 来实现；
- 全连接可以改写成特殊形式的卷积；
- 某些 `MatMul` 也可以进一步映射成卷积或卷积风格的 packed kernel。

这样做的主要目的不是改变数学含义，而是复用成熟的底层 kernel、pack 布局和缓存优化逻辑。对推理框架来说，算子“长什么样”不是最重要的，最后能否落到高效稳定的内核实现上才更关键。

### 1.6 Pipeline类

这一层对应的核心类是 `Pipeline`。`Pipeline` 不是单个算子，而是一条已经切分好的执行流水线，负责把 `Schedule::PipelineInfo` 里的算子信息真正落到 `Execution`、`Tensor` 和运行时后端上。

`source/core/Pipeline.hpp` 里最重要的三个阶段是：

- `encode()`：计算 shape、做 geometry transform，并把 op / tensor 信息写入缓冲；
- `allocMemory()`：为每个 op 创建 `Execution` 并分配内存；
- `execute()`：按流水线顺序真正跑算子。

```cpp
// source/core/Pipeline.hpp
class Pipeline : public NonCopyable {
public:
    ErrorCode encode(bool supportDebug = false, bool permitCodegen = false);
    ErrorCode allocMemory(bool firstMalloc, bool permitCodegen);
    ErrorCode execute();
};
```

这里可以把 `Pipeline` 理解成一次“从调度信息到可执行算子序列”的落地过程。单看注释其实已经很清楚：

```cpp
/** encode :
   1. compute shape for every op's inputs and outputs;
   2. geometry transform;
   3. copy op, inputs and outputs tensor info to mBuffer
*/
ErrorCode encode(bool supportDebug = false, bool permitCodegen = false);
```

也就是说，`Pipeline` 干的不是“再做一次调度”，而是把调度阶段已经决定好的结果继续往后推进：

1. 先把 shape 和几何变换补齐；
2. 再创建每个算子的 `Execution`；
3. 最后按顺序进入 `execute()`。

如果只看运行链路，它基本处在：

```text
Session::run()
    ↓
Pipeline::execute()
    ↓
Execution::onExecute()
```

所以 `Pipeline` 是 `Session` 下面真正把“算子序列”跑起来的那一层。

### 1.7 Session类

这一层对应的核心类是 `Session`。`Session` 负责把模型切成一组 `Pipeline`，再把当前运行时、输入输出张量、resize 状态和 cache 状态串起来。

`source/core/Session.hpp` 里可以直接看到它的职责：

- 持有 `RuntimeInfo`；
- 持有 `std::vector<std::shared_ptr<Pipeline>> mPipelines`；
- 提供 `run()`、`runWithCallBack()`、`resize()`、`getInput()`、`getOutput()` 等入口。

```cpp
// source/core/Session.hpp
class MNN_PUBLIC Session {
public:
    ErrorCode run() const;
    ErrorCode runWithCallBack(const TensorCallBackWithInfo& before,
                              const TensorCallBackWithInfo& after,
                              bool sync = false) const;
    ErrorCode resize();
};
```

`Session` 的关键不只是“有一组 `Pipeline`”，而是它把运行模式、后端、输入输出张量和 cache 策略放到了同一个上下文里。比如 `ModeGroup` 里就直接收了这些控制项：

```cpp
struct ModeGroup {
    Interpreter::SessionMode inputMode = Interpreter::Session_Input_Inside;
    Interpreter::SessionMode outputMode = Interpreter::Session_Output_Inside;
    Interpreter::SessionMode backendMode = Interpreter::Session_Backend_Fix;
    Interpreter::SessionMode resizeMode = Interpreter::Session_Resize_Direct;
    Interpreter::SessionMode memoryUsageMode = Interpreter::Session_Memory_Collect;
};
```

而 `source/core/Session.cpp` 里的 `createPipelineBackend()` 会进一步把调度结果落成具体后端：

```cpp
// source/core/Session.cpp
auto rt         = runtime.first.find(iter.first.info.type)->second.get();
auto cpuRuntime = runtime.second;
iter.first.cache.first.reset(rt->onCreate(iter.first.info.user));

if (iter.first.cache.first->type() == MNN_FORWARD_CPU && (!specialUsage)) {
    iter.first.cache.second = iter.first.cache.first;
} else {
    iter.first.cache.second.reset(cpuRuntime->onCreate(&defaultConfig, origin));
}
```

这里的设计重点是：

- 主后端负责按用户选择真正执行算子；
- CPU 备份后端负责 shape 计算和不支持算子的兜底；
- 一个 `Session` 里可以有多个 `Pipeline`，每个 `Pipeline` 共享同一套运行时上下文。

所以 `Session` 不是单纯的数据容器，而是一次推理的完整执行上下文。

### 1.8 Executor & ExecutorScope类

这一层对应的核心类是 `Executor` 和 `ExecutorScope`。`Executor` 负责 `Express` 图的 shape 推导、依赖收集和 `ComputeCache` 构建；`ExecutorScope` 则是线程局部的当前执行器入口。

先看 `Executor` 的公开接口：

```cpp
// include/MNN/expr/Executor.hpp
class MNN_PUBLIC Executor {
public:
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
    };
    Requirement getRequirement(Expr* expr) const;
    ErrorCode computeInfo(Expr* expr);
    void makeCache(const std::vector<EXPRP>& expr, bool forceCPU = false);
};
```

这里三件事分别对应三层职责：

- `getRequirement()`：判断一个 `Expr` 的输入到底是“只要 shape”还是“必须读内容”；
- `computeInfo()`：调用 `SizeComputer::computeOutputSize()` 推导输出张量信息；
- `makeCache()`：把当前 `Expr` 子图整理成 `ComputeCache`，并在里面构造临时 `Session`。

`express/Executor.cpp` 里的 `computeInfo()` 实现就是这条链路的核心：

```cpp
// express/Executor.cpp
bool res = SizeComputer::computeOutputSize(op, inputTensors, expr->inside()->mOutputTensors);
if (!res) {
    return COMPUTE_SIZE_ERROR;
}
for (int i = 0; i < expr->outputSize(); ++i) {
    auto tensor = expr->inside()->mOutputTensors[i];
    TensorUtils::setLinearLayout(tensor);
    auto shape  = expr->outputInfo(i);
    Utils::copyTensorToInfo(shape, tensor);
}
```

而 `_makeCache()` 则更像一次“小型调度器”，它会：

1. 从输出 `Expr` 逆向遍历依赖；
2. 收集每个算子的输入输出 `Tensor`；
3. 组装 `Schedule::ScheduleInfo`；
4. 最后直接构造一个临时 `Session` 塞进 `ComputeCache`。

```cpp
// express/Executor.cpp
scheduleInfo.pipelineInfo.resize(1);
auto& pipeline = scheduleInfo.pipelineInfo[0].second;
...
pipeline.emplace_back(std::move(opInfo));
...
cahce->mSession.reset(new Session(std::move(scheduleInfo), group, std::move(rt)));
```

所以 `Executor` 在 `Express` 层的作用，不只是“做 shape 推导”，还包括把动态图临时收敛成一次可执行的 `Session`。

`ExecutorScope` 则负责回答另一个问题：**当前这段代码到底应该使用哪个 `Executor`**。

```cpp
// express/ExecutorScope.cpp
const std::shared_ptr<Executor> ExecutorScope::Current() {
    auto exe = _getGlobalScope()->Content();
    if (exe) {
        return exe;
    }
    return Executor::getGlobalExecutor();
}
```

它的设计点在于线程局部作用域：

- 显式进入 `ExecutorScope(current)` 后，当前线程会临时绑定新的 `Executor`；
- 作用域结束时自动退出；
- 如果当前线程没有显式绑定，就回退到全局 `Executor`。

因此，`ExecutorScope` 的作用不是“创建执行器”，而是“决定当前这段代码使用哪个执行器”。这也是 `Express` 模式下能够同时支持全局执行器和局部执行上下文的关键。
