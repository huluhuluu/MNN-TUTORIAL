---
title: "MNN 工厂模式介绍"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-02-26T23:00:00+08:00
draft: false
description: "介绍MNN框架中工厂模式的设计与应用"
slug: "introduce-factory"
tags: ["mnn"]
categories: ["mnn"]

comments: true
---

# MNN 介绍

MNN 同时支持多种硬件后端、多种算子实现以及多套算子 Shape 推导逻辑，MNN 把“选择谁来创建对象”统一交给工厂模式，避免多分支判断的方式选择实例化对象：运行时由 `RuntimeCreator` 工厂创建，执行阶段由各个后端自己的 `Creator` 工厂创建，Shape 推导则由 `SizeComputerSuite` 统一管理。本文按这三层工厂模式来梳理 MNN 的实现。

- [MNN 介绍](#mnn-介绍)
  - [1. 工厂模式](#1-工厂模式)
  - [2. Runtime 工厂](#2-runtime-工厂)
    - [2.1 全局 RuntimeCreator 注册表](#21-全局-runtimecreator-注册表)
    - [2.2 注册过程](#22-注册过程)
    - [2.3 实例化过程](#23-实例化过程)
  - [3. Execution层工厂](#3-execution层工厂)
    - [3.1 全局注册表](#31-全局注册表)
    - [3.2 注册过程](#32-注册过程)
    - [3.3 实例化过程](#33-实例化过程)
  - [4. Shape 工厂](#4-shape-工厂)
    - [4.1 全局注册表](#41-全局注册表)
    - [4.2 注册过程](#42-注册过程)
    - [4.3 实例化过程](#43-实例化过程)
  - [5. 其它工程化做法：局部策略选择器](#5-其它工程化做法局部策略选择器)
  - [6. 小结](#6-小结)

## 1. 工厂模式

MNN 中至少有三类“对象创建”场景：

| 层级 | 被创建的对象 | 解决的问题 |
|------|--------------|------------|
| Runtime 层 | `Runtime` / `Backend` | 不同硬件后端如何接入统一调度框架 |
| Execution层 | `Execution` | 不同 `OpType` 如何绑定到对应后端实现 |
| Shape 层 | `SizeComputer` | 不同算子如何推导输出形状 |

这三层在组织方式上基本一致：

1. 维护一份 `类型 -> Creator` 的注册表；
2. 各实现模块通过注册宏把对应 `Creator` 写入注册表；
3. 执行阶段根据 `type` 或 `OpType` 查表并完成对象创建。

这种组织方式对应的工程效果主要有三点：

- **调用层与实现层分离**：调度逻辑依赖统一接口，不直接依赖具体实现类。
- **扩展点清晰**：新增后端或新增算子实现时，通常只需要补充 `Creator` 和对应注册逻辑。
- **编译边界明确**：`OpenCL`、`Metal`、`QNN` 这类后端可以按编译选项决定是否参与注册。

## 2. Runtime 工厂

这一层对应的核心类是 `RuntimeCreator`、`Runtime` 和 `Backend`。`RuntimeCreator` 负责把后端类型映射到具体 `Runtime` 创建逻辑，`Runtime` 负责管理某个后端族的资源、能力和 `Backend` 创建过程，`Backend` 则是后续执行阶段真正参与算子执行的对象。

### 2.1 全局 RuntimeCreator 注册表

`Runtime` 工厂入口位于 `source/core/Backend.cpp`。核心结构是一张以 `MNNForwardType` 为 key 的全局注册表：

```cpp
static std::map<MNNForwardType, std::pair<const RuntimeCreator*, bool>>& GetExtraCreator() {
    static std::once_flag gInitFlag;
    static std::map<MNNForwardType, std::pair<const RuntimeCreator*, bool>>* gExtraCreator;
    // 延迟初始化全局 RuntimeCreator 注册表
    std::call_once(gInitFlag,
                   [&]() { gExtraCreator = new std::map<MNNForwardType, std::pair<const RuntimeCreator*, bool>>; });
    return *gExtraCreator;
}
```
这里构建的是`后端类型`到`RutimeCreator`的映射表，`RuntimeCreator` 是一个抽象接口，定义了创建 `Runtime` 的方法，后端类型包括 `MNN_FORWARD_CPU`、`MNN_FORWARD_OPENCL`、`MNN_FORWARD_METAL`等。

对外暴露的两个核心接口如下：

```cpp
const RuntimeCreator* MNNGetExtraRuntimeCreator(MNNForwardType type);
bool MNNInsertExtraRuntimeCreator(MNNForwardType type, const RuntimeCreator* creator, bool needCheck);
```

这一层的职责很明确：维护 `后端类型 -> RuntimeCreator` 的映射，并在运行时按后端类型返回对应的创建入口。

### 2.2 注册过程

注册入口在 `source/core/Backend.cpp` 的 `registerBackend()`：

```cpp
void registerBackend() {
    std::call_once(s_flag, [&]() {
        // 先注册 CPU Runtime，再初始化 shape / geometry 相关工厂
        registerCPURuntimeCreator();
        SizeComputerSuite::init();
        GeometryComputer::init();
        // 其它后端按编译选项继续注册
    });
}
```

`CPU` 后端的注册逻辑位于 `source/backend/cpu/CPUBackend.cpp`：

```cpp
void registerCPURuntimeCreator() {
    MNNCoreFunctionInit();
    CPUBackend::initCreatorMap();
    registerCPUOps();
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
};
```

这段初始化顺序对应三件事：

1. 初始化 CPU 核心函数表；
2. 初始化 CPU 后端自己的 `OpType -> Creator` 表；
3. 把 `CPURuntimeCreator` 注册到全局 Runtime 工厂中。


### 2.3 实例化过程

`Express` 侧请求 `Runtime` 时，会进入 `express/Executor.cpp` 的 `Executor::_getOrCreateRuntime`：

```cpp
auto cre = MNNGetExtraRuntimeCreator(type);
if (nullptr == cre) {
    return nullptr;
}
Backend::Info info;
// 组装 Runtime 创建参数
info.type = type;
info.mode = Backend::Info::DIRECT;
info.numThread = numberThread;
info.user = (BackendConfig*)config;
std::shared_ptr<Runtime> rt(cre->onCreate(info));
```

这里的调用信息只有 `type` 和 `Backend::Info`：

- `type` 用来确定后端类型；
- `Backend::Info` 携带线程数、模式和用户配置；
- 真正的 `Runtime` 实例由 `Creator` 返回。

因此这一层解决的是后端接入问题。调度层不直接依赖具体后端类，只依赖统一的创建接口。

`RuntimeCreator::onCreate()` 返回的是 `Runtime`，例如 `source/backend/cpu/CPUBackend.cpp` 里的 `CPURuntimeCreator::onCreate()` 会创建 `CPURuntime`。真正的 `Backend` 实例化在对应 `Runtime` 的 `onCreate()` 里完成，再由后续执行流程使用。

`CPU` 的实现路径可以直接对上：

- `CPURuntimeCreator::onCreate()` 创建 `CPURuntime`
- `CPURuntime::onCreate()` 创建 `CPUBackend` 或其变体
- `CPUBackend` 再进入执行层的 `Creator` 分发

## 3. Execution层工厂

这一层对应的核心类是 `Execution` 和各后端自己的 `Creator`。`Execution` 表示一个算子在特定后端上的具体执行实现，`Creator` 负责根据 `OpType` 返回对应的 `Execution` 创建逻辑。

`Runtime` 层确定后端以后，下一步是确定当前 `Op` 由哪个 `Execution` 实现执行。这一层对应的是后端内部的算子工厂。

### 3.1 全局注册表

以 `CPU` 后端为例，`source/backend/cpu/CPUBackend.hpp` 中定义了 `Creator` 抽象接口：

```cpp
class Creator {
public:
    // 由具体后端 Creator 返回对应的 Execution 实现
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op,
                                Backend* backend) const = 0;
};

static bool addCreator(OpType t, Creator* c);
static std::map<OpType, CPUBackend::Creator*>* gCreator;
```

这一层的 key 已经不是后端类型，而是 `OpType`：

- `Runtime` 工厂确定后端；
- 后端工厂确定该后端下某个算子的执行实现。

### 3.2 注册过程

`CPU` 后端通过注册宏把 `Creator` 写入 `gCreator`：

```cpp
#define REGISTER_CPU_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static name _temp;                        \
        /* 把当前 opType 对应的 Creator 写入全局表 */ \
        CPUBackend::addCreator(opType, &_temp);   \
    }
```

典型注册形式如下：

```cpp
REGISTER_CPU_OP_CREATOR(CPUSoftmaxCreator, OpType_Softmax);
REGISTER_CPU_OP_CREATOR(CPUMatMulCreator, OpType_MatMul);
REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionCreator, OpType_Attention);
```

宏展开以后，实际还是调用 `CPUBackend::addCreator` 把 `Creator` 插入 `gCreator`：

```cpp
bool CPUBackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator;
    // 每个 OpType 只允许注册一个 Creator
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}
```

这一层和 `Runtime` 工厂的结构一致，只是映射关系从 `后端类型 -> Creator` 变成了 `OpType -> Creator`。

同样的组织方式也出现在 `OpenGL`、`Metal`、`QNN` 等后端。例如 `QNN` 后端直接使用：

```cpp
#define REGISTER_QNN_OP_CREATOR(name, opType)       \
    void ___##name##__##opType##__() {              \
        QnnBackend::addCreator(opType, new name);   \
    }
```

这说明这一层不是 `CPU` 特例，而是 MNN 后端扩展的通用组织方式。

### 3.3 实例化过程

当调度层已经把当前 `Op` 分配给 `CPUBackend`，最终会进入 `source/backend/cpu/CPUBackend.cpp` 的 `CPUBackend::onCreate`：

```cpp
Execution* CPUBackend::onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) {
    auto opType = op->type();
    auto map  = gCreator;
    // 按 OpType 查找对应 Creator
    auto iter = map->find(opType);
    if (iter == map->end() ) {
        MNN_PRINT("Don't support type [%s]\n", MNN::EnumNameOpType(op->type()));
        return nullptr;
    }
    // 委托 Creator 构造具体 Execution
    Execution* exe = iter->second->onCreate(inputs, outputs, op, this);
    return exe;
}
```

这里的执行过程可以直接拆成三步：

1. 根据 `op->type()` 取得 `OpType`；
2. 去 `gCreator` 查表；
3. 让对应的 Creator 创建真正的 `Execution`。

对应的调用链如下：

```text
调度器决定使用 CPUBackend
    ↓
CPUBackend::onCreate(op)
    ↓
gCreator[OpType]
    ↓
具体 Creator::onCreate(...) 的算子实例化
    ↓
返回具体 Execution
```

`CPUBackend` 在这里承担的是分发角色。它不直接依赖 `CPUSoftmax`、`CPUConvolution`、`CPUAttention` 这类具体实现，只负责查表并调用对应 `Creator`。

## 4. Shape 工厂

这一层对应的核心类是 `SizeComputer` 和 `SizeComputerSuite`。`SizeComputer` 负责单个算子的输出 shape、格式和部分属性推导，`SizeComputerSuite` 维护 `OpType` 到 `SizeComputer` 的映射关系，并在推导阶段按算子类型返回对应实例。

`Shape` 工厂和前面的 `Runtime`、执行层工厂是并行关系。它不参与算子执行实现选择，而是负责在执行前完成输出信息推导。

### 4.1 全局注册表

`SizeComputerSuite` 的接口定义位于 `source/shape/SizeComputer.hpp`，实际注册表管理位于 `source/shape/SizeComputer.cpp`。这一层同样维护一张按 `OpType` 索引的全局表：

```cpp
void SizeComputerSuite::init() {
    // 已初始化则直接返回
    if (nullptr != gInstance) {
        return;
    }
    // 创建全局单例
    gInstance = new SizeComputerSuite;
    // 按 OpType 范围分配注册表空间
    gInstance->mRegistry.resize(OpType_MAX + 1);
    // 初始状态全部置空
    ::memset(gInstance->mRegistry.data(), 0, gInstance->mRegistry.size() * sizeof(SizeComputer*));
    // 统一注册各个算子的 SizeComputer
    registerShapeOps();
}

void SizeComputerSuite::insert(SizeComputer* t, OpType type) {
    // 把当前 OpType 对应的 SizeComputer 写入表中
    mRegistry[type] = t;
}

SizeComputer* SizeComputerSuite::search(OpType name) {
    // 按 OpType 直接查表
    auto iter = mRegistry[name];
    if (iter == nullptr) {
        return nullptr;
    }
    return iter;
}
```

这套结构和前两节是一致的：

- `init()` 负责初始化注册表并触发统一注册；
- `insert()` 负责把具体 `SizeComputer` 写入指定 `OpType`；
- `search()` 负责在推导阶段按 `OpType` 查表。

### 4.2 注册过程

注册入口在 `source/shape/SizeComputer.hpp`：

```cpp
#define REGISTER_SHAPE(name, op)                          \
    void ___##name##__##op##__() {                        \
        name* _temp = new name;                           \
        /* 把当前 OpType 对应的 SizeComputer 写入注册表 */ \
        SizeComputerSuite* ts = SizeComputerSuite::get(); \
        ts->insert(_temp, op);                            \
    }
```

典型注册形式如下：

```cpp
REGISTER_SHAPE(ConvolutionSizeComputer, OpType_Convolution);
REGISTER_SHAPE(MatMulSizeComputer, OpType_MatMul);
REGISTER_SHAPE(ArgMaxComputer, OpType_ArgMax);
```

宏展开以后，本质上仍然是创建具体 `SizeComputer` 对象，并调用 `SizeComputerSuite::insert()` 写入全局表。这一步和前面两层的区别只在于注册对象类型不同，注册流程本身没有变化。

### 4.3 实例化过程

在 `Express` 侧，`Expr::requireInfo()`、`Executor::computeInfo()` 等流程最终都会进入 `SizeComputer::computeOutputSize()` 或 `SizeComputer::computeFlops()`。这两个静态函数内部都会先查表，再调用对应 `SizeComputer`：

```cpp
float SizeComputer::computeFlops(const MNN::Op* op,
                                 const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    auto computer       = computeFactory->search(op->type());
    if (nullptr != computer) {
        return computer->onComputeFlops(op, inputs, outputs);
    }
    ...
}
```

`computeOutputSize()` 的调用方式也是同一条路径：先按 `OpType` 取出对应 `SizeComputer`，再进入该算子的 `onComputeSize()`。

这一层的调用链可以整理成：

```text
Expr::requireInfo() / Executor::computeInfo()
    ↓
SizeComputer::computeOutputSize(...) / computeFlops(...)
    ↓
SizeComputerSuite::search(OpType)
    ↓
具体 SizeComputer::onComputeSize(...) / onComputeFlops(...)
```

对应关系可以直接理解为：

- 后端工厂负责执行实现选择；
- `Shape` 工厂负责输出信息推导。

## 5. 其它工程化做法：局部策略选择器

前面几节对应的是典型的注册表式工厂。除此之外，MNN 里还有一类更局部的选择逻辑，形式上更接近模块内部的策略分发器。 例如：`CPU` 卷积算子链路：

`OpType_Convolution` 首先仍然走标准的 `CPUBackend::Creator` 注册链路：

```cpp
class ConvolutionFactory : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op,
                                Backend* backend) const override {
        return ConvolutionFloatFactory::create(inputs, outputs, op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(ConvolutionFactory, OpType_Convolution);
```

第一层入口并没有变化，仍然是 `OpType -> Creator`。

但卷积不在这一步直接落到单一实现。`ConvolutionFactory` 继续把选择逻辑转交给 `ConvolutionFloatFactory::create()`：

```cpp
Execution* ConvolutionFloatFactory::create(const std::vector<Tensor*>& inputs,
                                           const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op,
                                           Backend* backend)
```

这个函数内部会继续根据卷积参数和运行时条件做二次分发，例如：

- 是否是多输入卷积
- 是否带量化参数
- 当前 `Backend` 是否处于 `low memory` 模式
- 是否可以走稀疏卷积
- 是否可以走 `Winograd`
- 是否适合走 `1x1 Strassen`
- 是否能走 `KleidiAI` / `OneDNN` 等特定实现

例如卷积带量化参数时，会继续进入 `ConvolutionIntFactory`：

```cpp
if (conv2d->quanParameter()->has_scaleInt()) {
    if (bytes < 4) {
        return nullptr;
    }
    return ConvolutionIntFactory::create(inputs[0], outputs[0], op, backend, quanCommon.get());
}
```

`ConvolutionIntFactory` 内部还会继续区分是否为 `group convolution`：

```cpp
if (1 == group) {
    return createUnit(input, output, op, backend, common,
                      conv2d->bias()->data(), conv2d->bias()->size());
}

return new ConvolutionGroup(backend, subConvolution);
```

这一条路径可以简化成：

```text
OpType_Convolution
    ↓
ConvolutionFactory
    ↓
ConvolutionFloatFactory::create(...)
    ↓
按卷积参数 / 精度 / 内存模式 / 指令集继续分支
    ↓
ConvolutionIntFactory / DenseConvolution / Winograd / Strassen / Group ...
```


## 6. 小结

工厂化结构把框架调度逻辑和具体实现逻辑分离，后端与算子的扩展点都落在 `Creator` 或对应注册入口上，关系如下：

```text
执行时实例化流程是：
MNNForwardType
    ↓
RuntimeCreator 工厂
    ↓
Runtime / Backend
    ↓
OpType
    ↓
Backend::Creator 工厂
    ↓
Execution

同时 执行时的形状计算：

OpType
    ↓
SizeComputer 工厂
    ↓
输出 shape / 格式推导
```
