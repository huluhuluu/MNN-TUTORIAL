---
title: "MNN Backend 介绍"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-02-26T23:00:00+08:00
draft: false
description: "介绍MNN框架的后端支持，以及后端有关的核心代码与功能"
slug: "introduce-backend"
tags: ["MNN", "Backend", "深度学习"]
categories: ["MNN端侧部署"]
comments: true
---

# MNN 介绍
本文将介绍MNN框架的后端支持，并介绍后端有关的核心代码与功能。
- [MNN 介绍](#mnn-介绍)
  - [1. 后端](#1-后端)
    - [1.1 CPU后端系列](#11-cpu后端系列)
      - [1.1.1 x86/x64-SSE4.1后端](#111-x86x64-sse41后端)
      - [1.1.2 x86/x64-AVX2后端](#112-x86x64-avx2后端)
      - [1.1.3 x86/x64-AVX512后端](#113-x86x64-avx512后端)
      - [1.1.4 ARMv7a后端](#114-armv7a后端)
      - [1.1.5 ARMv8后端](#115-armv8后端)
    - [1.2 GPU后端系列](#12-gpu后端系列)
      - [1.2.1 OpenCL后端](#121-opencl后端)
      - [1.2.2 Vulkan后端](#122-vulkan后端)
      - [1.2.3 Metal后端](#123-metal后端)
      - [1.2.4 CUDA后端](#124-cuda后端)
      - [1.2.5 OpenGL后端](#125-opengl后端)
    - [1.3 专用加速器](#13-专用加速器)
      - [1.3.1 CoreML后端](#131-coreml后端)
      - [1.3.2 HIAI后端](#132-hiai后端)
      - [1.3.3 NNAPI后端](#133-nnapi后端)
      - [1.3.4 QNN后端](#134-qnn后端)
      - [1.3.5 NeuroPilot 后端](#135-neuropilot-后端)
    - [1.4 后端相关目录结构](#14-后端相关目录结构)
      - [1.4.1 后端文件说明](#141-后端文件说明)
        - [1.4.1.1 Runtime - 运行时抽象层](#1411-runtime---运行时抽象层)
        - [1.4.1.2 Backend - 后端抽象基类](#1412-backend---后端抽象基类)
        - [1.4.1.3 Execution - 执行器抽象基类](#1413-execution---执行器抽象基类)
      - [1.4.2 后端调用设置](#142-后端调用设置)
        - [1.4.2.1 Core Function](#1421-core-function)

## 1. 后端

[MNN](https://github.com/alibaba/MNN)是阿里巴巴开源的高效轻量级深度学习推理框架，提供了较为**全面的后端支持**。 主要包括: 


| 架构/精度 | 类型           | Normal | FP16         | BF16         | Int8 |
| --------- | -------------- | ------ | ------------ | ------------ | ---- |
| CPU       | Native         | B      | C            | B            | B    |
| CPU       | x86/x64-SSE4.1 | A      | C            | C            | A    |
| CPU       | x86/x64-AVX2   | S      | C            | C            | A    |
| CPU       | x86/x64-AVX512 | S      | C            | C            | S    |
| CPU       | ARMv7a         | S      | S（ARMv8.2） | C            | S    |
| CPU       | ARMv8          | S      | S（ARMv8.2） | S（ARMv8.6） | S    |
| GPU       | OpenCL         | A      | S            | C            | S    |
| GPU       | Vulkan         | A      | A            | C            | A    |
| GPU       | Metal          | A      | S            | C            | S    |
| GPU       | CUDA           | A      | S            | C            | A    |
| NPU       | CoreML         | A      | C            | C            | C    |
| NPU       | HIAI           | A      | C            | C            | C    |
| NPU       | NNAPI          | B      | B            | C            | B    |
| NPU       | QNN            | C      | B            | C            | C    |

- **S（Support and work well）**：深度优化，推荐使用
- **A（Support and work well）**：支持良好，可以使用
- **B（Support but has bug or not optimized）**：支持但有bug或未优化，不推荐使用
- **C（Not Support）**：不支持

### 1.1 CPU后端系列

####  1.1.1 x86/x64-SSE4.1后端

**技术背景**：
SSE4.1（Streaming SIMD Extensions 4.1）是Intel在2007年推出的SIMD指令集扩展，提供了47条新指令，主要用于加速多媒体和浮点运算。

**适用设备**：一般是较老CPU

- Intel Core系列处理器（2008年后）
- AMD处理器（部分支持）
- 服务器和桌面PC
- 部分笔记本电脑

**技术特点**：

- 128位SIMD寄存器
- 支持打包整数和浮点运算
- 提供点积、混合、提取和插入操作



#### 1.1.2 x86/x64-AVX2后端

**技术背景**：
AVX2（Advanced Vector Extensions 2）是Intel在2013年推出的**256位SIMD指令集**，是AVX的增强版本，提供了更宽的向量寄存器和更多的指令。

**适用设备**：主流CPU

- Intel Haswell架构及以后的处理器
- AMD Excavator架构及以后的处理器
- 现代服务器和高性能工作站
- 游戏PC和高端笔记本

#### 1.1.3 x86/x64-AVX512后端

**技术背景**：
AVX512是Intel最新的**512位SIMD指令集**，首次出现在Xeon Phi处理器中，后来扩展到Xeon和Core系列。

**适用设备**：主流高性能CPU

- Intel Xeon Scalable处理器
- Intel Core i7/i9高端处理器（部分型号）
- 高性能计算服务器
- AI训练和推理专用硬件

#### 1.1.4 ARMv7a后端

**技术背景**：
ARMv7-A是ARM公司的**32位架构**，广泛应用于早期智能手机和嵌入式设备。支持NEON SIMD指令集。

**适用设备**：一般是比较老旧的移动端CPU

- 早期Android手机（2010-2015年）
- Raspberry Pi 2
- 部分嵌入式开发板
- 工业控制设备

#### 1.1.5 ARMv8后端

**技术背景**：
ARMv8-A是ARM的**64位架构**，是现代移动设备的主流架构，提供了更强的性能和更大的内存寻址空间。

**适用设备**：当前主流的移动端CPU

- 现代智能手机（iPhone 5s以后，Android旗舰机）
- 平板电脑
- ARM服务器（如AWS Graviton）
- Apple M系列芯片设备
- Raspberry Pi 3/4



### 1.2 GPU后端系列

#### 1.2.1 OpenCL后端

**技术背景**：
OpenCL（Open Computing Language）是Khronos Group制定的开放标准，用于异构计算平台的并行编程。

**适用设备**：除专用的GPU后端外，主流的GPU后端

- 支持OpenCL的GPU（NVIDIA、AMD、Intel）
- 部分移动GPU（Adreno、Mali、PowerVR）
- FPGA设备
- DSP处理器

#### 1.2.2 Vulkan后端

**技术背景**：
Vulkan是Khronos Group开发的低开销、跨平台的3D图形和计算API，提供了更直接的GPU控制。

**适用设备**：大部分GPU支持的后端

- 现代GPU（NVIDIA GTX 900系列以后）
- AMD GCN架构GPU
- Intel集成显卡（部分支持）
- Android设备（API 24+）


#### 1.2.3 Metal后端

**技术背景**：
Metal是Apple开发的低级图形和计算API，专为Apple设备优化，提供了接近硬件的性能。

**适用设备**：ios设备的GPU后端

- iPhone iPad Mac等

#### 1.2.4 CUDA后端

**技术背景**：
CUDA（Compute Unified Device Architecture）是NVIDIA开发的并行计算平台和编程模型，专为NVIDIA GPU设计。

**适用设备**：NVIDIA GPU的专用后端

- NVIDIA GeForce系列GPU
- NVIDIA Quadro专业卡
- NVIDIA Tesla计算卡
- NVIDIA Jetson嵌入式平台

#### 1.2.5 OpenGL后端

**技术背景**：OpenGL（Open Graphics Library）是一个图形渲染 API 标准，可以利用 GPU 并行计算能力处理非图形任务

**适用设备**：大部分GPU支持的后端

- 支持OpenCL的GPU（NVIDIA等）
- 部分移动GPU（Adreno等）

### 1.3 专用加速器

#### 1.3.1 CoreML后端

**技术背景**：
Core ML是Apple的机器学习框架，能够利用Apple设备的Neural Engine进行AI推理加速。

**适用设备**：当前主流的apple ai加速器

- iPhone（A11芯片以后，支持Neural Engine）
- iPad Pro（A12芯片等）
- Mac（M1芯片等）

#### 1.3.2 HIAI后端

**技术背景**：
HIAI（Huawei Intelligent Acceleration Infrastructure）是华为开发的AI加速平台，利用华为麒麟芯片的NPU。

**适用设备**：当前主流的华为 ai加速器

- 华为/荣耀手机（麒麟970以后）
- 华为平板电脑
- 华为笔记本电脑（部分型号）

#### 1.3.3 NNAPI后端

**技术背景**：
NNAPI（Neural Networks API）是Google在Android 8.1中引入的API，为Android设备提供统一的AI加速接口。

**适用设备**：在 Android 15 中已弃用。在专用供应商驱动程序的Android设备运行，缺乏时运行在 CPU 上

- Android 8.1+设备

#### 1.3.4 QNN后端

**技术背景**：
QNN（Qualcomm Neural Network SDK）是高通开发的AI推理SDK，专为高通Snapdragon平台优化。

**适用设备**：主流的高通芯片NPU后端

- 搭载Snapdragon处理器的设备
- 支持Hexagon DSP的设备
- 高通AI开发套件

#### 1.3.5 NeuroPilot 后端

**技术背景**：
 NeuroPilot SDK是联发科自研的专用 AI 加速硬件单元SDK，集成于天玑（Dimensity）系列 SoC 中。主要通过 NNAPI 间接调用 APU 硬件加速

**适用设备**：主流的天玑芯片APU后端

- 搭载天玑系列处理器的设备
- 支持 NeuroPilot SDK 且 Android 10+（API 29+）的联发科平台设备

### 1.4 后端相关目录结构

MNN的后端实现集中在`source/backend/`目录中：

```
source/backend/
├── cpu/          # CPU通用后端
|    └───x86_x64
|    		└─── avx/          # x86 AVX2优化
|    		└─── avx512/       # x86 AVX512优化
|    		└─── avxfma/       # x86 AVXfma 支持Fused Multiply-Add
|	 		└─── sse/          # x86 SSE优化
|    └─── arm
|    		└───arm32		   # 32为arm指令支持 如ARMv7-A后端
|    		└───arm64		   # 64为arm指令支持
├── arm82/        # ARMv8.2 支持
|    └─── asm
|    		└───arm32		   # 32为arm指令兼容
|    		└───arm64		   # 64为arm指令支持
├── nnapi/        # NNAPI后端
├── metal/        # Metal后端（iOS/macOS）
├── opencl/       # OpenCL后端
├── opengl/       # OpenGL后端
├── vulkan/       # Vulkan后端
├── cuda/         # CUDA后端
├── tensorrt
├── coreml/       # CoreML后端
├── hiai/         # HIAI后端
└── qnn/          # QNN后端
└── neuropilot/   # neuropilot后端
```

相应的后端需要设置指定的[编译宏](https://mnn-docs.readthedocs.io/en/latest/compile/cmake.html)，例如-DMNN_OPENCL=true表示启用OPENCL后端支持，但是具体调用需要在程序中设置，例如MNN-LLM需要设置… 具体见后续的代码梳理

#### 1.4.1 后端文件说明

MNN 的后端系统由以下几个核心类组成，大致执行顺序如下：

```
// 1. 创建 下面的关系只表示时间上的顺序依赖关系 如Backend需要调用Runtime创建
Runtime (运行时, 各类后端继承Runtime, 如class CPURuntime : public Runtime)
  ↓ 创建
Backend (后端, 各类后端继承Backend, 如class CPUBackend : public Backend)
  ↓ 创建
Execution (执行器, 大部分算子实现是通过继承Execution实现,如:class CPUAttention : public Execution;
			小部分为汇编优化过的代码, 如source/backend/cpu/x86_x64/avx512/_AVX512_MNNGemmFloatUnit16x8.S)
			
// 2. 重新计算输出形状 在/workspace/code/MNN/source/shape/SizeComputer.hpp内完成 

// 3. 根据输入长度变化 执行的resize操作 下面的关系只表示时间上的顺序依赖关系
Backend::onResizeBegin (resize的准备阶段)
  ↓
Execution::onResize (算子的resize)
  ↓
Backend::onResizeEnd (resize的收尾阶段)

// 4. 执行算子  下面的关系只表示时间上的顺序依赖关系
Backend::onExecuteBegin (execute的准备阶段)
  ↓
Execution::onExecute (算子的execute)
  ↓
Backend::onExecuteEnd (execute的收尾阶段)
```

##### 1.4.1.1 Runtime - 运行时抽象层

**位置**：`source/core/Backend.hpp`

Runtime 是硬件运行时的抽象层，负责管理整个后端的生命周期和资源配置。

**核心职责**：

- 创建 Backend 实例
- 管理线程池和内存分配器
- 提供垃圾回收机制
- 支持异步执行和并发控制

**关键接口**：

```cpp
class Runtime : public NonCopyable {
public:
    // 创建 Backend
    virtual Backend* onCreate(const BackendConfig* config, Backend* origin) const = 0;
    // 重置运行时配置
    virtual void onReset(int numberThread, const BackendConfig* config, bool full);
    
    // 垃圾回收
    virtual void onGabageCollect(int level) = 0;
    // 获取内存使用量
    virtual float onGetMemoryInMB();
	
    // 主要是cpuruntime使用 cpu后端实现了线程池(source/backend/cpu/ThreadPool.hpp)
    virtual void onConcurrencyBegin() const = 0;
    virtual void onConcurrencyEnd() const = 0;
	
    // 执行优化, mnn以算子图形式运行 下面是不同的运行方式
    enum CompilerType {
        Compiler_Geometry = 0, // 部分执行几何计算，分解形变算子，但不分解 BatchMatMul / Gather 等算子
        Compiler_Origin = 1, // 直接使用原始算子，不进行分解
        Compiler_Loop = 2, // 完全执行几何计算，仅此模式下，可以在算子不支持时自动回退到CPU计算
    };
private:    
    // 记录运行时信息 如kvcacheSizeLimit cpuIds等
    RuntimeHint mHint;
};
```



##### 1.4.1.2 Backend - 后端抽象基类

**位置**：`source/core/Backend.hpp`

Backend 是所有硬件后端的基类，定义了后端必须实现的接口。

**核心职责**：

- 管理内存分配和释放
- 创建 Execution
- 处理张量缓冲区操作
- 支持多种存储策略

**存储类型枚举**：
```cpp
enum StorageType {
    STATIC,           // 不可重用内存，分配后立即释放
    DYNAMIC,          // 可重用内存，优先重用已有内存
    DYNAMIC_SEPERATE, // 不可重用内存，但延迟释放
    DYNAMIC_IN_EXECUTION // 执行时动态分配
};
```

**关键接口**：

```cpp
class Backend : public NonCopyable {
public:
    // 创建 Execution 
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) = 0;

    // 执行算子Resize 的开始/结束的一些准备/收尾工作
    virtual void onResizeBegin();
    virtual ErrorCode onResizeEnd() = 0;
    // 执行算子execute 的开始/结束的一些准备/收尾工作
    // 例如OpenCLBackend中 
    virtual void onExecuteBegin() const = 0;
    virtual void onExecuteEnd() const = 0;

    // 内存管理
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) = 0;
    virtual bool onClearBuffer() = 0;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const = 0;
    
    // 把数据映射到指定后端 / 从指定后端映射回数据
    virtual void* onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor);
    virtual bool onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr);
    
    // 利用backend后端传递指针, llm中主要传递kvmeta(kv cache有关信息)
    void setMetaPtr(void* ptr);

private:
    // 枚举类 记录执行后端 如cpu opencl为: MNN_FORWARD_CPU MNN_FORWARD_OPENCL
    const MNNForwardType mType;
};
```

##### 1.4.1.3 Execution - 执行器抽象基类

**位置**：`source/core/Execution.hpp`

Execution 是算子执行的抽象基类，每个算子都有对应的 Execution 实现。

**核心职责**：

- 响应输入输出张量的形状变化
- 执行算子计算
- 支持克隆和共享权重

**关键接口**：
```cpp
class Execution : public NonCopyable {
public:
    // 根据输入seq_len长度变化 调整执行时用到的值
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs);

    // 执行计算
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) = 0;

    // 克隆执行器 这在llm推理中很有用 例如prefill和decode之间的seq_len输入长度不一样 就会触发克隆 mnn默认保存不同输入长度的计算图
    // 这里通常不克隆权重, 权重在Op* op中 这里会通过指针引用
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst);

    // 获取后端 例如在CPUAttention中利用这里获取kvmeta信息(mMeta = (KVMeta*)(backend->getMetaPtr());)
    Backend* backend() const;
};
```



#### 1.4.2 后端调用设置

除了后端架构中的 Runtime、Backend、Execution 等核心类，MNN中还有部分特定**基础**算子的优化代码，例如```source/backend/cpu/x86_x64/avx512/_AVX512_MNNGemmFloatUnit16x8.S``` 等 ，前者是汇编代码，后者是opencl的代码。

这里指的算子一般都是底层支持的基础算子，会在更上层的算子中被调用，例如CPUAttention中会调用core->Int8GemmKernel;以使用int8矩阵乘算子

##### 1.4.2.1 Core Function

下面以CPU的后端在实例化过程中调用的不同后端的指令集为例说明，在source/backend/cpu/compute/CommonOptFunction.h中定义了CoreFunctions

```cpp
struct CoreFunctions {
    // CPU 特性标志
    bool supportFp16arith = false;  // 支持 FP16 算术运算
    bool supportSDot = false;       // 支持 ARM S-Dot 指令
    bool supportI8mm = false;       // 支持 ARM I8MM 指令
    bool supportSME2 = false;       // 支持 ARM SME2 指令
    int  smeCoreNumber = 0;         // SME2 核心数量

    // 矩阵乘法相关函数
    void(*MNNGetMatMulPackMode)(int* eP, int *lP, int* hP);
    void(*MNNPackC4ForMatMul_A)(float* destOrigin, float const** sourceGroup,
                                 const int32_t* info, const int32_t* el);
    void(*MNNPackForMatMul_B)(float* dest, const float* source,
                              size_t h, size_t kernelsize, size_t ic, bool transpose);
    void(*MNNPackedMatMul)(float* C, const float* A, const float* B,
                           const size_t* parameter, const float* postParameters,
                           const float* bias, const float* k, const float* b);
    void(*MNNPackedMatMulRemain)(float* C, const float* A, const float* B,
                                 size_t eSize, const size_t* parameter,
                                 const float* postParameters, const float* bias,
                                 const float* k, const float* b);
    // 激活函数
    void(*MNNReluInt8)(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint);
    void(*MNNHardSwish)(float* dst, const float* src, size_t size);
    void(*MNNGelu)(float* dst, const float* src, size_t size, float* parameters);
    const float *beta, float epsilon, size_t size, bool RMSNorm);

    // 其他函数...
};
```

MNN 使用全局单例 `gCoreFunction` 来存储当前平台的最优函数实现，在程序启动时会选择各个函数的最优实现代码
```cpp
// 在source/backend/cpu/CPUBackend.cpp 中创建runtime时 会进入到不同指令集的初始化注册过程
{
#ifdef MNN_SUPPORT_BF16
extern void registerBF16Backend();
#endif
#ifdef ENABLE_ARMV82
extern void registerArm82RuntimeCreator();
#endif
void registerCPURuntimeCreator() {
    MNNCoreFunctionInit();
    CPUBackend::initCreatorMap();
    registerCPUOps();
#ifdef MNN_SUPPORT_BF16
    registerBF16Backend();
#endif
#ifdef MNN_USE_ARMV82
    registerArm82RuntimeCreator();
#endif
    // TODO: Merge _initCoreFunction MNNFunctionInit and cpuinfo_arm_init
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_CPU, new CPURuntimeCreator);
}

// 下面以armv82的初始化为例 在Arm82Functions::init()中
Arm82Functions::init(){
    
    // 部分特殊优化的底层代码
    FUNC_PTR_ASSIGN(gInstance->MNNPackC4ForMatMul_A, Arm82MNNPackForMatMul_A);
    /* 在声明中 这个代码是外部导入的, 路径位置source/backend/arm82/asm/arm64/Arm82MNNPackForMatMul_A.S
    extern "C" {
    // (UP_DIV(l,8), e, 8) -> (UP_DIV(e,eP), l, eP)
        void Arm82MNNPackForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
    }
    */
    
    // 其它代码...
}
```

在cpu后端使用时通过调用后端的core function执行底层算子，如

```cpp
// source/backend/cpu/CPUAttention.cpp
ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto gcore  = static_cast<CPUBackend *>(backend())->functions();
    auto core   = static_cast<CPUBackend*>(backend())->int8Functions();
    // 其它代码...
 	
    // 调用后端的core functions终端矩阵乘法
    gcore->MNNPackedMatMul(...);
    
    // 其它代码...
}
```

其它后端的底层代码调用可能不同，如opencl后端的.cl代码```source/backend/opencl/execution/cl/attention_buf.cl```等 通过运行时加载 Kernel 源码方式使用。