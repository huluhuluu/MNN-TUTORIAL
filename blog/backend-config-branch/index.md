---
title: "MNN BackendConfig 分支路径"
date: 2026-06-16T12:00:00+08:00
lastmod: 2026-06-16T12:00:00+08:00
draft: false
description: "对照 MNN 源码梳理 precision、memory、power 不同取值对应的执行分支"
slug: "backend-config-branch"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# MNN BackendConfig 分支路径
- [MNN BackendConfig 分支路径](#mnn-backendconfig-分支路径)
  - [1. 配置入口](#1-配置入口)
    - [1.1 JSON 默认值](#11-json-默认值)
    - [1.2 字符串写入 BackendConfig](#12-字符串写入-backendconfig)
  - [2. CPU 后端接收配置](#2-cpu-后端接收配置)
  - [3. 编译/运行时指令集](#3-编译运行时指令集)
    - [3.1 x86](#31-x86)
    - [3.2 ARM](#32-arm)
  - [4. precision 分支](#4-precision-分支)
    - [4.1 normal / high](#41-normal--high)
    - [4.2 low](#42-low)
    - [4.3 low\_bf16](#43-low_bf16)
  - [5. memory 分支](#5-memory-分支)
    - [5.1 入口：ConvolutionFloatFactory](#51-入口convolutionfloatfactory)
    - [5.2 int4 / int8 / fp16 / fp32 权重](#52-int4--int8--fp16--fp32-权重)
    - [5.3 权重量化加载：forceInt8](#53-权重量化加载forceint8)
    - [5.4 memory = high：反量化成 float](#54-memory--high反量化成-float)
    - [5.5 执行器选择：low 和 high 分叉](#55-执行器选择low-和-high-分叉)
    - [5.6 packed weight](#56-packed-weight)
    - [5.7 memory = low 的内存节省](#57-memory--low-的内存节省)
    - [5.8 MNN\_CPU\_WEIGHT\_DEQUANT\_GEMM 的反量化](#58-mnn_cpu_weight_dequant_gemm-的反量化)
    - [5.9 LLM load 时权重什么时候打包](#59-llm-load-时权重什么时候打包)
  - [6. power 分支](#6-power-分支)
  - [7. 导出精度和运行时精度](#7-导出精度和运行时精度)
  - [8. 速查表](#8-速查表)

本文记录 `BackendConfig` 里的 `precision`、`memory`、`power` 如何从 LLM 的 `config.json` 进入 MNN runtime，并最终影响 CPU 后端分支。

重点放在 Android 端 LLM 常见配置：

```json
{
  "backend_type": "cpu",
  "thread_num": 4,
  "precision": "low",
  "memory": "low",
  "power": "normal"
}
```

核心结论是，详细查看[速查表](#8-速查表)快速理解不同配置的主要影响：

| 字段 | LLM 默认值 | CPU 主影响 |
| 反量化/计算时机 | 主要表现 | CPU 主影响 |
|------------------|----------|------------|
| `precision = low` | 运行时 FP16 计算 | ARMv8.2 FP16 可用时创建 `Arm82Backend`，函数表从 FP32 换成 FP16 |
| `memory = low` | 不提前生成整层 `weightFloat`，执行期按 block 做缩放还原 | 对权重量化卷积，保留 int8/int4 权重，走低内存动态量化执行器 |
| `memory = high` | 先把量化权重反量化成 FP32 `weightFloat` | 避免低内存卷积分支，允许普通 dense/winograd 路径 |
| `power = normal` | 不涉及反量化 | 不主动指定大核 / 小核 |



## 1. 配置入口

### 1.1 JSON 默认值

`llm_demo` 读取模型目录里的 `config.json`。默认值在 `transformers/llm/engine/src/llmconfig.hpp`：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:353
std::string precision(bool mllm = false) const {
    if (mllm) return mllm_config_.value("precision", "low");
    return config_.value("precision", "low");
}
std::string power(bool mllm = false) const {
    if (mllm) return mllm_config_.value("power", "normal");
    return config_.value("power", "normal");
}
std::string memory(bool mllm = false) const {
    if (mllm) return mllm_config_.value("memory", "low");
    return config_.value("memory", "low");
}
```

LLM 默认是 `precision = low`、`memory = low`、`power = normal`。

### 1.2 字符串写入 BackendConfig

`transformers/llm/engine/src/llm.cpp` 的 `Llm::initRuntime()` 把字符串转换成 `BackendConfig` 枚举：

```cpp
// transformers/llm/engine/src/llm.cpp:166
ScheduleConfig config;
BackendConfig cpuBackendConfig;
config.type      = backend_type_convert(mConfig->backend_type());
config.numThread = mConfig->thread_num();

if (mConfig->power() == "high") {
    cpuBackendConfig.power = BackendConfig::Power_High;
} else if (mConfig->power() == "low") {
    cpuBackendConfig.power = BackendConfig::Power_Low;
}
if (mConfig->memory() == "high") {
    cpuBackendConfig.memory = BackendConfig::Memory_High;
} else if (mConfig->memory() == "low") {
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
}
if (mConfig->precision() == "high") {
    cpuBackendConfig.precision = BackendConfig::Precision_High;
} else if (mConfig->precision() == "low") {
    cpuBackendConfig.precision = BackendConfig::Precision_Low;
}
config.backendConfig = &cpuBackendConfig;
```

这里的字符串入口只处理 `"high"` 和 `"low"`。写 `"normal"` 时不会进入 `if`，保留 `BackendConfig` 构造时的默认枚举。

完整枚举在 `include/MNN/MNNForwardType.h`：

```cpp
// include/MNN/MNNForwardType.h:83
struct BackendConfig {
    enum MemoryMode { Memory_Normal = 0, Memory_High, Memory_Low };
    MemoryMode memory = Memory_Normal;

    enum PowerMode { Power_Normal = 0, Power_High, Power_Low };
    PowerMode power = Power_Normal;

    enum PrecisionMode { Precision_Normal = 0, Precision_High, Precision_Low, Precision_Low_BF16 };
    PrecisionMode precision = Precision_Normal;
};
```

## 2. CPU 后端接收配置

CPU 路径先把 `BackendConfig` 存进 `CPURuntime`。代码在 `source/backend/cpu/CPUBackend.cpp`：

```cpp
// source/backend/cpu/CPUBackend.cpp:222
CPURuntime::CPURuntime(const Backend::Info& info) {
    mPower   = BackendConfig::Power_Normal;
    mMemory  = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
        mFlags = info.user->flags;
    }
}
```

创建 backend 时，`CPURuntime::onCreate()` 取出 `precision` 和 `memory`：

```cpp
// source/backend/cpu/CPUBackend.cpp:306
auto precision = mPrecision;
auto memory = mMemory;
if (nullptr != config) {
    precision = config->precision;
    flags = config->flags;
    memory = config->memory;
}

if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
    res = new Arm82Backend(this, memory);
    break;
}
if (precision == BackendConfig::Precision_Low_BF16 && BF16Functions::get()) {
    res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU_EXTENSION);
    res->mCoreFunctions = BF16Functions::get();
    break;
}
res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, flags);
```

这一步的分工很清楚：

- `precision` 决定是否创建 `Arm82Backend` / BF16 extension / 普通 `CPUBackend`；
- `memory` 不决定 backend 类型，只是继续传给 backend；
- 后续算子创建器通过 `CPUBackend::memoryMode()` 读取 `memory`。

`CPUBackend` 构造函数只是保存这个值：

```cpp
// source/backend/cpu/CPUBackend.cpp:448
CPUBackend::CPUBackend(..., BackendConfig::MemoryMode memory, ...)
    : Backend(type) {
    mMemory = memory;
    ...
    mPrecisionMode = precision;
    mCoreFunctions = MNNGetCoreFunctions();
    mInt8CoreFunctions = MNNGetInt8CoreFunctions();
}
```

## 3. 编译/运行时指令集

CPU 后端常见指令集包括：

| 指令集 / 扩展 | 平台 | 主要能力 | MNN 中的典型体现 |
|---------------|------|----------|------------------|
| SSE / SSSE3 / SSE4.1 | x86 / x86_64 | x86 基础 SIMD 优化 | 编译 `MNNSSE`，运行期改写默认 `CoreFunctions` 的部分函数 |
| AVX2 / FMA | x86 / x86_64 | 更宽 SIMD 和 FMA，输出通道按 8 个一组排布和计算（MNN 里对应 `pack = 8`） | 编译 `MNNAVX` / `MNNAVXFMA`，运行期可创建 `AVX2Backend` |
| AVX512 | x86 / x86_64 | 更宽向量和 VNNI 等扩展，输出通道可按 16 个一组排布和计算（MNN 里对应 `pack = 16`） | 默认关闭，编译并运行期探测通过后在 `AVX2Functions` 内切换 |
| NEON | ARMv7 / AArch64 | ARM 基础 SIMD 优化 | 编译 ARM 汇编和 `CommonOptFunctionNeon.cpp`，定义 `MNN_USE_NEON` |
| ARMv8.2 FP16 / Arm82 | AArch64 为主 | FP16 arithmetic | `precision = low` 且运行期支持 FP16 时创建 `Arm82Backend`，`bytes = 2` |
| dot / i8mm / SME2 | AArch64 | int8 dot / matrix / SME2 优化 | 写入 `CoreFunctions` 能力位，供 int8、低内存 matmul 等 kernel 选择 |

这里的 `none` 或关闭某个开关，不是 `BackendConfig` 里的一个枚举。它的实际含义是：对应扩展 object、宏分支、扩展 Backend 或扩展函数表不可用，运行时只能落到普通 CPU 后端已有的函数表。

### 3.1 x86

x86 的主线是：

```text
CMake 开关
    -> 编译 SSE / AVX object
    -> 运行期探测 CPU flags
    -> 改写 CoreFunctions
    -> 必要时创建 AVX2Backend
```

顶层开关里，`MNN_USE_SSE` 默认打开；`MNN_AVX2` 默认打开；`MNN_AVX512` 默认关闭：

```cmake
# CMakeLists.txt:68
option(MNN_USE_SSE "Use SSE optimization for x86 if possiable" ON)

# CMakeLists.txt:264
option(MNN_AVX2 "Open AVX2 Compile for x86 if possible" ON)
option(MNN_AVX512 "Enable AVX512" OFF)
```

CPU CMake 里，只有 `MNN_USE_SSE` 打开时才会 include x86/x64 的 CMake。
```cmake
# source/backend/cpu/CMakeLists.txt:27
# X86_64 AVX/SSE
if (MNN_USE_SSE)
    include(${CMAKE_CURRENT_LIST_DIR}/x86_x64/CMakeLists.txt)
endif()
```

进入 `source/backend/cpu/x86_x64/CMakeLists.txt` 后，还要目标处理器匹配 x86，才会真正启用 SSE 分支：

```cmake
# source/backend/cpu/x86_x64/CMakeLists.txt:41
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)|(x86)")
    message(STATUS "${CMAKE_SYSTEM_PROCESSOR}: Open SSE")
    target_compile_options(MNNCPU PRIVATE -DMNN_USE_SSE)
    ...
    add_library(MNNX8664 OBJECT ${MNN_X8664_SRC})
    add_library(MNNSSE OBJECT ${MNN_SSE_SRC})
    target_compile_options(MNNX8664 PRIVATE -DMNN_USE_SSE)
    target_compile_options(MNNSSE PRIVATE -DMNN_USE_SSE)
```

`MNN_AVX2` 打开时，会额外编译 AVX / AVXFMA object，并给 `MNNX8664` 加上 `MNN_USE_AVX`。这个宏决定后面 `FunctionDispatcher.cpp` 里的 AVX 分支是否存在：

```cmake
# source/backend/cpu/x86_x64/CMakeLists.txt:81
if (MNN_AVX2)
    target_compile_options(MNNX8664 PRIVATE -DMNN_USE_AVX)
    add_library(MNNAVX OBJECT ${MNN_AVX_SRC})
    add_library(MNNAVXFMA OBJECT ${MNN_AVXFMA_SRC})
    target_compile_options(MNNAVX PRIVATE -DMNN_USE_SSE)
    target_compile_options(MNNAVXFMA PRIVATE -DMNN_USE_SSE)
endif()
```

非 MSVC 下，SSE object 用 `-msse4.1` 编译，AVX object 用 `-mavx2` / `-mfma`。这一步只是“把对应 kernel 编译出来”，还不等于运行时一定会用：

```cmake
# source/backend/cpu/x86_x64/CMakeLists.txt:93
else()
    target_compile_options(MNNSSE PRIVATE -msse4.1)
    if (MNN_AVX2)
        target_compile_options(MNNAVX PRIVATE -m64 -mavx2 -DMNN_X86_USE_ASM)
        target_compile_options(MNNAVXFMA PRIVATE -m64 -mavx2 -mfma -DMNN_X86_USE_ASM)
    endif()
endif()
```

运行时会再探测 CPU flags。`FunctionDispatcher.cpp` 先用 `libyuv::InitCpuFlags()` 看当前机器支持什么，再把默认 `CoreFunctions` 的部分函数改写成 SSE 版本；如果编译了 `MNN_USE_AVX` 且 CPU 支持 AVX2，再初始化 AVX2 函数表：

```cpp
// source/backend/cpu/x86_x64/FunctionDispatcher.cpp:41
void MNNFunctionInit() {
    auto cpuFlags = libyuv::InitCpuFlags();
    auto coreFunction = MNN::MNNGetCoreFunctions();
    if (cpuFlags & libyuv::kCpuHasSSSE3) {
        coreFunction->MNNGetMatMulPackMode = _SSEMNNGetMatMulPackMode;
        coreFunction->MNNPackedMatMul       = _SSE_MNNPackedMatMul;
        coreFunction->MNNPackedMatMulRemain = _SSE_MNNPackedMatMulRemain;
        ...
    }
#ifdef MNN_USE_AVX
    if (cpuFlags & libyuv::kCpuHasAVX2) {
        MNN::AVX2Functions::init(cpuFlags);
        gFunc.MNNExpC8 = _AVX_MNNExpC8;
        gFunc.MNNSoftmax = _AVX_MNNSoftmax;
        ...
    }
#endif
}
```

`AVX2Functions::init(...)` 会复制当前默认函数表，再替换成 AVX/AVX512 版本。这里的 `pack` 不是输入图片或矩阵的分片大小，而是 MNN 内部张量和权重布局里的通道成组数：输出通道按几个一组连续存放，矩阵乘内核也一次算这一组。普通 CPU 常见是 4 个输出通道一组，AVX2 改成 8 个；如果 AVX512 编译并探测可用，会进一步改成 16 个：

```cpp
// source/backend/cpu/x86_x64/AVX2Functions.cpp:31
bool AVX2Functions::init(int cpuFlags) {
    gAVX2CoreFunctions = new CoreFunctions;
    auto coreFunction = gAVX2CoreFunctions;
    gAVX2CoreInt8Functions = new CoreInt8Functions;
    *coreFunction = *MNNGetCoreFunctions();
    *gAVX2CoreInt8Functions = *MNNGetInt8CoreFunctions();
    ...
    coreFunction->pack = 8;
    ...
#ifdef MNN_AVX512
    if ((cpuFlags & libyuv::kCpuHasAVX512VNNI)
        || (cpuFlags & libyuv::kCpuHasAVX512VL)
        || ...) {
        coreFunction->pack = 16;
        ...
    }
#endif
    return true;
}
```

函数表准备好之后，`CPURuntime::onCreate()` 才会决定是否创建 `AVX2Backend`：

```cpp
// source/backend/cpu/CPUBackend.cpp:339
if (flags == MNN_CPU_USE_DEFAULT_BACKEND) {
    // Default don't use multi-thread init
    res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU);
    break;
}
#ifdef MNN_USE_SSE
if (AVX2Backend::isValid()) {
    res = new AVX2Backend(this, memory, flags);
    break;
}
#endif
res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, flags);
```

`AVX2Backend::isValid()` 只看 `AVX2Functions::get()` 是否已经初始化成功：

```cpp
// source/backend/cpu/x86_x64/AVX2Backend.cpp:27
bool AVX2Backend::isValid() {
    return nullptr != AVX2Functions::get();
}
```

所以几个“none / default”相关配置可以这样区分：

- `MNN_USE_SSE=OFF`：编译期不 include x86/x64 CMake，SSE/AVX object 和 `AVX2Backend` 分支都不会进来。
- `MNN_AVX2=OFF`：SSE 仍可编译，AVX / AVXFMA object 不编译，`MNN_USE_AVX` 分支不存在。
- `MNN_AVX512=OFF`：AVX2 仍可用，但 `AVX2Functions` 里不会切到 AVX512 的 `pack = 16` 路径。
- `MNN_CPU_USE_DEFAULT_BACKEND`：运行期 flag，表示创建普通 `CPUBackend`，跳过 `AVX2Backend` 这种扩展 backend 选择。

这就是“sse none”一类说法背后的实际效果：不是 JSON 里的 `precision/memory/power` 改了，而是编译产物或运行期 backend 选择少了一条 ISA 扩展路径。

### 3.2 ARM
ARM 的主线也类似，但要区分三类东西：

```text
NEON
    -> ARM 普通 CPU 优化

ARMv8.2 FP16 / Arm82
    -> precision = low 时可能切到 FP16 extension backend

dot / i8mm / SME2
    -> int8 / matrix kernel 能力位，不等于单独的 BackendConfig 选项
```

ARM 普通优化在 `source/backend/cpu/arm/CMakeLists.txt`。aarch64 或 armv7 会编译对应汇编和 `CommonOptFunctionNeon.cpp`，并定义 `MNN_USE_NEON`。这属于 ARM CPU 后端的基础优化层：

```cmake
# source/backend/cpu/arm/CMakeLists.txt:22
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?")
    message(STATUS "Enabling AArch32 Assemblies")
    add_library(MNNARM32 OBJECT ${MNN_AArch32_SRC} ${MNN_NEON_SRC})
    ...
    add_definitions(-DMNN_USE_NEON)
    target_compile_options(MNNARM32 PRIVATE -D__arm__)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
    message(STATUS "Enabling AArch64 Assemblies")
    ...
    add_library(MNNARM64 OBJECT ${MNN_AArch64_SRC} ${MNN_NEON_SRC} ...)
    ...
    add_definitions(-DMNN_USE_NEON)
    target_compile_options(MNNARM64 PRIVATE -D__aarch64__)
endif()
```

`MNN_SME2` 默认打开。aarch64 下会定义 `MNN_SME2`，并收集 `arm64/sme2_asm` 里的汇编。和 AVX2 一样，编译进去只是第一步，后面还要看运行期 CPU 是否支持：

```cmake
# source/backend/cpu/arm/CMakeLists.txt:44
if (MNN_SME2)
    add_definitions(-DMNN_SME2)
    FILE(GLOB MNN_SME2_AArch64_SRC ${MNN_SME2_AArch64_SRC} ${CMAKE_CURRENT_LIST_DIR}/arm64/sme2_asm/*.[sS])
    ...
endif()
```

ARMv8.2 FP16 是单独的 `MNN_ARM82` 开关。顶层默认打开：

```cmake
# CMakeLists.txt:260
option(MNN_ARM82 "Enable ARMv8.2's FP16 Compute" ON)
```

但 32 位 armv7 默认会强制关掉，除非打开 `MNN_SUPPORT_FP16_ARMV7`：

```cmake
# CMakeLists.txt:292
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?")
    if (NOT MNN_SUPPORT_FP16_ARMV7)
        message(STATUS "force turn off MNN_ARM82 when build for Android based on arm32 by MTL")
        SET(MNN_ARM82 OFF CACHE BOOL "Enable ARM82" FORCE)
    endif()
endif()
```

CPU CMake 中，只有 `MNN_ARM82` 打开且目标架构是 armv7/aarch64/arm64，才会 include `source/backend/arm82/CMakeLists.txt`：

```cmake
# source/backend/cpu/CMakeLists.txt:41
# ARM82 Assemblies
IF(MNN_ARM82)
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64" OR ARCHS STREQUAL "ARM64")
        target_compile_options(MNNCPU PRIVATE -DENABLE_ARMV82)
        include(${CMAKE_CURRENT_LIST_DIR}/../arm82/CMakeLists.txt)
        list(APPEND MNN_TARGETS MNN_Arm82)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Arm82>)
    ENDIF()
ENDIF()
```

`arm82` 目录里的源码和汇编会用 `-march=armv8.2-a+fp16` 编译：

```cmake
# source/backend/arm82/CMakeLists.txt:21
set_source_files_properties(${MNN_ARM82_SRCS} PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
set_source_files_properties(${MNN_ARM82_SRCS_ASM} PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16")
...
target_compile_options(MNN_Arm82 PRIVATE -march=armv8.2-a+fp16 -DENABLE_ARMV82)
```

运行时还会通过 `MNNGetCPUInfo()` 写入 `CoreFunctions` 的能力位。`fp16arith` 决定 ARM82 FP16 backend 能不能用；`dot / i8mm / sme2` 更多影响 int8、低内存 matmul 这类 kernel 的选择：

```cpp
// source/backend/cpu/compute/CommonOptFunction.cpp:4668
const MNNCPUInfo& gCPUInfo = *MNNGetCPUInfo();
gCoreFunction->supportFp16arith = gCPUInfo.fp16arith;
gCoreFunction->supportSDot = gCPUInfo.dot;
gCoreFunction->supportI8mm = gCPUInfo.i8mm;
gCoreFunction->supportSME2 = gCPUInfo.sme2;
gCoreFunction->smeCoreNumber = gCPUInfo.smeCoreNumber;
```

因此 `precision = low` 走 ARMv8.2 FP16 需要两件事同时满足：

```text
编译期：MNN_ARM82 打开，MNN_Arm82 object 被编进来
运行期：MNNGetCoreFunctions()->supportFp16arith 为 true
```

对应代码就是 `CPURuntime::onCreate()` 的 ARM82 分支：

```cpp
// source/backend/cpu/CPUBackend.cpp:324
#ifdef MNN_USE_ARMV82
auto core = MNNGetCoreFunctions();
if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
    res = new Arm82Backend(this, memory);
    res->mRelatedFunctions = &(res->functions()->int8MatmulRelatedFunctions);
    break;
}
#endif
```

## 4. precision 分支

### 4.1 normal / high

`Precision_Normal` 和 `Precision_High` 在 Android CPU 创建路径上都不会触发 ARMv8.2 FP16 分支，会创建普通 `CPUBackend`。

```cpp
// source/backend/cpu/CPUBackend.cpp:306
auto precision = mPrecision;
auto memory = mMemory;
if (nullptr != config) {
    precision = config->precision;
    flags = config->flags;
    memory = config->memory;
}

if (core->supportFp16arith && precision == BackendConfig::Precision_Low) {
    res = new Arm82Backend(this, memory);
    break;
}
if (precision == BackendConfig::Precision_Low_BF16 && BF16Functions::get()) {
    res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU_EXTENSION);
    res->mCoreFunctions = BF16Functions::get();
    break;
}
res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU, flags);
```

普通 `CPUBackend` 使用 `MNNGetCoreFunctions()`。默认 CPU 函数表里 `bytes = 4`：

```cpp
// source/backend/cpu/compute/CommonOptFunction.cpp:4579
gCoreFunction->MNNFp32ToLowp = nullptr;
gCoreFunction->MNNLowpToFp32 = nullptr;
gCoreFunction->bytes = 4; // sizeof(float)
gCoreFunction->pack = 4;
```

所以 CPU 上可以把它理解成普通 FP32 函数表。`Precision_High` 在 OpenCL / QNN 上含义更明显，通常表示不走 FP16 倾向。

### 4.2 low

`Precision_Low` 是 Android LLM 默认值。满足两个条件时会创建 `Arm82Backend`：

- 编译打开 `MNN_USE_ARMV82`；
- `MNNGetCoreFunctions()->supportFp16arith` 为 true。

`Arm82Backend` 继承 `CPUBackend`，但把函数表换成 ARMv8.2 FP16：

```cpp
// source/backend/arm82/Arm82Backend.cpp:27
Arm82Backend::Arm82Backend(const CPURuntime* runtime, BackendConfig::MemoryMode memory)
    : CPUBackend(runtime, BackendConfig::Precision_Low, memory, MNN_FORWARD_CPU_EXTENSION, 0) {
    mCoreFunctions = Arm82Functions::get();
    mInt8CoreFunctions = Arm82Functions::getInt8();
}
```

`Arm82Functions` 里会设置 FP16 转换函数，并把 `bytes` 改成 2：

```cpp
// source/backend/arm82/Arm82Functions.cpp:2645
FUNC_PTR_ASSIGN(gInstance->MNNFp32ToLowp, MNNQuantizeFP16);
FUNC_PTR_ASSIGN(gInstance->MNNLowpToFp32, MNNDequantizeFP16);
gInstance->bytes = 2;
gInstance->pack = 8;
```

因此 `precision = low` 的主影响是：算子优先在 FP16 tensor / FP16 kernel 的 CPU extension backend 上执行；不兼容的算子再回退。

### 4.3 low_bf16

`Precision_Low_BF16` 需要 `MNN_SUPPORT_BF16` 和 `BF16Functions::get()`：

```cpp
// source/backend/cpu/CPUBackend.cpp:333
if (precision == BackendConfig::Precision_Low_BF16 && BF16Functions::get()) {
    res = new CPUBackend(this, precision, memory, MNN_FORWARD_CPU_EXTENSION);
    res->mCoreFunctions = BF16Functions::get();
    break;
}
```

当前 `llm.cpp` 的 JSON 字符串入口没有把某个字符串映射到 `Precision_Low_BF16`，通常需要 C++ API 或测试工具直接设置枚举。

## 5. memory 分支

以卷积算子为例，梳理`memory` 的影响，大概理解为：

| memory | 枚举 | CPU 卷积里的核心语义 |
|--------|------|----------------------|
| `normal` | `Memory_Normal = 0` | 默认策略，不主动进入 `Memory_Low` 专属分支，也不享受 `Memory_High` 放宽 |
| `high` | `Memory_High = 1` | 倾向提前把量化权重反量化成 float，再走普通 dense / winograd / strassen |
| `low` | `Memory_Low = 2` | 倾向保留 int8/int4 权重，创建低内存动态量化执行器 |

LLM 构建时 `MNN_LOW_MEMORY` 必须编译进去。顶层 `CMakeLists.txt` 中，普通开关默认是 OFF：

```cmake
# CMakeLists.txt:79
option(MNN_LOW_MEMORY "Build MNN support low memory for weight quant model." OFF)
```

打开 LLM 时会强制打开：

```cmake
# CMakeLists.txt:109
IF (MNN_BUILD_LLM)
    set(MNN_LOW_MEMORY ON CACHE BOOL "<docstring>" FORCE)
    set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "<docstring>" FORCE)
ENDIF()
```

然后加上编译宏：

```cmake
# CMakeLists.txt:234
if(MNN_LOW_MEMORY)
    add_definitions(-DMNN_LOW_MEMORY)
endif()
```

### 5.1 入口：ConvolutionFloatFactory

普通 CPU 卷积从 `CPUConvolution.cpp` 进入 `ConvolutionFloatFactory::create(...)`：

```cpp
// source/backend/cpu/CPUConvolution.cpp:310
class ConvolutionFactory : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return ConvolutionFloatFactory::create(inputs, outputs, op, backend);
    }
};
```

`ConvolutionFloatFactory::create(...)` 先把 `memoryMode()` 映射成局部变量 `lowMemory`：

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:193
#ifdef MNN_LOW_MEMORY
bool lowMemory = static_cast<CPUBackend*>(backend)->memoryMode() == BackendConfig::Memory_Low;
#else
bool lowMemory = false;
#endif
```

如果编译打开 `MNN_CPU_WEIGHT_DEQUANT_GEMM`，源码条件是 `memoryMode() != Memory_High`。这会让 `Memory_Normal` 也被纳入 `lowMemory`。但后面的执行器选择还会再判断一次 `memoryMode()`：`Memory_Low` 仍然走 `DenseConvInt8TiledExecutor(..., true)`，真正新增出来的是 `Memory_Normal` 的 `DenseConvolutionTiledExecutor(..., weightQuantInfo)` 路径。

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:202
#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
lowMemory = lowMemory || (static_cast<CPUBackend*>(backend)->memoryMode() != BackendConfig::Memory_High);
#endif
```

### 5.2 int4 / int8 / fp16 / fp32 权重

常见权重格式有，

| 权重格式 | 模型里怎么存 | `ConvolutionCommon::load(...)` 后的典型形态 | 后续 CPU 路径 |
|----------|--------------|---------------------------------------------|--------------|
| int4 | 4bit 量化值，通常还带 `alpha/min` 或 scale 信息 | `forceInt8=true` 时保留在 `Int8Common::weight / alpha`，并标记 `canUseInt4` | `memory=low` 下可进 `DenseConvInt8TiledExecutor` 的 int4 权重路径 |
| int8 | 8bit 量化值，带 `alpha/min` 或 scale 信息 | `forceInt8=true` 时保留 `weight / alpha`；否则反量化成 `weightFloat` | low 走 int8 tiled；high 通常转成 float 后走普通卷积 |
| fp16 | half 存储的权重 | 加载时会转成 `weightFloat`，也就是 C++ `float` 数组 | 后续再按 backend 函数表 pack；ARM82 下 pack 时可能转成 FP16 |
| fp32 | 原始 float 权重 | 直接作为 `originWeight` 使用 | 后续按普通 dense / winograd / strassen pack |

fp16 权重在加载阶段会转成 `weightFloat`。代码里 `quan->type() == 3` 对应 read fp16 data，最后 `half` 转 `float`：

```cpp
// source/core/ConvolutionCommon.cpp:660
// read fp16 data
if (3 == quan->type()) {
    ...
    result->weightFloat.reset((int)weightLength);
    ...
    std::transform(halfWeight, halfWeight + weightLength, result->weightFloat.get(),
                   [](half_float::half h) { return float(h); });
    return result;
}
```

这里包括两个“精度层次”：

- 权重文件存储精度：int4 / int8 / fp16 / fp32；
- CPU 执行精度：由 backend 函数表决定，例如普通 CPU `bytes = 4`，ARM82 `bytes = 2`。

### 5.3 权重量化加载：forceInt8

卷积带 `quanParameter()` 时，会调用：

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:228
quanCommon = ConvolutionCommon::load(op, backend, forceFloat, lowMemory);
```

`ConvolutionCommon::load(...)` 的第四个参数对应的形参是 `forceInt8`：

```cpp
// source/core/ConvolutionCommon.hpp:29
static std::shared_ptr<Int8Common> load(
    const Op* op,
    Backend* backend = nullptr,
    bool forceFloat = false,
    bool forceInt8 = false,
    void* weightPtr = nullptr);
```

所以：

```text
memory = low
    -> lowMemory = true
    -> ConvolutionCommon::load(..., forceInt8 = true)
```

此时，量化权重会保留在 `Int8Common::weight / alpha`，提前返回，不进入 `Back to float` 分支：

```cpp
// source/core/ConvolutionCommon.cpp:744
if (forceInt8) {
    return result;
}
if (!quan->has_scaleInt() || forceFloat) {
    // Back to float
    result->weightFloat.reset((int)weightLength);
    ...
    result->weight.release();
    result->alpha.release();
}
```

`Back to float` 分支会把量化权重反量化成 C++ `float` 数组，存到 `Int8Common::weightFloat`。这里的 `float` 是 FP32 存储形态，表示“加载出来的权重临时用 FP32 表达”。它本身不等价于卷积计算一定使用 FP32；真正执行时用 FP32 还是 FP16，要继续看后面的 backend 函数表、`core->bytes` 和 packed weight 形态。

量化权重还原代码是：

```cpp
// source/core/ConvolutionCommon.cpp:752
result->weightFloat.reset((int)weightLength);
...
for (int o = 0; o < outputCount; ++o) {
    ...
    for (int v=0; v < partWeightSize; ++v) {
        dstW[v] = (float)srcW[v] * alpha + min;
    }
}
result->weight.release();
result->alpha.release();
```

`srcW[v]` 是 int8 量化权重值，`alpha` 是 scale，`min` 是非对称量化里的偏移。`dstW[v] = (float)srcW[v] * alpha + min` 得到的是一个 FP32 float 权重值。

总结，当 `memory = low` 时，这类量化卷积会避免先生成完整的 FP32 `weightFloat`，而是把量化权重留给后面的 int8/int4 低内存执行器。

### 5.4 memory = high：反量化成 float

当 `memory = high` 时，通常 `lowMemory = false`。这时 `ConvolutionCommon::load(...)` 不会在 `forceInt8` 处提前返回，而是继续进入 `Back to float` 分支。执行：

```text
模型文件里的 int4/int8 权重
    -> 读取量化值 weight 和 scale/alpha/min
    -> 按 dst = int_value * alpha + min 还原成 FP32 float
    -> 存入 Int8Common::weightFloat
```

回到卷积工厂后，`originWeight` 指向这份 float 权重：

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:242
originWeight     = quanCommon->weightFloat.get();
originWeightSize = quanCommon->weightFloat.size();
```

这里的“展开成 float”只发生在权重加载 / 初始化阶段，得到的是 FP32 `weightFloat`。它回答的是“量化权重先用什么形态交给普通卷积初始化”，不是最终计算精度。后续真正 pack 到 `mResource->mWeight` 时，还会看 backend 的 `core->bytes`：

- 普通 CPU / SSE / AVX2 函数表通常 `bytes = 4`，packed weight 按 FP32 存；
- ARM82 FP16 函数表 `bytes = 2`，`initWeight(...)` 会把 float cache 转成 lowp，也就是 FP16 存。

对应代码在 `ConvolutionTiledExecutor::initWeight(...)`：

```cpp
// source/backend/cpu/compute/ConvolutionTiledExecutor.cpp:24
void ConvolutionTiledExecutor::initWeight(const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function) {
    ...
    if (function->bytes < 4) {
        // Lowp
        function->MNNFp32ToLowp((float*)cache, (int16_t*)cache, outputCount * kernelSize * depth);
    }
}
```

所以 `memory=high` 会先生成一份 FP32 反量化权重作为初始化输入；最终常驻的 packed weight 是否是 FP32/FP16，由 CPU backend 的函数表决定。普通 CPU/SSE/AVX2 路径通常仍是 FP32 packed weight；ARM82 FP16 路径可以在 pack 阶段转成 FP16 存储和计算。

### 5.5 执行器选择：low 和 high 分叉

加载完权重后，`_createUnit(...)` 根据 `lowMemory` 和 `memoryMode()` 选择执行器：

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:154
#ifdef MNN_LOW_MEMORY
if (lowMemory && nullptr != weightQuantInfo.get() && originWeightSize == 0) {
    if (cpuBackend->memoryMode() == BackendConfig::Memory_Low) {
        return new DenseConvInt8TiledExecutor(backend, op, weightQuantInfo, true);
    } else {
        return new DenseConvolutionTiledExecutor(
            common, backend, originWeight, originWeightSize,
            bias, biasSize, weightQuantInfo);
    }
}
#endif
```

这里对应两条路径。

`memory = low`：

```text
Memory_Low
    -> lowMemory = true
    -> forceInt8 = true
    -> 不生成 weightFloat
    -> originWeightSize == 0
    -> DenseConvInt8TiledExecutor(..., isDynamicQuant = true)
```

非 `Memory_Low` 但 `lowMemory = true`：

```text
常见于 MNN_CPU_WEIGHT_DEQUANT_GEMM 打开且 memory = normal
    -> lowMemory = true
    -> forceInt8 = true
    -> 不生成 weightFloat
    -> originWeightSize == 0
    -> DenseConvolutionTiledExecutor(..., weightQuantInfo)
```

这条路径进的是 `DenseConvolutionTiledExecutor`，但因为 `originWeightSize == 0`，构造函数内部会走 `useInt8Weight` 分支，初始化量化权重资源：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:181
if (useInt8Weight) {
    // Quantize weight to int8
    auto allocSuccess = DenseConvolutionTiledExecutor::initQuantizeResource(
        int8Info, mResource, hU, hP, lU, lP, outputCount,
        srcCount, common->kernelX() * common->kernelY(), bytes);
    ...
}
```

`initQuantizeResource(...)` 会把 scale/bias 放进 `mResource->mDequantize.mScaleBias`，把 int8 weight 重排进 `mResource->mWeight`：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:47
resource->mDequantize.mScaleBias.reset(
    MNN::Tensor::createDevice<uint8_t>({scaleSize * 2 * bytes}));
...
resource->mWeight.reset(Tensor::createDevice<int8_t>(
    std::vector<int>{hU, lU * lP, hP}));
```

执行期再根据 `mResource->mDequantize.bits` 选择 int8 weight matmul，并取出 dequant scale/bias：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:458
DenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(
    &matmulUnit, &matmulRemain, &weightBytes, mResource->mDequantize.bits, core);
...
dequantAlpha = mResource->mDequantize.mScaleBias->host<uint8_t>();
dequantBias = dequantAlpha + scaleSize * bytes;
```

`memory = high`：

```text
Memory_High
    -> 通常 lowMemory = false
    -> forceInt8 = false
    -> 生成 weightFloat
    -> originWeightSize > 0
    -> 跳过上面的低内存分支
    -> 继续走 1x1 Strassen / Dense tiled / Winograd
```

这里`memory = high` 避开了 `lowMemory`分支，让权重量化卷积提前展开成 float，再进入普通卷积选择链路：

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:168
if (fastWay && cpuBackend->functions()->matmulBytes == 0) {
    return new Convolution1x1Strassen(common, backend, originWeight, originWeightSize, bias, biasSize);
}
if (cpuBackend->getRuntime()->hint().winogradMemoryUsed == 0 ||
    (!ConvolutionWinogradBridge::canUseWinograd(common))) {
    return new DenseConvolutionTiledExecutor(
        common, backend, originWeight, originWeightSize, bias, biasSize, nullptr);
}
return ConvolutionWinogradBridge::createWinogradImpl(...);
```

### 5.6 packed weight

普通 dense / winograd 路径不会直接拿 `[OC, IC, KH, KW]` 这种原始权重布局去做 GEMM。为了让矩阵乘更适合当前 CPU 的 pack 单元，初始化阶段会把权重重新排布成 packed layout。

以 `DenseConvolutionTiledExecutor` 为例：

- `originWeight` 是前面得到的 float 权重；
- `cache` 是临时 buffer，用来做 `k/ic` 维度转置；
- `mResource->mWeight` 是最终常驻的 packed weight；
- packed 后的 layout 按 `hU * hP`、`lU * lP`、`bytes` 组织，适合后面的 `MNNPackedMatMul` 直接读取。

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:192
mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
    {hU * hP, lU * lP, bytes}));
mValid = mValid && backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);

std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>(
    {outputCount, srcCount * common->kernelX() * common->kernelY(), (int)sizeof(float)}));
mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);

initWeight(mResource->mWeight->host<float>(), originWeight, cache->host<float>(),
           srcCount, outputCount, common->kernelX() * common->kernelY(), core);
backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
```

`cache` 用完会释放，`mResource->mWeight` 会常驻。它是为了加速执行期 GEMM 付出的内存成本。

`DenseConvolutionTiledExecutor` 构造函数只从 `originWeight` 读取数据，pack 到 `mResource->mWeight`，不会保存 `originWeight` 指针，也不会保存 `quanCommon`。`quanCommon` 是 `ConvolutionFloatFactory::create(...)` 里的局部 `shared_ptr`，传进 `_createUnit(...)` 只是为了初始化执行器；执行器创建完成并返回后，如果没有其它路径继续持有它，`Int8Common::weightFloat` 会跟着释放。因此对普通 dense 路径来说，`weightFloat` 是初始化期临时权重(执行器 pack 完后不再保存)，常驻的是 `mResource->mWeight`。

### 5.7 memory = low 的内存节省

`memory = low` 分支流程如下:

```text
config.json: "memory": "low"
    -> BackendConfig::Memory_Low
    -> ConvolutionFloatFactory: lowMemory = true
    -> ConvolutionCommon::load(..., forceInt8 = true)
    -> 保留 Int8Common::weight / alpha，不生成 weightFloat
    -> DenseConvInt8TiledExecutor(..., isDynamicQuant = true)
    -> 初始化期常驻 int8/int4 packed weight
    -> 执行期把输入动态量化成 int8，再走 int8/int4 GEMM
```

这里有两个关键点：

- 权重：不提前展开成整层 FP32 `weightFloat`，而是以 int8/int4 形态 pack 后常驻。
- 输入：仍然可能是 float 输入，但在执行期按当前输入 block 动态量化成 int8，再进入整数 GEMM。

入口在执行器选择。`memory = low` 的量化卷积会创建 `DenseConvInt8TiledExecutor`，第四个参数传 `true`：

```cpp
// source/backend/cpu/compute/ConvolutionFloatFactory.cpp:156
new DenseConvInt8TiledExecutor(backend, op, weightQuantInfo, true)
```

这会在构造函数里写成 `mDynamicQuant`。(如果权重是 int4 那么`mWeightBits` 会从 8 改成 4)：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:416
mResourceInt8->mWeightBits = 8;
if (quanCommon && quanCommon->canUseInt4) {
    shapeMain[4] = SRC_UNITMain / 2;
    shapeBranch[4] = SRC_UNITBranch / 2;
    mResourceInt8->mWeightBits = 4;
    mResourceInt8->mWeightAsymmetricQuant = true;
}
mResourceInt8->mDynamicQuant = isDynamicQuant ? true : false;
```

初始化时，`DenseConvInt8TiledExecutor` 的常驻资源是 `mWeightInt8`。这块内存里放 int8/int4 权重，以及每个 block 的量化参数；另外还会保存原始 bias 和权重 sum，用于后处理修正 zero/bias：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:435
auto quantlenMain = 2 * blockNum * ROUND_UP(mOcMain, UNITMain) * QUANT_INFO_BYTES;
auto weightlenMain = shapeMain[0] * shapeMain[1] * shapeMain[2] * shapeMain[3] * shapeMain[4];

mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>(
    {weightlenMain + quantlenMain + weightlenBranch + quantlenBranch}));
mResourceInt8->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4Main + ocUpHpBranch}));
mResourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>(
    {inputBlockNum * QUANT_INFO_BYTES * (ocUpHpMain + ocUpHpBranch)}));
```

所以 `memory = low` 避免常驻整层 FP32 `weightFloat` 或 FP32/FP16 packed weight，而是保存 int8/int4 形态，代价是执行期要多做输入动态量化和量化 GEMM 后处理。

执行时，进入 `onResize` / 执行准备阶段后，低内存路径会申请 int8 im2col buffer、输入 sum buffer、跨 block 的 int32 累加 buffer：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:877
mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, colBufferSize * im2colBytes}));
mTempSrcSum = bufferAlloc->alloc(
    threads * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
mAccumBuffer.reset(Tensor::createDevice<int32_t>(
    {threads, DST_XUNIT * ALIMAX(UNIT, gcore->pack)}));
```

如果当前路径需要动态量化输入，还会申请 `mQuantInput`、临时 min/max 和 `mQScaleZero`。这些是执行期 buffer，不是常驻模型权重：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:1009
mQuantInput.reset(Tensor::createDevice<int8_t>(
    {batch, mIm2ColParamter.ih, mIm2ColParamter.iw, ROUND_UP(inC, gcore->pack)}));
mTempMaxMinValueBuffer = bufferAlloc->alloc(tempSize * gcore->bytes);
mQScaleZero = bufferAlloc->alloc(tempSize * QUANT_INFO_BYTES);
```

接着，权重指针读的就是前面 pack 好的 `mWeightInt8`：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:1562
auto im2colPtr = mTempIm2ColBuffer->host<int8_t>();
auto weightDataPtr = mResourceInt8->mWeightInt8->host<int8_t>();
int im2colBytes = mIm2ColBasedInt8 == true ? 1 : gcore->bytes;
```

并继续进行计算，主线是：

```text
float input
    -> 执行期动态量化成 int8 input
    -> int8 input 与 int8/int4 weight 做整数 GEMM
    -> int32 accumulator
    -> 用 input scale、weight scale、zero/bias 修正还原到输出精度
```

先看输入量化。`mDynamicQuant = true` 后，执行期会按当前输入 block 计算量化 scale / bias，然后把输入写入 `mQuantInput`。非对称量化会计算 `qscale / qbias`，对称量化会计算 absmax 和 scale；两条路径的目的相同，都是给后面的 int8 GEMM 准备整数输入和还原参数：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:1612
auto qscale = (float*)(mQScaleZero.ptr() + tId * scaleCount * QUANT_INFO_BYTES);
auto qbias  = (float*)(mQScaleZero.ptr() + tId * scaleCount * QUANT_INFO_BYTES + (scaleCount / 2) * QUANT_INFO_BYTES);
...
// scale&bias:float32
gcore->MNNAsyQuantInfo(scalePtr, zeroPtr, qscale, qbias, (float*)minPtr, (float*)maxPtr, (float*)floatPtr, info);

// quant: float->int8_t
gcore->MNNAsyQuantFunc(dstInt8, (float*)floatPtr, qscale, qbias, info);
...
gcore->MNNQuantScale((float*)maxPtr, (float*)quantPtr, (float*)inputDequantScale, availableThreads, EP);
gcore->MNNDynamicQuant((float*)floatPtr, dstInt8, scale_ptr, LU, EP, LP, nullptr);
```

量化完成后，输入指针会切到 `mQuantInput`，即后面的 im2col/GEMM 的量化输入：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:1677
if (mResourceInt8->mDynamicQuant) {
    biasPtr = mResourceInt8->mOriginBias->host<uint8_t>();
}
if (mIm2ColBasedInt8 && mResourceInt8->mDynamicQuant) {
    ...
    im2colSrc = mQuantInput->host<uint8_t>();
}
```

GEMM kernel 也对应换成 int8 动态量化版本；如果常驻权重是 int4，则换成 `_W4` 版本。

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:902
if (!mMixedKernel) { // Dynamic Quant kernels, use single gemm kernel.
    mGemmKernel = mRelatedFunctions.Int8GemmKernel;
    ...
    if (mResourceInt8->mWeightBits == 4) {
        mGemmKernel = mRelatedFunctions.Int8GemmKernel_W4;
        ...
    }
}
```

真正调用 GEMM 前，会把动态输入量化得到的 `inputScale / inputBias`、输入 sum、跨 block 累加 buffer 塞进 `quanParam`。权重 scale/bias 则跟随 `weightPtrTid` 指向的 packed weight block 一起被 kernel 读取：

```cpp
// source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:2048
uint8_t* inputScale = nullptr; // input scale for batch dynamic quant.
uint8_t* inputBias = nullptr;
float* accumbuff = nullptr;
...
quanParam.inputScale = (float*)inputScale;
quanParam.inputBias = (float*)inputBias;
quanParam.srcKernelSum = ptrX;
if (mBlockNum > 1) {
    memset(accumbuff, 0, UNIT * 4 * DST_XUNIT);
    quanParam.accumBuffer = accumbuff;
}
gemmInt8(outputInTilePtr, im2colDstThread, weightPtrTid, blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
```

最后看 int8 GEMM 的普通 C++ fallback。主循环先做纯整数乘加，得到 `int32_t dstTemp`；后处理阶段再用 weight scale、input scale 和 zero/bias 修正把数值还原回输出域：

```cpp
// source/backend/cpu/compute/Int8FunctionsOpt.cpp:1429
static void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    ...
    const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
    const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
    ...
    dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
    ...
    float value = dstTemp[j] * scale_dz[j] + srcSumPtr[w] * weightBias_dz[j];
    if (post->inputScale) {
        value = dstTemp[j] * scale_dz[j] * inputScalePtr[w] + srcSumPtr[w] * weightBias_dz[j];
    }
    ...
    ((float*)dst_x)[j] = value;
}
```

这里的“还原”发生在 `int32 accumulator -> output value` 这一步：`scale_dz` 来自权重量化参数，`inputScalePtr` 来自执行期输入动态量化，`srcSumPtr * weightBias_dz` 和 `inputBias * weightKernelSum` 用来修正 zero/bias 项。

### 5.8 MNN_CPU_WEIGHT_DEQUANT_GEMM 的反量化

`MNN_CPU_WEIGHT_DEQUANT_GEMM` 的 `DEQUANT` 不是“加载模型时把权重全部反量化成 FP32”，而是实际打开策略：

```text
加载 / 初始化阶段
    -> 保留量化权重
    -> pack 成 int8 mResource->mWeight
    -> scale / bias 放进 mDequantize.mScaleBias

执行 GEMM 时
    -> 读当前 block 的 int8 weight
    -> 读当前 block 的 scale / bias
    -> 在 matmul kernel 内完成 dequant 相关计算
    -> 不生成整层常驻 FP32 weightFloat
```

执行器先根据 `mDequantize.bits` 把普通 matmul 函数换成 int8 weight matmul：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:150
void DenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(
    lowMemoryMatmulUnit* matmulUnit,
    lowMemoryMatmulRemain* matmulRemain,
    float* weightBytes,
    int32_t weightQuantBits,
    const CoreFunctions* core) {
    if (weightQuantBits == 8) {
        *matmulUnit = core->MNNPackedMatMul_int8;
        *matmulRemain = core->MNNPackedMatMulRemain_int8;
        *weightBytes  = 1;
    }
}
```

真正执行时，`DenseConvolutionTiledExecutor` 会从 `mDequantize.mScaleBias` 里取出当前输出通道对应的 `k / b`，再按 block 传给 `matmulUnit(...)`：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:580
auto k = reinterpret_cast<const uint8_t*>(dequantAlpha + ocIndex * bytes);
auto b = reinterpret_cast<const uint8_t*>(dequantBias + ocIndex * bytes);
...
for (int bk = 0; bk < blockNum; ++bk) {
    ...
    matmulUnit(_dstFloatPtr,
               (float*)(_APtr + eP * finishedL * bytes),
               (float*)(_weightPtr + wquantStride),
               paraParameters, relufp32, exeBiasPtr,
               (float*)(k + bk * ocUp4 * bytes),
               (float*)(b + bk * ocUp4 * bytes));
}
```

函数签名里最后两个参数就是这组 dequant 参数：

```cpp
// source/backend/cpu/compute/CommonOptFunction.cpp:192
void MNNPackedMatMul_int8(
    float* C, const float* A, const float* B,
    const size_t* parameter,
    const float* postParameters,
    const float* bias,
    const float* k,
    const float* b) {
    _MNNPackedMatMulRemain_int8(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}
```

`MNNPackedMatMul_int8(...)` 只是入口包装。继续查看调用的 `_MNNPackedMatMulRemain_int8(...)`，反量化计算就写在内层循环里：

```cpp
// source/backend/cpu/compute/CommonOptFunction.cpp:119
static void _MNNPackedMatMulRemain_int8(
    float* C, const float* A, const float* fB, size_t eSize,
    const size_t* parameter, const float* postParameters, const float* bias,
    int aStride, const float* k, const float* b) {
    auto B = reinterpret_cast<const int8_t*>(fB);
    ...
    auto alpha = k + y * 4;
    auto qbias  = b + y * 4;
    ...
    for (int z=0; z<l; ++z) {
        auto aZ = src + z * aStride;
        auto i8wZ = weight + z * 4;
        float wZ[4];
        wZ[0] = i8wZ[0] * alpha[0] + qbias[0];
        wZ[1] = i8wZ[1] * alpha[1] + qbias[1];
        wZ[2] = i8wZ[2] * alpha[2] + qbias[2];
        wZ[3] = i8wZ[3] * alpha[3] + qbias[3];
        summer[0] += wZ[0] * aZ[0];
        summer[1] += wZ[1] * aZ[0];
        summer[2] += wZ[2] * aZ[0];
        summer[3] += wZ[3] * aZ[0];
    }
}
```

这段代码里，`i8wZ` 是当前读到的 int8 权重，`alpha / qbias` 来自前面传入的 `k / b`。`wZ[i] = i8wZ[i] * alpha[i] + qbias[i]` 就是当前 4 个输出通道这一小段权重的反量化；随后立刻和输入 `aZ[0]` 相乘并累加到 `summer`，作为`float`精度。
所以，反量化是**按 GEMM block 读 int8 weight 和 scale/bias，在 kernel 内完成转换和乘加**。


所以 `memory = low` 与 `MNN_CPU_WEIGHT_DEQUANT_GEMM` 的区别是：前者把反量化/缩放融合在 int8 GEMM 的后处理里；后者是在 matmul 内临时构造当前权重 block 的 float `wZ`，再做 `float input * dequant(weight)`。

```text
memory = low:
    int8(input) * int8/int4(weight) -> int32 accumulator -> scale/bias -> output

MNN_CPU_WEIGHT_DEQUANT_GEMM + memory = normal:
    float(input) * dequant(int8 weight block) -> output
```

放回配置层面，对比是：

| 项目 | `memory = high` | `memory = low` | `MNN_CPU_WEIGHT_DEQUANT_GEMM` 打开且 `memory = normal` |
|------|-----------------|----------------|--------------------------------------------------|
| 量化权重加载 | `forceInt8 = false`，生成 FP32 `weightFloat` | `forceInt8 = true`，保留 `weight / alpha` | `forceInt8 = true`，保留 `weight / alpha` |
| 常驻权重 | FP32/FP16 packed weight，取决于 backend `bytes` | int8/int4 packed weight | int8 `mResource->mWeight` + `mDequantize.mScaleBias`，由 `DenseConvolutionTiledExecutor::initQuantizeResource(...)` 初始化 |
| 执行器 | 普通 dense / winograd / strassen | `DenseConvInt8TiledExecutor(..., true)` | `DenseConvolutionTiledExecutor(..., weightQuantInfo)` |
| 执行期成本 | 较少动态量化开销 | 需要输入动态量化、int8 im2col、scale/zero 等 buffer | GEMM 时走 int8 weight / dequant 相关 matmul，不是 `Memory_Low` 的动态量化执行器 |
| 适合场景 | 内存宽裕，追求更少运行时量化开销 | LLM 权重量化模型，优先降低常驻权重内存 | 编译启用 weight dequant GEMM 时，`normal` 的折中路径 |

### 5.9 LLM load 时权重什么时候打包

LLM 的主模型加载在 `Llm::load()` 里。CPU 常见路径下，`shapeMutable = true`，同时 `rearrange = true`：

```cpp
// transformers/llm/engine/src/llm.cpp:268
Module::Config module_config;
if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
    module_config.shapeMutable = false;
} else {
    module_config.shapeMutable = true;
}
module_config.rearrange    = true;
...
mRuntimeManager->setExternalFile(mConfig->llm_weight());
...
mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
```

`setExternalFile(...)` 指向外部权重文件；`Module::load(...)` 会进入 `PipelineModule / StaticModule` 的加载路径。`PipelineModule::load(...)` 先初始化常量 tensor：

```cpp
// express/module/PipelineModule.cpp:760
FileLoader fileLoader(modRuntimeCfgPtr->externalFile.c_str());
initConstTensors(sharedConst->allTensors, net, defaultBackend.get(), code, &fileLoader);
```

到了 `StaticModule` 构造函数，`rearrange = true` 会触发 `preRearrangeWeights(...)`：

```cpp
// express/module/StaticModule.cpp:342
if (config.rearrange) {
    mResource->mBuffer = preRearrangeWeights(
        scheduleInfo, bnCache.cache.first.get(), bnCache.cache.second.get(), config.base);
} else {
    mResource->mBuffer = std::move(buffer);
}
```

`preRearrangeWeights(...)` 的主线是：遍历图里的 op；对支持预处理的 op 创建 execution；如果 execution 支持 clone，就把这个 execution 放进 `executionCache`。卷积 / int8 卷积这类权重算子会在 execution 构造时走前面第 5 节讲过的权重加载和 pack 路径。

```cpp
// express/module/StaticModule.cpp:66
switch (op->type()) {
    case MNN::OpType_DepthwiseConvInt8:
    case MNN::OpType_ConvInt8:
    case MNN::OpType_ConvolutionDepthwise:
    case MNN::OpType_Convolution: {
        ...
        exe.reset(OpCommonUtils::createExecutionWithExternal(
            backend, info.inputs, info.outputs, op, &loader, tmpstorage));
        ...
        if (!exe->onClone(nullptr, op, nullptr)) {
            exe = nullptr;
            break;
        }
    }
}
```

预处理成功后，原始 op 里的权重字段会被清掉，说明后续不再依赖 flatbuffer 里那份原始权重字段反复创建执行器：

```cpp
// express/module/StaticModule.cpp:122
if (OpParameter_Convolution2D == op_table->main.type) {
    op_table->main.AsConvolution2D()->bias.clear();
    op_table->main.AsConvolution2D()->weight.clear();
    if (nullptr != op_table->main.AsConvolution2D()->symmetricQuan) {
        op_table->main.AsConvolution2D()->symmetricQuan->bias.clear();
        op_table->main.AsConvolution2D()->symmetricQuan->weight.clear();
    }
    if (nullptr != op_table->main.AsConvolution2D()->quanParameter) {
        op_table->main.AsConvolution2D()->quanParameter->alpha.clear();
        op_table->main.AsConvolution2D()->quanParameter->buffer.clear();
    }
}
```

然后 clone 出来的 execution 会进入 `executionCache`：

```cpp
// express/module/StaticModule.cpp:192
if (nullptr != exe) {
    // Clone Execution to reset op info
    Execution* dstExe;
    exe->onClone(exe->backend(), info.op, &dstExe);
    std::shared_ptr<Execution> dstExeP(dstExe);
    info.executionCache.insert(std::make_pair(info.op, dstExeP));
}
```

因此，对 LLM 的 CPU 路径，可以把 `load` 阶段理解成：

```text
Llm::load()
    -> 设置外部权重文件
    -> Module::load(...)
    -> initConstTensors 读取常量 / 外部权重
    -> StaticModule(config.rearrange = true)
    -> preRearrangeWeights 创建可 clone 的 execution
    -> 权重被加载并预处理成 execution/resource 可复用形态
```

随后推理时，LLM 不会每个 token 重新 `Module::load`。它会按 prefill / decode / speculative verify 等形态选择或创建 module clone：

```cpp
// transformers/llm/engine/src/llm.cpp:323
mModulePool[std::make_pair(verify_length, true)].reset(Module::clone(mModule.get()));
...
mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()));
mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;
```

执行时再按当前 `seqLenKey` 和 `isAllLogits` 选 module：

```cpp
// transformers/llm/engine/src/llm.cpp:456
std::shared_ptr<Module> selectModule = mModule;
if (mValidBlockSize.empty()) {
    if(mModulePool.find(moduleKey) == mModulePool.end()) {
        MNN_PRINT("Warning: module need new clone, cloning now.\n");
        mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
        mModulePool[moduleKey].reset(Module::clone(mModule.get()));
    }
    selectModule = mModulePool[moduleKey];
}
...
std::vector<Express::VARP> outputs = selectModule->onForward(inputs);
```

`StaticModule::clone(...)` 会共享 `mResource`，然后 clone session：

```cpp
// express/module/StaticModule.cpp:642
Module* StaticModule::clone(CloneContext* ctx) const {
    StaticModule* module(new StaticModule);
    module->mResource = mResource;
    module->mRuntimeManager = ctx->pRuntimeManager;
    ...
    module->mSession.reset(mSession->clone(std::move(rt), mResource->mSharedConst));
    module->resetInputOutputs();
    return this->cloneBaseTo(ctx, module);
}
```

输入 shape 变化时，`StaticModule::onForward(...)` 会进入 `_resize(...)`，最后由 `Session::resize()` 重新 encode / 分配动态内存：

```cpp
// express/module/StaticModule.cpp:566
std::vector<Express::VARP> StaticModule::onForward(const std::vector<Express::VARP>& inputs) {
    ...
    bool runResize = (!mShapeInferSeperate) || inputs.size() > 0;
    ...
    Variable::compute(inputs);
```

```cpp
// source/core/Session.cpp:284
bool firstMalloc = false;
if (mNeedResize) {
    ...
    for (auto& iter : mPipelines) {
        auto error = iter->encode(debug, permitCodegen);
        ...
    }
    mNeedResize = false;
    mNeedMalloc = true;
    firstMalloc = true;
}
if (mNeedMalloc) {
    ...
    for (auto& iter : mPipelines) {
        auto error = iter->allocMemory(firstMalloc, forbidReplace);
        ...
    }
}
```

因此，`Llm::load()` 阶段会读取权重，并在 `rearrange = true` 时尽量把支持的权重算子预处理成 execution/resource；后续推理会长期保留这些可复用权重资源。不同 shape 输入通常不会改变同一层权重的 packed 格式，变化的是执行计划、输入输出 tensor、临时 buffer 和 KV cache 相关资源。

## 6. power 分支

`power` 主要影响 CPU 选核。

`Power_Low` 会选择 `cpuInfo->groups[0].ids`：

```cpp
// source/backend/cpu/CPUBackend.cpp:187
case BackendConfig::Power_Low:
    mCpuIds = cpuInfo->groups[0].ids;
    break;
```

在 big.LITTLE 手机上，这通常对应小核 / 低功耗核心。

`Power_High` 会从后面的 CPU group 开始填充，优先选性能更高的核心：

```cpp
// source/backend/cpu/CPUBackend.cpp:190
case BackendConfig::Power_High: {
    int selectCPUSize = 0;
    int groupIndex = cpuInfo->groups.size() - 1;
    while (selectCPUSize < mThreadNumber && groupIndex >= 0) {
        auto& group = cpuInfo->groups[groupIndex];
        mCpuIds.insert(mCpuIds.end(), group.ids.begin(), group.ids.end());
        groupIndex--;
        selectCPUSize += group.ids.size();
    }
}
    break;
```

`Power_Low` 还有一个副作用：`CPUBackend` 构造函数里会跳过按 CPU group 计算线程算力比例的逻辑：

```cpp
// source/backend/cpu/CPUBackend.cpp:460
if (mThreadNumber <= 1 || mRuntime->mPower == BackendConfig::Power_Low) {
    break;
}
```

所以 `power = low` 不只是选小核，也会避免后续按 CPU group 做多线程算力分配。

## 7. 导出精度和运行时精度

LLM 导出阶段也有精度参数，`llmexport.py` 里 `--quant_bit` 默认是 4：

```python
# transformers/llm/export/llmexport.py:694
parser.add_argument('--quant_bit', type=int, default=4,
                    help='mnn quant bit, 4 or 8, default is 4.')
```

转 MNN 时：

```python
# transformers/llm/export/utils/mnn_converter.py:130
if quant_bit == 16:
    quant_args = ['--fp16']
else:
    quant_args = [
        '--weightQuantBits',
        str(quant_bit),
        '--weightQuantBlock',
        str(quant_block)
    ]
if quant_bit == 32:
    quant_args = []
```

对应关系：

| 参数 | 影响 |
|------|------|
| `--quant_bit 4` | 权重按 4bit 分组量化存储 |
| `--quant_bit 8` | 权重按 8bit 分组量化存储 |
| `--quant_bit 16` | 权重按 FP16 存储 |
| `--quant_bit 32` | 保留 FP32 权重 |

运行时的 `"precision": "low"` 是后端选择，决定 CPU 是否优先进入 `Arm82Backend`。

此外`embedding` 还有单独的 `--embed_bit`：

```python
# transformers/llm/export/llmexport.py:714
parser.add_argument('--embed_bit', type=int, default=16, choices=[16, 8, 4],
                    help='embedding export bit precision, choices are 16 (bf16), 8 (int8), 4 (int4), default is 16.')
```

默认 `embed_bit = 16` 时写 `embeddings_bf16.bin`；设置成 8 或 4 时写 `embeddings_int8.bin` / `embeddings_int4.bin`。

## 8. 速查表

`precision`：

| 取值 | LLM 字符串入口 | CPU 分支 | 主要效果 |
|------|----------------|----------|----------|
| `Precision_Normal = 0` | `"normal"` 或未命中 high/low | 普通 `CPUBackend` | 不主动进入 FP16 / BF16 |
| `Precision_High = 1` | `"high"` | 普通 `CPUBackend` | CPU 上仍是普通 FP32，OpenCL/QNN 上偏高精度 |
| `Precision_Low = 2` | `"low"` | ARMv8.2 可用时创建 `Arm82Backend` | FP16 函数表，`bytes = 2` |
| `Precision_Low_BF16 = 3` | LLM JSON 当前未直接映射 | BF16 可用时创建 CPU extension | BF16 函数表 |

`memory`：

| 取值 | LLM 字符串入口 | CPU 分支 | 主要效果 |
|------|----------------|----------|----------|
| `Memory_Normal = 0` | `"normal"` 或未命中 high/low | 普通策略 | 不进入 `Memory_Low` 专属分支，也不放宽到 high |
| `Memory_Normal + MNN_CPU_WEIGHT_DEQUANT_GEMM` | `"normal"` | `memoryMode() != Memory_High` 使 `lowMemory = true`，但仍不是 `Memory_Low` | 保留 int8 权重，走 `DenseConvolutionTiledExecutor(..., weightQuantInfo)`；GEMM 内按 block 做权重反量化为FP32精度|
| `Memory_High = 1` | `"high"` | `memoryMode() == Memory_High` | 权重量化卷积倾向提前展开 float，部分算子放宽内存限制 |
| `Memory_Low = 2` | `"low"` | `memoryMode() == Memory_Low` | 权重量化卷积保留 int8/int4 权重，走低内存动态量化执行器 |

`power`：

| 取值 | LLM 字符串入口 | CPU 分支 | 主要效果 |
|------|----------------|----------|----------|
| `Power_Normal = 0` | `"normal"` | 不主动改写 `mCpuIds` | 使用默认线程 / 调度策略 |
| `Power_High = 1` | `"high"` | 从高性能 CPU group 开始填充 `mCpuIds` | 更偏性能 |
| `Power_Low = 2` | `"low"` | 选择 `cpuInfo->groups[0].ids` | 更偏低功耗 |
