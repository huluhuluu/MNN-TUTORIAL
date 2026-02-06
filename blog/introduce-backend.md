# MNN 介绍

## 1. 后端

[MNN](https://github.com/alibaba/MNN)是阿里巴巴开源的高效轻量级深度学习推理框架，提供了较为**全面的后端支持**。 主要包括: 
MNN对每个后端的支持分为四个级别：


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

通常有backend runtime execution组成，关系是

例如

#### 1.4.2 后端调用设置

除了1.4.1中的文件 还有部分特定算子的优化代码，例如

是通过core设置 调用的

例如：