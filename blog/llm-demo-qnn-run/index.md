---
title: "以 llm_demo 为例的 QNN 后端运行"
date: 2026-05-27T12:00:00+08:00
lastmod: 2026-05-27T12:00:00+08:00
draft: false
description: "记录 MNN 的 llm_demo 在 Android 上通过 QNN 后端运行的最短流程"
slug: "llm-demo-qnn-run"
tags: ["mnn", "llm", "android", "qnn"]
categories: ["mnn"]

comments: true
math: true
---

# QNN 后端运行

- [QNN 后端运行](#qnn-后端运行)
  - [1. 环境](#1-环境)
  - [2. 交叉编译](#2-交叉编译)
    - [2.1 交叉编译 `llm_demo`](#21-交叉编译-llm_demo)
    - [2.2 导出 QNN 模型](#22-导出-qnn-模型)
  - [3. 推送并运行](#3-推送并运行)
    - [3.1 推送编译产物、模型和 QNN 依赖](#31-推送编译产物模型和-qnn-依赖)
    - [3.2 运行](#32-运行)


以 `transformers/llm/engine/demo/llm_demo.cpp` 为例，在 Android 设备上执行 QNN 后端流程：**服务器交叉编译 -> 构建 QNN 模型 -> `adb push` 推送到手机 -> 端侧设备运行**。需要准备下面环境：

- `adb devices` 能看到目标设备，本文测试设备为 `adb connect 127.0.0.1:40404`
- `ANDROID_NDK` 已配置，参考[交叉编译环境配置](../cross-compiler/index.md)
- `QNN_SDK_ROOT` 已配置，参考[QNN环境配置](http://github.com/huluhuluu/qnn-tutorial/blob/main/blog/qnn-setup/index.md)

整体构建流程参考[官方文档](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html#qnn-llm)。

## 1. 环境

以下面环境为例：

- 开发机(**编译设备**)：`Ubuntu 22.04`
- 测试机(**目标设备**)：`arm64-v8a` + `SM8750`
- QNN SDK：`/root/qairt/2.40.0.251030`
- 编译目录：`project/android/build_64`
- 手机测试目录：`/data/local/tmp/mnn-llm-qnn`
- 测试模型：`taobao-mnn/Qwen3-1.7B-MNN`

首先需要拉取仓库到编译设备,当前测试分支是`8d9f6131ac3ea61cde7e1d0edab52f56c7420926`：

```bash
➜ git clone https://github.com/alibaba/MNN
```

这次设备侧属性如下：

```bash
➜ adb connect 127.0.0.1:40404
➜ adb shell "cat /sys/devices/soc0/soc_id; getprop ro.soc.model; getprop ro.product.cpu.abi"
618
SM8750
arm64-v8a
```

在服务器准备原始测试模型[Qwen3-1.7B-MNN](https://huggingface.co/taobao-mnn/Qwen3-1.7B-MNN)，目录结构至少包含这些文件：

```text
Qwen3-1.7B-MNN/
├── config.json         # 运行时配置 包括后端设置、线程数、采样参数等
├── llm_config.json     # 导出时写入的模型元信息
├── tokenizer.txt       # 分词器文件
├── llm.mnn             # 模型计算图
└── llm.mnn.weight      # 模型权重
```
建议使用[hf-mirror](https://hf-mirror.com/)镜像站下载：

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh # 下载工具脚本

export HF_ENDPOINT=https://hf-mirror.com # 设置 Hugging Face 镜像站点
./hfd.sh taobao-mnn/Qwen3-1.7B-MNN # 下载模型到当前目录
```

## 2. 交叉编译

QNN 后端的运行需要准备两类产物：

- `Android` 设备侧运行的 `llm_demo`和所需的`MNN`编译产生动态库
- 导出的`QNN`模型和依赖的`QNN`动态库

### 2.1 交叉编译 `llm_demo`

QNN 版 `llm_demo` 编译时需要打开 `MNN_QNN` 和 `MNN_WITH_PLUGIN`。在**编译设备**执行：

```bash
# 要求 QNN_SDK_ROOT 已配置，指向 QAIRT SDK 根目录
cd ${MNN_ROOT}
cd project/android
mkdir -p build_64
cd build_64

# 交叉编译 Android arm64 版本的 QNN llm_demo
# 这里同步会编译模型转换工具，后续生成 QNN 模型时会用到
# 记住当前的目录
../build_64.sh -DMNN_QNN=ON -DMNN_WITH_PLUGIN=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
```

这次构建结束后，设备运行只用到了下面两个产物：

```bash
ls -lh llm_demo libMNN.so
```

### 2.2 导出 QNN 模型

导出`QNN`模型依赖主库编译产生的工具，需要先在服务器编译工具，**注意这不是交叉编译**
```bash
cd ${MNN_ROOT}
mkdir build && cd build
cmake .. -DMNN_QNN=ON -DMNN_QNN_CONVERT_MODE=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_LLM=ON
make -j256
```

`MNN`使用`python`脚本按 NPU 格式导出 MNN 模型。 先安装`python`依赖：

```bash
cd ${MNN_ROOT}  
cd ./transformers/llm/export
conda create -n mnn python=3.11 -y
conda activate mnn
uv pip install -r requirements.txt
```

再从`MNN`模型目录导出 `QNN` 模型，参数说明[参考](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html#qnn-llm), `soc`和`Hexagon`版本 [参考](https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=1601111740010412#supported-snapdragon-devices)（**这一步导出会出现部分导出错误的节点，MNN会把这部分计算交给CPU处理，导出没有失败**）
```bash
cd ${MNN_ROOT}
cd transformers/llm/export
export MODEL_DIR=/data/HF_MODELS/Qwen3-1.7B-MNN
python3 npu/generate_llm_qnn.py \
  --model "$MODEL_DIR" \
  --soc_id=69 \
  --dsp_arch=v79 \
  --mnn_path ../../../build/
  # 最后的参数是前面服务器上产生编译工具的路径，脚本会调用工具生成 QNN 模型并放在 `qnn/` 目录下
```

直接下载的 `taobao-mnn/Qwen3-1.7B-MNN` 使用 `tokenizer.txt`，`config_qnn.json` 不需要额外写 `tokenizer_file`。如果是自己用 `llmexport.py` 导出的模型，分词器可能是 `tokenizer.mtok`，这时需要给 `config_qnn.json` 补上：

```json
{
  "llm_model": "qnn/llm.mnn",
  "backend_type": "cpu",
  "thread_num": 1,
  "precision": "low",
  "chunk_limits": [
    128,
    1
  ],
  "memory": "low",
  "sampler_type": "penalty",
  "penalty": 1.1,
  "tokenizer_file": "tokenizer.mtok"
}
```

QNN 模型准备完成后，目录至少包含这些文件：

```text
Qwen3-1.7B-MNN/
├── config_qnn.json     # QNN 运行配置
├── llm_config.json     # 导出时写入的模型元信息
├── tokenizer.txt       # 直接下载模型的分词器文件
├── llm.mnn.weight      # 模型权重，也包含 tie_embeddings 使用的 embedding 数据
└── qnn/
    ├── llm.mnn         # QNN Plugin 替代模型
    ├── graph0.bin
    ├── graph1.bin
    └── ...             # QNN 离线产物
```

`config_qnn.json` 里的 `backend_type` 通常仍然是 `cpu`。这是 QNN LLM 离线产物的运行方式：QNN 图已经封装进 `qnn/llm.mnn`，外层还是通过 CPU 后端加载 Plugin，本次 `Qwen3-1.7B` 生成的 `qnn/` 目录约 `3.3G`。


## 3. 推送并运行

### 3.1 推送编译产物、模型和 QNN 依赖

`llm_demo` 不是静态单文件程序，除了可执行文件本身，还需要把 MNN 动态库、QNN 动态库、QNN 离线模型一起推送到手机。

下面命令按本次 `SM8750/v79` 测试环境整理, 按需要修改对应的环境变量：

```bash
export PHONE_DIR=/data/local/tmp/mnn-llm-qnn
export PHONE_MODEL_DIR=$PHONE_DIR/model/Qwen3-1.7B-MNN
export BUILD_DIR=/workspace/code/MNN/project/android/build_64
export MODEL_DIR=/data/HF_MODELS/Qwen3-1.7B-MNN
export QNN_SDK_ROOT=/root/qairt/2.40.0.251030
export HEXAGON_ARCH=79

# 创建手机侧目录
adb shell "rm -rf $PHONE_DIR && mkdir -p $PHONE_MODEL_DIR"

# 推送二进制和 MNN 动态库
# 如果是静态链接版本，llm_demo 会比较大，但不需要推其余动态库
adb push "$BUILD_DIR/llm_demo" "$PHONE_DIR/"
adb push "$BUILD_DIR/libMNN.so" "$PHONE_DIR/"
# adb push "$BUILD_DIR/libllm.so" "$PHONE_DIR/"
# adb push "$BUILD_DIR/libMNN_Express.so" "$PHONE_DIR/"

# 推送 QNN 运行库
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV${HEXAGON_ARCH}Stub.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/hexagon-v${HEXAGON_ARCH}/unsigned/libQnnHtpV${HEXAGON_ARCH}Skel.so" "$PHONE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so" "$PHONE_DIR/"

# 推送 QNN 运行需要的最小模型文件
adb push "$MODEL_DIR/config_qnn.json" "$PHONE_MODEL_DIR/"
adb push "$MODEL_DIR/llm_config.json" "$PHONE_MODEL_DIR/"
adb push "$MODEL_DIR/tokenizer.txt" "$PHONE_MODEL_DIR/"
adb push "$MODEL_DIR/llm.mnn.weight" "$PHONE_MODEL_DIR/"
adb push "$MODEL_DIR/qnn" "$PHONE_MODEL_DIR/"

# 准备一个文件输入，便于首次验证
printf 'hello\n' > prompt.txt
adb push prompt.txt "$PHONE_DIR/"

# 给可执行文件加权限
adb shell "chmod +x $PHONE_DIR/llm_demo"
```

`HEXAGON_ARCH` 必须和设备、QNN 模型生成参数匹配。本次如果误推 `V75` 库，会在 `logcat` 里看到 `Stub lib id mismatch`，随后 QNN 初始化失败。

### 3.2 运行

QNN LLM 运行时传入 `config_qnn.json`，不要传普通 `config.json`。

```bash
adb shell "cd $PHONE_DIR && \
export LD_LIBRARY_PATH=$PHONE_DIR && \
export ADSP_LIBRARY_PATH=$PHONE_DIR && \
./llm_demo $PHONE_MODEL_DIR/config_qnn.json $PHONE_DIR/prompt.txt"
```

测试后有正常输出：

```text
config path is /data/local/tmp/mnn-llm-qnn/model/Qwen3-1.7B-MNN/config_qnn.json
Prepare for tuning opt Begin
Prepare for tuning opt End
prompt file is /data/local/tmp/mnn-llm-qnn/prompt.txt
Hello! How can I assist you today?

prompt tokens num = 9
decode tokens num = 217
prefill speed = 37.52 tok/s
decode speed = 15.03 tok/s
```

`llm_demo` 还支持交互运行模式：

```bash
adb shell
cd /data/local/tmp/mnn-llm-qnn
export LD_LIBRARY_PATH=/data/local/tmp/mnn-llm-qnn
export ADSP_LIBRARY_PATH=/data/local/tmp/mnn-llm-qnn
./llm_demo /data/local/tmp/mnn-llm-qnn/model/Qwen3-1.7B-MNN/config_qnn.json
```

进入交互模式后，直接输入问题即可。
