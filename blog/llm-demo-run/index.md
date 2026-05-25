---
title: "以 llm_demo 为例的交叉编译与设备运行"
date: 2026-05-24T12:00:00+08:00
lastmod: 2026-05-24T12:00:00+08:00
draft: false
description: "记录 MNN 的 llm_demo 在 Android 上从交叉编译到 adb 运行的最短流程"
slug: "llm-demo-run"
tags: ["mnn", "llm", "android"]
categories: ["mnn"]

comments: true
math: true
---

# 交叉编译与设备运行

以 `transformers/llm/engine/demo/llm_demo.cpp` 在 Android 设备上执行流程：**服务器交叉编译 -> `adb push`推送到手机 -> 端侧设备运行**。 需要准备下面环境：
- `adb devices` 能看到目标手机，参考[远程ADB环境配置](../remote-adb/index.md)
- `ANDROID_NDK` 已配置，参考[交叉编译环境配置](../cross-compiler/index.md)

## 1. 环境

以下面环境为例：

- 开发机(**编译设备**)：`Ubuntu 22.04`
- 测试机(**目标设备**)：`arm64-v8a` + `SanpDragon 8 Gen 3`
- 编译目录：`project/android/build_64`
- 手机测试目录：`/data/local/tmp/mnn-llm-demo`

首先需要拉取仓库到编译设备
```bash
git clone https://github.com/alibaba/MNN
```

并且在编译设备上准备模型，模型以[Qwen3-4B-Instruct-MNN](https://huggingface.co/taobao-mnn/Qwen3-4B-Instruct-2507-MNN)，目录结构至少包含这些文件：

```text
Qwen3-4B-Instruct-2507-MNN/
├── config.json         # 运行时配置 包括后端设置、线程数、采样参数等
├── llm_config.json     # 导出时写入的模型元信息
├── tokenizer.mtok      # 分词器文件
├── llm.mnn             # 模型计算图
└── llm.mnn.weight      # 模型权重
```
建议使用[hf-mirror](https://hf-mirror.com/)镜像站下载：

```bash
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh # 下载工具脚本

export HF_ENDPOINT=https://hf-mirror.com # 设置 Hugging Face 镜像站点
./hfd.sh taobao-mnn/Qwen3-4B-Instruct-2507-MNN # 下载模型到当前目录
```

## 2. 交叉编译 `llm_demo`

`llm_demo` 位于 `transformers/llm/engine/demo/llm_demo.cpp`，编译它时至少要打开 `MNN_BUILD_LLM`。这一篇只保留能直接运行 LLM 的常用宏，在**编译设备**执行：

```bash
# 服务器端运行
cd /workspace/zjh/code/MNN/project/android
mkdir -p build_64
cd build_64

# 交叉编译 Android arm64 版本的 llm_demo
../build_64.sh \
  -DMNN_LOW_MEMORY=true \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
  -DMNN_BUILD_LLM=true \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=true \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

这一步结束后，当前目录下有下面几个产物：

```bash
ls -lh llm_demo libMNN.so libMNN_Express.so libllm.so
```

如果还要测试 OpenCL，可以继续加上 `-DMNN_OPENCL=true`。

## 3. 推送并运行
### 3.1 推送编译产物和模型

`llm_demo` 不是静态单文件程序，除了可执行文件本身，还需要把依赖的动态库 `.so` 一起推送到手机。

下面假设服务器上的模型目录是 `/data/HUGGINGFACE/Qwen3-4B-Instruct-MNN`，**需要按自己的模型下载路径替换**。

```bash
# 服务器端运行
export PHONE_DIR=/data/local/tmp/mnn-llm-demo # 手机侧测试目录
export MODEL_DIR=/data/HUGGINGFACE/Qwen3-4B-Instruct-MNN # 服务器侧模型目录

# 创建手机侧目录
adb shell "mkdir -p $PHONE_DIR/model"

# 推送二进制和动态库
# 执行时当前目录位于编译产物所在目录，即 `project/android/build_64`
adb push llm_demo "$PHONE_DIR/"
adb push libMNN.so "$PHONE_DIR/"
adb push libMNN_Express.so "$PHONE_DIR/"
adb push libllm.so "$PHONE_DIR/"

# 推送模型目录
adb push "$MODEL_DIR/" "$PHONE_DIR/model/"

# 准备一个文件输入，便于首次验证
printf 'hello\n' > prompt.txt
adb push prompt.txt "$PHONE_DIR/"

# 给可执行文件加权限
adb shell "chmod +x $PHONE_DIR/llm_demo"
```

### 3.2 运行

```bash
adb shell "cd $PHONE_DIR && \
export LD_LIBRARY_PATH=$PHONE_DIR && \
./llm_demo $PHONE_DIR/model/config.json $PHONE_DIR/prompt.txt"
```

正常情况下会看到类似输出：

```text
config path is /data/local/tmp/mnn-llm-demo/model/config.json
Prepare for tuning opt Begin
Prepare for tuning opt End
hello
A: Hello! How can I help you today?
```

`llm_demo`还支持交互运行模式：

```bash
adb shell
cd /data/local/tmp/mnn-llm-demo
# 需要先设置环境变量 程序才能找到编译出来的动态库
export LD_LIBRARY_PATH=/data/local/tmp/mnn-llm-demo
# 这里传入的参数是模型配置文件的路径
./llm_demo /data/local/tmp/mnn-llm-demo/model/config.json
```

进入交互模式后，直接输入问题即可。
