---
title: "LLM QNN 离线导出流程"
date: 2026-06-04T12:00:00+08:00
lastmod: 2026-06-04T12:00:00+08:00
draft: false
description: "按当前仓库代码梳理 MNN LLM QNN 模型从 llm.mnn 到 qnn/*.bin 的离线导出过程"
slug: "llm-qnn-export"
tags: ["mnn", "llm", "qnn"]
categories: ["mnn"]

comments: true
math: true
---

****# LLM QNN 离线导出流程
- [1. 入口参数](#1-入口参数)
- [2. Step1：生成 IO 样本](#2-step1生成-io-样本)
- [3. Step2：切图](#3-step2切图)
  - [3.1 QNN 离线模式](#31-qnn-离线模式)
  - [3.2 读取 testdir](#32-读取-testdir)
  - [3.3 生成候选子图列表](#33-生成候选子图列表)
    - [3.3.1 反向收集 op](#331-反向收集-op)
    - [3.3.2 按 break op 切图](#332-按-break-op-切图)
    - [3.3.3 补切动态 shape](#333-补切动态-shape)
    - [3.3.4 计算子图边界](#334-计算子图边界)
  - [3.4 过滤非卷积子图](#34-过滤非卷积子图)
  - [3.5 收集子图 IO](#35-收集子图-io)
  - [3.6 编译并替换 Plugin](#36-编译并替换-plugin)
  - [3.7 onForward 生成 graph 材料](#37-onforward-生成-graph-材料)
  - [3.8 合并多 shape](#38-合并多-shape)
  - [3.9 产物](#39-产物)
- [4. npu\_postreat.json 的作用](#4-npu_postreatjson-的作用)
- [5. Step3：生成 context binary](#5-step3生成-context-binary)
  - [5.1 读取后处理配置](#51-读取后处理配置)
  - [5.2 写 QNN context 配置](#52-写-qnn-context-配置)
  - [5.3 生成每个 graph 的 model library](#53-生成每个-graph-的-model-library)
  - [5.4 合成 context binary](#54-合成-context-binary)
  - [5.5 运行时使用 binary](#55-运行时使用-binary)
- [6. Step4：移动产物并写 config\_qnn.json](#6-step4移动产物并写-config_qnnjson)
- [7. 最终目录结构](#7-最终目录结构)

本文记录 `transformers/llm/export/npu/generate_llm_qnn.py` 将 MNN 格式模型导出到 QNN 后端的链路。主要把 `llm.mnn` / `llm.mnn.weight` 文件转成 QNN 运行需要的两类产物：

- `qnn/llm.mnn`：已经插入 `OpType_Plugin(type="QNN")` 的主模型；
- `qnn/*.bin`：QNN HTP context binary，运行时由 QNN Plugin 直接加载执行。

整体流程分为四步：

```text
generate_llm_qnn.py
  -> convert(args)
    -> makeIO(args)
    -> seperate(args)
    -> compile_qnn(args)
    -> output_qnn(args)
```

## 1. 入口参数

脚本入口在 `transformers/llm/export/npu/generate_llm_qnn.py`，常用参数如下：

| 参数 | 说明 |
|------|------|
| `--model` | MNN LLM 模型目录 |
| `--soc_id` | QNN 目标 SoC id |
| `--dsp_arch` | QNN DSP 架构，例如 `v75`、`v79` |
| `--mnn_path` | MNN build 目录，需要包含 `generateLlmIO` 和 `compilefornpu` |
| `--cache_path` | 临时工作目录，默认 `tmp` |
| `--chunk_size` | 写入 `config_qnn.json` 的 prefill shape，默认 `128` |

脚本大量通过 `os.getcwd()` 拼路径，所以运行目录很重要。常见执行方式是从仓库根目录进入导出目录后运行。`soc_id` 和 `dsp_arch` 需要和目标 SoC / Hexagon 版本对应，可以参考 [Qualcomm 官方设备信息](https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=1601111740010412#supported-snapdragon-devices)：

```bash
cd ${MNN_ROOT}
cd transformers/llm/export

python3 npu/generate_llm_qnn.py \
  --model /data/HF_MODELS/Qwen3-1.7B-MNN \
  --soc_id 69 \
  --dsp_arch v79 \
  --mnn_path ../../../build/
```

模型目录至少需要这些文件，可以直接使用官方提供的模型 [Qwen3-1.7B-MNN](https://huggingface.co/taobao-mnn/Qwen3-1.7B-MNN)：

```text
Qwen3-1.7B-MNN/
├── config.json
├── llm_config.json
├── tokenizer.txt
├── llm.mnn
└── llm.mnn.weight
```

`npu_convert.py` 还会读取 `QNN_SDK_ROOT`，所以开发机环境里需要先配置 QAIRT / QNN SDK。

## 2. Step1：生成 IO 样本

LLM QNN 导出从 `convert(args)` 方法开始。Step1 对应的是 `makeIO(args)`：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:67
def convert(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    print("Step1: Make IO")
    makeIO(args)
```

`makeIO(args)` 本身没有解析 `llm_config.json`，它只是拼出可执行文件 `generateLlmIO` 的路径（由主库编译产生，详见 [QNN 模型导出](../llm-demo-qnn-run/index.md#22-导出-qnn-模型)），并把模型目录、输出目录和 `chunk_size` 传进去：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:10
def makeIO(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "generateLlmIO")
    output = os.path.join(args.cache_path, 'testdir')
    print(os.popen(exe + " " + args.model + " " + output + ' %d' %args.chunk_size).read())
```

所以 Step1 实际执行的命令可以展开成：

```bash
<cwd>/<mnn_path>/generateLlmIO \
  <model_dir> \
  tmp/testdir \
  <chunk_size>
# generate_llm_qnn.py 会先创建 tmp；generateLlmIO 会创建 tmp/testdir 和子目录
```

这里 `<model_dir>` 是整个 MNN LLM 模型目录，不是单独的 `llm.mnn` 文件。`llm_config.json`、`llm.mnn`、`llm.mnn.weight` 等文件会由 `generateLlmIO` 在模型目录内部继续读取。

当前导出脚本会先通过 `os.makedirs(cache, exist_ok=True)` 创建 `tmp`。随后 `generateLlmIO` 内部会调用 `MNNCreateDir(outputDir)` 创建 `tmp/testdir`，再创建 `tmp/testdir/<chunk_size>` 和 `tmp/testdir/1`。如果单独手动运行 `generateLlmIO`，需要保证父目录 `tmp` 已经存在，因为这里的 `MNNCreateDir(...)` 不是递归创建目录。

`generate_llm_qnn.py` 只关心两个后续会被 `compilefornpu` 读取的样本目录：

| 用途 | 样本目录 |
|------|----------------|
| prefill | `testdir/<chunk_size>` |
| decode | `testdir/1` |

生成结果会落在 `tmp/testdir` 下，后续 `compilefornpu` 会按 `testdir/1`、`testdir/<chunk_size>` 读取这些输入输出样本。

生成结果包含两组不同长度的输入输出样本：
```bash
➜  export git:(master) ✗ ../../../build/generateLlmIO /data/HUGGINGFACE/Qwen3-1.7B-Eagle3-MNN tmp/testdir 128
blockSize=128 in main, 148
modelPath.c_str()=s /data/HUGGINGFACE/Qwen3-1.7B-Eagle3-MNN/llm.mnn in main, 152
llmConfigPath.c_str()=s /data/HUGGINGFACE/Qwen3-1.7B-Eagle3-MNN/llm_config.json in main, 153
Can't open file:/sys/devices/system/cpu/cpufreq/boost/affected_cpus
Can't open file:/sys/devices/system/cpu/cpufreq/ondemand/affected_cpus
CPU Group: [ 95  17  83  55  118  27  93  65  5  37  75  47  19  85  57  29  108  67  7  39  101  10  77  49  111  20  87  59  121  30  97  69  33  112  21  88  122  31  98  41  104  13  51  114  23  61  1  124  9  71  43  106  15  81  53  116  25  91  63  3  126  35  73  45  94  117  26  92  64  4  127  36  74  46  109  18  84  56  119  28  54  66  6  38  100  76  48  110  86  58  120  96  68  8  102  11  105  40  103  12  79  50  113  22  89  60  0  123  32  99  70  42  78  14  80  52  115  24  90  62  2  125  34  72  44  107  16  82 ], 1500000 - 2900000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Successfully generate tmp/testdir/128/input.mnn and tmp/testdir/128/output.mnn.
Successfully generate tmp/testdir/1/input.mnn and tmp/testdir/1/output.mnn.
```

输入由 `createInputsForLLM(...)` 构造，输出由这些输入跑一次 LLM 网络前向得到：


```cpp
// transformers/llm/engine/tools/generateLlmIO.cpp:117
{
    std::vector<MNN::Express::VARP> inputs;
    std::vector<MNN::Express::VARP> outputs;
    createInputsForLLM(blockSize, hiddenSize, attentionMaskType, false, inputs);
    outputs = net->onForward(inputs);
    saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, blockSize);
}

// transformers/llm/engine/tools/generateLlmIO.cpp:125
{
    std::vector<MNN::Express::VARP> inputs;
    std::vector<MNN::Express::VARP> outputs;
    createInputsForLLM(1, hiddenSize, attentionMaskType, true, inputs);
    outputs = net->onForward(inputs);
    saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, 1);
}
```

## 3. Step2：切图

`convert(args)` 的 Step2 直接调用 `seperate(args)`：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:76
print("Step2: Seperate Model")
seperate(args)
```

`seperate(args)` 先定位可执行文件 `compilefornpu`（由主库编译产生，详见 [QNN 模型导出](../llm-demo-qnn-run/index.md#22-导出-qnn-模型)）和原始 `llm.mnn`，再写一个临时 `qnn.json`。这里的 `seperate` 是源码里的函数名拼写，文档里保持不改：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:15
def seperate(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, 'llm.mnn')
    config = {
        "type":"QNN",
        "skips":[],
        "testdir":[],
        "cache":"qnn"
    }
    config['testdir'].append(os.path.join("testdir", '1'))
    config['testdir'].append(os.path.join("testdir", '%d' %args.chunk_size))
```

写出的配置内容对应下面这样：

```json
{
  "type": "QNN",
  "skips": [],
  "testdir": [
    "testdir/1",
    "testdir/<chunk_size>"
  ],
  "cache": "qnn"
}
```

然后它在 `tmp` 目录下执行 `compilefornpu`：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:29
cache = os.path.join(os.getcwd(), args.cache_path)
with open(os.path.join(cache, 'qnn.json'), 'w') as f:
    f.write(json.dumps(config, indent=4))

process = subprocess.Popen(
    exe + ' ' + model + ' qnn/llm.mnn qnn.json',
    cwd = cache,
    shell=True
)
```

命令展开后是：

```bash
<build>/compilefornpu \
  <model_dir>/llm.mnn \
  qnn/llm.mnn \
  qnn.json
```

`compilefornpu` 的入口在 `tools/cpp/compilefornpu.cpp`。这一步会在原图上做三件事：

- 先根据输入输出 tensor 找出真正需要计算的 op；
- 再把连续可下沉的 op 区间整理成 `SubModuleInfo`；
- 最后把每个可下沉区间替换成一个 `OpType_Plugin(type="QNN")`。

代码调用关系如下：

```text
generate_llm_qnn.py
└── seperate(args)                                      # 生成切图配置并启动 compilefornpu
    ├── 写 tmp/qnn.json                                # 告诉 compilefornpu 目标后端和测试 shape
    │   ├── type = "QNN"
    │   ├── testdir = ["testdir/1", "testdir/<chunk_size>"]
    │   └── cache = "qnn"
    │
    └── subprocess.Popen(...)
        └── compilefornpu <model_dir>/llm.mnn qnn/llm.mnn qnn.json
            │
            ├── main(argc, argv)                       # 切图工具主入口
            │   ├── 解析 qnn.json                      # 初始化 QNN 离线导出参数
            │   │   ├── gNPUType = MNN_CONVERT_QNN
            │   │   ├── gNeedOffline = true
            │   │   └── gCacheDir = "qnn"
            │   ├── 读取 testdir/*/input.mnn 和 output.mnn # 取得 decode / prefill 两组输入输出
            │   ├── Interpreter::createFromFile(srcMNN)    # 读取原始 llm.mnn
            │   ├── initConstTensorsNoAlloc(...)           # 标记常量 tensor
            │   └── _createSubModuleInfo(...)              # 生成候选子图列表
            │       ├── _collectNeededOps(...)             # 从输出反向追依赖 op
            │       ├── isBreakOp(...)                     # 遇到控制流 / 不支持 op 就切断
            │       ├── _splitSubModuleForShapeConst(...)  # shape 依赖输入内容时补切
            │       └── _computeTensorMask(...)            # 计算子图输入输出边界
            │
            ├── for 每组 testdir 输入 shape                  # 分别处理 decode / prefill
            │   ├── _getSubModuleIO(...)                    # 跑原图收集子图参考 IO
            │   │   └── 记录 KV state shape                 # KV 下沉时写 Plugin state
            │   ├── 过滤不含 Convolution 的子图              # 当前导出只保留卷积子图
            │   └── _compileSubModule(...)                  # 编译子图并生成 Plugin op
            │       ├── RuntimeManager(type = MNN_CONVERT_QNN)
            │       ├── setCache("qnn/graph*")              # 指定 QNN graph 源目录
            │       ├── Module::load(...)
            │       ├── Module::onForward(...)              # 触发 QNN convert backend 导出 graph
            │       └── 生成 OpType_Plugin(type="QNN")      # 用一个 Plugin 表示整段子图
            │
            ├── 用 Plugin 替换原始连续 op 区间               # 改写主图
            ├── _reIndexTensor(...) / _reOrderOp(...)       # 清理 tensor 并重排拓扑
            ├── 多 shape 结果用 _fuse(...) 合并 Plugin attr  # 合并 decode / prefill graph 信息
            ├── 写 qnn/llm.mnn                              # 插入 QNN Plugin 后的主模型
            └── 写 npu_postreat.json                        # 给 npu_convert.py 继续生成 context binary
                └── merge: qnn/graph*.bin -> [qnn/graph*, qnn/graph1_*, ...]
```

### 3.1 QNN 离线模式

`compilefornpu` 的命令格式是：

```cpp
// tools/cpp/compilefornpu.cpp:836
int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./compilefornpu src.mnn dst.mnn npu.json\n");
        return 0;
    }
    const char* srcMNN = argv[1];
    const char* dstMNN = argv[2];
```

当 `qnn.json` 里写的是 `"type": "QNN"` 时，代码会把后端类型设成 `MNN_CONVERT_QNN`，并打开 `gNeedOffline`：

```cpp
// tools/cpp/compilefornpu.cpp:859
gNPUName = document["type"].GetString();
if (gNPUName == "QNN") {
    MNN_PRINT("Convert for QNN, QualComn's NPU\n");
    gNPUType = MNN_CONVERT_QNN;
    gNeedOffline = true;
    gOfflieSrc = "";
    gOfflieDst = "bin";
}
```

`gNeedOffline = true` 之后，`_compileSubModule(...)` 会通过一次 `Module::onForward()` 触发 QNN convert 后端导出 graph 材料。

### 3.2 读取 testdir

`generate_llm_qnn.py` 写入了两组 `testdir`：

```json
"testdir": [
  "testdir/1",
  "testdir/<chunk_size>"
]
```

`compilefornpu` 会逐个读取里面的 `input.mnn` 和 `output.mnn`：

```cpp
// tools/cpp/compilefornpu.cpp:898
if (document.HasMember("testdir")) {
    auto testdir = document["testdir"].GetArray();
    for (auto iter = testdir.Begin(); iter != testdir.End(); iter++) {
        std::string dirname = iter->GetString();
        auto subinputs = MNN::Express::Variable::load((dirname + "/input.mnn").c_str());
        inputs.emplace_back(subinputs);
        inputNames.clear();
        for (int i=0; i<subinputs.size(); ++i) {
            inputNames.emplace_back(subinputs[i]->name());
        }
        auto outputs = MNN::Express::Variable::load((dirname + "/output.mnn").c_str());
        outputNames.clear();
        for (int i=0; i<outputs.size(); ++i) {
            outputNames.emplace_back(outputs[i]->name());
        }
    }
}
```
这里的 `input.mnn` / `output.mnn` 提供了：
- 主图输入名；
- 主图输出名；
- decode / prefill 两组真实输入 shape 和数据；

### 3.3 生成候选子图列表
`compilefornpu` 使用 `_createSubModuleInfo(...)` 生成需要转换的候选子图列表，流程如下。

#### 3.3.1 反向收集 op

切图的第一步是 `_collectNeededOps(...)`。它先给输入和输出 tensor 打标记，再从 `outputIndexes` 反向追溯数据依赖，直到遇到 `inputIndexes`：

```cpp
// tools/cpp/compilefornpu.cpp:261
static std::vector<SubModuleInfo> _createSubModuleInfo(
    const Net* net,
    const std::set<int>& inputIndexes,
    const std::set<int>& outputIndexes,
    const std::set<int>& noComputeIndexes,
    std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    std::vector<SubModuleInfo> submodule;
    auto selectOps = _collectNeededOps(net, inputIndexes, outputIndexes);
    // ...
}

// tools/cpp/compilefornpu.cpp:116
static std::vector<int> _collectNeededOps(
    const MNN::Net* net,
    const std::set<int>& inputIndexes,
    const std::set<int>& outputIndexes) {
    // 0: not set, 1: output, 2:input
    std::vector<int> tensorMask(net->tensorName()->size());
    ...
    for (auto v : outputIndexes) {
        tensorMask[v] = 1;
    }
    for (auto v : inputIndexes) {
        tensorMask[v] = 2;
    }
```

接着沿着输出到输入的方向反复传播标记：如果某个 op 的输出已经被标成需要保留，就把这个 op 标成需要计算，并继续把它的输入 tensor 标成依赖项：

```cpp
// tools/cpp/compilefornpu.cpp:134
do {
    change = false;
    for (int i=0; i<opMask.size(); ++i) {
        ...
        if (nullptr != op->outputIndexes()) {
            for (int j=0; j<op->outputIndexes()->size(); ++j) {
                auto index = op->outputIndexes()->data()[j];
                if (tensorMask[index] == 1) {
                    opMask[i] = 1;
                    change = true;
                }
            }
        }
        if (nullptr != op->inputIndexes() && opMask[i]) {
            for (int j=0; j<op->inputIndexes()->size(); ++j) {
                auto index = op->inputIndexes()->data()[j];
                if (tensorMask[index] != 2) {
                    tensorMask[index] = 1;
                }
            }
        }
    }
} while (change);
```

#### 3.3.2 按 break op 切图

`_createSubModuleInfo(...)` 会按原图 op 顺序扫描上一步采集的算子 `selectOps`：普通 op 缓存到 `current.opList`；遇到 break op，就先把前面的连续算子保存成一个子模块，再把这个 break op 单独保存成一个 `SubModuleInfo`：

```cpp
// tools/cpp/compilefornpu.cpp:267
for (int si=0; si<selectOps.size(); ++si) {
    auto i = selectOps[si];
    auto op = net->oplists()->GetAs<Op>(i);
    if (isBreakOp(op)) {
        if (current.opList.size() > 0) {
            submodule.emplace_back(std::move(current));
        }
        SubModuleInfo controlOp;
        controlOp.opList = {i};
        controlOp.isBreak = true;
        ...
        submodule.emplace_back(std::move(controlOp));
        continue;
    }
    current.opList.emplace_back(i);
}
```

`isBreakOp(...)` 的判别规则是，控制流算子属于 `break op`，例如 `OpType_While`、`OpType_If` 等：

```cpp
// tools/cpp/compilefornpu.cpp:102
static bool isBreakOp(const Op* op) {
    bool isWhileControlflow = false;
    if (op->type() == OpType_While && op->main_as_WhileParam() != nullptr) {
        isWhileControlflow = true;
    }
    if (op->type() == OpType_If || isWhileControlflow || op->type() == OpType_Where || op->type() == OpType_Segment || op->type() == OpType_Unique || op->type() == OpType_NonMaxSuppressionV2) {
        return true;
    }
    if (!_npuSupportOp(op)) {
        return true;
    }
    return false;
}
```

对当前 QNN LLM 导出，需要注意 `Attention + kv_cache`。默认配置下，使用 KV cache 的注意力算子会被 `_npuSupportOp(...)` 判成不支持，随后被 `isBreakOp(...)` 切断：

```cpp
// tools/cpp/compilefornpu.cpp:89
static bool _npuSupportOp(const Op* op) {
    if (gMaxKVSize > 0) {
        return true;
    }
    if (op->type() == OpType_Attention) {
        auto attn = op->main_as_AttentionParam();
        if (nullptr != attn && attn->kv_cache()) {
            return false;
        }
    }
    return true;
}
```

#### 3.3.3 补切动态 shape

第一次切图主要看 op 类型。接下来还会用 `_splitSubModuleForShapeConst(...)` 做第二次切分：

```cpp
// tools/cpp/compilefornpu.cpp:295
submodule = _splitSubModuleForShapeConst(submodule, net, sharedConst);
```

这一步处理的是“shape 推导依赖输入内容”的 op。判断入口在 `_findBreakIndex(...)`：

```cpp
// tools/cpp/compilefornpu.cpp:214
int inputNum = op->inputIndexes()->size();
auto dims = SizeComputer::needInputContent(op, inputNum);
for (auto index : dims) {
    if (index < inputNum) {
        if (constMask[op->inputIndexes()->data()[index]] != 1) {
            res.emplace_back(v);
            break;
        }
    }
}
```

含义是：如果某个 op 的 shape 计算需要读取输入 tensor 的内容，并且这个输入不是常量，就把这个 op 作为切分点。这样可以避免把动态 shape / 内容相关 shape 逻辑直接塞进 NPU 子图里。

#### 3.3.4 计算子图边界

`SubModuleInfo` 里真正描述切图边界的是：

```cpp
// tools/cpp/compilefornpu.cpp:56
struct SubModuleInfo {
    std::vector<int> opList;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<int> tensorMask;
    bool isBreak = false;
};
```

每个连续子图的 `inputs` / `outputs` 都是根据子图内部 op 的 tensor 使用情况计算出来。`_computeTensorMask(...)` 会先给 tensor 打标记，含义如下：

```text
1: 这个 tensor 被当前子图消费，是子图输入
2: 这个 tensor 由当前子图产生，是子图输出
3: 这个 tensor 既被当前子图消费又由当前子图产生，是子图内部中间 tensor
```

对应代码在 `_createSubModuleInfo(...)` 里：

```cpp
// tools/cpp/compilefornpu.cpp:299
if (!m.isBreak) {
    _computeTensorMask(m, net);
    for (int i=0; i<m.tensorMask.size(); ++i) {
        if (1 == m.tensorMask[i]) {
            if (noComputeIndexes.find(i) != noComputeIndexes.end()) {
                continue;
            }
            m.inputs.emplace_back(i);
            continue;
        }
        if (2 == m.tensorMask[i]) {
            m.outputs.emplace_back(i);
            continue;
        }
        if (3 == m.tensorMask[i]) {
            if (outputIndexes.find(i) != outputIndexes.end()) {
                m.outputs.emplace_back(i);
            }
        }
    }
}
```

产生的边界规则可以写成：

- 子图消费、但不是子图内部产生的 tensor，是子图输入；
- 子图产生、并且需要给后续 op 或最终输出使用的 tensor，是子图输出；
- 子图内部产生又内部消费的 tensor，是中间 tensor，默认不暴露；
- 如果某个中间 tensor 被后续子图需要，代码会把它补进前一个子图的 `outputs`，对应逻辑如下：

```cpp
// tools/cpp/compilefornpu.cpp:348
for (int sub=0; sub < moduleIndex; ++sub) {
    if (submodule[sub].tensorMask.empty()) {
        continue;
    }
    if (submodule[sub].tensorMask[index] == 2) {
        find = true;
        break;
    }
    if (submodule[sub].tensorMask[index] == 3) {
        submodule[sub].outputs.emplace_back(index);
        submodule[sub].tensorMask[index] = 2;
        find = true;
        break;
    }
}
```

`compilefornpu` 切图时不按层数、名字或者固定 pattern 切分，而是按 tensor 的数据依赖关系确定子图边界。

### 3.4 过滤非卷积子图

`_createSubModuleInfo(...)` 得到的是候选子图。还有一个过滤条件：如果候选子图里没有 `OpType_Convolution`，就重新标记成 `isBreak = true`：

```cpp
// tools/cpp/compilefornpu.cpp:1068
bool hasConvolution = false;
for (auto opIndex : subModulesInfo[i].opList) {
    auto op = net->oplists()->GetAs<Op>(opIndex);
    if (op->type() == OpType_Convolution) {
        hasConvolution = true;
        break;
    }
}
if (!hasConvolution) {
    subModulesInfo[i].isBreak = true;
}
```
这个过滤的效果是：只转换含有卷积算子的子图，其他子图直接保留在主图中。

### 3.5 收集子图 IO

切图完成后，主流程会按每一组 `testdir` 输入跑所有子图：

```cpp
// tools/cpp/compilefornpu.cpp:1040
for (int inputIndex=0; inputIndex < inputs.size(); ++inputIndex) {
    std::map<int, MNN::Express::VARP> stackes;
    ...
    std::vector<SubModuleIO> moduleIO(subModulesInfo.size());
    for (int i=0; i<subModulesInfo.size(); ++i) {
        auto& current = subModulesInfo[i];
        std::vector<MNN::Express::VARP> subInputs;
        for (auto index : current.inputs) {
            subInputs.emplace_back(stackes[index]);
        }
        moduleIO[i] = _getSubModuleIO(subInputs, current, bufferPair.first, bufferPair.second, srcMNN);
        for (int j=0; j<current.outputs.size(); ++j) {
            stackes.insert(std::make_pair(current.outputs[j], moduleIO[i].outputs[j]));
        }
    }
```

这里的 `stackes` 是一个 `tensor index -> VARP` 表。每跑完一个子图，就把它的输出放回 `stackes`，给后续子图继续使用。

`_getSubModuleIO(...)` 会用原始 MNN 模型加载当前子图的输入输出名，然后跑 CPU/default 路径得到参考输出：

```cpp
// tools/cpp/compilefornpu.cpp:399
static SubModuleIO _getSubModuleIO(
    std::vector<MNN::Express::VARP> inputs,
    const SubModuleInfo& info,
    const void* buffer,
    size_t bufferSize,
    std::string srcpath) {
    ...
    std::shared_ptr<MNN::Express::Module> m(
        MNN::Express::Module::load(inputNames, outputNames, (const uint8_t*)buffer, bufferSize, rtmgr),
        MNN::Express::Module::destroy);
    ...
    auto outputs = m->onForward(inputs);
    io.inputs = inputs;
    io.outputs.resize(outputs.size());
    for (int i=0; i<outputs.size(); ++i) {
        io.outputs[i] = MNN::Express::_Clone(outputs[i], true);
    }
    return io;
}
```

如果子图里有带 `kv_cache` 的 `Attention`，这里还会通过 callback 记录 KV state 的形状：

```cpp
// tools/cpp/compilefornpu.cpp:419
MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
    auto opName = info->name();
    if (info->type() != "Attention") {
        return true;
    }
    if (attentionNames.find(opName) != attentionNames.end()) {
        auto query = ntensors[0];
        auto key = ntensors[1];
        auto value = ntensors[2];
        int seq_len = query->length(1);
        auto numHead = query->length(2);
        auto headDim = query->length(3);
        auto kvNumHead = key->length(2);
        std::vector<int> kvDims = {kvNumHead, 1, 1, headDim};
        io.kvcache.emplace_back(kvDims);
    }
    return true;
};
```

### 3.6 编译并替换 Plugin

真正触发 QNN 导出的是 `_compileSubModule(...)`：

```cpp
// tools/cpp/compilefornpu.cpp:450
static std::unique_ptr<MNN::OpT> _compileSubModule(
    const SubModuleIO& io,
    SubModuleInfo& info,
    const void* buffer,
    size_t bufferSize,
    const std::string& path,
    std::string srcpath,
    const std::string& targetNpuPath,
    float& cpuTotal,
    float& npuTotal,
    int shapeIndex,
    std::string graphicName) {
```

该方法的核心调用在下面这段，其中 `config.type = MNN_CONVERT_QNN`：

```cpp
// tools/cpp/compilefornpu.cpp:470
MNN::ScheduleConfig config;
config.type = gNPUType;
std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(
    MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
rtmgr->setExternalFile((srcpath + ".weight").c_str());
rtmgr->setCache(path.c_str());
rtmgr->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, gMaxKVSize);
MNN::Express::Module::Config mdconfig;
mdconfig.shapeMutable = false;
std::shared_ptr<MNN::Express::Module> m(
    MNN::Express::Module::load(inputNames, outputNames, (const uint8_t*)buffer, bufferSize, rtmgr, &mdconfig),
    MNN::Express::Module::destroy);
auto predict = m->onForward(io.inputs);
if (gNeedOffline) {
    break;
}
```

这里的 `onForward(io.inputs)` 不只是为了拿一次输出。对于 QNN 离线导出，它的主要作用是驱动一次 `StaticModule` 前向，让 QNN convert 后端根据当前子图输入 shape 生成子图 graph 材料。因为 `gNeedOffline == true`，触发导出后会直接 `break`，不会继续做误差检查和测速。

导出完成后，`_compileSubModule(...)` 会创建一个新的 `Plugin` op：

```cpp
// tools/cpp/compilefornpu.cpp:532
std::unique_ptr<MNN::OpT> op(new OpT);
op->inputIndexes = info.inputs;
op->outputIndexes = info.outputs;
op->name = targetNpuPath;
op->type = MNN::OpType_Plugin;
op->main.type = MNN::OpParameter_Plugin;
op->main.value = new MNN::PluginT;
auto extra = op->main.AsPlugin();
extra->type = gNPUName;
```

这个 Plugin 保留了原子图的输入输出 tensor index，相当于用一个 op 占住原来整段子图的输入输出边界。写入的关键 attr 包括：

| attr | 作用 |
|------|------|
| `path` | 最终 QNN binary 的相对路径，例如 `qnn/graph0.bin` |
| `inputs` | Plugin 输入对应的 QNN tensor 名，一般是 `t<tensor_index>` |
| `outputs` | Plugin 输出对应的 QNN tensor 名 |
| `allGraphName` | `.bin` 内多个 graph 的名字顺序 |
| `allInputShape` | 当前 graph 对应的输入 shape |
| `o_<shapeIndex>_<outputIndex>` | 当前 shape 下输出 tensor 的形状、类型和 format |
| `state` | KV cache state 信息，存在 KV 下沉时使用 |

以 `inputs` / `outputs` 为例，代码直接用子图边界 tensor index 生成 QNN tensor 名：

```cpp
// tools/cpp/compilefornpu.cpp:547
attr->key = "inputs";
attr->list.reset(new ListValueT);
attr->list->s.resize(inputNames.size());
for (int i=0; i<inputNames.size(); ++i) {
    attr->list->s[i] = std::string("t") + std::to_string(info.inputs[i]);
}

// tools/cpp/compilefornpu.cpp:625
attr->key = "outputs";
attr->list.reset(new ListValueT);
attr->list->s.resize(outputNames.size());
for (int i=0; i<outputNames.size(); ++i) {
    attr->list->s[i] = std::string("t") + std::to_string(info.outputs[i]);
}
```

随后发生主图替换：

```cpp
// tools/cpp/compilefornpu.cpp:1124
for (int moduleIndex=0; moduleIndex<subModulesInfo.size(); ++moduleIndex) {
    auto moduleInfo = subModulesInfo[moduleIndex];
    if (moduleInfo.isBreak) {
        continue;
    }
    for (auto& index : moduleInfo.opList) {
        dstNet->oplists[index].reset();
    }
    dstNet->oplists[moduleInfo.opList[0]] = std::move(npuOps[moduleIndex]);
}
```

替换关系如下：

```text
原始连续 op 区间:
  opA -> opB -> opC

替换后:
  Plugin(type="QNN", inputs=原区间输入, outputs=原区间输出)
```

然后再调用 `_reIndexTensor(...)` 和 `_reOrderOp(...)` 清理未使用 tensor，并重新拓扑排序：

```cpp
// tools/cpp/compilefornpu.cpp:1140
_reIndexTensor(dstNet.get());
_reOrderOp(dstNet.get());
```

### 3.7 onForward 生成 graph 材料

`_compileSubModule(...)` 里的 `onForward(io.inputs)` 走的是一次真实的 MNN 模块执行入口，但在 `MNN_CONVERT_QNN` 后端下，执行的重点不是计算数值结果，而是触发 QNN graph 的构建和记录。

完整调用关系可以看成：

```text
_compileSubModule(...)
├── RuntimeManager(type = MNN_CONVERT_QNN)
├── rtmgr->setCache("qnn/graph0")
│   └── QnnRuntime::onSetCachePath(...)
│       └── QNNConvertor::OutputDir = "qnn/graph0"
├── Module::load(...)
└── m->onForward(io.inputs)
    └── StaticModule::onForward(...)
        ├── Variable::compute(inputs)
        ├── StaticModule::_resize(inputs)
        │   └── Session::resize()
        │       ├── Pipeline::encode(...)
        │       │   └── 把 ScheduleInfo 整理成 command buffer
        │       └── Pipeline::allocMemory(...)
        │           ├── QnnBackend::onResizeBegin()
        │           │   └── graphCreate(...)
        │           │       └── QNNConvertor::RecordBegin(...)
        │           ├── Execution::onResize(...)
        │           │   └── QNNCommonExecution::onEncode(...)
        │           │       ├── tensorCreateGraphTensor(...)
        │           │       │   └── QNNConvertor::RecordTensor(...)
        │           │       └── graphAddNode(...)
        │           │           └── QNNConvertor::RecordNode(...)
        │           └── QnnBackend::onResizeEnd()
        │               └── graphFinalize(...)
        │                   └── QNNConvertor::RecordEnd()
        └── StaticModule::_execute()
            └── Session::run()
                └── QNN convert 模式下 graphExecute(...) 不走真实 QNN 执行
```

创建 `RuntimeManager` 时，`config.type = gNPUType`。QNN 导出场景下，这个类型就是 `MNN_CONVERT_QNN`；`setCache(path)` 传入的是当前子图 graph 源目录，例如 `qnn/graph0`：

```cpp
// tools/cpp/compilefornpu.cpp:470
MNN::ScheduleConfig config;
config.type = gNPUType;
std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(
    MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
rtmgr->setExternalFile((srcpath + ".weight").c_str());
rtmgr->setCache(path.c_str());
```

`setCache(path)` 最终会进入 `QnnRuntime::onSetCachePath(...)`，并把这个路径记录到 `QNNConvertor::OutputDir`。后续所有 `RecordTensor(...)` / `RecordNode(...)` 写文件时都会落到这个目录：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1726
bool QnnRuntime::onSetCachePath(const char* path, int mode) {
#ifdef ENABLE_QNN_CONVERT_MODE
    MNN_ASSERT(path != nullptr);
    QNNConvertor::OutputDir = std::string(path);
    MNNCreateDir(path);
#endif
    return true;
}
```

`onForward(io.inputs)` 进入 `StaticModule` 后，先根据 `io.inputs` 里的真实输入张量更新子图输入 shape。`_resize(...)` 里会把外部 `VARP` 对应的 tensor 形状同步到 `mInputTensors`，一旦 shape 变化就标记 `Session` 需要 resize：

```cpp
// express/module/StaticModule.cpp:530
if (_resizeTensor(mInputTensors[i], inputTensor, mSession.get(), nullptr)) {
    mSession->setNeedResize();
}
...
code = mSession->resize();
```

所以 `testdir/1` 和 `testdir/<chunk_size>` 会在这里进入 `Session::resize()`，分别生成 decode / prefill 两份 graph 材料。

`Session::resize()` 接着会先调用 `encode`，再调用 `allocMemory`：

```cpp
// source/core/Session.cpp:285
if (mNeedResize) {
    bool debug = mCallBackMode == Interpreter::Session_Debug;
    for (auto& iter : mPipelines) {
        auto error = iter->encode(debug, permitCodegen);
        ...
    }
    mNeedResize = false;
    mNeedMalloc = true;
    firstMalloc = true;
}
if (mNeedMalloc) {
    for (auto& iter : mPipelines) {
        auto error = iter->allocMemory(firstMalloc, forbidReplace);
        ...
    }
}
```

`Pipeline::encode(...)` 的作用是把 `ScheduleInfo` 里的 op / tensor 依赖整理成 MNN 内部的 `command buffer`。后续 `allocMemory(...)` 会基于这些 command 创建后端算子 `Execution`，并调用每个 `Execution` 的 `onResize(...)`：

```cpp
// source/core/Pipeline.cpp:537
static ErrorCode _createExecutions(
    Schedule::PipelineInfo& mInfo,
    const std::string& externalFile,
    std::vector<std::shared_ptr<BufferStorage>>& extraStorage) {
    ...
    iter.execution.reset(OpCommonUtils::createExecutionWithExternal(
        mBackend.get(), iter.inputs, iter.outputs, iter.op, &loader, tmpStorage));
}

// source/core/Pipeline.cpp:962
mBackend->onResizeBegin();
...
auto code = iter.execution->onResize(iter.workInputs, iter.workOutputs);
...
auto code = mBackend->onResizeEnd();
```

在 QNN 后端下，这些 `Execution` 通常就是对应的 QNN 执行器，例如 `QNNConvolution`。后续会在 `onResize()` 里把它们编码成 QNN graph 节点。

`onResize()` 前后还会配合两个 backend 生命周期回调：`QnnBackend::onResizeBegin()` 创建 QNN graph；`QnnBackend::onResizeEnd()` finalize graph：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1141
void QnnBackend::onResizeBegin() {
    clean();
    createContextAndGraph();
    return;
}

// source/backend/qnn/backend/QNNBackend.cpp:1147
ErrorCode QnnBackend::onResizeEnd() {
    buildOutputCast();
    buildOutputDequant();
    finalizeGraph();
    ...
}
```

QNN convert 模式的“记录动作”是通过 QNN API 函数表替换实现的。`QNN_INTERFACE_VER_TYPE` 本身是一张函数表，`QnnBackend` 后续调用 `graphCreate(...)`、`graphAddNode(...)`、`tensorCreateGraphTensor(...)` 时，都是从 `mQnnInterface` 这张表里取函数。

在普通 QNN 推理模式下，`createQnnContext()` 会从真实 QNN SDK 里拿 interface provider；在 `ENABLE_QNN_CONVERT_MODE` 下，`qnnInterface` 会被设置成 MNN 自己定义的 `gQnnConvertorInterface`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:43
#ifndef ENABLE_QNN_CONVERT_MODE
{
    QnnInterface_t** interfaceProviders = nullptr;
    uint32_t numProviders = 0;
    if (QNN::QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
        ...
    }
    ...
    qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
}
#else
    qnnInterface = QNN::gQnnConvertorInterface;
#endif
```

`QnnRuntime::create(...)` 会把 `gContext.interface` 传进 `QnnRuntime`，构造函数里再保存到 `mQnnInterface`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1711
QnnRuntime* QnnRuntime::create(const Backend::Info& info) {
    if (QNN::gContext.deviceHandle == nullptr){
        QNN::createQnnContext();
    }
    return new QnnRuntime(info, gContext.interface, gContext.logHandle, gContext.backendHandle, gContext.deviceHandle);
}

// source/backend/qnn/backend/QNNBackend.cpp:1635
QnnRuntime::QnnRuntime(
    const Backend::Info& info,
    QNN_INTERFACE_VER_TYPE qnnInterface,
    Qnn_LogHandle_t qnnLogHandle,
    Qnn_BackendHandle_t qnnBackendHandle,
    Qnn_DeviceHandle_t qnnDeviceHandle) {
    ...
    mQnnInterface = qnnInterface;
}
```

所以 `QnnBackend` 里的调用代码不需要区分“真实 SDK”还是“离线导出”。它仍然写成：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1489
CALL_QNN(mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig));

// source/backend/qnn/backend/QNNBackend.cpp:1532
CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, staticTensor));
```

区别只在 `mQnnInterface` 指向哪张函数表。`gQnnConvertorInterface` 把这些 QNN API 槽位填成了 convertor 函数：

```cpp
// source/backend/qnn/convertor/QNNConvertorInterface.cpp:96
QNN_INTERFACE_VER_TYPE gQnnConvertorInterface = {
    ...
    /* graphCreate */                            QnnConvertorGraph_Create,
    /* graphAddNode */                           QnnConvertorGraph_AddNode,
    /* graphFinalize */                          QnnConvertorGraph_Finalize,
    ...
    /* tensorCreateGraphTensor */                QnnConvertorTensor_CreateGraphTensor,
    ...
};
```

真实 QNN SDK 返回的 interface 和 `gQnnConvertorInterface` 是同一个类型：`QNN_INTERFACE_VER_TYPE`。这个类型本质上是一组函数指针槽位，只要每个槽位填进去的函数签名匹配，后续调用方就不需要知道这个函数来自真实 QNN SDK，还是来自 MNN 的 convertor。

可以用一个简化版结构理解：

```cpp
struct QnnInterfaceLike {
    Qnn_ErrorHandle_t (*graphCreate)(...);
    Qnn_ErrorHandle_t (*graphAddNode)(Qnn_GraphHandle_t graphHandle, Qnn_OpConfig_t opConfig);
    Qnn_ErrorHandle_t (*graphFinalize)(...);
    Qnn_ErrorHandle_t (*tensorCreateGraphTensor)(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor);
};
```

真实 QNN SDK 会提供一张类似这样的表：

```cpp
QnnInterfaceLike realQnnInterface = {
    QnnGraph_Create,
    QnnGraph_AddNode,
    QnnGraph_Finalize,
    QnnTensor_CreateGraphTensor,
};
```

而 MNN convert 模式自己构造了一张同类型的表：

```cpp
QnnInterfaceLike convertInterface = {
    QnnConvertorGraph_Create,
    QnnConvertorGraph_AddNode,
    QnnConvertorGraph_Finalize,
    QnnConvertorTensor_CreateGraphTensor,
};
```

`QnnBackend` 后续只持有 `mQnnInterface`，调用时等价于：

```cpp
auto fn = mRuntime->mQnnInterface.graphAddNode;
fn(mQnnGraphHandle, opConfig);
```

如果 `mQnnInterface` 保存的是 `gQnnConvertorInterface`，这里的 `fn` 就是 `QnnConvertorGraph_AddNode`，调用链等价于：

```text
mRuntime->mQnnInterface.graphAddNode(...)
    == gQnnConvertorInterface.graphAddNode(...)
    == QnnConvertorGraph_AddNode(...)
    -> QNNConvertor::RecordNode(opConfig)
```

这些 convertor 函数不会真正调用 QNN SDK，而是转成 `QNNConvertor::Record*`：

```cpp
// source/backend/qnn/convertor/QNNConvertorInterface.cpp:58
Qnn_ErrorHandle_t QnnConvertorGraph_Create(...) {
    *graphHandle = NOTNULL;
    QNNConvertor::RecordBegin(graphName);
    return QNN_SUCCESS;
}

// source/backend/qnn/convertor/QNNConvertorInterface.cpp:76
Qnn_ErrorHandle_t QnnConvertorGraph_AddNode(Qnn_GraphHandle_t graphHandle, Qnn_OpConfig_t opConfig) {
    QNNConvertor::RecordNode(opConfig);
    return QNN_SUCCESS;
}

// source/backend/qnn/convertor/QNNConvertorInterface.cpp:90
Qnn_ErrorHandle_t QnnConvertorTensor_CreateGraphTensor(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor) {
    QNNConvertor::RecordTensor(tensor);
    return QNN_SUCCESS;
}
```

QNN convert 模式复用了原来的 backend 建图路径，同时把 QNN SDK 函数表换成了 convertor 函数表。

每个 QNN `Execution` 的 `onResize(...)` 会继续调用自己的 `onEncode(...)`，把当前 MNN op 翻译成 QNN tensor 和 QNN node。公共入口在 `QNNCommonExecution`：

```cpp
// source/backend/qnn/execution/QNNCommonExecution.cpp:19
ErrorCode QNNCommonExecution::onResize(
    const std::vector<Tensor *> &inputs,
    const std::vector<Tensor *> &outputs) {
    this->setNodeName(mOp, inputs, outputs);
    ErrorCode result = this->onEncode(inputs, outputs);
    ...
    return NO_ERROR;
}
```

例如卷积 op 会通过注册器创建 `QNNConvolution`：

```cpp
// source/backend/qnn/execution/QNNConvolution.cpp:748
class QNNConvolutionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(
        const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs,
        const MNN::Op* op,
        Backend* backend) const override {
        ...
        return new QNNConvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConvolutionCreator, OpType_Convolution)
```

在具体 `onEncode(...)` 里，权重、bias、临时 tensor 会通过 `createStaticTensor(...)` / `createStaticFloatTensor(...)` 注册成 QNN tensor，算子本身会通过 `addNodeToGraph(...)` 注册成 QNN node：

```cpp
// source/backend/qnn/execution/QNNCommonExecution.cpp:68
void QNNCommonExecution::createStaticTensor(
    const std::string & name,
    Qnn_DataType_t dataType,
    const std::vector<uint32_t> & dimensions,
    const void * buffer,
    Qnn_QuantizeParams_t quantizeParam) {
    std::string tensorName = mNodeName + "_" + name;
    std::shared_ptr<QNNTensorWrapper> tensorWrapper =
        QNNTensorWrapper::createStaticTensor(tensorName, dataType, dimensions, buffer, quantizeParam);
    mBackend->addStaticTensorToGraph(tensorWrapper->getNativeTensor());
    ...
}

// source/backend/qnn/execution/QNNCommonExecution.cpp:167
mBackend->addNodeToGraph(
    mOpConfigVersion,
    mNodeName.c_str(),
    mPackageName.c_str(),
    mNodeType.c_str(),
    mParams,
    mInputs,
    mOutputs);
```

这些调用最终进入 `QnnBackend`，再通过 `mQnnInterface` 转到 convertor 函数表里的 `tensorCreateGraphTensor(...)` 和 `graphAddNode(...)`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1472
void QnnBackend::addNodeToGraph(
    Qnn_OpConfigVersion_t version,
    const char* nodeName,
    const char* packageName,
    const char* nodeType,
    std::vector<Qnn_Param_t> & params,
    std::vector<Qnn_Tensor_t> & inputs,
    std::vector<Qnn_Tensor_t> & outputs) {
    ...
    CALL_QNN(mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig));
}

// source/backend/qnn/backend/QNNBackend.cpp:1530
void QnnBackend::addStaticTensorToGraph(Qnn_Tensor_t * staticTensor) {
    MNN_ASSERT(staticTensor->v1.type == QNN_TENSOR_TYPE_STATIC);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, staticTensor));
}
```

`QNNConvertor` 的 `RecordBegin(...)` / `RecordTensor(...)` / `RecordNode(...)` / `RecordEnd(...)` 会把这些调用整理成内部 `QNNCommand`。每个 `QNNCommand` 再交给 `Translate(cmd)`，写进当前 graph 的 `.cpp`：

```cpp
// source/backend/qnn/convertor/QNNConvertor.cpp:57
void QNNConvertor::RecordBegin(const char* graphName) {
    MNN_ASSERT(!(QNNConvertor::OutputDir.empty()));
    QNNTranslator::GraphNameSymbol = GetLastDirName(QNNConvertor::OutputDir);
    ...
    std::string cppFilePath = MNNFilePathConcat(
        QNNConvertor::OutputDir,
        QNNTranslator::GraphNameSymbol + ".cpp");
    QNNConvertor::CppFilePointer = std::fopen(cppFilePath.c_str(), "w");
}

// source/backend/qnn/convertor/QNNConvertor.cpp:79
void QNNConvertor::RecordTensor(const Qnn_Tensor_t * tensor) {
    ...
    QNNConvertor::Translate(cmd);
    if (cmd.commandTensor.type == Qnn_Convertor_Tensor_t::TENSOR_STATIC) {
        QNNConvertor::DumpBuffer(
            cmd.commandTensor.name,
            cmd.commandTensor.clientBuf.data,
            cmd.commandTensor.clientBuf.dataSize);
    }
}

// source/backend/qnn/convertor/QNNConvertor.cpp:124
void QNNConvertor::RecordNode(const Qnn_OpConfig_t & opConfig) {
    ...
    QNNConvertor::Translate(cmd);
}
```

每个记录函数都包括`Translate(cmd)`方法，其本身分两步：

1. 调用 `QNNTranslator::TranslateCommand(cmd)`，把 `QNNCommand` 翻译成一组 C++ 源码行；
2. 把这些源码行追加到 `CppBuffer`，再通过 `fwrite(...)` 写入 `CppFilePointer`。

```cpp
// source/backend/qnn/convertor/QNNConvertor.cpp:159
void QNNConvertor::Translate(const QNNCommand & cmd) {
    std::vector<std::string> cppLines = QNNTranslator::TranslateCommand(cmd);
    for (const std::string& line : cppLines) {
        QNNConvertor::CppBuffer.append(line);
        QNNConvertor::CppBuffer.push_back('\n');
    }
    size_t written = std::fwrite(
        QNNConvertor::CppBuffer.data(),
        1,
        QNNConvertor::CppBuffer.size(),
        QNNConvertor::CppFilePointer);
    ...
    QNNConvertor::CppBuffer.clear();
}
```

`TranslateCommand(...)` 按命令类型分发：

```cpp
// source/backend/qnn/convertor/QNNConvertor.cpp:195
std::vector<std::string> QNNTranslator::TranslateCommand(const QNNCommand & cmd) {
    switch (cmd.type) {
        case QNNCommandTypeBegin:
            return QNNTranslator::TranslateBegin();
        case QNNCommandTypeTensor:
            return QNNTranslator::TranslateTensor(cmd.commandTensor);
        case QNNCommandTypeNode:
            return QNNTranslator::TranslateNode(cmd.commandNode);
        case QNNCommandTypeEnd:
            return QNNTranslator::TranslateEnd();
        ...
    }
}
```

这几类命令对应的输出内容是：

| 命令 | 翻译函数 | 写入 `.cpp` 的内容 |
|------|----------|--------------------|
| `QNNCommandTypeBegin` | `TranslateBegin()` | `#include`、`QnnModel_composeGraphs(...)` 函数头、`QnnModel graph*` 初始化 |
| `QNNCommandTypeTensor` | `TranslateTensor(...)` | `Qnn_Tensor_t` 定义、维度、数据类型、量化参数、`addTensor(...)` |
| `QNNCommandTypeNode` | `TranslateNode(...)` | node 的 params / inputs / outputs 数组，以及 `graph.addNode(...)` |
| `QNNCommandTypeEnd` | `TranslateEnd()` | `getGraphInfoFromModels(...)`、`QnnModel_freeGraphsInfo(...)` 和函数收尾 |

以 tensor 和 node 为例，`TranslateTensor(...)` 会生成 `Qnn_Tensor_t` 并在需要时写入 `addTensor(...)`：

```cpp
// source/backend/qnn/convertor/QNNConvertor.cpp:254
std::vector<std::string> QNNTranslator::TranslateTensor(const QNNCommandTensor& cmdT) {
    ...
    result.push_back("  Qnn_Tensor_t " + tensorNameSymbol + " = QNN_TENSOR_INIT;");
    ...
    result.push_back("  " + tensorNameSymbol + ".v1.name = \"" + sName +"\";");
    result.push_back("  " + tensorNameSymbol + ".v1.type = " + QNNTranslator::MapTensorType(cmdT.type) + ";");
    result.push_back("  " + tensorNameSymbol + ".v1.dataType = " + QNNTranslator::MapDataType(cmdT.dataType) + ";");
    ...
    if (shouldBeAdded) {
        result.push_back("  VALIDATE(" + QNNTranslator::GraphNameSymbol + ".addTensor(\"" + sName + "\", " + tensorNameSymbol + "), err);");
    }
    return result;
}
```

`TranslateNode(...)` 会生成 `params_*`、`inputs_*`、`outputs_*`，再写一行 `addNode(...)`：

```cpp
// source/backend/qnn/convertor/QNNConvertor.cpp:303
std::vector<std::string> QNNTranslator::TranslateNode(const QNNCommandNode& cmdN) {
    ...
    APPEND_VECTOR(result, QNNTranslator::TranslateNodeParamArray(...));
    APPEND_VECTOR(result, QNNTranslator::TranslateNodeInputArray(...));
    APPEND_VECTOR(result, QNNTranslator::TranslateNodeOutputArray(...));
    result.push_back("  VALIDATE(" + QNNTranslator::GraphNameSymbol + ".addNode(QNN_OPCONFIG_VERSION_1, \"" + sName + "\", \"" + std::string(cmdN.packageName) + "\", \"" + std::string(cmdN.typeName) + "\",");
    ...
    return result;
}
```

`.raw` 文件不是 `Translate(cmd)` 写的，而是 `RecordTensor(...)` 遇到 `TENSOR_STATIC` 时额外调用 `DumpBuffer(...)` 写出来：

```cpp
// source/backend/qnn/convertor/QNNConvertor.cpp:117
if (cmd.commandTensor.type == Qnn_Convertor_Tensor_t::TENSOR_STATIC) {
    QNNConvertor::DumpBuffer(
        cmd.commandTensor.name,
        cmd.commandTensor.clientBuf.data,
        cmd.commandTensor.clientBuf.dataSize);
}

// source/backend/qnn/convertor/QNNConvertor.cpp:173
void QNNConvertor::DumpBuffer(const char * name, const void * buffer, size_t size) {
    std::string dataPath = MNNFilePathConcat(QNNConvertor::OutputDir, std::string(name) + ".raw");
    FILE* fp = std::fopen(dataPath.c_str(), "wb");
    ...
    std::fwrite(buffer, 1, size, fp);
}
```

所以 Step2 生成的 graph 源目录可以这样理解：

- `graph*.cpp`：`RecordBegin/RecordTensor/RecordNode/RecordEnd -> Translate(cmd)` 逐段写出来的 QNN model 源码；
- `*.raw`：`RecordTensor(...)` 遇到 static tensor 时，把常量 buffer 单独 dump 出来的二进制数据。

因此这一句代码：

```cpp
// tools/cpp/compilefornpu.cpp:479
auto predict = m->onForward(io.inputs);
```

会让 QNN 后端看到具体输入 / 输出 tensor 形状、常量权重和每个 op 的参数，并把这些信息记录成 `qnn/graph*/graph*.cpp` 和 `qnn/graph*/*.raw`。

### 3.8 合并多 shape

因为 `testdir` 有 decode 和 prefill 两组输入，main 会为每组输入都生成一个 `dstNet`：

```cpp
// tools/cpp/compilefornpu.cpp:1040
for (int inputIndex=0; inputIndex < inputs.size(); ++inputIndex) {
    ...
    allNets.emplace_back(std::move(dstNet));
}
```

第一组 shape 生成 `graph0`，第二组 shape 会生成类似 `graph1_0` 的 graph 名：

```cpp
// tools/cpp/compilefornpu.cpp:1094
if (inputIndex == 0) {
    srcPath = gCacheDir + "/" + gGraphName +  std::to_string(npuIndex);
    graphicName = gGraphName +  std::to_string(npuIndex);
} else {
    srcPath = gCacheDir + "/" + gGraphName + std::to_string(inputIndex) + "_" +  std::to_string(npuIndex);
    graphicName = gGraphName + std::to_string(inputIndex) + "_" +  std::to_string(npuIndex);
}
```

最后 `_fuse(...)` 会把后续 shape 的 Plugin attr 合并回第一组 `dstNet`：

```cpp
// tools/cpp/compilefornpu.cpp:1145
auto dstNet = allNets[0].get();
for (int i=1; i<allNets.size(); ++i) {
    _fuse(dstNet, allNets[i].get());
    allNets[i].reset();
}
```

`_fuse(...)` 会跳过 `inputs` / `outputs`，只追加 `allGraphName`、`allInputShape`、`o_<shapeIndex>_<outputIndex>` 这类 shape 相关 attr：

```cpp
// tools/cpp/compilefornpu.cpp:690
for (auto&& srcAttr : src->attr) {
    if (srcAttr->key == "inputs" || srcAttr->key == "outputs") {
        continue;
    }
    auto dstIter = dstKeys.find(srcAttr->key);
    if (dstIter == dstKeys.end()) {
        dst->attr.emplace_back(std::move(srcAttr));
        continue;
    }
    if (dstIter->second->list != nullptr && srcAttr->list != nullptr) {
        dstIter->second->list->s.insert(dstIter->second->list->s.end(), srcAttr->list->s.begin(), srcAttr->list->s.end());
        dstIter->second->list->i.insert(dstIter->second->list->i.end(), srcAttr->list->i.begin(), srcAttr->list->i.end());
    }
}
```

这一步结束后，`qnn/llm.mnn` 里的同一个 `Plugin` 会同时保存 decode / prefill 两组 graph：

```text
Plugin(type="QNN")
├── path: qnn/graph0.bin
├── allGraphName: [graph0, graph1_0, ...]
├── allInputShape: [decode 输入 shape..., prefill 输入 shape...]
└── o_0_0 / o_1_0 / ...: 不同 shape 下的输出描述
```

### 3.9 产物

Step2 结束时，真正落盘的产物有三类：

```text
tmp/
├── qnn/
│   ├── llm.mnn                 # 插入 Plugin(type="QNN") 后的主图
│   ├── graph0/                 # 第一组 shape 的 QNN graph 源目录
│   │   ├── graph0.cpp
│   │   └── *.raw
│   └── graph1_0/               # 第二组 shape 的 QNN graph 源目录
│       ├── graph1_0.cpp
│       └── *.raw
└── npu_postreat.json           # Step3 合成 context binary 的 merge 配置
```
## 4. npu_postreat.json 的作用

`compilefornpu` 还会写 `npu_postreat.json`，里面最重要的是 `merge`：

```json
{
  "type": "QNN",
  "merge": {
    "qnn/graph0.bin": [
      "qnn/graph0",
      "qnn/graph1_0"
    ]
  },
  "cache": "qnn"
}
```

`merge` 的含义是：

```text
最终生成的 QNN binary -> 参与合并的 graph 源目录列表
```

一个 `Plugin` 可能对应多组 shape。离线阶段先为每个 shape 生成 graph 源目录，再通过 `merge` 合成一个 context binary。运行时通过 `shapeIndex` 选择 `.bin` 里的具体 graph handle。

## 5. Step3：生成 context binary

`convert(args)` 的 Step3 调用 `compile_qnn(args)`：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:81
print("Step3: Compile to QNN")
compile_qnn(args)
```

`compile_qnn(args)` 会从 `args.mnn_path` 反推出源码树里的 `source/backend/qnn/npu_convert.py` 路径，并在 `tmp` 目录下执行它：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:39
def compile_qnn(args):
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    process = subprocess.Popen(
        "python3 " + exe + ' npu_postreat.json %d ' %args.soc_id + ' ' + args.dsp_arch,
        cwd = cache,
        shell=True
    )
```

命令展开后是：

```bash
python3 <repo>/source/backend/qnn/npu_convert.py \
  npu_postreat.json \
  <soc_id> \
  <dsp_arch>
```

到 Step3 时，`compilefornpu` 已经把 MNN 子图导出成 QNN graph 的源码材料。`npu_convert.py` 会继续调用 QNN SDK，把这些材料编译成运行时可直接加载的 HTP context binary。

整体调用关系是：

```text
npu_postreat.json
└── merge: qnn/graph0.bin -> [qnn/graph0, qnn/graph1_0, ...]  # 一个 Plugin 的多组 shape graph
    ├── for each graph 源目录
    │   ├── tar *.raw -> graph*.bin                            # 打包 graph 依赖的 raw 常量数据
    │   ├── qnn-model-lib-generator
    │   │   ├── -c graph*.cpp                                  # QNN graph C++ 源码
    │   │   ├── -b graph*.bin                                  # 上一步打包的 raw 数据
    │   │   └── -> x86_64-linux-clang/libgraph*.so             # 生成 QNN model library
    │   └── 记录 graph name 和 lib 路径
    │
    └── qnn-context-binary-generator
        ├── --model libgraph0.so,libgraph1_0.so,...             # 多个 shape graph 一起写入 context
        ├── --backend <QNN_SDK_ROOT>/lib/.../libQnnHtp.so        # 目标 HTP backend
        ├── --config_file ./context_config.json                 # 指向 HTP backend extension 配置
        ├── --binary_file graph0
        └── --output_dir qnn
            └── qnn/graph0.bin                                  # 最终 Plugin 加载的 context binary
```

### 5.1 读取后处理配置

`npu_convert.py` 一开始从环境变量和命令参数读取输入：

```python
# source/backend/qnn/npu_convert.py:8
qnn_sdk = os.environ["QNN_SDK_ROOT"]
with open(sys.argv[1]) as f:
    post_treat = json.load(f)
soc_id = int(sys.argv[2])
dsp_arch = sys.argv[3]
```

`sys.argv[1]` 就是 `npu_postreat.json`。`soc_id` 和 `dsp_arch` 来自导出命令参数，用来写入 HTP backend 的设备配置。

随后脚本定位 QNN SDK 的两个工具，并取出 `merge`：

```python
# source/backend/qnn/npu_convert.py:15
qnn_bin_path = os.path.join(qnn_sdk, 'bin', 'x86_64-linux-clang')
qnnModelLibGenerator = os.path.join(qnn_bin_path, 'qnn-model-lib-generator')
qnnContextBinaryGenerator = os.path.join(qnn_bin_path, 'qnn-context-binary-generator')
merges = post_treat["merge"]
```

`merge` 的 key 是最终要生成的 context binary 路径，例如 `qnn/graph0.bin`；value 是这个 binary 内要合并的 graph 源目录，例如 `qnn/graph0` 和 `qnn/graph1_0`。

### 5.2 写 QNN context 配置

脚本会生成两个配置文件。

第一个是 `context_config.json`，它只负责告诉 `qnn-context-binary-generator` 去哪里加载 HTP backend extension：

```python
# source/backend/qnn/npu_convert.py:23
context_config = {
    "backend_extensions": {
        "shared_library_path": os.path.join(qnn_sdk, "lib","x86_64-linux-clang","libQnnHtpNetRunExtensions.so"),
        "config_file_path": "./htp_backend_extensions.json"
    }
}
htp_so = os.path.join(qnn_sdk, 'lib','x86_64-linux-clang','libQnnHtp.so')
```

第二个是 `htp_backend_extensions.json` 的主体配置：

```python
# source/backend/qnn/npu_convert.py:33
htp_backend_extensions = {
    "graphs": [
        {
            "vtcm_mb": 8,
            "O": 3.0,
            "fp16_relaxed_precision": 1,
            "hvx_threads": 4
        }
    ],
    "devices": [
        {
            "soc_id": soc_id,
            "dsp_arch": dsp_arch,
            "cores": [
                {
                    "core_id": 0,
                    "perf_profile": "burst",
                    "rpc_control_latency": 100
                }
            ]
        }
    ],
    "context": {
        "weight_sharing_enabled": True
    }
}
```

这些配置会进入 QNN context binary 生成过程。`graphs` 里是 HTP graph 编译参数；`devices` 里绑定目标 SoC 和 DSP 架构；`weight_sharing_enabled` 表示同一个 context 里的多个 graph 可以共享权重。

### 5.3 生成每个 graph 的 model library

这里直接消费 Step2 落到 `tmp/qnn/graph*/` 下的文件。以 `tmp/qnn/graph0/` 为例，Step3 不是重新读取原始 `llm.mnn` 做切图，而是把这个目录当作一个 QNN graph 的编译工作目录：

| 文件 | 来源 | Step3 中的用途 |
|------|------|----------------|
| `graph0.cpp` | Step2 的 QNN convert backend 生成 | 作为 `qnn-model-lib-generator -c` 的输入，里面是创建 tensor、node、graph 的 QNN C++ 源码 |
| `*.raw` | Step2 记录 static tensor 时 dump 出来 | 先在同目录打包成 `graph0.bin`，再作为 `qnn-model-lib-generator -b` 的输入 |
| `graph0.bin` | Step3 临时打包生成 | 只是 model library 编译阶段使用的 raw 数据包，不是最终运行时加载的 context binary |
| `x86_64-linux-clang/libgraph0.so` | Step3 编译生成 | 作为 `qnn-context-binary-generator --model` 的输入 |

这里容易混淆的是两个 `.bin`。`tmp/qnn/graph0/graph0.bin` 是 Step3 临时生成的 raw 打包文件；`tmp/qnn/graph0.bin` 才是最终给 Plugin attr `path = qnn/graph0.bin` 使用的 QNN context binary。前者位于 graph 源目录内部，后者位于 `qnn/` 目录下。

接下来脚本遍历 `post_treat["merge"]`。每个 `key` 对应一个最终 `.bin`，每个 `src` 对应一个 graph 源目录：

```python
# source/backend/qnn/npu_convert.py:60
for key in post_treat["merge"]:
    srcs = merges[key]
    dst = key
    dstname = key.split('/')
    dstname = dstname[len(dstname)-1]
    dstname = dstname.replace('.bin', '')
```

以 `qnn/graph0.bin -> [qnn/graph0, qnn/graph1_0]` 为例，`dstname` 会变成 `graph0`。这个名字后续会作为 `qnn-context-binary-generator --binary_file graph0` 的参数。

然后对每个 graph 源目录打包 `.raw`：

```python
# source/backend/qnn/npu_convert.py:69
for i,src in enumerate(srcs):
    graphname = src.split('/')
    graphname = graphname[len(graphname)-1]
    graphs.append(graphname)
    workdir = os.path.join(os.getcwd(), src)
    workdirs.append(workdir)
    print(subprocess.run("tar -cf " + graphname + '.bin' + ' *.raw', cwd=workdir, capture_output=True, text=True, shell=True))
```

`graphname` 是 `graph0`、`graph1_0` 这样的名字。`workdir` 目录里已经有 Step2 导出的 `graphname.cpp` 和若干 `.raw` 文件，`.raw` 是 QNN graph 依赖的常量数据。脚本先把这些 `.raw` 打包成 `graphname.bin`，给下一步的 `qnn-model-lib-generator` 使用。

随后调用 `qnn-model-lib-generator`：

```python
# source/backend/qnn/npu_convert.py:80
compile_cmd = 'python3 ' + qnnModelLibGenerator + ' -c ' + os.path.join(workdir, graphname + '.cpp') + ' -b ' + os.path.join(workdir, graphname + '.bin') + ' -t x86_64-linux-clang -o ' + workdir
print(os.popen(compile_cmd).read())
libs.append(os.path.join(workdir, 'x86_64-linux-clang', 'lib' + graphname + '.so'))
```

这一步的输入是：

- `graphname.cpp`：Step2 由 QNN convert backend 导出的 graph 构建代码；
- `graphname.bin`：刚才打包出来的 raw 常量数据；
- `x86_64-linux-clang`：生成 host 侧工具可加载的 model library。

输出是 `workdir/x86_64-linux-clang/libgraphname.so`。这个 `.so` 还不是运行时最终加载的产物，它只是 `qnn-context-binary-generator` 的输入。

### 5.4 合成 context binary

同一个 `key` 下的所有 graph 源目录都编译出 `libgraph*.so` 后，脚本会把 graph 名字写入 `htp_backend_extensions.json`：

```python
# source/backend/qnn/npu_convert.py:85
htp_backend_extensions['graphs'][0]['graph_names'] = graphs
with open('htp_backend_extensions.json', 'w') as f:
    f.write(json.dumps(htp_backend_extensions, indent=4))
```

`graphs` 顺序很重要。`compilefornpu` 写入 Plugin attr 时，也会保存同一组 `allGraphName`：

```cpp
// tools/cpp/compilefornpu.cpp:556
attr->key = "allGraphName";
attr->list.reset(new ListValueT);
attr->list->s = {graphicName};
```

多 shape 合并时，`_fuse(...)` 会把后续 shape 的 `allGraphName` 追加进去。运行时 QNN Plugin 会读取 `allGraphName`，并按这个顺序重新排列 context binary 里的 graph 信息：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:913
std::vector<std::string> allGraphName;
auto allGraphNameAttr = ctx->getAttr("allGraphName");
if (allGraphNameAttr && allGraphNameAttr->list() && allGraphNameAttr->list()->s()) {
    auto graphNames = allGraphNameAttr->list()->s();
    for (int i = 0; i < graphNames->size(); ++i) {
        allGraphName.push_back(graphNames->GetAsString(i)->str());
    }
}
```

最后脚本把所有 `libgraph*.so` 拼成逗号分隔的 `--model` 参数，调用 `qnn-context-binary-generator`：

```python
# source/backend/qnn/npu_convert.py:88
libsStr = ""
for i in range(0, len(libs)):
    if i > 0:
        libsStr+=','
    libsStr += libs[i]
print(os.popen(qnnContextBinaryGenerator + ' --model ' + libsStr + ' --backend '+ htp_so + ' --binary_file ' + dstname + ' --config_file ./context_config.json ' + ' --output_dir ' + cache_dir).read())
```

展开后大致是：

```bash
qnn-context-binary-generator \
  --model <tmp>/qnn/graph0/x86_64-linux-clang/libgraph0.so,<tmp>/qnn/graph1_0/x86_64-linux-clang/libgraph1_0.so \
  --backend <QNN_SDK_ROOT>/lib/x86_64-linux-clang/libQnnHtp.so \
  --binary_file graph0 \
  --config_file ./context_config.json \
  --output_dir qnn
```

输出目录由 `npu_postreat.json` 里的 `"cache": "qnn"` 决定：

```python
# source/backend/qnn/npu_convert.py:19
cache_dir = 'res'
if 'cache' in post_treat:
    cache_dir = post_treat['cache']
```

最终产物会写到：

```text
tmp/qnn/graph0.bin
```

这就是 Plugin attr 里的 `path: qnn/graph0.bin` 对应的文件。

Step3 结束后，`npu_convert.py` 会按 `clean_tmp = True` 清掉 `tmp/qnn/graph*/` 这些 graph 源目录。`context_config.json`、`htp_backend_extensions.json`、`qnn.json`、`npu_postreat.json` 和 `testdir/` 会继续留在 `tmp` 根目录，直到 Step4 最后 `shutil.rmtree(args.cache_path)` 删除整个临时目录。

此时 `tmp` 大致是：

```text
tmp/
├── testdir/
│   ├── 1/
│   └── <chunk_size>/
├── qnn.json
├── npu_postreat.json
├── context_config.json
├── htp_backend_extensions.json
└── qnn/
    ├── llm.mnn
    └── graph*.bin
```

### 5.5 运行时使用 binary

运行时 QNN Plugin 初始化时，会从 Plugin attr 里读 `path` 和 `allGraphName`，然后调用 `RawExecutorWrapper::compileModel(...)`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:911
auto path = MNNFilePathConcat(ctx->dir_path(), ctx->getAttr("path")->s()->str());
...
mRawExecutor.reset(new RawExecutorWrapper());
return mRawExecutor->compileModel(path, binaryOffset, binarySize, allGraphName);
```

`compileModel(...)` 会读取 context binary，并通过 QNN API 从 binary 里恢复 context 和 graph handle：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:797
CALL_QNN(QNN::gContext.systemInterface.systemContextGetBinaryInfo(systemContextHandle, buffer, size, &binaryInfo,&binarySize));
copyMetadataToGraphsInfo(binaryInfo, mGraphsInfo, mGraphCount);

// source/backend/qnn/backend/QNNBackend.cpp:816
CALL_QNN(QNN::gContext.interface.contextCreateFromBinary(QNN::gContext.backendHandle, QNN::gContext.deviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, mQnnProfileHandle));
```

然后按 `allGraphName` 的顺序取出每个 graph handle：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:826
for (int i = 0; i < mGraphCount; ++i) {
    auto it = graphInfoMap.find(allGraphName[i]);
    MNN_ASSERT(it != graphInfoMap.end());
    sortedGraphsInfo[i] = it->second;
}
for (int i = 0; i < mGraphCount; i++) {
    CALL_QNN(QNN::gContext.interface.graphRetrieve(mQnnContextHandle, mGraphsInfo[i]->graphName, &(mQnnGraphHandleVec[i])));
}
```

所以 Step3 生成的 `qnn/graph0.bin` 不是普通权重文件，而是一个 QNN HTP context binary。它里面可以包含多个 graph，分别对应 decode / prefill 等不同输入 shape。执行时，Plugin 通过 `shapeIndex` 选择 `mQnnGraphHandleVec[shapeIndex]` 去调用 QNN graph。

## 6. Step4：移动产物并写 config_qnn.json

最后 `convert(args)` 调用 `output_qnn(args)` 做收尾：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:85
print("Step4: Move result file to ", args.model)
output_qnn(args)
```

`output_qnn(args)` 先把临时目录里的 `qnn` 移到模型目录（`tmp/qnn -> <model_dir>/qnn`）：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:47
def output_qnn(args):
    if os.path.exists(os.path.join(args.model, 'qnn')):
        shutil.rmtree(os.path.join(args.model, 'qnn'))
    shutil.move(os.path.join(args.cache_path, 'qnn'), os.path.join(args.model, 'qnn'))
```

然后脚本直接写出一个新的 `config_qnn.json`，不是读取 `config.json` 后做增量修改：

```python
# transformers/llm/export/npu/generate_llm_qnn.py:51
config_npu = {
    "llm_model": "qnn/llm.mnn",
    "backend_type": "cpu",
    "thread_num": 1,
    "precision": "low",
    "chunk_limits":[args.chunk_size, 1],
    "memory": "low",
    "sampler_type": "penalty",
    "penalty": 1.1
}
with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
    f.write(json.dumps(config_npu, indent = 4))
shutil.rmtree(args.cache_path)
```

其中最关键的字段是：

```json
{
  "llm_model": "qnn/llm.mnn",
  "chunk_limits": [128, 1]
}
```

实际运行时还需要确认 `config_qnn.json` 里有正常的运行配置，例如：

```json
{
  "llm_model": "qnn/llm.mnn",
  "backend_type": "cpu",
  "thread_num": 1,
  "precision": "low",
  "chunk_limits": [128, 1],
  "memory": "low",
  "sampler_type": "penalty",
  "penalty": 1.1
}
```

这里 `backend_type` 写成 `cpu` 是正常的。外层 MNN 主图仍然由 CPU backend 执行；真正下沉到 QNN 的部分已经变成 `Plugin(type="QNN")`，运行时会由 Plugin 机制加载 `qnn/*.bin` 并调用 QNN graph。

## 7. 最终目录结构

完成后，模型目录会变成类似这样：

```text
Qwen3-1.7B-MNN/
├── config.json
├── config_qnn.json
├── llm_config.json
├── tokenizer.txt
├── llm.mnn
├── llm.mnn.weight
└── qnn/
    ├── llm.mnn
    └── graph*.bin
```

整个离线执行链路如下：

```text
generate_llm_qnn.py 负责编排；
generateLlmIO 负责造多 shape 样本；
compilefornpu 负责切图、导出 QNN graph 材料并改写 MNN 主图；
npu_convert.py 负责调用 QNN SDK 生成最终 HTP context binary。
```
