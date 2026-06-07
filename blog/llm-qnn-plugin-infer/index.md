---
title: "LLM QNN 插件推理流程"
date: 2026-06-04T12:00:00+08:00
lastmod: 2026-06-04T12:00:00+08:00
draft: false
description: "按当前仓库代码梳理 MNN LLM 从 response 到 QNN graphExecute 的运行时路径"
slug: "llm-qnn-plugin-infer"
tags: ["mnn", "llm", "qnn"]
categories: ["mnn"]

comments: true
math: true
---

# LLM QNN 插件推理流程

- [LLM QNN 插件推理流程](#llm-qnn-插件推理流程)
  - [1. `config_qnn.json` 配置](#1-config_qnnjson-配置)
  - [2. LLM 生成链路](#2-llm-生成链路)
  - [3. `prefill` / `decode` 形状](#3-prefill--decode-形状)
  - [4. 算子前向](#4-算子前向)
  - [5. `resize` 阶段](#5-resize-阶段)
    - [5.1 `ShapePlugin`](#51-shapeplugin)
    - [5.2 创建 `QNN` 算子](#52-创建-qnn-算子)
    - [5.3 加载 `QNN` 图](#53-加载-qnn-图)
    - [5.4 `PluginExecuteRaw::resize()`](#54-pluginexecuterawresize)
  - [6. `execute` 阶段](#6-execute-阶段)
    - [6.1 重新确认 `shapeIndex`](#61-重新确认-shapeindex)
    - [6.2 准备输入输出 `Tensor`](#62-准备输入输出-tensor)
    - [6.3 拷贝输入、执行 `QNN`、拷回输出](#63-拷贝输入执行-qnn拷回输出)
    - [6.4 `invokModel()`执行入口](#64-invokmodel执行入口)
    - [6.5 `graphExecute()` 执行点](#65-graphexecute-执行点)
  - [7. 小结](#7-小结)


本篇梳理移动设备侧运行 `QNN` 后端主图 `qnn/llm.mnn` 时的代码路径。离线编译阶段会把部分子图编译成 `QNN` 后端使用的 `qnn/*.bin` 二进制文件，并把主图改写成调用插件算子 `OpType_Plugin(type="QNN")` 的形式。推理时，核心调用链路如下：

```text
[LLM 生成层]
llm->response()
└── generate(input_ids)
    ├── embedding()
    └── forwardVec()
        └── forwardRaw()
            │
            ▼
[MNN Express Module 层]
Module::onForward()
├── [阶段 1: resize / 形状推导]
│   └── StaticModule::_resize()
│       └── ShapePlugin
│           └── PluginShapeRaw::compute()
│               └── 根据 allInputShape 选择 shapeIndex，并读取 o_<shapeIndex>_* 输出描述
│
└── [阶段 2: execute / 真正执行]
    └── StaticModule::_execute()
        └── CPUPlugin::onExecute()
            └── PluginExecuteRaw::compute()
                └── RawExecutorWrapper::invokModel()
                    └── QNN graphExecute()
```

## 1. `config_qnn.json` 配置

`QNN` 模型运行时一般使用 `config_qnn.json` 配置文件：

```json
{
  "llm_model": "qnn/llm.mnn",
  "backend_type": "cpu",
  "chunk_limits": [128, 1]
}
```

`LlmConfig::llm_model()` 会把 `llm_model` 拼到模型目录下：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:279
std::string llm_model() const {
    return base_dir_ + config_.value("llm_model", "llm.mnn");
}
```

所以，最终加载主图路径就是 `<model_dir>/qnn/llm.mnn`，也就是 `Llm::load()` 里会用这个路径加载 `Module`：

```cpp
// transformers/llm/engine/src/llm.cpp:280
std::string model_path = mConfig->llm_model();
...
mRuntimeManager->setExternalFile(mConfig->llm_weight());
mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
mRuntimeManager->setExternalFile("");
```

这里加载的 `qnn/llm.mnn` 是一个带 `QNN` 插件算子的主图。同时，`Llm::setRuntimeHint()` 会设置 `NPU` 文件目录：

```cpp
// transformers/llm/engine/src/llm.cpp:153
rtg->setExternalPath(mConfig->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
```

`npu_model_dir()` 默认返回模型的根目录：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:311
std::string npu_model_dir() const {
    return base_dir_ + config_.value("npu_model_dir", "");
}
```

这个路径会先写入 `RuntimeManager` 中：

```cpp
// transformers/llm/engine/src/llm.cpp:153
rtg->setExternalPath(mConfig->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);

// express/Executor.cpp:242
void Executor::RuntimeManager::setExternalPath(std::string path, int type) {
    if (type == MNN::Interpreter::EXTERNAL_NPU_FILE_DIR) {
        mInside->mContent->mNpuDir = path;
        return;
    }
    mInside->mContent->modes.setExternalPath(path, type);
}
```

后面 `StaticModule` 构造时会把它写到 `CPU` 后端的 `pNPUModelDirPath`：

```cpp
// express/module/StaticModule.cpp:340
bnCache.cache.first->pNPUModelDirPath = rtm->getInside()->mContent->mNpuDir;
bnCache.cache.second->pNPUModelDirPath = rtm->getInside()->mContent->mNpuDir;
```

并在 `QNN` 插件读取 `path` 属性时，使用该目录拼出完整 `.bin` 路径，例如：

```text
<model_dir>/qnn/graph0.bin
```

## 2. LLM 生成链路

文本输入最常见的入口是：

```cpp
llm->response(prompt);
```

对应源码里，`response(string)` 会先按配置套 `chat template`，再使用 `tokenizer` 分词为 `token id`，最后转到 `response(input_ids)` 方法：

```cpp
// transformers/llm/engine/src/llm.cpp:912
void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    auto prompt = user_content;
    if (mConfig->use_template()) {
        prompt = mPrompt->applyTemplate(user_content, true);
    }
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}
```

在 `response(input_ids)` 方法中，会初始化一些记录的数据，随后继续走 `generate(input_ids)` 接口生成输出：

```cpp
// transformers/llm/engine/src/llm.cpp:900
void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_ids, max_new_tokens);
}
```

在 `generate(input_ids)` 里会根据 `prompt` 长度决定是一次 `embedding`，还是按 `mBlockSize` 分块：

```cpp
// transformers/llm/engine/src/llm.cpp:786
mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end());
if(!passExecute) {
    if (0 == mBlockSize || input_ids.size() <= mBlockSize) {
        auto hidden_states = embedding(input_ids);
        return generate(hidden_states, max_tokens);
    }
    int total_size = (int)input_ids.size();
    int loop_size = UP_DIV(total_size, mBlockSize);
    for (int i = 0; i < loop_size; i++) {
        ...
        auto input_embeds = embedding(chunk_ids);
        generate(input_embeds, 0);
    }
}
generate(max_tokens);
```

## 3. `prefill` / `decode` 形状

`Llm::set_config()` 读取 `chunk_limits` 后，会排序并把最大值作为 `mBlockSize`，代码块如下：

```cpp
// transformers/llm/engine/src/llm.cpp:104
if (mConfig->config_.document.HasMember("chunk_limits")) {
    auto& size_limit = mConfig->config_.document["chunk_limits"];
    ...
    for (auto iter = size_limit.GetArray().begin(); iter != size_limit.GetArray().end(); iter++) {
        mValidBlockSize.emplace_back(iter->GetInt());
    }
    ...
    std::sort(mValidBlockSize.begin(), mValidBlockSize.end());
    mBlockSize = mValidBlockSize[mValidBlockSize.size()-1];
}
```

以 `chunk_limits: [128, 1]` 为例，内部结果是：

```text
mValidBlockSize = [1, 128]
mBlockSize = 128
```

这和离线导出阶段的两组形状对上：

| 阶段 | `seq len` | 作用 |
|------|---------|------|
| `prefill` | `128` | `prompt` 分块或 `pad` 后的大块前向 |
| `decode` | `1` | 自回归单 `token` 前向 |

`forwardVec(input_embeds)` 会按 `mBlockSize` 拆分长 `prompt`：

```cpp
// transformers/llm/engine/src/llm.cpp:572
std::vector<VARP> Llm::forwardVec(MNN::Express::VARP input_embeds) {
    int seq_len = input_embeds->getInfo()->dim[mSeqLenIndex];
    if (0 == mBlockSize) {
        mMeta->add = seq_len;
        auto attention_mask = gen_attention_mask(seq_len);
        auto position_ids = gen_position_ids(seq_len);
        auto res = forwardRaw(input_embeds, attention_mask, position_ids);
        return res;
    }
    auto blockNumber = seq_len / mBlockSize;
    auto blockRemain = seq_len % mBlockSize;
```

整块部分直接按 `mBlockSize` 调 `forwardRaw()`：

```cpp
// transformers/llm/engine/src/llm.cpp:604
for (int i=0; i<blockNumber; ++i) {
    logits.clear();
    mMeta->add = blockSize;
    auto embed = embeddings[i];
    auto attention_mask = gen_attention_mask(blockSize);
    auto position_ids = gen_position_ids(blockSize);
    logits = forwardRaw(embed, attention_mask, position_ids);
    ...
    updateContext(blockSize, 0);
}
```

剩余长度如果不是合法形状，会 `pad` 到 `mValidBlockSize` 中最近的合法大小：

```cpp
// transformers/llm/engine/src/llm.cpp:623
if (!mValidBlockSize.empty()) {
    forwardSize = mValidBlockSize[mValidBlockSize.size()-1];
    for (int j=mValidBlockSize.size()-2; j>=0; --j) {
        if (mValidBlockSize[j] < blockRemain) {
            break;
        }
        forwardSize = mValidBlockSize[j];
    }
    if (blockRemain < forwardSize) {
        hasPad = true;
        auto dim = input_embeds->getInfo()->dim;
        dim[mSeqLenIndex] = forwardSize;
        auto newEmbed = _Input(dim, NCHW);
        ::memcpy(newEmbed->writeMap<void>(), input_embeds->readMap<void>(), input_embeds->getInfo()->size * sizeof(float));
        ::memset(newEmbed->writeMap<float>() + input_embeds->getInfo()->size, 0, (newEmbed->getInfo()->size - input_embeds->getInfo()->size) * sizeof(float));
        input_embeds = newEmbed;
    }
}
```

比如剩余 `prompt` 长度是 `80`，会 `pad` 到 `128`；`decode` 阶段长度是 `1`，直接匹配 `1`。

这个形状需要匹配离线导出时 `compilefornpu` 写入插件属性的 `allInputShape`。

## 4. 算子前向

`forwardRaw()` 构造主模型输入：

```cpp
// transformers/llm/engine/src/llm.cpp:448
std::vector<Express::VARP> Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    Express::VARP logitsIndex;
    bool inDecode = mContext->gen_seq_len > 0;
    bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
    auto seqLen = hiddenState->getInfo()->dim[mSeqLenIndex];
    int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
    ...
std::vector<Express::VARP> inputs {
    hiddenState,
    mask,
    inputPos,
    logitsIndex
};
```

然后选择 `Module` 模块：

```cpp
// transformers/llm/engine/src/llm.cpp:480
std::vector<Express::VARP> inputs {hiddenState, mask, inputPos, logitsIndex};
inputs.insert(inputs.end(), extraArgs.begin(), extraArgs.end());
std::vector<Express::VARP> outputs = selectModule->onForward(inputs);
```

这里的 `selectModule` 通常有两种：

- `prefill` 走主 `mModule`；
- `decode` 会按 `(seqLenKey, isAllLogits)` 从 `mModulePool` 里取 `clone` 出来的模块。

对应代码里，默认先使用 `mModule`，在没有 `mValidBlockSize` 的路径下会按输入长度等 `key` 复制模块：

```cpp
// transformers/llm/engine/src/llm.cpp:455
auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
std::shared_ptr<Module> selectModule = mModule;
if (mValidBlockSize.empty()) {
    if(mModulePool.find(moduleKey) == mModulePool.end()) {
        MNN_PRINT("Warning: module need new clone, cloning now.\n");
        mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
        mModulePool[moduleKey].reset(Module::clone(mModule.get()));
    }
    selectModule = mModulePool[moduleKey];
}
```

进入 `Module::onForward()` 后，实际对象通常是 `StaticModule`。`StaticModule::onForward()` 的核心顺序是：

```cpp
// express/module/StaticModule.cpp:566
std::vector<Express::VARP> StaticModule::onForward(const std::vector<Express::VARP>& inputs) {
    ...
    Variable::compute(inputs);
    ...
    if (runResize) {
        code = _resize(inputs);
    }
    if (NO_ERROR == code && runCompute) {
        code = _execute();
    }
```

`_resize()` 会把输入 `VARP` 绑定到 `Session` 输入张量，并触发 `mSession->resize()`，从而向下执行 `Pipeline::encode()` 和 `Pipeline::allocMemory()`，完成创建算子 `Execution` 类的准备。对于 `QNN` 图来说，这里是在准备 `OpType_Plugin` 算子：

```cpp
// express/module/StaticModule.cpp:521
for (int i = 0; i < inputs.size(); ++i) {
    ...
    if (_resizeTensor(mInputTensors[i], inputTensor, mSession.get(), nullptr)) {
        mSession->setNeedResize();
    }
}
mSession->getInfo(Interpreter::RESIZE_STATUS, &curStatus);
code = mSession->resize();
```

最终 `_execute()` 会调用 `Session` 的 `run()` 方法，执行流水线中的算子。这里使用 `QNN` 图时，对应的是 `OpType_Plugin` 算子的执行：

```cpp
// express/module/StaticModule.cpp:550
ErrorCode StaticModule::_execute() {
    ErrorCode code;
    if (mResource->mModes.callBackMode == Interpreter::Session_Debug) {
        ...
        code = mSession->runWithCallBack(debug->before, debug->after);
    } else {
        code = mSession->run();
    }
    return code;
}
```

所以 `QNN` 插件会经历两个不同阶段：

- `resize` 阶段：走 `ShapePlugin -> PluginShapeRaw::compute()`，确定输出形状；
- `execute` 阶段：走 `CPUPlugin::onExecute() -> PluginExecuteRaw::compute()`，真正调用 `QNN` 计算图。

## 5. `resize` 阶段

上一节已经走到 `StaticModule::_resize()`：

```text
StaticModule::onForward()
└── _resize(inputs)
    └── mSession->resize()
        └── Pipeline::encode()
            ├── ShapePlugin / PluginShapeRaw::compute()
            └── CPUPluginCreator -> CPUPlugin -> PluginExecuteRaw::init()
```

这里要分清两件事：

- `ShapePlugin` 负责输出形状推导；
- `CPUPluginCreator` 负责创建真正参与后续 `execute` 的 `Execution`。

这两件事都发生在 `resize` 这条链路上，所以第 5 节先看形状推导，再看 `QNN` 内核的注册和创建。

### 5.1 `ShapePlugin`

通用插件形状入口在 `source/shape/ShapePlugin.cpp`。它会从 `Plugin` 参数里读出 `type`，再用这个 `type` 找到对应的形状推导内核：

```cpp
// source/shape/ShapePlugin.cpp:35
const Plugin* plugin_param = op->main_as<Plugin>();
std::shared_ptr<plugin::InferShapeKernel> kernel =
    getInferShapeKernel(plugin_param->type()->str());
...
plugin::InferShapeContext ctx(inputs, outputs);
for (const Attribute* attr : *(plugin_param->attr())) {
    ctx.setAttr(attr->key()->str(), attr);
}
bool status = kernel->compute(&ctx);
```

`QNN` 插件的 `type` 是 `"QNN"`。这个 `type` 对应的形状推导内核和计算内核都在 `QNNBackend.cpp` 里注册：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1837
#ifdef MNN_WITH_PLUGIN
plugin::InferShapeKernelRegister::add("QNN", []() {
    return new plugin::shape_inference::PluginShapeRaw;
});
plugin::ComputeKernelRegistry<plugin::backend::PluginExecuteRaw::KernelT>::add("QNN", []() {
    return new plugin::backend::PluginExecuteRaw;
});
#endif
```

这里注册了两条不同阶段的入口：

| 注册项 | 阶段 | 作用 |
|--------|------|------|
| `InferShapeKernelRegister::add("QNN", ...)` | `resize` / 形状推导 | 用 `allInputShape` 匹配当前输入形状，并写出输出形状 |
| `ComputeKernelRegistry::add("QNN", ...)` | `resize` / `execute` | 创建 `PluginExecuteRaw`，后面加载 `QNN` 上下文二进制并执行计算图 |


`QNN` 注册的 `InferShapeKernelRegister` 是 `PluginShapeRaw::compute()`，它会先调用 `computeIndex()`。这一步的输入是当前插件算子的输入张量形状：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:361
static bool computeIndex(PluginContext* ctx, int & index) {
    const std::vector<Tensor *> & inputs = ctx->inputs();
    auto attrAllShape = ctx->getAttr("allInputShape");
    if (nullptr == attrAllShape || nullptr == attrAllShape->list() || nullptr == attrAllShape->list()->i()) {
        MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allInputShape' attr.\n");
        return false;
    }
    int dimSum = 0;
    for (int i = 0; i < inputs.size(); i++) {
        auto inputDim = inputs[i]->dimensions();
        dimSum += inputDim;
    }
    ...
}
```

随后与离线 `compilefornpu` 写进插件属性的多组合法输入 `allInputShape` 对比，找到匹配当前输入形状的 `shapeIndex`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:378
auto indexNumber = attrAllShape->list()->i()->size() / dimSum;
for (int si=0; si<indexNumber; ++si) {
    auto dstSi = attrAllShape->list()->i()->data() + si * dimSum;
    bool valid = true;
    for (int i=0; i<inputs.size(); ++i) {
        auto inputDim = inputs[i]->dimensions();
        for (int j = 0; j < inputDim; j++) {
            if (inputs[i]->length(j) != dstSi[j]) {
                valid = false;
                break;
            }
        }
        dstSi += inputDim;
        if (!valid) {
            break;
        }
    }
    if (valid) {
        index = si;
        return true;
    }
}
```

匹配成功后得到的 `shapeIndex` 是后面 `QNN` 路径的关键索引：

```text
shapeIndex
├── resize 阶段: 读取 o_<shapeIndex>_<outputIndex> 写输出形状
├── PluginExecuteRaw::resize(): 记录 mShapeIndex
└── execute 阶段: 选择 mQnnGraphHandleVec[mShapeIndex]
```

因此，运行时输入形状必须能在 `allInputShape` 里找到，否则 `QNN` 插件的形状推导会失败。

当匹配到 `shapeIndex` 后，`PluginShapeRaw::compute()` 会读取 `o_<shapeIndex>_<outputIndex>`，把离线记录的输出类型、形状和格式写回当前输出张量：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:403
bool PluginShapeRaw::compute(InferShapeContext* ctx) {
    ...
    int shapeIndex = 0;
    if (!(computeIndex(ctx, shapeIndex))) {
        MNN_ERROR("MNN_QNN: Failed to compute shape for Plugin Op.\n");
        return false;
    }

    std::string prefix = "o_" + std::to_string(shapeIndex) + "_";
    for (int i=0; i<ctx->outputs().size(); ++i) {
        auto dst = ctx->output(i);
        std::string key = prefix + std::to_string(i);
        auto attr = ctx->getAttr(key.c_str());
        ...
        auto blob = attr->tensor();
        dst->setType(blob->dataType());
        if (nullptr != blob->dims()) {
            dst->buffer().dimensions = blob->dims()->size();
            for (int j=0; j<blob->dims()->size(); ++j) {
                dst->setLength(j, blob->dims()->data()[j]);
            }
        } else {
            dst->buffer().dimensions = 0;
        }
        TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
    }
    return true;
}
```

### 5.2 创建 `QNN` 算子

形状推导完成后，`Pipeline::encode()` 还会为每个算子创建后端 `Execution`。这里不是直接手写调用 `new CPUPlugin(...)`，而是走 `MNN` 后端的通用算子创建路径：

```text
Pipeline::encode()
└── OpCommonUtils::createExecutionWithExternal(...)
    └── backend->onCreate(inputs, outputs, op)
        └── CPUBackend::onCreate(...)
            └── gCreator[OpType_Plugin]->onCreate(...)
                └── CPUPluginCreator::onCreate(...)
                    └── new CPUPlugin(ctx)
                        └── getCPUComputeKernel("QNN")
                            └── ComputeKernelRegistry<CPUComputeKernel>::get("QNN")
                                └── new PluginExecuteRaw
```

在 `Pipeline::encode()` 里，当前 `command` 还没有 `Execution` 时，会调用 `createExecutionWithExternal(...)`：

```cpp
// source/core/Pipeline.cpp:563
std::shared_ptr<BufferStorage> tmpStorage;
if (nullptr == iter.execution) {
    iter.execution.reset(OpCommonUtils::createExecutionWithExternal(
        mBackend.get(), iter.inputs, iter.outputs, iter.op, &loader, tmpStorage));
}
```

`OpCommonUtils::createExecutionWithExternal(...)` 对没有外部权重重建需求的算子，会直接转到后端的 `onCreate(...)`：

```cpp
// source/core/OpCommonUtils.cpp:664
Execution* OpCommonUtils::createExecutionWithExternal(
    Backend* backend,
    const std::vector<Tensor*>& inputs,
    const std::vector<Tensor*>& outputs,
    const MNN::Op* op,
    FileLoader* externalFile,
    std::shared_ptr<BufferStorage>& tmpstore) {
    ...
    if (!hasExternal) {
        return backend->onCreate(inputs, outputs, op);
    }
```

当前模型运行在 `CPU` 后端上，所以这里进入 `CPUBackend::onCreate(...)`。它用 `op->type()` 去 `gCreator` 表里找创建器：

```cpp
// source/backend/cpu/CPUBackend.cpp:720
Execution* CPUBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) {
    ...
    auto opType = op->type();
    ...
    auto map  = gCreator;
    auto iter = map->find(opType);
    if (iter == map->end() ) {
        MNN_PRINT("Don't support type [%s]\n", MNN::EnumNameOpType(op->type()));
        return nullptr;
    }
    Execution* exe = nullptr;
    if (exe == nullptr) {
        exe = iter->second->onCreate(inputs, outputs, op, this);
    }
    return exe;
}
```

`OpType_Plugin` 对应的创建器在 `CPUPlugin.cpp` 文件尾部注册：

```cpp
// source/backend/cpu/CPUPlugin.cpp:98
REGISTER_CPU_OP_CREATOR(CPUPluginCreator, OpType_Plugin);
```

这个宏展开后的核心动作是把 `CPUPluginCreator` 放进 `CPUBackend::gCreator`：

```cpp
// source/backend/cpu/CPUBackend.hpp:218
#define REGISTER_CPU_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static name _temp;\
        CPUBackend::addCreator(opType, &_temp); \
    }
```

所以对于 `OpType_Plugin(type="QNN")`，最终会调用 `CPU` 后端里的 `CPUPluginCreator::onCreate(...)`：

```cpp
// source/backend/cpu/CPUPlugin.cpp:70
class CPUPluginCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        ...
        const Plugin* plugin_param = op->main_as<Plugin>();
        const std::string& op_type = plugin_param->type()->str();
        std::unique_ptr<plugin::CPUKernelContext> ctx(
            new plugin::CPUKernelContext(op_type, backend, inputs, outputs, static_cast<CPUBackend *>(backend)->pNPUModelDirPath));

        for (const Attribute* attr : *(plugin_param->attr())) {
            ctx->setAttr(attr->key()->str(), attr);
        }
        return new CPUPlugin(std::move(ctx));
    }
};
```

这里把三类信息塞进了 `CPUKernelContext`：

- `op_type`：这里是 `"QNN"`；
- `pNPUModelDirPath`：模型目录，后面用来拼 `QNN` 计算图 `qnn/graph0.bin`；
- 插件属性：包括 `path`、`inputs`、`outputs`、`allGraphName`、`allInputShape`。

`CPUPluginCreator::onCreate(...)` 最后一行真正进入 `CPUPlugin` 构造函数。构造函数里会通过 `op_type` 找到 `QNN` 计算内核，并立刻调用 `kernel_->init(...)`：

```cpp
// source/backend/cpu/CPUPlugin.cpp:27
class CPUPlugin : public Execution {
public:
    CPUPlugin(std::unique_ptr<plugin::CPUKernelContext> ctx)
        : Execution(ctx->backend()), ctx_(std::move(ctx)) {
        kernel_ = getCPUComputeKernel(ctx_->op_type());
        MNN_CHECK(nullptr != kernel_.get(),
                  "CPU compute kernel has not been registered for plugin op.");
        kernel_->init(ctx_.get());
        mNeedAllocIO = kernel_->needAllocIO();
    }
```

`getCPUComputeKernel(...)` 本身只是从 `ComputeKernelRegistry` 里按名字取工厂函数：

```cpp
// source/backend/cpu/CPUPlugin.cpp:21
static std::shared_ptr<plugin::CPUComputeKernel> getCPUComputeKernel(
    const std::string& name) {
    return std::shared_ptr<plugin::CPUComputeKernel>(
        plugin::ComputeKernelRegistry<plugin::CPUComputeKernel>::get(name));
}
```

其 `get(name)` 会查 `gFactoryMap`，然后执行对应工厂函数：

```cpp
// source/plugin/PluginKernel.cpp:48
PluginKernelT* ComputeKernelRegistry<PluginKernelT>::get(const std::string& name) {
    auto* gFactoryMap = ComputeKernelRegistry<PluginKernelT>::getFactoryMap();
    if (!gFactoryMap->count(name)) {
        MNN_PRINT("Factory has not been registered for name %s.", name.c_str());
        return nullptr;
    }
    return gFactoryMap->at(name)();
}
```

`QNN` 在前面注册过 `"QNN"` 这个工厂函数：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1841
plugin::ComputeKernelRegistry<plugin::backend::PluginExecuteRaw::KernelT>::add("QNN", []() {
    return new plugin::backend::PluginExecuteRaw;
});
```

所以这里的实际对象就是 `PluginExecuteRaw`。因此 `CPUPlugin` 构造函数里的 `kernel_->init(ctx_.get())` 会动态分发到 `PluginExecuteRaw::init(...)`，执行加载 `qnn/graph*.bin`。

### 5.3 加载 `QNN` 图

`PluginExecuteRaw::init()` 先确保 `QNN` 全局上下文已经创建，然后用模型目录和插件属性 `path` 拼出 `QNN` 图的上下文二进制路径：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:907
bool init(CPUKernelContext* ctx) override {
    if (QNN::gContext.deviceHandle == nullptr){
        QNN::createQnnContext();
    }
    auto path = MNNFilePathConcat(ctx->dir_path(), ctx->getAttr("path")->s()->str());
```

这里的路径关系是：

```text
ctx->dir_path()              = <model_dir>
ctx->getAttr("path")->s()    = qnn/graph0.bin
最终 path                    = <model_dir>/qnn/graph0.bin
```

随后读取 `allGraphName`。一个 `.bin` 里可能包含多个计算图，比如 `prefill` 和 `decode`，所以运行时必须知道计算图名字顺序：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:913
std::vector<std::string> allGraphName;
auto allGraphNameAttr = ctx->getAttr("allGraphName");
if (allGraphNameAttr && allGraphNameAttr->list() && allGraphNameAttr->list()->s()) {
    auto graphNames = allGraphNameAttr->list()->s();
    for (int i = 0; i < graphNames->size(); ++i) {
        allGraphName.push_back(graphNames->GetAsString(i)->str());
    }
} else {
    MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allGraphName' attr.\n");
    return false;
}
```

最后创建 `RawExecutorWrapper`，把上下文二进制交给 `compileModel(...)`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:952
mRawExecutor.reset(new RawExecutorWrapper());
return mRawExecutor->compileModel(path, binaryOffset, binarySize, allGraphName);
```

`compileModel(...)` 先读入 `.bin`。如果插件属性里有 `offset` / `size`，就从文件指定位置读取；否则直接 `mmap` 整个文件：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:772
bool compileModel(const std::string& path, size_t offset, size_t size, const std::vector<std::string>& allGraphName) {
    mPath = path;
    void* buffer = nullptr;
    std::vector<char> bufferVec(size, 0);
    MMapReader reader;
    if (size > 0) {
        buffer = bufferVec.data();
        std::unique_ptr<FileLoader> binaryFile(new FileLoader(path.c_str()));
        binaryFile->offset((int64_t)offset);
        binaryFile->read((char *)buffer, (int64_t)size);
    } else {
        reader.open(path.c_str());
        buffer = reader.addr();
        size = reader.size();
    }
```

接着通过 `QNN system API` 解析二进制里的计算图元信息：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:790
QnnSystemContext_Handle_t systemContextHandle = nullptr;
QNN::gContext.systemInterface.systemContextCreate(&systemContextHandle);
Qnn_ContextBinarySize_t binarySize = 0;
const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
CALL_QNN(QNN::gContext.systemInterface.systemContextGetBinaryInfo(
    systemContextHandle, buffer, size, &binaryInfo, &binarySize));
copyMetadataToGraphsInfo(binaryInfo, mGraphsInfo, mGraphCount);
QNN::gContext.systemInterface.systemContextFree(systemContextHandle);
```

然后校验二进制，并从二进制创建 `QNN` 上下文：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:807
auto error = QNN::gContext.interface.contextValidateBinary(
    QNN::gContext.backendHandle,
    QNN::gContext.deviceHandle,
    mQnnContextConfig,
    buffer,
    size);
...
CALL_QNN(QNN::gContext.interface.contextCreateFromBinary(
    QNN::gContext.backendHandle,
    QNN::gContext.deviceHandle,
    mQnnContextConfig,
    buffer,
    size,
    &mQnnContextHandle,
    mQnnProfileHandle));
```

最后按 `allGraphName` 的顺序整理 `mGraphsInfo`，并取回每个计算图句柄：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:820
std::vector<GraphInfo*> sortedGraphsInfo(mGraphCount, nullptr);
std::map<std::string, GraphInfo*> graphInfoMap;
for (int i = 0; i < mGraphCount; ++i) {
    graphInfoMap[mGraphsInfo[i]->graphName] = mGraphsInfo[i];
}

for (int i = 0; i < mGraphCount; ++i) {
    auto it = graphInfoMap.find(allGraphName[i]);
    MNN_ASSERT(it != graphInfoMap.end());
    sortedGraphsInfo[i] = it->second;
}
for (int i = 0; i < mGraphCount; i++) {
    CALL_QNN(QNN::gContext.interface.graphRetrieve(
        mQnnContextHandle,
        mGraphsInfo[i]->graphName,
        &(mQnnGraphHandleVec[i])));
}
```

这一步结束后，`QNN` 插件已经持有：

```text
RawExecutorWrapper
├── mQnnContextHandle
├── mGraphsInfo[shapeIndex]
└── mQnnGraphHandleVec[shapeIndex]
```

后面 `execute` 阶段不再重新创建 `QNN` 计算图，而是根据 `shapeIndex` 选择已经取回的计算图句柄。

### 5.4 `PluginExecuteRaw::resize()`

`CPUPlugin::onResize()` 会把当前输入输出张量重置到 `CPUKernelContext`，再调用 `QNN` 内核的 `resize()`：

```cpp
// source/backend/cpu/CPUPlugin.cpp:49
ErrorCode CPUPlugin::onResize(const std::vector<Tensor*>& inputs,
                              const std::vector<Tensor*>& outputs) {
    ctx_->reset(inputs, outputs);
    auto success = kernel_->resize(ctx_.get());
    ...
}
```

`PluginExecuteRaw::resize()` 会再次通过 `computeIndex()` 得到当前 `shapeIndex`，并保存到成员变量 `mShapeIndex`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:956
bool resize(CPUKernelContext* ctx) override {
    int shapeIndex = 0;
    if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
        MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
        return false;
    }
    mShapeIndex = shapeIndex;
```

随后读取插件属性里的 `inputs` / `outputs`。这些名字一般是 `t<tensor_index>`，由离线 `compilefornpu` 按主图张量索引写入：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:964
auto inputs = ctx->getAttr("inputs")->list();
auto inputTensor = ctx->inputs();
MNN_ASSERT(inputs->s()->size() == inputTensor.size());
mInputs.resize(inputs->s()->size());
mRealInputs.resize(inputTensor.size());
for (int i=0; i<inputs->s()->size(); ++i) {
    mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::CAFFE));
    mInputs[i].second = inputs->s()->GetAsString(i)->str();
    mInputs[i].first = mRealInputs[i].get();
}
```

输出也是同样处理：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:974
auto outputs = ctx->getAttr("outputs")->list();
auto outputTensor = ctx->outputs();
mOutputs.resize(outputs->s()->size());
MNN_ASSERT(outputs->s()->size() == outputTensor.size());
mRealOutputs.resize(outputTensor.size());
for (int i=0; i<outputs->s()->size(); ++i) {
    mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::CAFFE));
    mOutputs[i].second = outputs->s()->GetAsString(i)->str();
    mOutputs[i].first = mRealOutputs[i].get();
}
```

这里保存的是 `MNN` 张量和 `QNN` 张量名的对应关系：

```text
mRealInputs / mRealOutputs: 临时 Tensor，后面交给 QNN graphExecute 使用
mInputs / mOutputs: (临时 Tensor 指针, QNN tensor name)
mShapeIndex: 当前输入 shape 对应的 graph index
```

到这里，`resize` 阶段完成了三件事：输出形状已经确定，`QNN` 上下文二进制已经加载，当前形状的输入输出映射也准备好了。

## 6. `execute` 阶段

`StaticModule::_execute()` 最后会走到 `Session::run()`。当流水线执行到 `OpType_Plugin(type="QNN")` 时，`CPU` 后端调用的是 `CPUPlugin::onExecute()`：

```cpp
// source/backend/cpu/CPUPlugin.cpp:59
ErrorCode CPUPlugin::onExecute(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) {
    if (kernel_->compute(ctx_.get())) {
        return NO_ERROR;
    } else {
        MNN_ERROR("Plugin kernel compute failed with false returned.\n");
        return INVALID_VALUE;
    }
}
```

对 `QNN` 插件来说，这里的 `kernel_` 就是前面 `init` 阶段创建好的 `PluginExecuteRaw`。

### 6.1 重新确认 `shapeIndex`

`PluginExecuteRaw::compute()` 开头会重新调用 `computeIndex()`，确认当前输入形状对应哪个计算图：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:988
bool compute(CPUKernelContext* ctx) override {
    AUTOTIME;
    int shapeIndex = 0;
    if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
        MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
        return false;
    }
    std::string graphName = ctx->getAttr("allGraphName")->list()->s()->GetAsString(shapeIndex)->str();
```

当前源码后面传给 `invokModel()` 的是成员变量 `mShapeIndex`，也就是 `resize` 阶段记录的 `shape index`。`compute()` 里重新算出的局部 `shapeIndex` 主要用于读取 `allGraphName` 和调试打印。

### 6.2 准备输入输出 `Tensor`

`compute()` 里会重新创建一次实时输入输出张量映射。源码注释写的是 `in-time inputs and outputs tensor`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:997
// compute and alloc real in-time inputs and outputs tensor
{
    auto inputs = ctx->getAttr("inputs")->list();
    auto inputTensor = ctx->inputs();
    MNN_ASSERT(inputs->s()->size() == inputTensor.size());
    mInputs.resize(inputs->s()->size());
    mRealInputs.resize(inputTensor.size());
    for (int i=0; i<inputs->s()->size(); ++i) {
        mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::CAFFE));
        mInputs[i].second = inputs->s()->GetAsString(i)->str();
        mInputs[i].first = mRealInputs[i].get();
    }
    ...
}
```

这和 `resize` 阶段的映射逻辑有重复。`resize` 阶段先建立一次映射并记录 `mShapeIndex`；`execute` 阶段再基于当前 `ctx->inputs()` / `ctx->outputs()` 建立实时 `Tensor`，保证后面的拷贝和 `QNN` 的 `clientBuf` 绑定使用的是本次运行的缓冲区。

### 6.3 拷贝输入、执行 `QNN`、拷回输出

真正执行前，`MNN` 会先把当前输入张量拷到 `mRealInputs`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1024
auto inputTensor = ctx->inputs();
auto outputTensor = ctx->outputs();

for (int i=0; i<mInputs.size(); ++i) {
    ctx->backend()->onCopyBuffer(inputTensor[i], mRealInputs[i].get());
}
mRawExecutor->invokModel(mInputs, mOutputs, mShapeIndex);
for (int i=0; i<mOutputs.size(); ++i) {
    ctx->backend()->onCopyBuffer(mRealOutputs[i].get(), outputTensor[i]);
}
```
随后在 `invokModel()` 里执行，最后把结果从 `mRealOutputs` 拷回 `ctx->outputs()`。

### 6.4 `invokModel()`执行入口

`RawExecutorWrapper::invokModel()` 先根据 `shapeIndex` 选择 `QNN` 计算图：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:844
void invokModel(const std::vector<std::pair<const MNN::Tensor *, std::string>>& inputs, std::vector<std::pair<const MNN::Tensor *, std::string>>& outputs, int shapeIndex) {
    GraphInfo* graph = mGraphsInfo[shapeIndex];
    Qnn_GraphHandle_t qnnGraphHandle = mQnnGraphHandleVec[shapeIndex];
```

然后按张量名把 `MNN` 缓冲区绑定到 `QNN` 输入张量的 `clientBuf`：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:849
for (int i=0; i<inputs.size(); ++i) {
    auto t = inputs[i].first;
    bool find = false;
    for (int j=0; j<graph->numInputTensors; ++j) {
        auto& dstT = graph->inputTensors[j];
        if (inputs[i].second == dstT.v1.name) {
            dstT.v1.clientBuf.data = t->host<void>();
            dstT.v1.clientBuf.dataSize = t->usize();
            find = true;
            break;
        }
    }
}
```

输出张量也是同样按名字绑定：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:868
for (int i=0; i<outputs.size(); ++i) {
    auto t = outputs[i].first;
    bool find = false;
    for (int j=0; j<graph->numOutputTensors; ++j) {
        auto& dstT = graph->outputTensors[j];
        if (outputs[i].second == dstT.v1.name) {
            dstT.v1.clientBuf.data = t->host<void>();
            dstT.v1.clientBuf.dataSize = t->usize();
            find = true;
            break;
        }
    }
}
```

这里的名字来自离线阶段写进插件属性的 `inputs` / `outputs`。

### 6.5 `graphExecute()` 执行点

绑定完输入输出缓冲区后，`RawExecutorWrapper::invokModel()` 会调用 `QNN` 接口 `graphExecute()` 执行计算图：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:887
CALL_QNN(QNN::gContext.interface.graphExecute(
    qnnGraphHandle,
    graph->inputTensors,
    graph->numInputTensors,
    graph->outputTensors,
    graph->numOutputTensors,
    mQnnProfileHandle,
    nullptr));
MNN::QNN::doProfile(QNN::gContext.interface, mQnnProfileHandle);
```

到这一行，计算已经交给了 `QNN` 后端：

- 通过 `shapeIndex` 选中了正确的 `QNN` 计算图句柄；
- 通过 `inputs` / `outputs` 名字把缓冲区绑到 `QNN` 张量；
- 把执行交给 `QNN` 运行时。

`.bin` 里面的 `HTP` 计算图执行由 `QNN` 后端负责。`MNN` 运行时在这里主要做计算图选择、缓冲区绑定和结果回拷。

## 7. 小结

把第 4 到第 6 节串起来看，`QNN` 插件的运行时路径是：

```text
StaticModule::onForward()
├── _resize()
│   ├── ShapePlugin
│   │   └── PluginShapeRaw::compute()
│   │       ├── allInputShape -> shapeIndex
│   │       └── o_<shapeIndex>_* -> 输出 shape
│   │
│   └── Pipeline::encode() 创建 Plugin Execution
│       └── CPUPluginCreator
│           └── CPUPlugin
│               └── PluginExecuteRaw::init()
│                   └── RawExecutorWrapper::compileModel()
│                       ├── mmap/read qnn/graph*.bin
│                       ├── contextCreateFromBinary()
│                       └── graphRetrieve()
│
└── _execute()
    └── CPUPlugin::onExecute()
        └── PluginExecuteRaw::compute()
            ├── 准备实时输入输出 Tensor
            ├── onCopyBuffer(input -> mRealInputs)
            ├── RawExecutorWrapper::invokModel(mShapeIndex)
            │   ├── mQnnGraphHandleVec[mShapeIndex]
            │   ├── 绑定 QNN input/output clientBuf
            │   └── graphExecute()
            └── onCopyBuffer(mRealOutputs -> output)
```
