---
title: "LLM 加载流程"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-05-24T23:00:00+08:00
draft: false
description: "按当前仓库代码梳理 MNN LLM 从配置读取到模块就绪的完整加载过程"
slug: "llm-load"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# LLM 加载流程


- [LLM 加载流程](#llm-加载流程)
  - [1. LLM配置](#1-llm配置)
  - [2. LLM加载整体流程](#2-llm加载整体流程)
  - [3. Runtime 初始化](#3-runtime-初始化)
  - [4. tokenizer、embedding 和 sampler 初始化](#4-tokenizerembedding-和-sampler-初始化)
    - [4.1 tokenizer 初始化](#41-tokenizer-初始化)
    - [4.2 context.json 和 chat template](#42-contextjson-和-chat-template)
    - [4.3 DiskEmbedding 初始化](#43-diskembedding-初始化)
    - [4.4 Sampler 初始化](#44-sampler-初始化)
  - [5. Module 加载与执行层装配](#5-module-加载与执行层装配)
    - [5.1 加载策略](#51-加载策略)
    - [5.2 加载模型](#52-加载模型)
    - [5.3 网络解析](#53-网络解析)
    - [5.4 把图拆成静态执行模块](#54-把图拆成静态执行模块)
    - [5.5 创建`Session`](#55-创建session)
      - [5.5.1 预重排权重](#551-预重排权重)
    - [5.6 创建 `Pipeline`](#56-创建-pipeline)
    - [5.7 执行具体算子](#57-执行具体算子)
    - [5.8 总览](#58-总览)
  - [6. 其它准备](#6-其它准备)
    - [6.1 解码策略](#61-解码策略)
    - [6.2 预分配常用变量](#62-预分配常用变量)
    - [6.3 加载结束](#63-加载结束)


本篇针对`transformers/llm/engine/src/llm.cpp` 的 `Llm::load()`接口，梳理 MNN LLM 如何把 `config.json`、`llm_config.json`、`llm.mnn` 和 `llm.mnn.weight` 装配成可执行状态，源码执行顺序如下。

```text
`Llm::createLLM(config_path)`
    ↓
`LlmConfig`
    ↓
`Llm::load()`
    ↓
`initRuntime()`
    ↓
`Tokenizer::createTokenizer(...)`
    ↓
读取 `context.json`
    ↓
构造 `DiskEmbedding` / `Prompt` / `Sampler`
    ↓
`Module::load(...)`
    ↓
`loadInternal(...)`
    ↓
`PipelineModule::load(...)`
    ↓
`StaticModule(...)`
    ↓
`Session(...)`
    ↓
`Pipeline(...)`
    ↓
初始化 `ModulePool` / `mask` / `position ids`
```

## 1. LLM配置

LLM 的入口通常是：

```cpp
// transformers/llm/engine/src/llm.cpp:72
Llm* llm = Llm::createLLM(config_path);
```

入口逻辑在 `Llm::createLLM`：

```cpp
// transformers/llm/engine/src/llm.cpp:72
Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual() || config->is_audio() || config->has_talker()) {
        llm = new Omni(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}
```

这里先做了一次很轻量的模型分流：

- 纯文本模型返回 `Llm`
- 多模态/语音模型返回 `Omni`

此时**模型还没有真正加载**，只读取配置并做基础成员的初始化：

```cpp
// transformers/llm/engine/src/llm.cpp:930
Llm::Llm(std::shared_ptr<LlmConfig> config) : mConfig(config) {
    mExecutor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    mContext.reset(new LlmContext);
    mMeta.reset(new KVMeta);
    mMeta->layer_nums = mConfig->layer_nums();
    mMeta->attn_scale = mConfig->attn_scale();
    mGenerateParam.reset(new GenerationParams);
}
```

该阶段会构建LLM类运行时频繁使用的核心对象：

- `mExecutor`：Express 执行器；
- `mContext`：记录 prompt 长度、decode 长度、耗时、token 历史；
- `mMeta`：KV Cache 元信息；
- `mGenerateParam`：各个生成策略使用参数

## 2. LLM加载整体流程

先看 `Llm::load()` 的主骨架：

```cpp
// transformers/llm/engine/src/llm.cpp:238
bool Llm::load() {
    initRuntime();

    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
    // 2. load context
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    mSampler.reset(Sampler::createSampler(mContext, mConfig));

    Module::Config module_config;
    if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = true;
    }
    module_config.rearrange = true;

    mRuntimeManager->setExternalFile(mConfig->llm_weight());
    mModule.reset(Module::load(inputNames, outputNames, mConfig->llm_model().c_str(), mRuntimeManager, &module_config));
    mRuntimeManager->setExternalFile("");

    setSpeculativeConfig();
    mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);
    ...
}
```

这段代码会：

1. 先初始化 Runtime，准备后端、线程和 hint；
2. 再构造 tokenizer、embedding、prompt 和 sampler；
3. 然后准备 `Module::Config` 和模型路径；
4. 接着装载 `llm.mnn`；
5. 最后准备解码策略 `Generation` 和 `ModulePool`。

## 3. Runtime 初始化

Runtime 初始化在 `initRuntime()`方法中：

```cpp
// transformers/llm/engine/src/llm.cpp:166
ScheduleConfig config;
BackendConfig cpuBackendConfig;
config.type      = backend_type_convert(mConfig->backend_type());
config.numThread = mConfig->thread_num();
config.backendConfig = &cpuBackendConfig;
if(config.type == 3){
    // opencl need set numThread = 64(buffer mode)
    config.numThread |= 64;
}
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

// transformers/llm/engine/src/llm.cpp:191
mRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
setRuntimeHint(mRuntimeManager);
```

首先会做两件事：

1. 把 JSON 配置翻译成 MNN 的 `ScheduleConfig/BackendConfig`；
2. 创建 `RuntimeManager`，后续 `Module::load()` 就绑定到这个运行时上。

紧接着会调用 `setRuntimeHint()`，把很多 LLM 相关配置设置，例如：

```cpp
// transformers/llm/engine/src/llm.cpp:136
rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->config_.value("dynamic_option", 0));
rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
rtg->setExternalPath(cachePath, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
rtg->setHint(MNN::Interpreter::MMAP_FILE_SIZE, mConfig->mmap_size());
```
会配置：attention 执行模式、动态量化相关选项、KV Cache 元信息地址、权重 mmap 路径、prefix cache 路径、mmap 文件大小。


## 4. tokenizer、embedding 和 sampler 初始化

Runtime 就绪后，`load()` 开始准备推理前端所需的组件。

### 4.1 tokenizer 初始化

```cpp
// transformers/llm/engine/src/llm.cpp:243
mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
if (mTokenizer == nullptr) {
    MNN_ERROR("[Error]: Failed to load tokenizer from: %s\n", mConfig->tokenizer_file().c_str());
    return false;
}
```

`Tokenizer::createTokenizer()` 会先读 tokenizer 文件的头部，判断它是哪一种格式：

```cpp
// transformers/llm/engine/src/tokenizer.cpp:71
Tokenizer* Tokenizer::createTokenizer(const std::string& filename) {
    ...
}
```

然后按类型创建不同的分词器：

- `Sentencepiece`
- `Tiktoken`
- `BertTokenizer`
- `HuggingfaceTokenizer`
- `PipelineTokenizer`（`.mtok` 二进制格式）

### 4.2 context.json 和 chat template

如果配置目录里有 `context.json`，`load()` 会把它读出来并 merge 到 `jinja.context`，然后调用：

```cpp
// transformers/llm/engine/src/llm.cpp:246
std::ifstream contextFile(mConfig->context_file());
if (contextFile.is_open()) {
    std::ostringstream contextStream;
    contextStream << contextFile.rdbuf();
    auto contextStr = contextStream.str();
    rapidjson::Document contextDoc;
    contextDoc.Parse(contextStr.c_str());
    if (!contextDoc.HasParseError()) {
        std::string config_json = R"({
            "jinja": {
                "context": )" + contextStr + R"(
            }
        })";
        mConfig->config_.merge(config_json.c_str());
    }
}
```

完成提示词拼接后，`Llm` 会用 `Prompt::createPrompt(...)` 统一封装模板对象。

### 4.3 DiskEmbedding 初始化

LLM 的词向量不是直接塞在图里，而是通过 `DiskEmbedding` 从独立文件或权重文件里读取：

```cpp
// transformers/llm/engine/src/llm.cpp:264
mDiskEmbedding.reset(new DiskEmbedding(mConfig));
mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
```

`DiskEmbedding` 没有把 embedding lookup 写成一个常规算子，而是直接：

1. 根据 token id 在磁盘文件里定位 offset；
2. 读出对应向量；
3. 如有量化则现场反量化；
4. 写入输入 embedding Tensor。

这个设计减少了 embedding 的常驻内存占用。

如果继续顺着 `DiskEmbedding` 往下追，当前仓库里的实际读盘入口在：

```cpp
// transformers/llm/engine/src/diskembedding.cpp:96
void DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
}
```

对应 `embedding()` 里会按 token 循环执行“计算偏移 -> 读权重 -> 反量化/类型转换 -> 写入目标 Tensor”这条路径。也就是说，加载阶段把 `DiskEmbedding` 构造出来以后，后面推理阶段就能直接按 token 做随机读取，而不需要把整张 embedding 表常驻在内存里。

### 4.4 Sampler 初始化

```cpp
// transformers/llm/engine/src/llm.cpp:266
mSampler.reset(Sampler::createSampler(mContext, mConfig));
```

这里把采样器也提前准备好。这样后续 Prefill 得到 logits 后，就能直接进入 decode，不需要在推理热路径里临时构建采样规则。

`Sampler::createSampler()` 中真正的组装逻辑在构造函数和 `configSampler()`：

```cpp
// transformers/llm/engine/src/sampler.cpp:158
Sampler::Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    mContext = context;
    // mConfig = getSamplerConfig(config);
    mConfig.max_all_tokens = config->max_all_tokens();
    mConfig.max_new_tokens = config->max_new_tokens();
    mConfig.type = config->sampler_type();
    mConfig.configSampler(mConfig.type, config);
}

/* ----------SamplerConfig---------- */
void Sampler::SamplerConfig::configSampler( std::string sampler_type, std::shared_ptr<LlmConfig> llmConfig) {
    if (sampler_type == "greedy"){
        this->configGreedy(llmConfig);
    } else if (sampler_type == "temperature"){
        this->configTemperature(llmConfig);
    } else if (sampler_type == "topK"){
        this->configTopK(llmConfig);
    } else if (sampler_type == "topP"){
        this->configTopP(llmConfig);
    } else if (sampler_type == "minP"){
        this->configMinP(llmConfig);
    } else if (sampler_type == "tfs"){
        this->configTFS(llmConfig);
    } else if (sampler_type == "typical"){
        this->configTypical(llmConfig);
    } else if (sampler_type == "penalty"){
        this->configPenalty(llmConfig);
    } else if (sampler_type == "mixed"){
        this->configMixed(llmConfig);
    }
}
```

这里把 `temperature`、`topK`、`topP`、`penalty` 等采样参数直接保存。

## 5. Module 加载与执行层装配

前面在 `llm.cpp` 里看到的模型加载部分：

```cpp
// transformers/llm/engine/src/llm.cpp:282
std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
std::vector<std::string> outputNames {"logits"};

mRuntimeManager->setExternalFile(mConfig->llm_weight());
mModule.reset(Module::load(inputNames, outputNames, mConfig->llm_model().c_str(), mRuntimeManager, &module_config));
mRuntimeManager->setExternalFile("");
```

这里只是加载入口。真正重要的是：`Module::load()` 往下会继续经过 `loadInternal()`、`PipelineModule`、`StaticModule`、`Session`、`Pipeline` 这几层，最后把 `llm.mnn` 装配成可执行对象，流程如下：

```text
`Module::load(...)`
    ↓
读取 `llm.mnn`，绑定外部权重文件
    ↓
`loadInternal(...)`
    ↓
解析 `Net` 元信息，构造 `NetModule`
    ↓
`PipelineModule::load(...)`
    ↓
把静态图整理成 `ScheduleInfo + sharedConst`
    ↓
`StaticModule(...)`
    ↓
创建 `Session`，准备输入输出和权重重排
    ↓
`Session(...)`
    ↓
为每条 pipeline 创建 `Pipeline`
    ↓
`Pipeline::encode / allocMemory / execute`
    ↓
具体算子的 `Execution`
```

### 5.1 加载策略

在进入 `Module::load()` 之前，`llm.cpp` 先准备了 `module_config`：

```cpp
// transformers/llm/engine/src/llm.cpp:269
if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
    module_config.shapeMutable = false;
} else {
    module_config.shapeMutable = true;
}
module_config.rearrange = true;
```

对应的配置结构在：

```cpp
// include/MNN/expr/Module.hpp:56
struct Config {
    bool dynamic = false;
    bool shapeMutable = true;
    bool rearrange = false;
    BackendInfo* backend = nullptr;
    const Module* base = nullptr;
};
```

这几个字段分别决定：

- `dynamic`：走动态 `Expr` 图加载，还是走静态模块加载；
- `shapeMutable`：输入 shape 是否允许频繁变化；
- `rearrange`：是否在加载阶段预重排权重；
- `base`：是否复用已有基础模块，例如 LoRA 场景。

对当前 `LLM` 路径来说，最关键的是后两个：

- `shapeMutable` 会影响后面 `Session` 的输入模式和 `resize` 策略；
- `rearrange` 会影响 `StaticModule` 是否提前做权重重排。

这一层还没有真正读模型，它只是先把“后面按什么方式装配执行模块”定下来。

### 5.2 加载模型

`Module::load(const char* fileName, ...)` 本身先把模型文件读进内存：

```cpp
// express/module/Module.cpp:355
FileLoader loader(fileName, true);
loader.read();
loader.merge(buffer);
```

然后补运行时和外部权重路径，这里的运行时基于上一篇提到的`config`创建：

```cpp
// express/module/Module.cpp:356
AutoStorage<uint8_t> buffer;
{
    FileLoader loader(fileName, true);
    if (!loader.valid()) {
        MNN_ERROR("Error for open %s\n", fileName);
        return nullptr;
    }
    loader.read();
    if (!loader.valid()) {
        return nullptr;
    }
    loader.merge(buffer);
    if (buffer.get() == nullptr) {
        return nullptr;
    }
}

// express/module/Module.cpp:372
if (nullptr == rtMgr.get()) {
    rtMgr.reset(_createDefaultRuntimeManager(config));
}
if (rtMgr->getInside()->mContent->mExternalFile.empty()) {
    rtMgr->setExternalFile(std::string(fileName) + ".weight");
}
auto res = loadInternal(inputs, outputs, buffer.get(), buffer.size(), rtMgr, config);
```

这里的流程包括两个关键点：

1. `llm.mnn` 会先整体读成内存 buffer；
2. `llm.mnn.weight` 不在这里直接解析，而是通过 `RuntimeManager::setExternalFile()` 传下去，供后面常量张量和权重重排阶段使用。

接下来真正做网络解析的是 `loadInternal()`方法。

### 5.3 网络解析

`loadInternal()` 在 `express/module/Module.cpp` 里，核心骨架如下：

```cpp
// express/module/Module.cpp:397
auto net = GetNet(buffer);
std::shared_ptr<Module::Info> info(new Module::Info);
info->inputNames = inputs;
info->outputNames = outputs;

std::shared_ptr<Module> m(PipelineModule::load(inputs, outputs, buffer, length, rtMgr, config));
return new NetModule(m, info, net, length, ...);
```

`loadInternal()` 对应的是“把 FlatBuffers 模型解释成 MNN 内部模块”的入口层。它做的事情可以压成四步：

1. 校验 `MNN` buffer 是否合法；
2. 解析 `Net` 里的 `uuid`、`version`、`bizCode`、`metaData`；
3. 整理输入输出名字与输入信息，如调用侧未显式传入输入输出name，会扫描 `net->oplists()`、`net->tensorName()`、`net->outputName()`，自动推导真实输入输出；
4. 调用 `PipelineModule::load(...)` 继续往下构建真正的执行模块。

这里的`NetModule` 是一个最外层包装：

- 持有 `Module::Info` 这里包含了属性`runTimeManager`；
- 在 `onForward()` 前后做 `RuntimeExecuteWrap`；
- 调完内部模块后清缓存；
- 对外暴露统一的 `Module` 接口。

### 5.4 把图拆成静态执行模块

`LLM` 加载路径会继续进入 `PipelineModule::load(...)` 的静态加载分支：

```cpp
// express/module/PipelineModule.cpp:682
auto net = GetNet(buffer);
bool needGeometry = net->usage() != Usage_INFERENCE_STATIC;
sharedConst.reset(new Schedule::ScheduleInfo);
initConstTensors(sharedConst->allTensors, net, defaultBackend.get(), code, &fileLoader);
return new StaticModule(info.inputs, info.outputs, std::move(buffers), std::move(scheduleInfo), sharedConst, std::move(modes), std::move(rt), config);
```

`PipelineModule::load()` 对应的是“把图整理成可执行调度信息”，这里的重点是为下一层 `StaticModule`准备直接可用的数据结构：

- `sharedConst`：共享常量张量和默认后端信息；
- `Schedule::ScheduleInfo`：调度后的 pipeline 信息；
- `oplists`：每个 op 的输入输出 `Tensor`、`Op` 指针和执行缓存信息；
- `needGeometry`：是否还需要做 geometry transform。


按 `express/module/PipelineModule.cpp` 的真实代码顺序看，这一层可以再细分成下面几步：

```text
复制 `llm.mnn` buffer 到 `BufferStorage`
    ↓
创建 `subGraphMap`
    ↓
初始化 `sharedConst`
    ↓
`initConstTensors(...)` 解析常量张量
    ↓
收集输入/输出 tensor index
    ↓
`_createSubModuleInfo(...)` 划分子模块
    ↓
`_createSubModule(...)` 为每个子模块构造执行模块
    ↓
生成最外层 `PipelineModule`
```

其中：
1. `initConstTensors(...)` 会结合外部 `llm.mnn.weight` 文件，把网络里的常量张量准备好。
2. `preReplaceConstTensor`：`PipelineModule::load()` 会尝试把常量张量预拷贝或预替换到目标后端。
3. `subModules`：支持按子图拆成多个子模块，再由最外层 `PipelineModule` 统一调度。
   
### 5.5 创建`Session`

上一层的`_createSubModule`会把各个子模块最终构造成 `StaticModule`类，`StaticModule` 的构造函数在：

```cpp
// express/module/StaticModule.cpp:311
StaticModule::StaticModule(..., Schedule::ScheduleInfo&& scheduleInfo, ..., std::shared_ptr<Executor::RuntimeManager> rtm, const Module::Config& config) {
    Session::createPipelineBackend(scheduleInfo.pipelineInfo[0], rt);
    if (config.rearrange) {
        mResource->mBuffer = preRearrangeWeights(scheduleInfo, ...);
    } else {
        mResource->mBuffer = std::move(buffer);
    }
    mSession.reset(new Session(std::move(scheduleInfo), mResource->mModes, std::move(rt)));
    resetInputOutputs();
    if (canResize && (!config.rearrange)) {
        mSession->resize();
    }
}
```

`StaticModule` 类进行真正的执行装配：

1. 先根据 `ScheduleInfo` 创建 pipeline 对应的后端；
2. 如果 `rearrange=true`，就在加载阶段预重排权重；
3. 构造 `Session`；
4. 绑定输入输出 `Tensor`；
5. 条件满足时，提前做一次 `resize()`。

#### 5.5.1 预重排权重
`preRearrangeWeights(...)` 对应 `config.rearrange=true` 时的加载阶段权重预处理，分成下面几步：
```cpp
// express/module/StaticModule.cpp:37
std::map<const std::string, std::shared_ptr<Execution>> base_executions;
if (base != nullptr) {
    auto static_module = getStaticModule(base);
    auto session = static_module->getSession();
    std::vector<Schedule::OpCacheInfo> op_caches = session->getPipelineInfo(0).second;
    // 从 base module 的 executionCache 里按 op name 收集可复用 Execution
}

// express/module/StaticModule.cpp:54
FileLoader loader(scheduleInfo.externalWeightPath.c_str());
auto&& pipelineInfo = scheduleInfo.pipelineInfo[0].second;
```
1. 处理`clone`场景：从 `base module` 的 `Session` 里收集已有 `executionCache`。可以直接`clone` 部分已经预处理过的算子`Execution`，不需要重新读取和重排权重。

2. 进入预处理分支，包括下面的算子类型：

```cpp
// express/module/StaticModule.cpp:66
switch (op->type()) {
    case MNN::OpType_DepthwiseConvInt8:
    case MNN::OpType_ConvInt8:
    case MNN::OpType_ConvolutionDepthwise:
    case MNN::OpType_Convolution:
        ...
    case MNN::OpType_Attention:
        ...
    case MNN::OpType_LayerNorm:
        ...
}
```

3. 以卷积为例，权重预处理入口是调用 `OpCommonUtils::createExecutionWithExternal(...)`：

```cpp
// express/module/StaticModule.cpp:109
exe.reset(OpCommonUtils::createExecutionWithExternal(backend, info.inputs, info.outputs, op, &loader, tmpstorage));
if (exe.get() == nullptr) {
    exe.reset(OpCommonUtils::createExecutionWithExternal(backupBackend, info.inputs, info.outputs, op, &loader, tmpstorage));
}
```

`createExecutionWithExternal(...)` 中先判断当前 `Op` 是否使用 external data。如果是卷积、`Scale` 或 `LayerNorm` 这类带外部权重的算子，它会调用 `_RebuildExternalOp(...)` 临时重建一个**带权重信息**的 `Op`，再交给 `backend->onCreate(...)`：

```cpp
// source/core/OpCommonUtils.cpp:669
bool hasExternal = false;
switch (op->main_type()) {
    case OpParameter_Convolution2D:
        hasExternal = USE_EXTERNAL_DATA(op->main_as_Convolution2D());
        break;
    case OpParameter_Scale:
        hasExternal = USE_EXTERNAL_DATA(op->main_as_Scale());
        break;
    case OpParameter_LayerNorm:
        hasExternal = USE_EXTERNAL_DATA(op->main_as_LayerNorm());
        break;
}

// source/core/OpCommonUtils.cpp:692
bool res = _RebuildExternalOp(externalFile, op, builder);
auto newOp = flatbuffers::GetRoot<MNN::Op>(builder.GetBufferPointer());
execution = backend->onCreate(inputs, outputs, newOp);
```

具体来说，对于卷积，`_RebuildExternalOp(...)` 会根据 `Convolution2D.external` 里的 offset/size 读外部权重文件。PTQ / sparse 量化卷积会把 `quanParameter->buffer`、`alpha` 等字段读回 `Op`；bias、index 这类字段则按 `external` 里的长度决定是否读取。非量化卷积会把 `externalPath` 写回新的 `Op`，并把 bias 读出来，后面的 `ConvolutionCommon::load` 再继续按 `externalPath` 加载权重：

```cpp
// source/core/OpCommonUtils.cpp:604
case OpParameter_Convolution2D: {
    std::unique_ptr<Convolution2DT> param(origin->main_as_Convolution2D()->UnPack());
    if (param->quanParameter) {
        external->offset(param->external[0]);
        param->quanParameter->buffer.resize(param->external[1]);
        external->read((char*)param->quanParameter->buffer.data(), param->external[1]);
        param->quanParameter->alpha.resize(param->external[2] / sizeof(float));
        external->read((char*)param->quanParameter->alpha.data(), param->external[2]);
        ...
    } else {
        param->quanParameter.reset(new IDSTQuanT);
        param->quanParameter->type = 8;
        externalPathFbb = builder.CreateString(external->path());
        param->bias.resize(param->external[2] / sizeof(float));
        external->read((char*)param->bias.data(), param->external[2]);
    }
}
```

重建op后，会继续调用`backend->onCreate(...)`方法创建 `Execution` 。以 CPU dense convolution 为例，会进入 `DenseConvolutionTiledExecutor`类的初始化。这里的卷积会走 `im2col + GEMM` 思路，所以卷积核最终要变成 matmul 的 B 矩阵。原始权重可以理解成：

```text
weight[outputChannel, inputChannel, kernelY, kernelX]
```

CPU 后端需要的不是这个连续布局，而是按 matmul kernel 的 pack 参数整理过的 `mResource->mWeight`：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:163
auto outputCount = (int)biasSize;
core->MNNGetMatMulPackMode(&eP, &lP, &hP);
auto srcCount = originWeightSize / outputCount / common->kernelX() / common->kernelY();
auto kernelSize = common->kernelX() * common->kernelY();
auto hU = UP_DIV(outputCount, hP);
auto lU = UP_DIV(srcCount, lP) * kernelSize;

// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:192
mResource->mWeight.reset(Tensor::createDevice<uint8_t>({hU * hP, lU * lP, bytes}));
initWeight(mResource->mWeight->host<float>(), originWeight, cache->host<float>(), srcCount, outputCount, kernelSize, core);
```

float 权重的重排分两步。第一步先在每个输出通道内交换 `kernel` 和 `input channel` 的顺序：

```cpp
// source/backend/cpu/compute/ConvolutionTiledExecutor.cpp:24
void ConvolutionTiledExecutor::initWeight(...) {
    // Swap k, ic
    int dims[4] = { depth, kernelSize, kernelSize, depth };
    for (int o = 0; o < outputCount; ++o) {
        auto dO = cache + o * depth * kernelSize;
        auto sO = source + o * depth * kernelSize;
        MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
    }
}
```

也就是从每个输出通道下的：

```text
[inputChannel, kernelY * kernelX]
```

转成：

```text
[kernelY * kernelX, inputChannel]
```

第二步再调用 `MNNPackForMatMul_B(...)`，把这个二维矩阵打包成 CPU matmul kernel 最容易读取的 B 矩阵布局：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:25
void DenseConvolutionTiledExecutor::initWeight(...) {
    ConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize, depth, true);
}
```

把原始卷积核从 `outputChannel/inputChannel/kernel` 顺序，改成后端 GEMM 内核按 `hP/lP` 分块读取的布局。这样执行阶段只需要做输入的 `im2col`/blit，再直接和预打包好的 `mResource->mWeight` 做 packed matmul，不需要第一次 forward 时再整理权重。

如果是 int8/int4 权重，重排对象还包括量化信息。`initQuantizeResource(...)` 会把 int4 先展开成 int8，然后按 `hU/lU/hP/lP` 写入 `resource->mWeight`，同时把 scale/bias 保存到 `mDequantize.mScaleBias`：

```cpp
// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:55
if (int8Info->canUseInt4) {
    // Revert int4 to int8
    blob.resize(int8Info->weight.size() * 2);
    ...
}

// source/backend/cpu/compute/DenseConvolutionTiledExecutor.cpp:70
resource->mWeight.reset(Tensor::createDevice<int8_t>({hU, lU * lP, hP}));
for (int y = 0; y < outputCount; ++y) {
    int yo = y / hP;
    int yi = y % hP;
    auto srcY = srcWInt8 + y * srcChannel * kernelSize;
    auto dstY = dstWInt8 + yo * lP * hP * lU + yi;
    for (int iz = 0; iz < srcChannel; ++iz) {
        for (int k = 0; k < kernelSize; ++k) {
            int sx = iz * kernelSize + k;
            int dx = iz + k * srcChannel;
            dstY[dx * hP] = srcY[sx];
        }
    }
}
```

此时`Execution` 里已经持有适合 `Backend` 执行的 `mResource->mWeight、bias、scale` 等资源。当后续 `Session::resize()` / `Pipeline::allocMemory()` 遇到同一个 `Op`，可以直接复用 `info.executionCache` 里的 `Execution`，不再重新进行 external weight 解析和 pack 权重。

卷积分支创建成功后，还会清掉 `op_table` 里的原始权重字段：

```cpp
// express/module/StaticModule.cpp:122
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
```

这一步的含义是：权重已经被 backend 的 `Execution` 使用并重排，原始 `Op` 里的 raw weight / bias / 量化 buffer 等不再继续保留。



### 5.6 创建 `Pipeline`

`StaticModule` 里构造 `Session` 后，`Session` 构造函数会继续为每条 pipeline 创建 `Pipeline` 对象：

```cpp
// source/core/Session.cpp:157
for (auto& iter : mInfo.pipelineInfo) {
    createPipelineBackend(iter, mRuntime);
    std::shared_ptr<Pipeline> newPipeline(
        new Pipeline(mInfo.externalWeightPath, std::move(iter), ..., rt, cpuRuntime.get(), geoMask));
    mPipelines.emplace_back(std::move(newPipeline));
}
```

`Session` 这一层负责把调度结果落成真正的执行流水线：

- `Session` 持有运行时 `RuntimeInfo`；
- `Session` 会把调度阶段生成的 `pipelineInfo` 落成一组真正的 `Pipeline`；
- 每个 `Pipeline` 不在构造函数里直接执行算子，而是在 `Session::resize()` 和 `Session::run()` 里分阶段执行 `encode`、`allocMemory`、`execute`等算子生成、分配内存、执行等命令。

对 `LLM` 推理路径来说，可以把 `Session` 理解成“执行层总控”：

- `resize()` 负责让所有 `Pipeline` 完成 command 生成、`Execution` 创建和内存分配；
- `run()` 负责按顺序调用每条 `Pipeline::execute()`；
- `clone(...)` 则允许 `ModulePool` 在加载完成后快速复制出多份可独立运行的执行实例。

源码里这两个阶段直接调用所有执行流水线的接口：

```cpp
// source/core/Session.cpp:272
ErrorCode Session::resize() {
    if (mNeedResize) {
        for (auto& iter : mPipelines) {
            iter->encode(debug, permitCodegen);
        }
        mNeedMalloc = true;
    }
    if (mNeedMalloc) {
        for (auto& iter : mPipelines) {
            iter->allocMemory(firstMalloc, forbidReplace);
        }
    }
}

// source/core/Session.cpp:243
ErrorCode Session::run() const {
    for (auto& iter : mPipelines) {
        iter->execute();
    }
}
```

### 5.7 执行具体算子

`Pipeline` 再往下就是具体执行阶段，这里的 `encode()` 是把调度结果整理成 MNN 内部的 `command buffer`，其包括算子、输入输出等属性：

```cpp
// source/core/Pipeline.cpp:201
ErrorCode Pipeline::encode(bool supportDebug, bool permitCodegen) {
    if (!mInfo.first.needComputeGeometry) {
        // Static Model just copy info to command buffer
        cmd->op = info.op;
        cmd->inputs = info.inputs;
        cmd->outputs = info.outputs;
        info.executeBuffer.command = {cmd};
    } else {
        GeometryComputerUtils::shapeComputeAndGeometryTransform(...);
    }
}
```
`encode()`方法主要是把 `OpCacheInfo` 里的 `op / inputs / outputs` 拷到 `executeBuffer.command`。如果需要 `geometry transform`，它会调用 `GeometryComputerUtils::shapeComputeAndGeometryTransform(...)`，在 resize 阶段重新做 shape compute、常量计算和 command buffer 生成。

`Session`的`resize`方法还会调用 `Pipeline::allocMemory()`方法，为每个 command 创建具体 `Execution`，绑定 tensor backend，并分配输入、中间结果和输出 tensor 的内存。

执行时调用`Pipeline::execute()` 方法按顺序真正执行这些 command。模型已经从“图结构”落成了“可执行算子序列”。

从`Pipeline`的执行视角如下：

```text
`encode()`
    ↓
把 `OpCacheInfo` 整理成 command buffer
    ↓
`allocMemory()`
    ↓
为每条 command 创建 `Execution`，绑定 backend，并分配 Tensor 内存
    ↓
`execute()`
    ↓
循环调用 `Execution::onExecute(...)`
```
### 5.8 总览
把前面几层连起来，可以把 `Module::load(...)` 之后的对象关系分成两段看。

第一段是模块包装和调度信息生成：

- `NetModule` 是 `Module::load(...)` 返回的最外层包装，保存模型 `Info`，并在 `onForward()` 里转调内部真实模块；
- `PipelineModule` 负责把 `Net` 整理成输入输出索引、执行顺序、子模块和 `ScheduleInfo`；
- 静态加载路径下，`PipelineModule::load(...)` 最终会创建 `StaticModule`。

第二段才是实际执行层：

- `StaticModule` 管输入输出和 `Session`
- `Session` 管多条 `Pipeline`
- `Pipeline` 管 command buffer
- `Execution` 才是真正单个算子的后端实现

## 6. 其它准备
### 6.1 解码策略

主 `Module` 加载完成后，`llm.cpp` 还会继续准备与解码策略相关对象：

```cpp
// transformers/llm/engine/src/llm.cpp:312
setSpeculativeConfig();
mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);
```

其中包括`ArGeneration、LookaheadGeneration、MtpGeneration、EagleGeneration`等多种解码策略，然后初始化模型池 `ModulePool`，这是输入长度`seq_len`+`isAllLogists`(当前是否需要返回全量 logits)到`Module`类的映射，如果不存在对应长度需要`clone`模型：

```cpp
// transformers/llm/engine/src/llm.cpp:318
if (mInSpec) {
    mModulePool[std::make_pair(verify_length, true)].reset(Module::clone(mModule.get()));
}
mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()));
mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;
```

### 6.2 预分配常用变量

加载最后还会创建常用的输入变量缓存如`attention mask` 和 `position ids`：

```cpp
// transformers/llm/engine/src/llm.cpp:332
logitsLastIdx = _var<int>({-1}, {1});
logitsAllIdx = _var<int>({0}, {1});

mAttentionMaskVarVec.resize(decode_type_num);
mPositionIdsVarVec.resize(decode_type_num);
```

然后针对 decode 场景预写好常用 `mask` 和 `position id` `Tensor`，避免单 token decode 时反复重新建图、重新分配小 Tensor。

### 6.3 加载结束

`load()` 成功返回时，程序里不只包含“模型”，同时已经具备了完整的推理状态，此时已经就绪的核心对象

- `mRuntimeManager`：已绑定好后端、线程、mmap、KV hint；
- `mTokenizer`：已能编码/解码和套用模板；
- `mDiskEmbedding`：已能按 token 从磁盘拉 embedding；
- `mSampler`：已按配置搭好采样流水线；
- `mModule`：已完成图加载；
- `mModulePool`：已准备好 prefill/decode 所需模块副本；
- `mMeta`：已带上层数、KV 状态、attention scale；
- `mGenerationStrategy`：已选好 AR / Lookahead / MTP / Eagle。
