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

这一篇直接看 `transformers/llm/engine/src/llm.cpp` 里的 `Llm::load()`，梳理 MNN LLM 如何把 `config.json`、`llm_config.json`、`llm.mnn` 和 `llm.mnn.weight` 装配成可执行状态。

加载阶段的核心不是“读文件”本身，而是把模型路径、运行时选项、tokenizer、embedding、采样器、`Module` 和 speculative decoding 相关对象一次性准备好。下面按源码里的实际执行顺序展开。

```text
`Llm::createLLM(config_path)`
    ↓
`LlmConfig`
    ↓
`Llm::load()`
    ↓
`checkFile(...)`
    ↓
`initRuntime()`
    ↓
`Tokenizer::createTokenizer(...)`
    ↓
读取 `context.json`
    ↓
构造 `DiskEmbedding` / `Sampler`
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

后面各节就按这条链路展开：先看入口和 Runtime，再看 tokenizer / embedding / sampler，最后重点看 `Module::load()` 往下是如何把 `llm.mnn` 变成真正可执行的 `Session + Pipeline`。

- [LLM 加载流程](#llm-加载流程)
  - [1. 从 createLLM 开始](#1-从-createllm-开始)
  - [2. load 的整体主流程](#2-load-的整体主流程)
  - [3. Runtime 初始化](#3-runtime-初始化)
  - [4. tokenizer、embedding 和 sampler 初始化](#4-tokenizerembedding-和-sampler-初始化)
  - [5. Module 加载与 ModulePool 初始化](#5-module-加载与-modulepool-初始化)
  - [6. 加载结束时内存里已经有什么](#6-加载结束时内存里已经有什么)

## 1. 从 createLLM 开始

LLM 的入口通常是：

```cpp
// transformers/llm/engine/src/llm.cpp
Llm* llm = Llm::createLLM(config_path);
```

入口逻辑在 `Llm::createLLM`：

```cpp
// transformers/llm/engine/src/llm.cpp
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

注意此时**模型还没有真正加载**，只是把配置读进来，并选好了对象类型。

构造函数本身只做基础成员初始化：

```cpp
// transformers/llm/engine/src/llm.cpp
Llm::Llm(std::shared_ptr<LlmConfig> config) : mConfig(config) {
    mExecutor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    mContext.reset(new LlmContext);
    mMeta.reset(new KVMeta);
    mMeta->layer_nums = mConfig->layer_nums();
    mMeta->attn_scale = mConfig->attn_scale();
    mGenerateParam.reset(new GenerationParams);
}
```

构造阶段先把几个后面会一直复用的核心对象建起来：

- `mExecutor`：Express 执行器；
- `mContext`：记录 prompt 长度、decode 长度、耗时、token 历史；
- `mMeta`：KV Cache 元信息；
- `mGenerateParam`：生成阶段共享参数。

## 2. load 的整体主流程

先看 `Llm::load()` 的主骨架：

```cpp
// transformers/llm/engine/src/llm.cpp
bool Llm::load() {
    std::string tokenizer_path = mConfig->tokenizer_file();
    std::string model_path = mConfig->llm_model();
    std::string weight_path = mConfig->llm_weight();
    if (!checkFile(tokenizer_path, "tokenizer file") ||
        !checkFile(model_path, "LLM model file") ||
        !checkFile(weight_path, "LLM weight file")) {
        return false;
    }

    initRuntime();

    mTokenizer.reset(Tokenizer::createTokenizer(tokenizer_path));
    ...

    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    mSampler.reset(Sampler::createSampler(mContext, mConfig));

    mRuntimeManager->setExternalFile(weight_path);
    mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
    mRuntimeManager->setExternalFile("");

    setSpeculativeConfig();
    mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);
    ...
}
```

这段代码已经把加载阶段的职责说得很清楚了：

1. 先校验模型文件是否存在；
2. 再初始化 Runtime；
3. 然后构造 tokenizer、embedding、sampler；
4. 接着装载 `llm.mnn`；
5. 最后准备 speculative decoding 和 `ModulePool`。

如果继续顺着 `load()` 的真实代码往下看，这一段的组织方式其实很规整：

```cpp
// transformers/llm/engine/src/llm.cpp
std::string tokenizer_path = mConfig->tokenizer_file();
std::string model_path = mConfig->llm_model();
std::string weight_path = mConfig->llm_weight();
...
initRuntime();
...
mTokenizer.reset(Tokenizer::createTokenizer(tokenizer_path));
...
std::ifstream contextFile(mConfig->context_file());
...
mDiskEmbedding.reset(new DiskEmbedding(mConfig));
if (mConfig->has_ple()) {
    mPleEmbedding.reset(new DiskEmbedding(mConfig, mConfig->ple_embed_file(), ...));
}
setChatTemplate();
mSampler.reset(Sampler::createSampler(mContext, mConfig));
...
if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
    module_config.shapeMutable = false;
}
...
if (mConfig->speculative_type() == "mtp") {
    needHiddenState = true;
}
...
mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
...
setSpeculativeConfig();
mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);
```

开头三行先把路径类配置取出来：

- `tokenizer_file()` 对应 tokenizer 文件；
- `llm_model()` 对应 `llm.mnn`；
- `llm_weight()` 对应 `llm.mnn.weight`。

这些 getter 定义在 `transformers/llm/engine/src/llmconfig.hpp`，真正落地使用是在这里的 `load()`。紧接着的 `checkFile(...)` 也说明了一点：对当前纯文本 `LLM` 路径，tokenizer、模型图和权重文件都是硬依赖，缺任何一个都会直接返回失败。

再往下第一步是 `initRuntime()`。这一层不直接碰模型图，而是先把运行时环境准备好。`backend_type`、`thread_num`、`precision`、`power`、`memory` 这些字段会在 `transformers/llm/engine/src/llm.cpp` 的 `initRuntime()` 里转成 `ScheduleConfig + BackendConfig`；`attention_mode`、`dynamic_option`、`use_mmap`、`kvcache_mmap`、`prefix_cache_path` 这些更偏执行策略的配置，则继续在同文件的 `setRuntimeHint()` 里写进 `RuntimeManager`。

Runtime 就绪以后，`load()` 开始装前端组件。`Tokenizer::createTokenizer(tokenizer_path)` 的实现位于 `transformers/llm/engine/src/tokenizer/tokenizer.cpp`，它会先判断 tokenizer 类型，再创建对应 tokenizer。随后 `context_file()` 指向的 `context.json` 会在这里被读出，并 merge 到 `jinja.context`，最后由 `setChatTemplate()` 把模板信息真正写进 tokenizer。

接下来是 embedding 和 sampler。`mDiskEmbedding.reset(new DiskEmbedding(mConfig))` 会把 `embedding_file`、`tie_embeddings`、`ple_embed_file`、`ple_quant` 这些信息交给 `transformers/llm/engine/src/diskembedding.cpp`；`mSampler.reset(Sampler::createSampler(mContext, mConfig))` 则把 `sampler_type`、`topK`、`topP`、`temperature` 和 penalty 系列配置交给 `transformers/llm/engine/src/sampler.cpp`。也就是说，到这一段结束时，文本前端和采样前端都已经准备好了。

模型真正开始装载是在 `Module::load(...)` 之前。这里会先根据配置修正 `module_config` 和输入输出签名：`backend_type` 会影响 `module_config.shapeMutable`，`speculative_type == "mtp"` 会强制追加 `hidden_states` 输出，`has_ple()`、`has_deepstack()` 则会继续追加额外输入。这里已经不是单纯“把 `llm.mnn` 读进来”，而是在为后面的静态执行模块准备正确的接口定义。

最后一段才是 speculative decoding 的初始化。`setSpeculativeConfig()` 会先根据当前模型能力和配置决定是否进入 speculative 路径；`GenerationStrategyFactory::create(...)` 再根据 `speculative_type` 选择 `ArGeneration`、`MtpGeneration` 或 `EagleGeneration`。像 `draft_predict_length`、`eagle_model` 这类配置，也是在这一层之后才真正接入 `transformers/llm/engine/src/speculative_decoding/generate.cpp` 和 `eagle.cpp`。

这样顺着代码看，`Llm::load()` 更像总装配入口：前半段处理路径和运行时，中间段装 tokenizer、embedding、sampler，后半段装 `Module` 和 speculative generation。每一段都不复杂，但合在一起才构成完整的加载流程。

## 3. Runtime 初始化

### 3.1 先检查必要文件

在真正加载前，`load()` 会先确认关键文件都存在：

```cpp
// transformers/llm/engine/src/llm.cpp
std::string tokenizer_path = mConfig->tokenizer_file();
std::string model_path = mConfig->llm_model();
std::string weight_path = mConfig->llm_weight();
if (!checkFile(tokenizer_path, "tokenizer file") ||
    !checkFile(model_path, "LLM model file") ||
    !checkFile(weight_path, "LLM weight file")) {
    return false;
}
```

这里的处理方式是直接在 `load()` 开头检查关键文件，缺任何一个就立刻返回失败。

### 3.2 initRuntime 创建 RuntimeManager

Runtime 初始化在 `initRuntime()`：

```cpp
// transformers/llm/engine/src/llm.cpp
ScheduleConfig config;
BackendConfig cpuBackendConfig;
config.type      = backend_type_convert(mConfig->backend_type());
config.numThread = mConfig->thread_num();
config.backendConfig = &cpuBackendConfig;

mRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
setRuntimeHint(mRuntimeManager);
```

这里有两个直接影响后续执行的点：

1. 把 JSON 配置翻译成 MNN 的 `ScheduleConfig/BackendConfig`；
2. 创建 `RuntimeManager`，后续 `Module::load()` 就会绑定到这个运行时上。

### 3.3 setRuntimeHint 把配置下推到底层 Runtime

紧接着会调用 `setRuntimeHint()`，把很多 LLM 相关配置继续下推：

```cpp
// transformers/llm/engine/src/llm.cpp
rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->config_.value("dynamic_option", 0));
rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
rtg->setExternalPath(cachePath, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
rtg->setHint(MNN::Interpreter::MMAP_FILE_SIZE, mConfig->mmap_size());
```

到这一步，下面几类运行时信息已经全部下推到底层 Runtime：

- attention 执行模式；
- 动态量化相关选项；
- KV Cache 元信息地址；
- 权重 mmap 路径；
- prefix cache 路径；
- mmap 文件大小。

因此加载阶段不只是反序列化模型，还把模型和底层 Runtime 策略绑到了一起。

## 4. tokenizer、embedding 和 sampler 初始化

Runtime 就绪后，`load()` 开始准备推理前端所需的组件。

### 4.1 tokenizer 初始化

```cpp
// transformers/llm/engine/src/llm.cpp
mTokenizer.reset(Tokenizer::createTokenizer(tokenizer_path));
if (mTokenizer == nullptr) {
    MNN_ERROR("[Error]: Failed to load tokenizer from: %s\n", tokenizer_path.c_str());
    return false;
}
```

`Tokenizer::createTokenizer()` 会先读 tokenizer 文件的头部，判断它是哪一种格式：

```cpp
// transformers/llm/engine/src/tokenizer/tokenizer.cpp
line_str >> magic_number;
line_str >> tokenizer_type;
```

然后按类型创建：

- `Sentencepiece`
- `Tiktoken`
- `BertTokenizer`
- `HuggingfaceTokenizer`
- `PipelineTokenizer`（`.mtok` 二进制格式）

`Tokenizer::createTokenizer()` 本身也是一层小型工厂。

### 4.2 context.json 和 chat template

如果配置目录里有 `context.json`，`load()` 会把它读出来并 merge 到 `jinja.context`，然后调用：

```cpp
setChatTemplate();
```

这意味着加载阶段已经把“怎么拼 prompt”这件事配置好了，后续 `response(prompt)` 不需要再现拼。

### 4.3 DiskEmbedding 初始化

LLM 的词向量不是直接塞在图里，而是通过 `DiskEmbedding` 从独立文件或权重文件里读取：

```cpp
// transformers/llm/engine/src/llm.cpp
mDiskEmbedding.reset(new DiskEmbedding(mConfig));
if (mConfig->has_ple()) {
    mPleEmbedding.reset(new DiskEmbedding(mConfig, mConfig->ple_embed_file(), mConfig->ple_embed_dim(), mConfig->ple_quant()));
}
```

`DiskEmbedding` 的设计很重要，它说明 MNN-LLM 并没有把 embedding lookup 写成一个常规前向算子，而是直接：

1. 根据 token id 在磁盘文件里定位 offset；
2. 读出对应向量；
3. 如有量化则现场反量化；
4. 写入输入 embedding Tensor。

这个设计直接减少了 embedding 常驻内存占用。

如果继续顺着 `DiskEmbedding` 往下追，当前仓库里的实际读盘入口在：

```cpp
// transformers/llm/engine/src/diskembedding.cpp
void DiskEmbedding::seek_read(uint8_t* dst, size_t size, size_t offset) {
    mFile->offset(offset);
    mFile->read((char*)dst, size);
}
```

对应 `embedding()` 里会按 token 循环执行“计算偏移 -> 读权重 -> 反量化/类型转换 -> 写入目标 Tensor”这条路径。也就是说，加载阶段把 `DiskEmbedding` 构造出来以后，后面推理阶段就能直接按 token 做随机读取，而不需要把整张 embedding 表常驻在内存里。

### 4.4 Sampler 初始化

```cpp
// transformers/llm/engine/src/llm.cpp
mSampler.reset(Sampler::createSampler(mContext, mConfig));
```

这里把采样器也提前准备好。这样后续 Prefill 得到 logits 后，就能直接进入 decode，不需要在推理热路径里临时构建采样规则。

`Sampler::createSampler()` 本身比较薄，真正的组装逻辑在构造函数和 `buildPipeline()`：

```cpp
// transformers/llm/engine/src/sampler.cpp
Sampler::Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config)
    : mContext(context), mRng(std::random_device{}()) {
    mConfig.type = config->sampler_type();
    mConfig.configSampler(mConfig.type, config);
    buildPipeline();
}
```

这说明加载阶段已经把 `temperature`、`topK`、`topP`、`penalty` 等采样步骤组织成固定流水线了，后面 decode 只需要直接复用。

## 5. Module 加载与执行层装配

前面在 `llm.cpp` 里看到的：

```cpp
// transformers/llm/engine/src/llm.cpp
std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
std::vector<std::string> outputNames {"logits"};

mRuntimeManager->setExternalFile(weight_path);
mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
mRuntimeManager->setExternalFile("");
```

这里只是加载入口。真正重要的是：`Module::load()` 往下会继续经过 `loadInternal()`、`PipelineModule`、`StaticModule`、`Session`、`Pipeline` 这几层，最后把 `llm.mnn` 装配成可执行对象。

先把这一段主流程记住，后面每个小节都只是对照这条链路解释各层职责：

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

### 5.1 `Module::Config` 先决定加载策略

在进入 `Module::load()` 之前，`llm.cpp` 先准备了 `module_config`：

```cpp
// transformers/llm/engine/src/llm.cpp
if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
    module_config.shapeMutable = false;
} else {
    module_config.shapeMutable = true;
}
module_config.rearrange = true;
```

对应的配置结构在：

```cpp
// include/MNN/expr/Module.hpp
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

### 5.2 `Module::load()` 先读模型，再进入 `loadInternal()`

`Module::load(const char* fileName, ...)` 本身先把模型文件读进内存：

```cpp
// express/module/Module.cpp
FileLoader loader(fileName, true);
loader.read();
loader.merge(buffer);
```

然后补运行时和外部权重路径：

```cpp
// express/module/Module.cpp
if (nullptr == rtMgr.get()) {
    rtMgr.reset(_createDefaultRuntimeManager(config));
}
if (rtMgr->getInside()->mContent->mExternalFile.empty()) {
    rtMgr->setExternalFile(std::string(fileName) + ".weight");
}
auto res = loadInternal(inputs, outputs, buffer.get(), buffer.size(), rtMgr, config);
```

这里先对照上面的流程看两个关键点：

1. `llm.mnn` 会先整体读成内存 buffer；
2. `llm.mnn.weight` 不在这里直接解析，而是通过 `RuntimeManager::setExternalFile()` 传下去，供后面常量张量和权重重排阶段使用。

接下来真正做网络解析的是 `loadInternal()`。

### 5.3 `loadInternal()` 把 `Net` 变成 `PipelineModule`

`loadInternal()` 在 `express/module/Module.cpp` 里，核心骨架如下：

```cpp
// express/module/Module.cpp
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
3. 整理输入输出名字与输入信息；
4. 调用 `PipelineModule::load(...)` 继续往下构建真正的执行模块。

所以 `NetModule` 更像一个最外层包装：

- 持有 `Module::Info`；
- 在 `onForward()` 前后做 `RuntimeExecuteWrap`；
- 调完内部模块后清缓存；
- 对外暴露统一的 `Module` 接口。

也就是说，到 `loadInternal()` 为止，模型已经从“文件 buffer”变成了“带元信息的模块对象”，但还没有落到底层 `Session`。

如果顺着 `express/module/Module.cpp` 再往下看，`loadInternal()` 这里还有两个容易忽略的点：

1. 它会先把 `runTimeManager` 塞进 `Module::Info`，后面 `NetModule::onForward()` 才能在执行前用 `RuntimeExecuteWrap` 绑定当前运行时；
2. 如果调用侧没有显式传入输入输出名字，它还会扫描 `net->oplists()`、`net->tensorName()`、`net->outputName()`，自动推导真实输入输出。

所以 `loadInternal()` 不只是“转发给 `PipelineModule::load()`”，它同时也是 `Net` 元信息、输入输出签名和运行时句柄的整理入口。

### 5.4 `PipelineModule::load()` 负责把图拆成静态执行模块

当前 `LLM` 这条路径默认不是 `dynamic`，因此最终会进入 `PipelineModule::load(...)` 的静态加载分支：

```cpp
// express/module/PipelineModule.cpp
auto net = GetNet(buffer);
bool needGeometry = net->usage() != Usage_INFERENCE_STATIC;
sharedConst.reset(new Schedule::ScheduleInfo);
initConstTensors(sharedConst->allTensors, net, defaultBackend.get(), code, &fileLoader);
return new StaticModule(info.inputs, info.outputs, std::move(buffers), std::move(scheduleInfo), sharedConst, std::move(modes), std::move(rt), config);
```

`PipelineModule::load()` 对应的是“把图整理成可执行调度信息”这一层。当前 `LLM` 默认走静态图路径，因此这里的重点不是再解释 `Net`，而是准备后面 `StaticModule` 直接可用的数据结构：

- `sharedConst`：共享常量张量和默认后端信息；
- `Schedule::ScheduleInfo`：调度后的 pipeline 信息；
- `oplists`：每个 op 的输入输出 `Tensor`、`Op` 指针和执行缓存信息；
- `needGeometry`：是否还需要做 geometry transform。

对 `LLM` 这类静态图来说，`PipelineModule::load()` 的核心作用可以概括成一句话：把 `Net` 里的算子、张量、常量权重和后端配置整理成 `StaticModule` 能直接执行的 `ScheduleInfo`。

如果按 `express/module/PipelineModule.cpp` 的真实代码顺序看，这一层可以再细分成下面几步：

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

其中有三块对 `LLM` 特别关键：

1. `initConstTensors(...)`
   这里会结合外部 `llm.mnn.weight` 文件，把网络里的常量张量先准备好。也就是说，后面 `StaticModule` 看到的很多权重 Tensor，在这一层其实已经被解析进 `sharedConst->allTensors` 了。

2. `preReplaceConstTensor`
   如果后端不是直接用 CPU 常量，`PipelineModule::load()` 还会尝试把常量张量预拷贝或预替换到目标后端，减少运行期第一次执行时的常量搬运开销。

3. `subModules`
   `PipelineModule` 不是永远只包一个 `StaticModule`。当前 `LLM` 这类单图路径通常会落成一个主要静态模块，但这层设计本身支持按子图拆成多个子模块，再由最外层 `PipelineModule` 统一调度。

因此这一层的重点不是“执行模型”，而是把原始图做成更适合执行层消费的模块树和常量池。

### 5.5 `StaticModule` 才是真正持有 `Session` 的执行模块

`StaticModule` 的构造函数在：

```cpp
// express/module/StaticModule.cpp
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

`StaticModule` 对应的是“真正持有执行对象”的这一层。它已经开始做真正的执行装配：

1. 先根据 `ScheduleInfo` 创建 pipeline 对应的后端；
2. 如果 `rearrange=true`，就在加载阶段预重排权重；
3. 构造 `Session`；
4. 绑定输入输出 `Tensor`；
5. 条件满足时，提前做一次 `resize()`。

也就是说，`StaticModule` 是 `Module` 体系里第一个真正“持有可执行 `Session`”的对象。

这里再顺着构造函数看，`StaticModule` 还额外做了几件和推理阶段直接相关的事情：

- 设置 `mModes.inputMode`
  如果 `shapeMutable=true`，输入模式会设成 `Session_Input_User`；否则走 `Session_Input_Inside`。这会直接影响后面 `onForward()` 是把输入 Tensor 直接引用进去，还是先按 Session 内部输入模式绑定。

- 记录 `mOutputFromInput` 和 `mOutputFromTensor`
  这一步会区分“哪些输出其实是输入透传”和“哪些输出需要从 `Session` 输出 Tensor 里取”。后面 `onForward()` 组装 `VARP` 输出时就依赖这两组索引。

- 记录 `mUseContentInputs`
  如果某些 shape 推导依赖输入内容而不是只依赖输入 shape，那么 `StaticModule` 会强制把输入模式改成 `Session_Input_User`，保证 `resize()` 阶段能看到真实输入内容。

所以 `StaticModule` 的职责不只是“包一层 `Session`”，它还负责把 `Module` 接口和底层 `Session` 的输入输出语义对齐起来。

它在 `onForward()` 里做的事情也很直接：

```cpp
// express/module/StaticModule.cpp
Variable::compute(inputs);
code = _resize(inputs);
code = _execute();
```

其中：

- `_resize()` 会把输入 `VARP` 绑定到 `Session` 输入张量，并在需要时触发 `Session::resize()`；
- `_execute()` 最终调用的是 `mSession->run()`。

所以从 `Llm` 的角度看，后面 `mModule->onForward(inputs)` 的那一跳，实质上已经进入 `StaticModule::onForward()` 这条执行路径了。

这里的 `_resize()` 值得单独点一下，因为 `LLM` 在 `prefill/decode/speculative verify` 之间会频繁切换不同的 `seq_len`。`StaticModule::_resize()` 实际做的事情包括：

1. 把外部 `VARP` 转成底层输入 `Tensor`；
2. 检查输入后端是否变化，必要时标记 `inputBackendChange`；
3. 判断是否需要复制输入内容、是否需要重新申请内存；
4. 调用 `Session::setNeedResize()` / `setNeedMalloc()`；
5. 最后真正执行一次 `mSession->resize()`。

这也是为什么前面 `Llm::load()` 里要提前 clone 多份 `Module` 放进 `mModulePool`。不同 `seq_len` 的路径如果复用同一个静态模块，`resize` 和内部缓存切换会更频繁。

### 5.6 `Session` 继续向下创建 `Pipeline`

`StaticModule` 里构造 `Session` 后，`Session` 构造函数会继续为每条 pipeline 创建 `Pipeline` 对象：

```cpp
// source/core/Session.cpp
for (auto& iter : mInfo.pipelineInfo) {
    createPipelineBackend(iter, mRuntime);
    std::shared_ptr<Pipeline> newPipeline(
        new Pipeline(mInfo.externalWeightPath, std::move(iter), ..., rt, cpuRuntime.get(), geoMask));
    mPipelines.emplace_back(std::move(newPipeline));
}
```

`Session` 这一层负责把调度结果落成真正的执行流水线。这里可以直接理解成：

- `Session` 持有运行时 `RuntimeInfo`；
- `Session` 会把调度阶段生成的 `pipelineInfo` 落成一组真正的 `Pipeline`；
- 每个 `Pipeline` 之后都会负责 `encode`、`allocMemory`、`execute`。

因此 `Session` 这一层主要解决的是“如何管理一次推理里完整的流水线执行”。

对当前 `LLM` 路径来说，可以把 `Session` 理解成“执行层总控”：

- `resize()` 负责让所有 `Pipeline` 完成 shape 推导、`Execution` 创建和内存分配；
- `run()` 负责按顺序调用每条 `Pipeline::execute()`；
- `clone(...)` 则允许 `ModulePool` 在加载完成后快速复制出多份可独立运行的执行实例。

前面 `mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()))` 这类代码，最终复制的核心对象就是这里的 `Session + Pipeline`。

### 5.7 `Pipeline` 才开始面对具体算子和 `Execution`

`Pipeline` 再往下就是具体执行阶段。它的关键职责很明确：

```cpp
// source/core/Pipeline.cpp
ErrorCode Pipeline::encode(bool supportDebug, bool permitCodegen) {
    // shape compute + geometry transform + command buffer
}
```

如果当前图还需要 geometry transform，`encode()` 会进入：

```cpp
GeometryComputerUtils::shapeComputeAndGeometryTransform(...)
```

否则就直接把已有的 op / tensor 信息拷进 command buffer。

后续 `allocMemory()` 会为每个 command 创建具体 `Execution` 并分配内存，`execute()` 则按顺序真正执行这些 command。也就是说，到 `Pipeline` 这一层，模型已经从“图结构”彻底落成了“可执行算子序列”。

如果继续按 `source/core/Pipeline.cpp` 的真实代码往下拆，`Pipeline` 可以按三个阶段理解：

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

这三个阶段分别解决不同问题：

1. `encode()`
   这一层主要负责把图变成 command。对于纯静态图，如果 `needComputeGeometry=false`，很多 command 会直接从已有 `OpCacheInfo` 拷进 `executeBuffer.command`；如果还需要 geometry transform，则会调用 `GeometryComputerUtils::shapeComputeAndGeometryTransform(...)` 重新生成 command buffer。

2. `allocMemory()`
   这一层最关键，因为真正的后端 `Execution` 是在这里创建的。`_createExecutions(...)` 会为每个 command 选择主后端或备用后端，并调用 `OpCommonUtils::createExecutionWithExternal(...)` 创建具体算子的 `Execution`。也就是说，到了这一步，`Convolution`、`Attention`、`MatMul` 这些算子才真正绑定到对应后端实现。

3. `execute()`
   真正执行时流程反而最直接：`_enterExecute()` -> 遍历 command -> `cmd.execution->onExecute(...)` -> `_exitExecute()`。从 `LLM` 上层看到的一次 `forwardRaw()`，最终就是落到这里逐条执行 command。

这里也能看出整个执行层的层次关系：

- `StaticModule` 管输入输出和 `Session`
- `Session` 管多条 `Pipeline`
- `Pipeline` 管 command buffer
- `Execution` 才是真正单个算子的后端实现

把这几层连起来，可以得到 `llm.mnn` 到真正执行对象的主路径：

```text
`Llm::load()`
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
`encode / allocMemory / execute`
    ↓
具体算子的 `Execution`
```

这样再回头看 `mModule.reset(Module::load(...))` 就比较清楚了：这里并不是只得到一个“模型句柄”，而是已经把 `Net`、常量权重、运行时、`Session` 和 `Pipeline` 这几层都装配起来了。

### 5.8 speculative decoding 的模块准备

主 `Module` 加载完成后，`llm.cpp` 还会继续准备 speculative decoding 相关对象：

```cpp
// transformers/llm/engine/src/llm.cpp
setSpeculativeConfig();
mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);
```

然后初始化 `ModulePool`：

```cpp
// transformers/llm/engine/src/llm.cpp
if (mInSpec) {
    mModulePool[std::make_pair(verify_length, true)].reset(Module::clone(mModule.get()));
}
mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()));
mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;
```

这里要注意，`MNN-LLM` 不是只维护一个 `Module`，而是会根据：

- `Prefill / Decode`
- 序列长度
- 是否返回 `all logits`

预先维护一组可复用模块实例。

这一步和后面 `forwardRaw()` 的模块选择是直接对应的。当前代码里 `forwardRaw()` 会根据：

- `seqLenKey`：当前是 `prefill` 还是某种长度的 `decode`
- `isAllLogists`：当前是否需要返回全量 logits

组合成 `moduleKey`，再到 `mModulePool` 里取对应模块实例。因此 `load()` 阶段提前准备 `ModulePool`，本质上是在减少运行期 clone 和重新构图的开销。

### 5.9 预分配 `attention mask` 和 `position ids`

加载最后还会创建常用的输入变量缓存：

```cpp
// transformers/llm/engine/src/llm.cpp
logitsLastIdx = _var<int>({-1}, {1});
logitsAllIdx = _var<int>({0}, {1});

mAttentionMaskVarVec.resize(decode_type_num);
mPositionIdsVarVec.resize(decode_type_num);
```

然后针对 decode 场景预写好常用 `mask` 和 `position id` `Tensor`。这么做的目的很直接：避免单 token decode 时反复重新建图、重新分配小 Tensor。

## 6. 加载结束时内存里已经有什么

`load()` 成功返回时，程序里已经不只是“有一个模型”这么简单，而是已经具备了完整的推理起跑状态：

### 6.1 已经就绪的核心对象

- `mRuntimeManager`：已绑定好后端、线程、mmap、KV hint；
- `mTokenizer`：已能编码/解码和套用模板；
- `mDiskEmbedding`：已能按 token 从磁盘拉 embedding；
- `mSampler`：已按配置搭好采样流水线；
- `mModule`：已完成图加载；
- `mModulePool`：已准备好 prefill/decode 所需模块副本；
- `mMeta`：已带上层数、KV 状态、attention scale；
- `mGenerationStrategy`：已选好 AR / Lookahead / MTP / Eagle。

### 6.2 但还没有真正做的事情

加载完成后，下面这些事情还没发生：

- 还没对用户 prompt 做 tokenizer 编码；
- 还没真正执行 prefill；
- 还没生成第一个 token；
- 还没产生任何有效 KV Cache 内容。

这里可以把边界划清楚：加载阶段负责把推理环境装好，真正把 prompt 喂进去并开始产出 token 是下一篇[LLM 推理流程](../llm-infer/index.md)的内容。
