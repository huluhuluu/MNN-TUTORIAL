---
title: "LLM 配置"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-05-24T23:00:00+08:00
draft: true
description: "按当前仓库代码整理 MNN LLM 的配置加载、运行时选项和量化相关字段"
slug: "llm-config"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# LLM 配置

这篇直接按当前仓库里的 `transformers/llm/engine/src/llmconfig.hpp` 和 `llm.cpp` 来整理 MNN LLM 的配置项。这里的配置不只是一个 `config.json`，它同时负责模型文件定位、Runtime 选项、生成行为和导出阶段写入的模型元信息。

先不要一上来就陷进字段细节里。按当前仓库代码，这部分最适合先看“配置是如何被消费的”，也就是先看一遍调用流程，再对照流程分析各类字段。

```text
`config.json` / `llm_config.json`
    ↓
`LlmConfig` 构造并 merge
    ↓
路径类配置
    ↓
`Llm::load()` 读取模型、tokenizer、context
    ↓
运行时配置
    ↓
`initRuntime()` / `setRuntimeHint()`
    ↓
推理行为配置
    ↓
`Sampler::createSampler()` / `generate_init()` / `GenerationStrategy`
    ↓
量化与存储配置
    ↓
`DiskEmbedding` / `mmap` / `KV Cache`
```

后面这篇就按这条链路往下拆：先看配置文件怎么合并，再看路径、运行时、生成行为和量化信息分别在哪一层被消费。

- [LLM 配置](#llm-配置)
  - [1. 配置文件是如何加载的](#1-配置文件是如何加载的)
  - [2. 模型与资源路径配置](#2-模型与资源路径配置)
    - [2.1 主模型文件](#21-主模型文件)
    - [2.2 Embedding 与上下文文件](#22-embedding-与上下文文件)
    - [2.3 多模态与扩展模型](#23-多模态与扩展模型)
  - [3. 运行时配置](#3-运行时配置)
    - [3.1 后端与线程](#31-后端与线程)
    - [3.2 Runtime Hint](#32-runtime-hint)
    - [3.3 mmap 与外部目录](#33-mmap-与外部目录)
  - [4. 推理行为配置](#4-推理行为配置)
    - [4.1 生成长度与 KV 行为](#41-生成长度与-kv-行为)
    - [4.2 Prompt 模板与采样配置](#42-prompt-模板与采样配置)
    - [4.3 分块与 speculative decoding](#43-分块与-speculative-decoding)
  - [5. 量化与存储配置](#5-量化与存储配置)
    - [5.1 词向量/权重文件的量化元信息](#51-词向量权重文件的量化元信息)
    - [5.2 运行时动态量化/Attention 选项](#52-运行时动态量化attention-选项)
    - [5.3 离线特征量化工具](#53-离线特征量化工具)
  - [6. 一份最小可运行配置](#6-一份最小可运行配置)

## 1. 配置文件是如何加载的

配置入口类是 `LlmConfig`。构造函数里做了两步 merge：

```cpp
// transformers/llm/engine/src/llmconfig.hpp
LlmConfig(const std::string& path) {
    if (has_suffix(path, ".json")) {
        std::ifstream config_file(path);
        config_ = ujson::json::parse(...);
        base_dir_ = base_dir(path);
    } else {
        // 兼容旧用法
    }
    base_dir_ = config_.value("base_dir", base_dir_);

    std::ifstream llm_config_file(llm_config());
    if (llm_config_file.is_open()) {
        auto llm_config_ = ujson::json::parse(...);
        config_.merge(llm_config_);
    }
    mllm_config_ = config_.contains("mllm") ? config_["mllm"] : ujson::json();
}
```

所以配置分成三层理解最清楚：

1. **启动配置**：通常是用户传入的 `config.json`；
2. **模型配置**：默认会继续读取同目录下的 `llm_config.json`；
3. **多模态子配置**：如果有 `mllm` 字段，会单独保存给视觉/音频侧使用。

这意味着一份部署目录里，通常至少会有：

```text
model_dir/
├── config.json        # 运行时和部署侧配置
├── llm_config.json    # 导出时写入的模型结构配置
├── llm.mnn
├── llm.mnn.weight
├── tokenizer.txt / tokenizer.mtok
└── embeddings_bf16.bin
```

这里可以直接区分成两层：`config.json` 更偏运行时和部署侧配置，`llm_config.json` 更偏模型结构和导出元信息。

继续按当前代码往下看，这个构造函数实际做的是：

1. 先读取用户传入的 `config.json`，并确定 `base_dir_`；
2. 再根据 `llm_config()` 找到同目录下的 `llm_config.json`；
3. 调用 `config_.merge(llm_config_)` 把模型导出时写入的元信息合并进来；
4. 最后单独取出 `mllm` 子配置，给多模态路径复用。

因此 `LlmConfig` 不是单纯保存一份 JSON，而是把部署配置、模型结构信息和多模态子配置整理成了统一的运行时视图。后面 `Llm::load()`、`DiskEmbedding`、`Sampler`、`RuntimeManager` 读到的都是这份 merge 之后的配置。

## 2. 模型与资源路径配置

`LlmConfig` 里有一组纯路径配置，用来告诉 `Llm::load()` 去哪里找文件。

### 2.1 主模型文件

最核心的三个路径如下：

```cpp
// transformers/llm/engine/src/llmconfig.hpp
std::string llm_model() const { return base_dir_ + config_.value("llm_model", "llm.mnn"); }
std::string llm_weight() const { return base_dir_ + config_.value("llm_weight", "llm.mnn.weight"); }
std::string tokenizer_file() const { return base_dir_ + config_.value("tokenizer_file", "tokenizer.txt"); }
```

在 `Llm::load()` 里，这三者是硬依赖：

```cpp
// transformers/llm/engine/src/llm.cpp
if (!checkFile(tokenizer_path, "tokenizer file") ||
    !checkFile(model_path, "LLM model file") ||
    !checkFile(weight_path, "LLM weight file")) {
    return false;
}
```

对纯文本 LLM，最少要准备：

- `llm_model`
- `llm_weight`
- `tokenizer_file`

### 2.2 Embedding 与上下文文件

除了主图和权重，MNN-LLM 还会额外读这些资源：

| 配置项 | 默认值 | 用途 |
|--------|--------|------|
| `embedding_file` | `embeddings_bf16.bin` | 磁盘词向量表，`DiskEmbedding` 读取 |
| `context_file` | `context.json` | 聊天模板上下文，会合并到 `jinja.context` |
| `llm_config` | `llm_config.json` | 导出阶段写入的模型结构信息 |
| `npu_model_dir` | `""` | NPU 后端缓存/编译结果目录 |

`context_file` 经常会被忽略，但当前代码里它确实会在 `Llm::load()` 中继续读取并 merge：

```cpp
// transformers/llm/engine/src/llm.cpp
std::ifstream contextFile(mConfig->context_file());
if (contextFile.is_open()) {
    auto contextJson = ujson::json::parse(contextStr);
    std::string config_json = R"({
        "jinja": {
            "context": )" + contextStr + R"(
        }
    })";
    mConfig->config_.merge(ujson::json::parse(config_json));
}
```

如果聊天模板依赖额外上下文字段，这部分不是硬编码在引擎里的，而是通过 `context.json` 注入。

### 2.3 多模态与扩展模型

`LlmConfig` 还定义了很多可选文件，比如：

- `visual_model`
- `audio_model`
- `talker_model`
- `mtp_model`
- `eagle_model`
- `ple_embed_file`

`LlmConfig` 不只服务于普通自回归文本模型，也覆盖：

- 多模态 Omni 模型；
- speculative decoding 的 MTP / Eagle 草稿模型；
- Gemma4 这类带 PLE embedding 的模型。

## 3. 运行时配置

运行时配置的核心目标是：**告诉 MNN 用什么后端、什么精度、什么内存策略去执行模型**。

### 3.1 后端与线程

运行时最常见的是这组字段：

```cpp
// transformers/llm/engine/src/llmconfig.hpp
std::string backend_type() const { return config_.value("backend_type", "cpu"); }
int thread_num() const { return config_.value("thread_num", 4); }
std::string precision() const { return config_.value("precision", "low"); }
std::string power() const { return config_.value("power", "normal"); }
std::string memory() const { return config_.value("memory", "low"); }
```

它们会在 `Llm::initRuntime()` 中转换成 `ScheduleConfig + BackendConfig`：

```cpp
// transformers/llm/engine/src/llm.cpp
ScheduleConfig config;
config.type      = backend_type_convert(mConfig->backend_type());
config.numThread = mConfig->thread_num();
config.backendConfig = &cpuBackendConfig;
```

然后再通过 `Executor::RuntimeManager::createRuntimeManager(config)` 创建运行时。

### 3.2 Runtime Hint

`Llm::setRuntimeHint()` 会继续把很多配置转成 Interpreter Hint：

```cpp
// transformers/llm/engine/src/llm.cpp
int legacyAttentionMode = mConfig->config_.value("quant_qkv", 8);
int attentionMode = mConfig->config_.value("attention_mode", legacyAttentionMode);
rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->config_.value("dynamic_option", 0));
rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
```

这里有三个非常关键的点：

1. `attention_mode` 是当前代码真正读取的 attention 执行模式配置；
2. `quant_qkv` 只是旧字段兼容入口；
3. `dynamic_option` 会直接作为动态量化相关 Hint 传给 Interpreter。

这些字段不是停留在 JSON 层，而是会直接影响底层 Runtime 的执行策略。

### 3.3 mmap 与外部目录

和大模型部署关系最大的，通常是 mmap 相关配置：

```cpp
bool use_mmap() const { return config_.value("use_mmap", false); }
bool use_cached_mmap() const { return config_.value("use_cached_mmap", true); }
bool kvcache_mmap() const { return config_.value("kvcache_mmap", false); }
std::string tmp_path() const { return config_.value("tmp_path", ""); }
std::string prefix_cache_path() const { return config_.value("prefix_cache_path", "prefixcache"); }
int mmap_size() const { return config_.value("mmap_size", 1024); }
```

这些配置最终对应到：

- 权重是否通过外部文件映射加载；
- KV Cache 是否落盘/映射；
- Prefix Cache 写到哪个目录；
- mmap 预留文件大小。

在 `setRuntimeHint()` 中可以看到真实映射关系：

```cpp
// transformers/llm/engine/src/llm.cpp
if (mConfig->kvcache_mmap()) {
    rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
}
rtg->setExternalPath(cachePath, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
if (mConfig->use_mmap()) {
    rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
}
rtg->setHint(MNN::Interpreter::MMAP_FILE_SIZE, mConfig->mmap_size());
```

## 4. 推理行为配置

这部分配置不决定“模型是什么”，而决定“模型怎么生成”。

先看这一层的配置消费流程：

```text
`LlmConfig`
    ↓
`generate_init()`                 # `reuse_kv` / `prompt_cache`
`setChatTemplate()`              # `prompt_template` / `chat_template`
`Sampler::createSampler(...)`    # `temperature` / `topK` / `topP`
`forwardVec(...)`                # `chunk` / `chunk_limits`
`GenerationStrategyFactory`      # `speculative_type` / `eagle_model`
```

后面各小节就按这条链路往下对照。

### 4.1 生成长度与 KV 行为

最基础的一组如下：

```cpp
// transformers/llm/engine/src/llmconfig.hpp
int max_all_tokens() const { return config_.value("max_all_tokens", 2048); }
int max_new_tokens() const { return config_.value("max_new_tokens", 512); }
bool reuse_kv() const { return config_.value("reuse_kv", false); }
bool prompt_cache() const { return config_.value("prompt_cache", false); }
bool all_logits() const { return config_.value("all_logits", false); }
```

它们分别影响：

- 最长上下文和最长新生成 token 数；
- 多轮对话时是否复用内存中的 KV Cache；
- 是否开启文本级 Prompt Cache；
- Prefill 是否保留全量 logits。

其中 `reuse_kv` 在 `generate_init()` 中最明显：

```cpp
// transformers/llm/engine/src/llm.cpp
if (!mConfig->reuse_kv()) {
    mContext->all_seq_len = 0;
    mContext->history_tokens.clear();
    mMeta->remove = mMeta->previous;
}
```

如果它为 `false`，每次新请求都会清空历史 KV；如果为 `true`，则允许连续复用。

### 4.2 Prompt 模板与采样配置

LLM 的 prompt 模板不只是 `prompt_template` 一个字段，`LlmConfig` 里还定义了：

- `system_prompt`
- `system_prompt_template`
- `user_prompt_template`
- `assistant_prompt_template`
- `chat_template`
- `prompt_template`
- `use_template`

如果 tokenizer 支持 jinja 模板，`Llm::setChatTemplate()` 会把这些信息交给 tokenizer。

采样相关配置则基本都在 `Sampler` 里消费：

- `sampler_type`
- `temperature`
- `topK`
- `topP`
- `minP`
- `tfsZ`
- `typical`
- `repetition_penalty`
- `presence_penalty`
- `frequency_penalty`
- `penalty_window`
- `logit_bias`
- `banned_tokens`

这部分会在 `Sampler::buildPipeline()` 中拼成一条采样流水线。

如果把这些配置重新放回当前仓库的实际调用路径里，可以整理成下面这条“配置消费链”：

```text
`LlmConfig`
    ↓
`Llm::load()`
    ↓
`setChatTemplate()`          # prompt / chat_template / system_prompt
`Sampler::createSampler()`   # topK / topP / temperature / penalty
`DiskEmbedding(...)`         # embedding_file / tie_embeddings / ple_quant
`initRuntime()`              # backend_type / thread_num / precision / memory
    ↓
`setRuntimeHint()`           # attention_mode / dynamic_option / mmap / kvcache
```

也就是说，配置字段虽然都定义在 `LlmConfig` 里，但真正消费它们的对象并不相同：

- 模板相关字段主要进入 `Tokenizer`；
- 采样相关字段主要进入 `Sampler`；
- 运行时字段主要进入 `RuntimeManager`；
- embedding 和量化元信息主要进入 `DiskEmbedding`。

### 4.3 分块与 speculative decoding

如果 prompt 很长，MNN-LLM 不一定一次性 prefill 完，它还支持分块：

```cpp
mBlockSize = mConfig->config_.value("chunk", 0);
if (mConfig->config_.contains("chunk_limits")) {
    ...
}
```

对应到推理时，`forwardVec()` 会把输入 embedding 按 block 切开逐段执行。

另外，speculative decoding 相关配置也在这里：

- `speculative_type`
- `draft_predict_length`
- `draft_match_strictness`
- `draft_selection_rule`
- `ngram_match_maxlen`
- `ngram_update`
- `draft_model`
- `mtp_model`
- `eagle_model`

`GenerationStrategyFactory::create(...)` 会根据这些字段决定使用：

- `ArGeneration`
- `LookaheadGeneration`
- `MtpGeneration`
- `EagleGeneration`

如果把这些生成相关字段继续和后面的执行路径对应起来，可以再整理成下面这张“配置到执行层”的关系表：

| 配置项 | 主要消费位置 | 对执行路径的影响 |
|--------|--------------|------------------|
| `all_logits` | `Llm::forwardRaw()` | 决定 `logitsIndex` 取最后一个位置还是全量位置，也影响 `mModulePool` key |
| `chunk` / `chunk_limits` | `Llm::forwardVec()` | 决定 `prefill` 是整段一次前向，还是按 block 拆成多次 `forwardRaw()` |
| `speculative_type` | `GenerationStrategyFactory::create()` | 决定 decode 阶段使用 `ArGeneration`、`MtpGeneration` 还是 `EagleGeneration` |
| `draft_predict_length` | speculative generation | 决定 speculative verify 阶段一次验证多少个候选 token |
| `eagle_model` / `eagle_fc` / `eagle_d2t` | `EagleGeneration::load()` | 决定 `Eagle` draft 分支加载哪几个额外模块 |

这几项配置看起来都只是 JSON 字段，但它们的影响并不止于“逻辑分支不同”，而是会进一步影响：

- `ModulePool` 里预先 clone 几份 `Module`
- `forwardRaw()` 选哪个模块实例
- `prefill` 是否拆成多次 `Session::run()`
- speculative 路径是否额外加载一套 draft `Module`

## 5. 量化与存储配置

“量化配置”在 MNN-LLM 里其实分成三层，不要只盯着一个字段看。

### 5.1 词向量/权重文件的量化元信息

`DiskEmbedding` 会读取 `tie_embeddings` 或 `ple_quant`：

```cpp
// transformers/llm/engine/src/diskembedding.cpp
auto tie_embeddings = quant_info.size() == 5 ? quant_info : config->tie_embeddings();
...
mWeightOffset = tie_embeddings[0];
mQuantBit     = tie_embeddings[3];
mQuantBlock   = tie_embeddings[4];
```

这说明导出器会把 embedding 的量化布局写进配置里，运行时再按这个元信息去反量化读取磁盘权重。

因此 `tie_embeddings` / `ple_quant` 更接近导出阶段生成的模型元数据，不适合按普通运行时参数来理解。

### 5.2 运行时动态量化/Attention 选项

`llm.cpp` 当前真正消费的运行时量化相关字段是：

- `attention_mode`
- `quant_qkv`（兼容旧名）
- `dynamic_option`

如果要调整 runtime 行为，重点看的是这几个字段，而不是旧样例里的 `quant_kv`。

### 5.3 离线特征量化工具

MNN-LLM 还有单独的离线量化工具 `transformers/llm/engine/tools/quantize_llm.cpp`。它不是简单改配置，而是：

1. 用一批 prompt 跑模型；
2. 统计中间 Tensor 的范围；
3. 计算 scale / zero；
4. 把 `TensorQuantInfo` 写回模型描述。

激活量化不是简单补几个 JSON 字段，而是要配合离线分析工具改写模型描述。

## 6. 一份最小可运行配置

结合 `transformers/llm/config.json` 和当前代码路径，最小文本模型配置可以写成这样：

```json
{
  "llm_model": "llm.mnn",
  "llm_weight": "llm.mnn.weight",
  "tokenizer_file": "tokenizer.txt",
  "embedding_file": "embeddings_bf16.bin",
  "backend_type": "cpu",
  "thread_num": 4,
  "precision": "low",
  "memory": "low",
  "power": "normal",
  "max_new_tokens": 512,
  "reuse_kv": false,
  "use_mmap": true
}
```

把这份配置和源码对应起来，可以直接按下面理解：

- `llm_model/llm_weight/tokenizer_file`：让 `Llm::load()` 能找到模型；
- `backend_type/thread_num/precision/memory/power`：让 `initRuntime()` 能正确创建 RuntimeManager；
- `max_new_tokens/reuse_kv`：决定生成阶段行为；
- `use_mmap`：决定权重是否通过映射方式加载。

这一层的重点不是记字段列表，而是看清楚配置如何把模型资源、Runtime、生成策略和量化元信息串起来。后面的加载流程可以接着看[LLM 加载流程](../llm-load/index.md)。
