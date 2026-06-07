---
title: "LLM 配置"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-06-04T12:00:00+08:00
draft: false
description: "按当前仓库代码整理 MNN LLM 的配置加载、运行时选项和量化相关字段"
slug: "llm-config"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# LLM 配置

`MNN`的`LLM`的推理流程涉及很多配置信息，本篇梳理 `transformers/llm/engine/src/llmconfig.hpp` 和 `llm.cpp` 中提到到 `MNN LLM` 的配置项`config.json`，它同时负责模型文件定位、Runtime 选项、生成行为和导出阶段写入的模型元信息。 

- [LLM 配置](#llm-配置)
  - [1. 配置文件加载](#1-配置文件加载)
  - [2. 各项参数](#2-各项参数)
    - [2.1 模型文件](#21-模型文件)
    - [2.2 运行时配置](#22-运行时配置)
    - [2.3 推理行为配置](#23-推理行为配置)
  - [3. 配置实例](#3-配置实例)

## 1. 配置文件加载
通常一份部署目录中会包括：

```text
model_dir/
├── config.json        # 运行时和部署侧配置
├── llm_config.json    # 导出时写入的模型结构配置
├── llm.mnn            # 模型文件
├── llm.mnn.weight     # 权重文件
├── tokenizer.txt / tokenizer.mtok # tokenizer 文件
└── embeddings_bf16.bin            # 磁盘词向量表
```

配置入口类是 `LlmConfig`。构造函数里做了两步 merge：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:234
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
        config_.merge_and_clear(llm_config_);
    }
    mllm_config_ = config_.contains("mllm") ? config_["mllm"] : ujson::json();
}
```

主要包括：

1. **启动配置**：通常是用户显示传入的 `config.json`，包括运行时和部署侧配置；
2. **模型配置**：默认会继续读取同目录下的 `llm_config.json`，包括模型结构和导出元信息；
3. **多模态子配置**：如果有 `mllm` 字段，会单独保存给视觉/音频侧使用。

配置类的构造函数实际做的是：

1. 先读取用户传入的 `config.json`，并确定 `base_dir_`；
2. 再根据 `llm_config()` 找到同目录下的 `llm_config.json`；
3. 调用 `config_.merge_and_clear(llm_config_)` 把模型导出时写入的元信息合并进来；
4. 最后单独取出 `mllm` 子配置，给多模态路径复用。

## 2. 各项参数

`LlmConfig` 包含了各项参数，如模型文件参数、运行时参数等等，简单说明如下。

### 2.1 模型文件

参数中涉及模型文件的参数有: 模型文件、权重文件和 tokenizer 文件：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:279
std::string llm_model() const { return base_dir_ + config_.value("llm_model", "llm.mnn"); }
std::string llm_weight() const { return base_dir_ + config_.value("llm_weight", "llm.mnn.weight"); }
std::string tokenizer_file() const { return base_dir_ + config_.value("tokenizer_file", "tokenizer.txt"); }
```

除了主图和权重外，MNN-LLM 还会额外读这些资源：

| 配置项 | 默认值 | 用途 |
|--------|--------|------|
| `embedding_file` | `embeddings_bf16.bin` | 磁盘词向量表，`DiskEmbedding` 读取 |
| `context_file` | `context.json` | 聊天模板上下文，会合并到 `jinja.context` |
| `llm_config` | `llm_config.json` | 导出阶段写入的模型结构信息 |
| `npu_model_dir` | `""` | NPU 后端缓存/编译结果目录 |

如果聊天模板依赖额外上下文字段，这部分不是硬编码在引擎里的，而是通过 `context.json` 注入：

```cpp
// transformers/llm/engine/src/llm.cpp:246
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


`LlmConfig` 还定义了很多可选文件，用来支持多模态与扩展模型等，比如：

- `visual_model`：视觉子模型路径，用于图像/视频侧 embedding 或多模态编码。
- `audio_model`：音频子模型路径，用于语音输入相关的特征提取。
- `talker_model`：语音/多模态输出侧的 talker 子模型路径。
- `mtp_model`：MTP speculative decoding 使用的 draft 模型路径。
- `eagle_model`：Eagle speculative decoding 使用的 draft 模型路径。

这些字段覆盖多模态 Omni 模型、推测解码的 MTP / Eagle 草稿模型，以及 talker 一类的扩展分支。

### 2.2 运行时配置

运行时配置的核心目标是：**配置MNN后端、精度、内存策略等**, 常见字段有：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:343
std::string backend_type(bool mllm = false) const {
    if (mllm) return mllm_config_.value("backend_type", "cpu");
    return config_.value("backend_type", "cpu");
}
int thread_num(bool mllm = false) const {
    if (mllm) return mllm_config_.value("thread_num", 4);
    return config_.value("thread_num", 4);
}
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

此外，`Llm::setRuntimeHint()` 会继续把很多配置转成 Interpreter Hint：

```cpp
// transformers/llm/engine/src/llm.cpp:132
int legacyAttentionMode = mConfig->config_.value("quant_qkv", 8);
int attentionMode = mConfig->config_.value("attention_mode", legacyAttentionMode);
rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->config_.value("dynamic_option", 0));
rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
```

这些字段会直接影响底层 Runtime 的执行策略，例如：

1. `attention_mode` 代码读取的 attention 执行模式配置；
2. `quant_qkv` 兼容旧字段入口；
3. `dynamic_option` 会直接作为动态量化相关 Hint 传给 Interpreter。


### 2.3 推理行为配置

这部分配置决定“模型怎么生成token”,从配置到实际类参数设置的流程如下：

```text
`LlmConfig`
    ↓
`generate_init()`                 # `reuse_kv`
`Prompt::createPrompt()`         # `system_prompt` / `prompt_template` / `chat_template`
`Sampler::createSampler(...)`    # `temperature` / `topK` / `topP`
`forwardVec(...)`                # `chunk` / `chunk_limits`
`GenerationStrategyFactory`      # `speculative_type` / `eagle_model`
```

最基础的一组生成参数如下：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:325
int max_all_tokens() const { return config_.value("max_all_tokens", 2048); }
int max_new_tokens() const { return config_.value("max_new_tokens", 512); }
bool reuse_kv() const { return config_.value("reuse_kv", false); }
bool all_logits() const { return config_.value("all_logits", false); }
```

它们分别影响：

- 最长上下文和最长新生成 token 数；
- 多轮对话时是否复用内存中的 KV Cache；
- Prefill 是否保留全量 logits。

此外，LLM 的 prompt 模板不只是 `prompt_template` 一个字段，`LlmConfig` 里还定义了下面的模板：

- `system_prompt`：默认系统提示词内容。
- `system_prompt_template`：system 角色的模板。
- `user_prompt_template`：user 角色的模板。
- `assistant_prompt_template`：assistant 角色的模板。
- `chat_template`：完整对话模板，通常用于多轮消息拼接。
- `prompt_template`：单轮 prompt 包装模板，兼容较简单的输入格式。
- `use_template`：是否启用模板拼接。


采样相关配置则基本都在 `transformers/llm/engine/src/sampler.hpp`的`Sampler` 类里使用：

- `sampler_type`：采样器类型，例如 penalty / mixed 等策略入口。
- `temperature`：采样温度，控制 logits 分布平滑程度。
- `topK`：只在概率最高的前 K 个 token 中采样。
- `topP`：nucleus sampling 阈值，按累计概率截断候选 token。
- `minP`：按最大概率比例过滤过低概率 token。
- `tfsZ`：tail free sampling 参数。
- `typical`：typical sampling 参数。
- `repetition_penalty`：重复 token 惩罚。
- `presence_penalty`：出现过的 token 惩罚。
- `frequency_penalty`：按出现频率增加惩罚。
- `penalty_window`：重复惩罚的历史窗口长度。
- `logit_bias`：对指定 token 的 logits 做偏置调整。
- `banned_tokens`：禁止生成的 token 列表。

如果 prompt 很长，MNN-LLM 还支持分块 prefill：

```cpp
// transformers/llm/engine/src/llm.cpp:101
if (mConfig->config_.document.HasMember("chunk")) {
    mBlockSize = mConfig->config_.document["chunk"].GetInt();
}
if (mConfig->config_.document.HasMember("chunk_limits")) {
    ...
}
```

另外，推测解码相关配置包括：

- `speculative_type`：选择推测解码类型，例如 MTP / Eagle / lookahead。
- `draft_predict_length`：draft 阶段一次预测的 token 数。
- `draft_match_strictness`：主模型验证 draft token 时的匹配严格程度。
- `draft_selection_rule`：draft 候选 token 的选择规则。
- `ngram_match_maxlen`：lookahead / ngram 路径里最大匹配长度。
- `ngram_update`：是否更新 ngram 相关缓存。
- `draft_model`：通用 draft 模型路径。
- `mtp_model`：MTP draft 模型路径。
- `eagle_model`：Eagle draft 模型路径。

`GenerationStrategyFactory::create(...)` 会根据这些字段决定使用哪些解码策略：

- `ArGeneration`：自回归贪心解码策略。
- `LookaheadGeneration`：lookahead / ngram 相关生成策略。
- `MtpGeneration`：MTP 推测解码策略。
- `EagleGeneration`：Eagle3 推测解码策略。


## 3. 配置实例

以 [Qwen3-1.7B-MNN](https://huggingface.co/taobao-mnn/Qwen3-1.7B-MNN/blob/main/config.json) 的配置为例：
```json
{
    "llm_model": "llm.mnn",
    "llm_weight": "llm.mnn.weight",
    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "memory": "low",
    "sampler_type": "mixed",
    "mixed_samplers": [
        "penalty",
        "topK",
        "topP",
        "min_p",
        "temperature"
    ],
    "penalty": 1.1,
    "temperature": 0.6,
    "topP": 0.95,
    "topK": 20,
    "min_p": 0
}
```
这个配置里包括了 模型文件、运行时选项和采样策略的配置，实际运行时可以按需配置参数。
