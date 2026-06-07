---
title: "LLM 推理流程"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-05-24T23:00:00+08:00
draft: true
description: "按当前仓库代码梳理 MNN LLM 的 token 处理、Prefill/Decode 和 KV Cache 管理"
slug: "llm-infer"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# LLM 推理流程

- [LLM 推理流程](#llm-推理流程)
  - [1. 推理主链路](#1-推理主链路)
  - [2. 入口：从 prompt 到 input ids](#2-入口从-prompt-到-input-ids)
  - [3. Prefill：把 prompt 写进 KV](#3-prefill把-prompt-写进-kv)
    - [3.1 token id 变成 embedding](#31-token-id-变成-embedding)
    - [3.2 forwardVec 生成动态输入](#32-forwardvec-生成动态输入)
    - [3.3 forwardRaw 选择模块并执行](#33-forwardraw-选择模块并执行)
  - [4. Decode：围绕 KV 做增量生成](#4-decode围绕-kv-做增量生成)
    - [4.1 GenerationStrategy 选择](#41-generationstrategy-选择)
    - [4.2 AR decode 循环](#42-ar-decode-循环)
  - [5. 动态输入：attention mask 和 position ids](#5-动态输入attention-mask-和-position-ids)
    - [5.1 attention mask](#51-attention-mask)
    - [5.2 position ids](#52-position-ids)
  - [6. KV Cache 状态推进](#6-kv-cache-状态推进)
    - [6.1 KVMeta](#61-kvmeta)
    - [6.2 reuse_kv](#62-reuse_kv)
    - [6.3 prefix cache](#63-prefix-cache)
  - [7. 分块 prefill](#7-分块-prefill)
  - [8. 总览](#8-总览)

本篇按当前仓库 `transformers/llm/engine/src/llm.cpp` 和 `transformers/llm/engine/src/speculative_decoding/generate.cpp` 的真实代码路径，梳理一次文本推理从 `response()` 到 `Prefill`、`Decode`、KV Cache 更新的过程。

上一篇 [LLM 加载流程](../llm-load/index.md) 已经整理了 `Module::load(...) -> StaticModule -> Session -> Pipeline -> Execution` 的装配过程。这一篇只关心加载完成后，一次推理请求如何穿过这套执行层。

## 1. 推理主链路

推理主链路可以先按阶段看，后面再展开关键代码：

| 阶段 | 入口 | 主要动作 |
|------|------|----------|
| 文本前端 | `Llm::response(...)` | 套 chat template，调用 `tokenizer_encode(...)` 得到 `input_ids` |
| 请求初始化 | `generate_init(...)` | 清理请求级状态，按 `reuse_kv` 决定是否移除旧 KV |
| Prefill | `generate(input_ids, max_tokens)` | `embedding(input_ids)` 后调用 `forwardVec(...)`，把 prompt 写入 KV 并得到第一轮 logits |
| 动态输入 | `forwardVec(...)` | 设置 `mMeta->add`，生成 `attention_mask` 和 `position_ids` |
| 模型前向 | `forwardRaw(...)` | 按 `(seq_len, all_logits)` 选择 `Module`，执行 `Module::onForward(...)` |
| 状态提交 | `mMeta->sync()` | 把本轮 `add/remove/reserve` 提交到 KV 状态 |
| Decode | `GenerationStrategy::generate(...)` | 循环执行 `sample -> tokenizer_decode -> forwardVec({token})` |

这篇按这张表展开。执行层内部的 `StaticModule -> Session -> Pipeline -> Execution` 已经在 [LLM 加载流程](../llm-load/index.md) 里单独整理，这里只在需要理解输入输出时提到。

## 2. 入口：从 prompt 到 input ids

文本入口在 `Llm::response(const std::string& user_content, ...)`：

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

多轮对话入口也是同一条思路，只是 `prompt` 来自 `ChatMessages`：

```cpp
// transformers/llm/engine/src/llm.cpp:921
void Llm::response(const ChatMessages& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = mPrompt->applyTemplate(chat_prompts);
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}
```

真正进入生成前，会先执行 `generate_init(...)`：

```cpp
// transformers/llm/engine/src/llm.cpp:900
void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_ids, max_new_tokens);
}
```

`generate_init(...)` 会清理一次请求级状态：

```cpp
// transformers/llm/engine/src/llm.cpp:687
void Llm::generate_init(std::ostream* os, const char* end_with) {
    mContext->os = os;
    mContext->end_with = end_with;
    mContext->generate_str.clear();
    mContext->gen_seq_len = 0;
    mContext->prefill_us  = 0;
    mContext->decode_us   = 0;
    mContext->current_token = -1;
    mContext->sample_us = 0;
    mContext->status = LlmStatus::RUNNING;
    if (!mConfig->reuse_kv()) {
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
    }
    mContext->output_tokens.clear();
}
```

这里要注意 `reuse_kv()` 的影响。如果没有复用 KV，新请求开始时会把 `all_seq_len`、`history_tokens` 清掉，并通过 `mMeta->remove = mMeta->previous` 告诉底层 KV Cache：旧 KV 需要移除。

## 3. Prefill：把 prompt 写进 KV

`generate(const std::vector<int>& input_ids, int max_tokens)` 是 `Prefill` 的入口：

```cpp
// transformers/llm/engine/src/llm.cpp:753
std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    ...
    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end());
    if(!passExecute) {
        if (0 == mBlockSize || input_ids.size() <= mBlockSize) {
            auto hidden_states = embedding(input_ids);
            return generate(hidden_states, max_tokens);
        }
        ...
    }
    generate(max_tokens);
    mContext->prompt_len = static_cast<int>(input_ids.size());
    return mContext->output_tokens;
}
```

这里先把 prompt token 写入 `history_tokens`。如果没有命中 prefix cache 的跳过执行路径，就会先做 embedding，再进入 `generate(input_embeds, max_tokens)`。

### 3.1 token id 变成 embedding

`embedding(input_ids)` 在 `llm.cpp` 里只创建输出 `VARP`，实际查表在 `DiskEmbedding`：

```cpp
// transformers/llm/engine/src/llm.cpp:1002
VARP Llm::embedding(const std::vector<int>& input_ids) {
    int hidden_size = mConfig->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    mDiskEmbedding->embedding(input_ids, res->writeMap<float>());
    return res;
}
```

`DiskEmbedding::embedding(...)` 会按 token id 定位到 embedding 文件中的偏移：

```cpp
// transformers/llm/engine/src/diskembedding.cpp:96
void DiskEmbedding::embedding(const std::vector<int>& input_ids, float* dst) {
    ...
    seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
    ...
}
```

如果 embedding 是 int8 / int4 量化存储，这里会按 block 读取 `scale/zero` 并反量化到 `float`；如果是 fp16，则会读出 half 后转成 `float`。所以 `token id -> hidden_state` 这一步不是 MNN 图里的 `Gather` 算子，而是在图外由 `DiskEmbedding` 手工完成。

### 3.2 forwardVec 生成动态输入

`generate(input_embeds, max_tokens)` 先执行一次 `forwardVec(input_embeds)`：

```cpp
// transformers/llm/engine/src/llm.cpp:840
std::vector<int> Llm::generate(MNN::Express::VARP input_embeds, int max_tokens) {
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    mContext->prompt_len = seqLen;

    Timer _t;
    forwardVec(input_embeds);
    if(mGenerateParam->outputs.size() < 1) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    updateContext(seqLen, 0);
    mContext->prefill_us += _t.durationInUs();
    MNN::Express::ExecutorScope::Current()->gc();
    ...
}
```

没有分块时，`forwardVec(input_embeds)` 的路径很短：

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
    ...
}
```

这里有三个关键动作：

- `mMeta->add = seq_len`：告诉 KV 层本轮要新增 `seq_len` 个 token；
- `gen_attention_mask(seq_len)`：按当前上下文长度生成 mask；
- `gen_position_ids(seq_len)`：按当前上下文长度生成 position ids。

### 3.3 forwardRaw 选择模块并执行

真正进入模型前向的是 `forwardRaw(...)`：

```cpp
// transformers/llm/engine/src/llm.cpp:448
std::vector<Express::VARP> Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    bool inDecode = mContext->gen_seq_len > 0;
    bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
    auto seqLen = hiddenState->getInfo()->dim[mSeqLenIndex];
    int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
    isAllLogists = seqLenKey == 1 ? false : isAllLogists;
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
    std::shared_ptr<Module> selectModule = mModule;
    if (mValidBlockSize.empty()) {
        if(mModulePool.find(moduleKey) == mModulePool.end()) {
            mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
            mModulePool[moduleKey].reset(Module::clone(mModule.get()));
        }
        selectModule = mModulePool[moduleKey];
    }
    ...
}
```

`forwardRaw(...)` 会根据当前阶段选择不同模块实例：

- `Prefill`：`seqLenKey = mPrefillKey`；
- 普通 decode：`seqLenKey = 1`，并且 `isAllLogists=false`；
- speculative verify：可能选择 `verify_length`，并打开 `all_logits`。

随后它把输入整理成模型图需要的 4 个基础输入：

```cpp
// transformers/llm/engine/src/llm.cpp:466
if (isAllLogists) {
    logitsIndex = logitsAllIdx;
} else {
    logitsIndex = logitsLastIdx;
}
if (mMeta->add != seqLen) {
    logitsIndex = logitsAllIdx;
}

std::vector<Express::VARP> inputs {hiddenState, mask, inputPos, logitsIndex};
inputs.insert(inputs.end(), extraArgs.begin(), extraArgs.end());
std::vector<Express::VARP> outputs = selectModule->onForward(inputs);
```

所以推理图实际看到的是：

- `hiddenState`：embedding 后的输入；
- `mask`：attention mask；
- `inputPos`：position ids；
- `logitsIndex`：告诉图只取最后一个 logits，还是取全量 logits。

`selectModule->onForward(inputs)` 再往下就是加载篇里的执行层。这里不再重复展开，只需要记住：`forwardRaw(...)` 负责把 LLM 运行时生成的 `hiddenState / mask / inputPos / logitsIndex` 交给当前选中的 `Module`，真正的后端执行由 `StaticModule`、`Session` 和 `Pipeline` 接管。

前向结束后，`forwardRaw(...)` 会保存本轮输出，并推进 KV 状态：

```cpp
// transformers/llm/engine/src/llm.cpp:491
mGenerateParam->input_embeds = hiddenState;
mGenerateParam->outputs = outputs;

// transformers/llm/engine/src/llm.cpp:547
mMeta->sync();
return outputs;
```

这一步结束后，`Prefill` 已经完成：

- prompt 对应的 KV 已经写入；
- `mGenerateParam->outputs` 里保存了第一轮 logits；
- `mContext->all_seq_len` 在 `generate(input_embeds, ...)` 里通过 `updateContext(seqLen, 0)` 增加；
- 后续 `Decode` 可以直接从 `mGenerateParam->outputs[0]` 开始采样。

## 4. Decode：围绕 KV 做增量生成

### 4.1 GenerationStrategy 选择

解码策略在加载阶段已经通过工厂创建：

```cpp
// transformers/llm/engine/src/speculative_decoding/generate.cpp:19
std::shared_ptr<Generation> GenerationStrategyFactory::create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec) {
    if(canSpec) {
        if(config->speculative_type() == "lookahead") {
            res.reset(new LookaheadGeneration(llm, context, config));
        } else if(config->speculative_type() == "mtp") {
            res.reset(new MtpGeneration(llm, context, config));
        } else if(config->speculative_type() == "eagle") {
            res.reset(new EagleGeneration(llm, context, config));
        } else {
            res.reset(new ArGeneration(llm, context, config));
        }
    } else {
        res.reset(new ArGeneration(llm, context, config));
    }
    return res;
}
```

这篇先看最基础的 `ArGeneration`。`Lookahead`、`MTP`、`Eagle` 会在 speculative decoding 相关文章里单独拆。

### 4.2 AR decode 循环

`generate(input_embeds, max_tokens)` 完成 prefill 后，会进入：

```cpp
// transformers/llm/engine/src/llm.cpp:892
if (0 < max_tokens) {
    mGenerateParam->max_new_tokens = max_tokens;
    mGenerationStrategy->generate(*mGenerateParam);
}
```

普通 AR decode 的核心循环在 `ArGeneration::generate(...)`：

```cpp
// transformers/llm/engine/src/speculative_decoding/generate.cpp:42
void ArGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;
    int len = 0;
    while (len < max_token) {
        if(mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }
        mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
        mLlm->updateContext(0, 1);
        if (mLlm->is_stop(mContext->current_token)) {
            ...
            break;
        }
        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        ...
        auto outputs = mLlm->forwardVec({mContext->current_token});
        ...
        mLlm->updateContext(1, 0);
        mContext->decode_us += _t.durationInUs();
        len++;
    }
}
```

单轮 decode 可以按这个顺序理解：

1. 从上一轮 logits 采样得到 `current_token`；
2. 把 token 写入 `history_tokens` 和 `output_tokens`；
3. `updateContext(0, 1)`：生成 token 数加 1；
4. 如果 token 是 stop token，结束；
5. `tokenizer_decode(...)` 把 token 变成文本并输出；
6. `forwardVec({current_token})` 把这个 token 作为下一轮模型输入；
7. `updateContext(1, 0)`：KV 上下文长度加 1；
8. 下一轮继续从新的 logits 采样。

这里的两个 `updateContext(...)` 分工很明确：

```cpp
// transformers/llm/engine/src/llm.cpp:660
void Llm::updateContext(int seq_len, int gen_len) {
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len += gen_len;
}
```

- `all_seq_len` 描述 KV / 上下文里已经提交了多少 token；
- `gen_seq_len` 描述本轮 response 已经生成了多少 token。

所以 decode 不是简单 while 循环。每轮都要同时推进三类状态：

- 文本状态：`generate_str`、`output_tokens`；
- 模型输入状态：下一轮的 `input_embeds`、mask、position ids；
- KV 状态：`mMeta->add`、`mMeta->sync()`、`all_seq_len`。

## 5. 动态输入：attention mask 和 position ids

### 5.1 attention mask

`gen_attention_mask(seq_len)` 的第一行就是当前 KV 长度：

```cpp
// transformers/llm/engine/src/llm.cpp:1023
VARP Llm::gen_attention_mask(int seq_len) {
    int kv_seq_len = mContext->all_seq_len + seq_len;
    ...
}
```

如果配置是 float mask，并且 `attention_type == "mix"`，会生成 full attention 和 sliding window attention 两份 mask：

```cpp
// transformers/llm/engine/src/llm.cpp:1027
if (mConfig->attention_type() == "mix") {
    const int sliding_window = mConfig->sliding_window();
    attentionMask = _Input({2, 1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
    auto full_attn_ptr = attentionMask->writeMap<float>();
    ...
    auto sliding_attn_ptr = full_attn_ptr + seq_len * kv_seq_len;
    ...
}
```

普通 float mask 会在 CPU 上走一个轻量路径：

```cpp
// transformers/llm/engine/src/llm.cpp:1061
kv_seq_len = seq_len;
if (mAttentionMaskVarVec.size() > 0) {
    if(seq_len == 1) {
        return mAttentionMaskVarVec[0];
    }
    if (mAttentionMaskVarVec.size() > 1 && seq_len == mDraftLength) {
        return mAttentionMaskVarVec[1];
    }
}

// transformers/llm/engine/src/llm.cpp:1072
if (mConfig->backend_type() == "cpu") {
    attentionMask = _Input({}, NCHW, halide_type_of<float>());
    auto ptr = attentionMask->writeMap<float>();
    ptr[0] = 0;
} else {
    attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
    ...
}
```

也就是说，decode 单 token 时不会每次都创建完整 `[1, 1, 1, kv_seq_len]` mask。CPU 路径可以用一个标量 mask 表示 lower triangular 逻辑，部分固定长度 decode 还会复用加载阶段预分配的 `mAttentionMaskVarVec`。

非 float mask 会生成 int mask，主要服务 GLM / GLM2 这类模型：

```cpp
// transformers/llm/engine/src/llm.cpp:1088
attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
...
bool is_glm2 = mConfig->attention_mask() == "glm2";
ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
```

### 5.2 position ids

`gen_position_ids(seq_len)` 同样按模型类型分支。

GLM 分支生成 `{1, 2, seq_len}`：

```cpp
// transformers/llm/engine/src/llm.cpp:1117
if (mConfig->attention_mask() == "glm") {
    if (needNewVar(positionIds, 2, seq_len)) {
        positionIds = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
    }
    ...
}
```

普通模型在单 token decode 时会复用 `mPositionIdsVarVec[0]`：

```cpp
// transformers/llm/engine/src/llm.cpp:1136
bool is_glm2 = mConfig->attention_mask() == "glm2";
if (seq_len == 1) {
    auto ptr = mPositionIdsVarVec[0]->writeMap<int>();
    ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
    return mPositionIdsVarVec[0];
}
```

多 token prefill 则按当前上下文长度顺序写：

```cpp
// transformers/llm/engine/src/llm.cpp:1150
positionIds = _Input({seq_len}, NCHW, halide_type_of<int>());
auto ptr = positionIds->writeMap<int>();
for (int i = 0; i < seq_len; i++) {
    ptr[i] = i + mContext->all_seq_len;
}
```

所以 mask 和 position ids 都不是固定输入，而是由当前 `all_seq_len`、`gen_seq_len`、`seq_len`、模型类型和后端共同决定。

## 6. KV Cache 状态推进

### 6.1 KVMeta

`KVMeta` 定义在 `kvmeta.hpp`：

```cpp
// transformers/llm/engine/src/kvmeta.hpp:17
struct KVMeta {
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    std::vector<int> reserveHost;
    void sync();
};
```

它不是 KV 数据本身，而是 KV Cache 的“状态变更描述”：

- `previous`：上一轮已经提交的 KV 长度；
- `add`：这轮前向新增多少 token；
- `remove`：这轮需要删除多少历史 token；
- `reserve/n_reserve`：删除历史时保留的片段；
- `file_name/file_flag/seqlen_in_disk`：prefix cache 读写相关信息；
- `layer_index/layer_nums`：后端逐层处理 KV 时使用。

每次 `forwardRaw(...)` 结束都会调用：

```cpp
// transformers/llm/engine/src/llm.cpp:547
mMeta->sync();
```

`sync()` 的逻辑很直接：

```cpp
// transformers/llm/engine/src/llm.cpp:37
void KVMeta::sync() {
    int revertNumber = 0;
    for (int i=0; i<n_reserve; ++i) {
        revertNumber += reserve[2*i+1];
    }
    previous = previous - remove + add + revertNumber;
    n_reserve = 0;
    reserve = nullptr;
    remove = 0;
    add = 0;
}
```

也就是把本轮 `remove/add/reserve` 提交到 `previous`，然后清空临时字段。这样上层只需要在前向前设置 `mMeta->add/remove`，前向结束后统一 `sync()`。

### 6.2 reuse_kv

`reuse_kv` 控制新请求是否复用内存中的 KV：

```cpp
// transformers/llm/engine/src/llm.cpp:702
if (!mConfig->reuse_kv()) {
    mContext->all_seq_len = 0;
    mContext->history_tokens.clear();
    mMeta->remove = mMeta->previous;
}
```

如果 `reuse_kv=false`，新请求会清空上层上下文，并通过 `mMeta->remove` 告诉底层删除旧 KV。否则新的 prompt 会接在已有上下文后面。

### 6.3 prefix cache

prefix cache 是磁盘级 KV 复用，通过 `setPrefixCacheFile(...)` 开启：

```cpp
// transformers/llm/engine/src/llm.cpp:964
bool Llm::setPrefixCacheFile(const std::string& filename, int flag) {
    mPrefixCacheFileName = filename;
    mCallIndex = 0;
    mPrefixCacheMode = true;
    mIsPrefixFileExist = true;
    for(int i = 0; i < mConfig->layer_nums(); i++) {
        auto k_file = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i) + "_sync.k";
        ...
        auto v_file = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i) + "_sync.v";
        ...
    }
    return mIsPrefixFileExist;
}
```

在 `generate(input_ids, ...)` 里，第一次调用会根据文件是否存在决定写 cache 还是跳过执行：

```cpp
// transformers/llm/engine/src/llm.cpp:759
if(mPrefixCacheMode) {
    mCallIndex++;
    if(mCallIndex == 1) {
        passExecute = mIsPrefixFileExist;
        if(!mIsPrefixFileExist) {
            mMeta->file_name = mPrefixCacheFileName;
            mMeta->file_flag = KVMeta::PendingWrite;
        }
        mPrefixLength = input_ids.size();
    } else if(mCallIndex == 2) {
        if(mIsPrefixFileExist) {
            mMeta->file_name = mPrefixCacheFileName;
            mMeta->file_flag = KVMeta::PendingRead;
            mMeta->seqlen_in_disk = mPrefixLength;
        }
    }
}
```

这套逻辑主要服务固定长 prefix，例如固定 system prompt。第一次没有 cache 文件时写入；后续命中文件时可以读回 prefix KV，减少重复 prefill。

## 7. 分块 prefill

如果配置了 `chunk` / `chunk_limits`，`mBlockSize` 会生效。长 prompt 不会一次性 forward，而是按 block 切开：

```cpp
// transformers/llm/engine/src/llm.cpp:792
int total_size = (int)input_ids.size();
int loop_size = UP_DIV(total_size, mBlockSize);
for (int i = 0; i < loop_size; i++) {
    auto start = i * mBlockSize;
    auto end = (i+1) * mBlockSize;
    if (end >= total_size) {
        end = total_size;
    }
    std::vector<int> chunk_ids(input_ids.begin() + start, input_ids.begin() + end);
    auto input_embeds = embedding(chunk_ids);
    generate(input_embeds, 0);
}
```

`forwardVec(input_embeds)` 内部还有一层 block 逻辑，支持把 embedding 拆成多段，并在最后一段按 `mValidBlockSize` 做 pad：

```cpp
// transformers/llm/engine/src/llm.cpp:583
auto blockNumber = seq_len / mBlockSize;
auto blockRemain = seq_len % mBlockSize;
...
for (int i=0; i<blockNumber; ++i) {
    mMeta->add = blockSize;
    auto attention_mask = gen_attention_mask(blockSize);
    auto position_ids = gen_position_ids(blockSize);
    logits = forwardRaw(embed, attention_mask, position_ids);
    updateContext(blockSize, 0);
}
...
if (blockRemain != 0) {
    mMeta->add = blockRemain;
    ...
    logits = forwardRaw(input_embeds, attention_mask, position_ids);
}
```

分块 prefill 的目的很直接：控制长 prompt 的峰值内存，同时兼容某些后端的固定输入长度。最后一块如果被 pad，`forwardRaw(...)` 会把 `logitsIndex` 切到 `logitsAllIdx`，并通过 `validLogitStart/validLogitSize` 只采样真实 token 对应的 logits。

## 8. 总览

推理阶段几个对象的职责如下：

| 对象 | 作用 |
|------|------|
| `mPrompt` | 负责 chat template |
| `mTokenizer` | 负责 `encode / decode` |
| `mDiskEmbedding` | 负责 `token id -> hidden_state` |
| `mModulePool` | 按 `(seq_len, all_logits)` 保存不同模块实例 |
| `mGenerateParam` | 保存当前 logits、输入 embedding 和采样范围 |
| `mGenerationStrategy` | 负责 decode 循环 |
| `mContext` | 记录文本、token、耗时和 `all_seq_len / gen_seq_len` |
| `mMeta` | 把 KV Cache 的 `add/remove/reserve/read/write` 状态传给后端 |

这一层的重点是：`Prefill` 把 prompt 写进 KV 并得到第一轮 logits；`Decode` 每轮只新增一个 token，但每轮都会重新生成 mask / position ids，并通过 `KVMeta` 把新增 token 提交到底层 KV Cache。
