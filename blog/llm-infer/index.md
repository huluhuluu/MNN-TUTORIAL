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

前一篇[LLM 加载流程](../llm-load/index.md)停在 `load()` 结束，这一篇继续往下看真正的生成路径。当前仓库里的推理过程可以直接拆成两段：`Prefill` 负责把整段 prompt 写入 KV Cache，`Decode` 负责围绕已有 KV 做增量生成。下面按 `transformers/llm/engine/src/llm.cpp` 和 `speculative_decoding/generate.cpp` 的代码路径展开。

这里也先不急着看零散函数，先看一遍普通文本生成的主调用流程。把这条链路记住，再往下对照代码会顺很多。

```text
`response(prompt / chat)`
    ↓
`apply_chat_template(...)`
    ↓
`tokenizer_encode(...)`
    ↓
`generate_init(...)`
    ↓
`generate(input_ids, max_new_tokens)`
    ↓
`embedding(input_ids)`
    ↓
`forwardVec(input_embeds)`
    ↓
`gen_attention_mask(...)` / `gen_position_ids(...)`
    ↓
`forwardRaw(...)`
    ↓
`Module::onForward(inputs)`
    ↓
`sample(logits)`
    ↓
`tokenizer_decode(token)`
    ↓
`forwardVec({token})`
    ↓
循环直到 `stop / timeout / max_new_tokens`
```

可以把这条链分成三段来看：

- 文本前端：模板拼接和 tokenizer；
- `Prefill`：首轮整段 prompt 前向；
- `Decode`：围绕已有 KV 的单 token 递推。

后面各节就按这三段顺序展开，不再一上来就扎进单个函数细节。

- [LLM 推理流程](#llm-推理流程)
  - [1. 从 response 到 generate](#1-从-response-到-generate)
  - [2. Token 处理流程](#2-token-处理流程)
    - [2.1 prompt 模板与 tokenizer](#21-prompt-模板与-tokenizer)
    - [2.2 embedding：token id 如何变成输入向量](#22-embeddingtoken-id-如何变成输入向量)
    - [2.3 prefill：一次性吃下整段 prompt](#23-prefill一次性吃下整段-prompt)
    - [2.4 decode：按 token 递推生成](#24-decode按-token-递推生成)
  - [3. attention mask 与 position ids](#3-attention-mask-与-position-ids)
    - [3.1 attention mask](#31-attention-mask)
    - [3.2 position ids](#32-position-ids)
  - [4. KV Cache 管理](#4-kv-cache-管理)
    - [4.1 KVMeta 是状态核心](#41-kvmeta-是状态核心)
    - [4.2 reuse\_kv、prompt\_cache 和 prefix cache](#42-reuse_kvprompt_cache-和-prefix-cache)
      - [`reuse_kv`](#reuse_kv)
      - [`prompt_cache`](#prompt_cache)
      - [prefix cache](#prefix-cache)
    - [4.3 分块 prefill 与长上下文处理](#43-分块-prefill-与长上下文处理)
  - [5. 整体调用链总结](#5-整体调用链总结)

## 1. 从 response 到 generate

文本生成最常见的入口是：

```cpp
// transformers/llm/engine/src/llm.cpp
llm->response(prompt);
```

字符串版本的 `response()` 很薄，主要做两件事：

```cpp
// transformers/llm/engine/src/llm.cpp
auto prompt = user_content;
if (mConfig->use_template()) {
    prompt = apply_chat_template(user_content);
}
std::vector<int> input_ids = tokenizer_encode(prompt);
response(input_ids, os, end_with, max_new_tokens);
```

然后进入生成入口：

```cpp
// transformers/llm/engine/src/llm.cpp
generate_init(os, end_with);
generate(input_ids, max_new_tokens);
```

顶层推理路径可以先按下面理解：

```text
response(string/chat)
    ↓
套模板
    ↓
tokenizer_encode
    ↓
generate_init
    ↓
generate(input_ids)
    ↓
prefill
    ↓
decode loop
```

如果继续顺着 `llm.cpp` 往下看，这里还会再分成三段：

- `response(...)` 负责模板拼接和 token 化；
- `generate(...)` 负责把 token 变成 embedding，并执行 `Prefill`；
- `GenerationStrategy::generate(...)` 负责围绕已有 logits 和 KV 进入 `Decode` 循环。

也就是说，表面上只有一个 `response()`，底下实际已经拆成“文本前端”“模型前向”“采样递推”三段逻辑。

## 2. Token 处理流程

### 2.1 prompt 模板与 tokenizer

如果启用了模板，用户输入不会直接送给 tokenizer，而是先过一层模板：

```cpp
// transformers/llm/engine/src/llm.cpp
std::string Llm::apply_chat_template(const std::string& user_content) const {
    return mTokenizer->apply_chat_template(user_content, mConfig->system_prompt());
}
```

对多轮消息则是：

```cpp
// transformers/llm/engine/src/llm.cpp
std::string Llm::apply_chat_template(const ChatMessages& chat_prompts) const {
    return mTokenizer->apply_chat_template(chat_prompts, true);
}
```

接着进入：

```cpp
// transformers/llm/engine/src/llm.cpp
std::vector<int> Llm::tokenizer_encode(const std::string& user_content) {
    return mTokenizer->encode(user_content);
}
```

因此 Token 处理的第一步不是“编码字符串”，而是：

1. 先把聊天历史按模板拼成完整 prompt；
2. 再把完整 prompt 编码成 token id。

### 2.2 embedding：token id 如何变成输入向量

拿到 `input_ids` 后，模型并不会直接吃 int token，而是先调用：

```cpp
// transformers/llm/engine/src/llm.cpp
VARP Llm::embedding(const std::vector<int>& input_ids) {
    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    mDiskEmbedding->embedding(input_ids, res->writeMap<float>());
    return res;
}
```

这里的 embedding 不是普通神经网络前向，而是 `DiskEmbedding` 直接查磁盘文件：

```cpp
// transformers/llm/engine/src/diskembedding.cpp
seek_read(mWeight.get(), mTokenSize, mWeightOffset + token * mTokenSize);
```

如果 embedding 权重是量化存储的，这里还会顺手完成反量化。`token id -> hidden_state` 这一步是在运行时手工完成的，不是图里的一层 `Gather`。

### 2.3 prefill：一次性吃下整段 prompt

先看 `Prefill` 子流程。这里的重点不是把每个函数都展开，而是先看清楚“整段 prompt 是怎么一次性写进 KV 的”：

```text
`input_ids`
    ↓
`embedding(input_ids)`
    ↓
`forwardVec(input_embeds)`
    ↓
`gen_attention_mask(seq_len)`
`gen_position_ids(seq_len)`
    ↓
`forwardRaw(...)`
    ↓
`Module::onForward(inputs)`
    ↓
得到首轮 `logits`
    ↓
`mMeta->sync()` + `updateContext(seqLen, 0)`
```

对照这条流程，再看 `generate(MNN::Express::VARP input_embeds, int max_tokens)` 的前半段：

```cpp
// transformers/llm/engine/src/llm.cpp
auto outputs = forwardVec(input_embeds);
updateContext(seqLen, 0);
mContext->prefill_us += _t.durationInUs();
```

`forwardVec(input_embeds)` 对应的就是这段主路径里“构造动态输入并进入执行层”：

```cpp
// transformers/llm/engine/src/llm.cpp
mMeta->add = seq_len;
auto attention_mask = gen_attention_mask(seq_len);
auto position_ids = gen_position_ids(seq_len);
auto res = forwardRaw(input_embeds, attention_mask, position_ids, extraArgs);
```

最后进入模型前向：

```cpp
// transformers/llm/engine/src/llm.cpp
std::vector<Express::VARP> outputs = selectModule->onForward(inputs);
```

这一跳往下就进入执行层了。对当前静态图路径，可以把它概括成：

```text
`Module::onForward(inputs)`
    ↓
`NetModule::onForward(...)`
    ↓
`StaticModule::onForward(...)`
    ↓
`Session::run()`
    ↓
`Pipeline::execute()`
    ↓
具体算子的 `Execution::onExecute()`
```

其中真正负责执行的是 `StaticModule::onForward()`：

```cpp
// express/module/StaticModule.cpp
std::vector<Express::VARP> StaticModule::onForward(const std::vector<Express::VARP>& inputs) {
    Variable::compute(inputs);
    ErrorCode code = NO_ERROR;
    if (runResize) {
        code = _resize(inputs);
    }
    if (NO_ERROR == code && runCompute) {
        code = _execute();
    }
    ...
}
```

这一层会先把输入 `VARP` 收敛成底层 `Tensor`，然后分两步：

- `_resize(inputs)`：根据当前输入 shape 绑定输入张量，并在需要时触发 `Session::resize()`；
- `_execute()`：最终调用 `mSession->run()`，真正执行整条流水线。

更底层的 `Module -> Session -> Pipeline` 装配过程，上一篇[LLM 加载流程](../llm-load/index.md)已经单独展开，这里只保留 `Prefill` 相关的执行主线。

Prefill 后，`mGenerateParam->outputs` 里就有第一轮 logits，接下来可以开始采样第一个新 token。

这说明 `Prefill` 除了算出第一轮 logits，还顺手完成了三件事：

- 把当前 prompt 写进 KV；
- 更新 `mContext->all_seq_len`；
- 把 `mGenerateParam->outputs`、`validLogitStart`、`validLogitSize` 等后续 decode 需要的状态准备好。

这里再往下补一层 `forwardRaw()`，就能看清为什么 `Prefill` 和 `Decode` 会走不同的模块实例：

```cpp
// transformers/llm/engine/src/llm.cpp
bool inDecode = mContext->gen_seq_len > 0;
bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
```

`forwardRaw()` 会根据当前是不是 `decode`、当前 `seq_len` 是多少、当前是否需要 `all_logits`，从 `mModulePool` 里选择不同模块：

- `prefill` 走 `mPrefillKey`
- 普通单 token `decode` 走 `(1, false)`
- speculative verify 则可能走 `(verify_length, true)`

所以从推理层往下看，`Prefill` 和 `Decode` 表面上都在调 `Module::onForward()`，但底下未必是同一个 `StaticModule/Session` 实例。

### 2.4 decode：按 token 递推生成

再看 `Decode`。它和 `Prefill` 最大的区别是：每轮只新增一个 token，但会持续复用已经存在的 KV。

先看单轮 `Decode` 子流程：

```text
上一步 `logits`
    ↓
`sample(...)`
    ↓
得到 `current_token`
    ↓
`tokenizer_decode(current_token)`
    ↓
`forwardVec({current_token})`
    ↓
`gen_attention_mask(1)` / `gen_position_ids(1)`
    ↓
`forwardRaw(...)`
    ↓
下一轮 `logits`
```

最基础的 decode 策略是 `ArGeneration::generate()`：

```cpp
// transformers/llm/engine/src/speculative_decoding/generate.cpp
while (len < max_token) {
    mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    mLlm->updateContext(0, 1);
    if (mLlm->is_stop(mContext->current_token)) {
        break;
    }
    auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
    auto outputs = mLlm->forwardVec({mContext->current_token});
    mLlm->updateContext(1, 0);
    len++;
}
```

这个循环的逻辑非常标准：

1. 对上一步 logits 采样；
2. 把 token 记进历史；
3. 判断是否命中 stop token；
4. 解码成字符串输出；
5. 把这个 token 再喂回模型做下一步前向。

注意这里的 `updateContext` 分成了两次：

- `updateContext(0, 1)`：表示“新生成了 1 个 token”；
- `updateContext(1, 0)`：表示“这个 token 已经被作为输入写进 KV Cache”。

这两个计数拆开后，MNN-LLM 才能区分：

- 当前已经生成了多少 token；
- 当前 KV 里一共有多少 token。

这里可以再补一句和 `Prefill` 的区别：`Prefill` 是整段输入一次性写入，`Decode` 是单 token 递推，每轮都只更新很小一段动态输入，但底层依旧会走 `forwardVec -> forwardRaw -> Module::onForward` 这条主链。

所以 `Decode` 的关键点不是“循环调用模型”这么简单，而是每一步都只新增一个 token，同时围绕已有 KV 更新 mask、位置和上下文状态。

如果把 `Decode` 和加载阶段的执行层结构对应起来，可以整理成：

```text
`ArGeneration::generate()`
    ↓
`Llm::sample(...)`
    ↓
`Llm::forwardVec({token})`
    ↓
`Llm::forwardRaw(...)`
    ↓
从 `mModulePool` 选择 decode 模块
    ↓
`NetModule::onForward(...)`
    ↓
`StaticModule::_resize()` / `_execute()`
    ↓
`Session::run()`
    ↓
`Pipeline::execute()`
    ↓
各算子的 `Execution::onExecute()`
```

这样再回头看 `Decode`，它并不是“上层 while 循环 + 一个黑盒前向”，而是每一轮都会完整穿过执行层，只不过输入长度已经缩成了单 token。

## 3. attention mask 与 position ids

这两个输入是理解 LLM 推理最重要的动态张量。

### 3.1 attention mask

`gen_attention_mask(seq_len)` 会根据配置决定 mask 的形态。

如果是 float mask，并且是普通 causal attention，大致逻辑是：

```cpp
// transformers/llm/engine/src/llm.cpp
attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < kv_seq_len; j++) {
        ptr[kv_seq_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
    }
}
```

如果是 decode 单 token，CPU 上还会走一个特化路径，复用更小的 mask 变量，减少内存和构图开销。

此外它还支持：

- `attention_type = mix`：同时生成 full attention 和 sliding window attention；
- `attention_mask = glm/glm2`：按 ChatGLM 的规则生成 int mask；
- 分块 prefill：`kv_seq_len = all_seq_len + seq_len`。

### 3.2 position ids

`gen_position_ids(seq_len)` 的逻辑和 mask 一样，也是按模型类型分支：

```cpp
// transformers/llm/engine/src/llm.cpp
if (mConfig->attention_mask() == "glm") {
    ...
} else {
    if (seq_len == 1) {
        ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i + mContext->all_seq_len;
        }
    }
}
```

如果配置了 `is_mrope()`，则会生成三路 position ids：

```cpp
positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
```

所以 position ids 并不是一个固定写死的张量，它和：

- 当前是否在 prefill / decode；
- 当前总上下文长度；
- 模型是否为 GLM / MRoPE；

都有关系。

## 4. KV Cache 管理

### 4.1 KVMeta 是状态核心

KV 状态核心是 `KVMeta`：

```cpp
// source/core/KVMeta.hpp
struct KVMeta {
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
    float attn_scale = 0.0f;
    void sync() {
        previous = previous - remove + add + revertNumber;
        ...
    }
};
```

这个结构不是实际存 KV 数据，而是描述“这轮前向之后，KV 长度应该怎么变化”：

- `previous`：上一次已经提交的 KV 长度；
- `add`：本轮新增多少 token；
- `remove`：本轮要删掉多少 token；
- `reserve/n_reserve`：做历史裁剪时要保留哪几段；
- `file_name/file_flag`：prefix cache 落盘/读盘相关信息。

每次 `forwardRaw()` 结束后都会调用：

```cpp
// transformers/llm/engine/src/llm.cpp
mMeta->sync();
```

因此 KV 状态推进不是散落在各处，而是集中通过 `KVMeta` 完成。

### 4.2 reuse_kv、prompt_cache 和 prefix cache

这里要把三套“复用历史”的机制分开看。

#### `reuse_kv`

这是最直接的一种，控制是否在新请求开始时清空内存中的 KV：

```cpp
// transformers/llm/engine/src/llm.cpp
if (!mConfig->reuse_kv()) {
    mContext->all_seq_len = 0;
    mContext->history_tokens.clear();
    mMeta->remove = mMeta->previous;
}
```

#### `prompt_cache`

这是文本级别的增量 Prefill。它会比较本轮 prompt 和上轮 prompt 的公共前缀，只对 delta 部分重新 prefill：

```cpp
// transformers/llm/engine/src/llm.cpp
auto prompt_for_compare = mTokenizer->apply_chat_template(chat_prompts, false);
while (text_common < text_max && mCachedPromptText[text_common] == prompt_for_compare[text_common])
    text_common++;
```

然后再把公共前缀映射到 token 边界，构造 `delta` token 列表，只喂增量部分。

这套机制的重点是：**它不是盲目复用 KV，而是先比较文本前缀，再决定能复用多少**。

#### prefix cache

这是磁盘级别的 KV 缓存，通过 `setPrefixCacheFile()` 开启：

```cpp
// transformers/llm/engine/src/llm.cpp
mMeta->file_name = mPrefixCacheFileName;
mMeta->file_flag = KVMeta::PendingWrite; // or PendingRead
```

第一次运行可以把 prefix KV 写成 `.k/.v` 文件，下一次则可以直接读回。这个机制主要服务于固定系统 prompt 或固定长前缀场景。

### 4.3 分块 prefill 与长上下文处理

如果配置了 `chunk` 或 `chunk_limits`，`forwardVec(input_embeds)` 不会一次性处理整个 prompt，而是：

1. 把 embedding 按 block 切开；
2. 每个 block 单独生成 mask / position ids；
3. 逐块执行前向；
4. 最后一块必要时做 pad。

对应代码大致是：

```cpp
// transformers/llm/engine/src/llm.cpp
auto blockNumber = seq_len / mBlockSize;
auto blockRemain = seq_len % mBlockSize;
embeddings = MNN::Express::_Split(input_embeds, sizeSplits);
for (int i=0; i<blockNumber; ++i) {
    mMeta->add = blockSize;
    logits = forwardRaw(embed, attention_mask, position_ids, blockExtraArgs);
    updateContext(blockSize, 0);
}
```

这套逻辑的目标很直接：控制超长 prompt 在 prefill 阶段的内存占用，同时兼容部分后端的固定长度执行模式。

## 5. 整体调用链总结

把整条链路合在一起，文本推理流程可以总结成：

```text
response(prompt/chat)
    ↓
apply_chat_template
    ↓
tokenizer_encode
    ↓
DiskEmbedding::embedding
    ↓
gen_attention_mask / gen_position_ids
    ↓
forwardRaw -> Module::onForward
    ↓
得到 prefill logits，并建立首轮 KV Cache
    ↓
GenerationStrategy::generate
    ↓
sample(logits)
    ↓
tokenizer_decode(token)
    ↓
forwardVec({token}) 做下一轮 decode
    ↓
KVMeta::sync() 推进 KV 状态
    ↓
循环直到 stop / timeout / max_new_tokens
```

这一层的重点可以压成一句话：`Prefill` 负责把 prompt 写进 KV，`Decode` 负责围绕 KV 做增量生成，而 `KVMeta`、`attention_mask` 和 `position_ids` 是这条链路里最关键的运行时状态。
