---
title: "Eagle 推理流程"
date: 2026-05-24T23:00:00+08:00
lastmod: 2026-05-24T23:00:00+08:00
draft: true
description: "按当前仓库代码梳理 MNN LLM 中 Eagle speculative decoding 的加载、draft 生成与验证流程"
slug: "llm-eagle-infer"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# Eagle 推理流程

这是 `MNN-LLM` 里 `Eagle speculative decoding` 的代码路径整理。前面的 [LLM 加载流程](../llm-load/index.md) 和 [LLM 推理流程](../llm-infer/index.md) 主要讲的是普通 `AR decode`，这一篇单独看 `speculative_type = "eagle"` 时，`MNN` 是怎么额外加载 draft 模块、生成候选树、做 tree decoding，再把接受结果同步回主模型 KV 的。

- [Eagle 推理流程](#eagle-推理流程)
  - [1. Eagle 入口在哪里](#1-eagle-入口在哪里)
  - [2. Eagle 相关配置](#2-eagle-相关配置)
  - [3. Eagle 模块加载](#3-eagle-模块加载)
  - [4. Eagle 推理主流程](#4-eagle-推理主流程)
    - [4.1 从首 token 开始](#41-从首-token-开始)
    - [4.2 draft tree 生成](#42-draft-tree-生成)
    - [4.3 tree decoding 验证](#43-tree-decoding-验证)
    - [4.4 接受结果并回写 KV](#44-接受结果并回写-kv)
  - [5. Eagle 与普通 AR 的差异](#5-eagle-与普通-ar-的差异)

## 1. Eagle 入口在哪里

`LLM` 在加载阶段会统一创建生成策略对象，入口在：

```cpp
// transformers/llm/engine/src/speculative_decoding/generate.cpp
std::shared_ptr<Generation> GenerationStrategyFactory::create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec) {
    std::shared_ptr<Generation> res;
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

也就是说，只要：

- 当前运行环境允许 speculative decoding；
- 并且配置里 `speculative_type = "eagle"`；

后面的生成逻辑就不再走普通 `ArGeneration`，而是切到 `EagleGeneration`。

## 2. Eagle 相关配置

`Eagle` 路径依赖的主要配置在 `llmconfig.hpp`：

```cpp
// transformers/llm/engine/src/llmconfig.hpp
std::string speculative_type() const {
    return config_.value("speculative_type", "");
}
std::string eagle_model() const {
    return base_dir_ + config_.value("eagle_model", "eagle.mnn");
}
std::string eagle_fc() const {
    return base_dir_ + config_.value("eagle_fc", "eagle_fc.mnn");
}
std::string eagle_d2t() const {
    return base_dir_ + config_.value("eagle_d2t", "eagle_d2t.mnn");
}
int eagle_depth() const {
    return config_.value("eagle_depth", 3);
}
int eagle_topk() const {
    return config_.value("eagle_topk", 1);
}
```

这里几项配置的作用可以直接对应到实现：

- `eagle_model`：主 `Eagle` draft 模块，输入 `input_embed + hidden_states + mask + position_ids + logits_index`；
- `eagle_fc`：把主模型 hidden states 变成 `Eagle` draft 模块需要的 hidden states；
- `eagle_d2t`：draft token 到真实 token 的映射表；
- `eagle_depth`：draft tree 的深度；
- `eagle_topk`：每层扩展多少个候选分支。

## 3. Eagle 模块加载

`EagleGeneration::load()` 会在普通主模型之外，再额外加载两份模块和一份映射表：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
void EagleGeneration::load(Module::Config module_config) {
    mEagleMeta.reset(new KVMeta);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleMeta.get());

    std::vector<std::string> inputNames{"input_embed", "hidden_states", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits", "out_hidden_states"};
    mEagleModules.resize(2);
    mEagleModules[0].reset(Module::load(inputNames, outputNames, mLlm->mConfig->eagle_model().c_str(), mLlm->mRuntimeManager, &module_config));

    mEagleModules[1].reset(Module::load({"fc_hidden"}, {"hidden_states"}, mLlm->mConfig->eagle_fc().c_str(), mLlm->mRuntimeManager, &module_config));

    mD2t = Express::Variable::load(mLlm->mConfig->eagle_d2t().c_str())[0];

    mTopK = mLlm->mConfig->eagle_topk();
    mDepth = mLlm->mConfig->eagle_depth();
    mTreePosition = _Input({1, mTopK}, NCHW, halide_type_of<int>());
}
```

这一层可以直接理解成：

1. 主模型仍然由 `Llm::mModule` 负责；
2. `Eagle` 自己额外持有两份 `Module`：
   - `mEagleModules[0]`：draft logits + hidden states 模块
   - `mEagleModules[1]`：hidden state 投影模块
3. 同时维护自己独立的一份 `mEagleMeta`，单独管理 draft 路径上的 KV 状态。

所以 `Eagle` 并不是“在主模型 logits 上多做一个 topk”，而是另外挂了一条 draft 模型分支。

这里还要注意 `mEagleMeta` 这一层。`EagleGeneration::load()` 里会把：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
mEagleMeta.reset(new KVMeta);
mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleMeta.get());
```

也就是说，`Eagle` draft 模块不会直接复用主模型的 `mMeta`，而是自己维护一份 draft 路径上的 KV 状态。这样主模型 KV 和 draft 模型 KV 才能分别推进、分别回滚。

## 4. Eagle 推理主流程

`EagleGeneration::generate()` 的主流程在：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
void EagleGeneration::generate(GenerationParams& param) {
    mEaglePastLen = 0;
    mEagleRemove  = mEagleMeta->previous;
    ...
    auto sampleToken  = mLlm->sample(param.outputs[0]);
    ...
    auto draftInfo  = topkGenerate(inputIds, hiddenStates, inputEmbeds);
    while (true) {
        auto decodingInfo = treeDecoding(draftInfo);
        auto acceptInfo = evaluatePosterior(draftInfo, decodingInfo[0]);
        ...
        draftInfo = updateDraft(acceptInfo, decodingInfo[1]);
    }
}
```

如果把这段代码压成一条执行链，大致是：

```text
主模型 prefill logits
    ↓
先采样出第一个 token
    ↓
Eagle draft 生成候选树
    ↓
主模型 tree decoding 验证候选树
    ↓
选择接受路径
    ↓
把接受 token 回写主模型 KV
    ↓
再基于新的 hidden states 继续生成下一轮 draft
```

### 4.1 从首 token 开始

`Eagle` 不是从空状态直接建树，而是先消费主模型 `prefill` 后的第一轮 logits：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
auto sampleToken  = mLlm->sample(param.outputs[0]);
mContext->current_token = sampleToken;
mContext->history_tokens.push_back(mContext->current_token);
mContext->output_tokens.push_back(mContext->current_token);
mLlm->updateContext(0, 1);
```

然后把这个 token 对应的 embedding 拼到输入尾部，作为 `Eagle` draft 的起点：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
auto cur_embed  = mLlm->embedding({sampleToken});
auto pre_embeds = _Split(inputEmbeds, {1, seqLen - 1}, 0);
inputEmbeds     = _Concat({pre_embeds[1], cur_embed}, 0);
```

这说明 `Eagle` 路径仍然依赖主模型先给出首 token，后面的 speculative 部分是在这个基础上展开的。

### 4.2 draft tree 生成

draft tree 的生成入口是 `topkGenerate(...)`：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
EagleGeneration::DraftInfo EagleGeneration::topkGenerate(const std::vector<int>& inputIds, VARP hiddenStates, VARP inputEmbeds) {
    TokenTree tokenTree(mTopK, d2tPtr);
    ...
    auto inputHidden = mEagleModules[1]->forward(hiddenStates);
    outputs      = eagleForward(inputEmbeds, inputHidden);
    ...
    auto topKV = MNN::Express::_TopKV2(lastP, MNN::Express::_Scalar<int>(mTopK));
    tokenTree.init(indices, scores);
    ...
    for (int d = 0; d < mDepth - 1; d++) {
        inputEmbeds   = mLlm->embedding(tokenTree.getIds());
        auto attentionMask = getMask(tokenTree.getMask(), seqLen);
        outputs = eagleForwardRaw({inputEmbeds, inputHidden, attentionMask, mTreePosition, mLlm->logitsAllIdx});
        ...
        tokenTree.grow(indices, scores);
    }
    auto output = tokenTree.finalize(sampleToken, mLlm->mDraftLength);
    ...
}
```

这里的核心点有三个：

1. 先通过 `mEagleModules[1]` 把主模型 hidden states 映射到 `Eagle` draft 模型所需空间；
2. 再用 `mEagleModules[0]` 递归生成每层 top-k 候选；
3. 借助 `TokenTree` 累积候选树、mask、位置和最终检索索引。

这一步的输出不是简单一串草稿 token，而是一个 `DraftInfo`：

- `draftTokens`
- `retrieveIndices`
- `attentionMask`
- `positionIds`

也就是后面 tree decoding 验证整棵树所需的全部输入。

这里如果只看 `topkGenerate(...)` 里的主函数，还不太容易看出 `TokenTree` 到底在做什么。把 `tokentree.hpp` 的职责拆开会更清楚：

- `TokenTree::init(...)`
  用首轮 `topk` 结果创建第一层候选节点；
- `TokenTree::grow(...)`
  对当前活跃叶子节点继续展开下一层候选，并按累计 logprob 做裁剪，只保留最优 `topK` 条分支；
- `TokenTree::finalize(...)`
  把整棵候选树整理成主模型验证所需的四类输入：
  - `draftTokens`
  - `positionIds`
  - `attentionMask`
  - `retrieveIndices`

其中 `retrieveIndices` 很关键。它保存的是“每条候选路径在扁平化 draft token 序列中的索引路径”，后面 `evaluatePosterior(...)` 才能逐路径比较“主模型采样结果”和“draft tree 里的候选 token”是否一致。

### 4.3 tree decoding 验证

draft tree 建好以后，会进入 `treeDecoding(...)`：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
VARPS EagleGeneration::treeDecoding(const EagleGeneration::DraftInfo& drafInfo) {
    auto inputEmbeds   = mLlm->embedding(drafInfo.draftTokens);
    int inputLen = drafInfo.draftTokens.size();
    mLlm->mMeta->add = inputLen;
    auto outputs = mLlm->forwardRaw(inputEmbeds, drafInfo.attentionMask, drafInfo.positionIds);
    return outputs;
}
```

这一层做的事情很直接：把 draft tree 里的所有候选 token 一次性交给主模型，再让主模型跑一次 tree decoding，返回：

- `logits`
- 以及后续继续生成 draft 时需要的 hidden states

所以 `Eagle` 的核心思想不是“draft 模型直接决定最终输出”，而是：

- draft 模型负责提出一整棵候选树；
- 主模型负责一次性验证这棵树。

这里再补一层 `treeDecoding()` 里的 KV 语义：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
int inputLen = drafInfo.draftTokens.size();
mLlm->mMeta->add = inputLen;
auto outputs = mLlm->forwardRaw(inputEmbeds, drafInfo.attentionMask, drafInfo.positionIds);
```

这说明主模型在 tree decoding 阶段会先把“整棵 draft tree 扁平化后的 token 序列”都视为待验证输入，一次性把这些候选写进当前前向路径里。也正因为如此，后面才需要 `updateDraft()` 通过 `remove + reserve` 把 KV 收敛回真正接受的那条路径。

### 4.4 接受结果并回写 KV

主模型验证完 draft tree 后，会通过 `evaluatePosterior(...)` 选出最佳接受路径：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
auto sampleTokens = MNN::Express::_ArgMax(logits, -1);
...
if (samples[sampleIdx] != drafInfo.draftTokens[draftIdx]) {
    break;
}
...
acceptInfo.acceptIndices = std::move(bestCandidate);
acceptInfo.acceptTokens  = std::move(acceptTokens);
```

然后在 `updateDraft(...)` 里把接受结果回写到主模型 KV 状态：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp
mLlm->updateContext(acceptLen, acceptLen);
mLlm->mMeta->remove = acceptInfo.sampleTokens.size();
mLlm->mMeta->n_reserve = acceptLen;
mLlm->mMeta->reserve = new int[mLlm->mMeta->n_reserve * 2];
for (size_t i = 0; i < acceptLen; i++) {
    mLlm->mMeta->reserve[2 * i] = acceptInfo.acceptIndices[i];
    mLlm->mMeta->reserve[2 * i + 1] = 1;
}
```

这里的逻辑非常关键：

- `sampleTokens.size()` 表示这次 tree decoding 验证过的整棵候选树长度；
- `acceptLen` 表示最终真正接受了多少个 token；
- `remove + reserve` 这组 `KVMeta` 操作会把主模型 KV 从“整棵树的验证状态”收敛回“只保留接受路径”的状态。

随后 `updateDraft(...)` 还会用接受路径对应的 hidden states 继续调用 `topkGenerate(...)`，进入下一轮 draft tree 生成。

这里如果再顺着 `evaluatePosterior(...)` 和 `updateDraft(...)` 看一层，`Eagle` 的“验证后收敛”可以整理成下面这条链：

```text
`treeDecoding()` 得到整树 logits
    ↓
`_ArgMax(logits, -1)` 取每个位置的主模型预测
    ↓
`retrieveIndices` 逐路径比较
    ↓
选出最长匹配候选路径
    ↓
生成 `acceptIndices + acceptTokens`
    ↓
`mMeta->remove = sampleTokens.size()`
`mMeta->reserve = acceptIndices`
    ↓
主模型 KV 从“整树验证态”回收成“接受路径态”
```

这一层正是 `Eagle` 和普通 `AR decode` 差异最大的地方。普通 `AR` 不需要在一次前向后做路径回收；`Eagle` 因为先验证了一整棵候选树，所以必须显式告诉 `KVMeta`：

- 哪些候选位置只是验证用的，后面要删掉；
- 哪些位置属于真正接受路径，需要保留下来。

## 5. Eagle 与普通 AR 的差异

如果只和普通 `ArGeneration` 对比，`Eagle` 主要多了下面几层：

- 多加载了两份 draft 相关 `Module`；
- 维护了独立的 `mEagleMeta`；
- 多了一套 `TokenTree`、`DraftInfo`、`AcceptInfo` 数据结构；
- 生成时不再是“单 token -> 单 forward”，而是“draft tree -> tree decoding -> 接受路径”。

可以把两者的主路径对比成：

```text
普通 AR
`logits`
    ↓
`sample`
    ↓
单 token `forwardVec({token})`
    ↓
下一轮 `logits`


Eagle
`logits`
    ↓
`sample`
    ↓
`topkGenerate()` 生成 draft tree
    ↓
`treeDecoding()` 用主模型整树验证
    ↓
`evaluatePosterior()` 选接受路径
    ↓
`updateDraft()` 回写主模型 KV
    ↓
下一轮 draft tree
```

所以 `Eagle` 的收益点不在于替代主模型，而在于：

- 用更轻的 draft 路径快速提出多 token 候选；
- 再让主模型一次验证多 token；
- 尽量提高单次主模型前向所能确认的 token 数量。

这也是为什么代码最后会统计：

- `Tree Decoding Time`
- `Eagle Generate Time`
- `Compression Ratio`

因为这条路径的核心指标就是：一次验证到底能多确认多少 token。
