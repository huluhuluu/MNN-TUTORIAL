---
title: "Eagle 推理流程"
date: 2026-05-24T23:00:00+08:00
lastmod: 2026-05-24T23:00:00+08:00
draft: false
description: "按当前仓库代码梳理 MNN LLM 中 Eagle speculative decoding 的加载、draft 生成与验证流程"
slug: "llm-eagle-infer"
tags: ["mnn"]
categories: ["mnn"]

comments: true
math: true
---

# Eagle 推理流程

本篇记录`MNN`的`Eagle3`推理流程，主要内容包括：

- [Eagle 推理流程](#eagle-推理流程)
  - [1. Eagle 入口](#1-eagle-入口)
  - [2. Eagle 配置](#2-eagle-配置)
  - [3. Eagle 加载](#3-eagle-加载)
  - [4. Eagle 推理主流程](#4-eagle-推理主流程)
    - [4.1 首 token](#41-首-token)
    - [4.2 draft tree 生成](#42-draft-tree-生成)
    - [4.3 TokenTree 接口](#43-tokentree-接口)
    - [4.4 tree decoding 验证](#44-tree-decoding-验证)
    - [4.5 接受token](#45-接受token)

## 1. Eagle 入口

`LLM` 在加载阶段会统一创建生成策略对象，入口在：

```cpp
// transformers/llm/engine/src/speculative_decoding/generate.cpp:19
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
使用`Eagle3`进行推理的前提是：
- 配置文件中 `speculative_type = "eagle"`；

## 2. Eagle 配置

`Eagle` 路径依赖的主要配置在 `llmconfig.hpp`：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:576
std::string speculative_type() const {
    return config_.value("speculative_type", "");
}
int draft_predict_length() const {
    return config_.value("draft_predict_length", 3);
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

- `eagle_model`：主 `Eagle` draft 模块，输入 `(input_embed + hidden_states, mask, position_ids,  logits_index)`；
- `eagle_fc`：把主模型浅层、中层、深层的 hidden states 变成 `Eagle` draft 模块需要的 hidden states；
- `eagle_d2t`：draft token 到真实 token 的映射表；
- `draft_predict_length`：每轮从草稿树里取多少个 draft node 交给主模型验证，默认是 `3`；
- `eagle_depth`：草稿树的深度；
- `eagle_topk`：每层扩展多少个候选分支。

## 3. Eagle 加载

`EagleGeneration::load()` 会在普通主模型之外，再额外加载两份模块和一份映射表，对应上面的`eagle_model/eagle_fc/eagle_d2t`,加载代码如下：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:28
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

在`Eagle3`推理情况下：

1. 目标模型仍然由 `Llm::mModule` 负责；
2. `Eagle3`草稿模型额外持有两份 `Module`：
   - `mEagleModules[0]`：主模型，一层注意力层+一层线性层lm_head
   - `mEagleModules[1]`：hidden state 投影模块
3. `Eagle3`草稿模型维护草稿模型到目标模型token的词汇映射表 `mD2t`

另外，草稿模型额外维护一份 draft 路径上的 KV 状态 `mEagleMeta`，在`EagleGeneration::load()` 里会进行初始化：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:29
mEagleMeta.reset(new KVMeta);
mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleMeta.get());
```

## 4. Eagle 推理主流程

`EagleGeneration::generate()` 的流程大致是：
```text
主模型 prefill logits
    ↓
先采样出第一个 token
    ↓
Eagle draft 生成草稿树
    ↓
主模型 tree decoding 验证草稿树
    ↓
选择接受路径
    ↓
把接受 token 回写主模型 KV
    ↓
再基于新的验证得到的 hidden states 继续生成下一轮 draft
```

对应大致代码结构如下：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:298
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

### 4.1 首 token

`Eagle` 不是从空状态直接建树，而是先消费主模型 `prefill` 后的第一轮 logits：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:305
auto sampleToken  = mLlm->sample(param.outputs[0]);
mContext->current_token = sampleToken;
mContext->history_tokens.push_back(mContext->current_token);
mContext->output_tokens.push_back(mContext->current_token);
mLlm->updateContext(0, 1);
```

然后把这个 token 对应的 embedding 拼到输入尾部，作为 `Eagle` draft 的起点：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:317
auto cur_embed  = mLlm->embedding({sampleToken});
auto pre_embeds = _Split(inputEmbeds, {1, seqLen - 1}, 0);
inputEmbeds     = _Concat({pre_embeds[1], cur_embed}, 0);
```
这里给草稿模型的输入是上一个token的hidden state+新一个的embedding，第一个token没有前序hidden state, 所以pre_embeds需要切掉第一个token的embedding，拼上新token的embedding作为草稿树的输入。

### 4.2 draft tree 生成

draft tree 的生成入口是 `topkGenerate(...)`：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:110
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

这一步的输出是一个 `DraftInfo`结构体，包括：

- `draftTokens`: 主模型要验证的 token 列表
- `retrieveIndices`：每条候选路径怎么从 draftTokens 里取，是一个二维列表，第一维是路径数，第二维是每条路径对应的 token 索引；
- `attentionMask`：树形解码的 attention mask，主模型验证时需要用到
- `positionIds`：树形解码的 position id，主模型验证时需要用到

由草稿树`TokenTree` 负责把多轮 `topk` 候选整理成主模型一次性验证需要的 `draftTokens / attentionMask / positionIds / retrieveIndices`。

### 4.3 TokenTree 接口

`TokenTree` 定义在 `transformers/llm/engine/src/speculative_decoding/tokentree.hpp`。它是一个纯 CPU 侧的草稿树整理器。`EagleGeneration::topkGenerate(...)` 每轮拿到 `Eagle` 模型输出的 `topk` token 后，会把候选交给 `TokenTree` 维护。

基础数据结构是 `TokenTreeNode` 和 `TreeOutputs`：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:24
class TokenTreeNode : public std::enable_shared_from_this<TokenTreeNode> {
public:
    int mTokenId, mNodeId, mDepth;
    double mLogProb, mCumulativeLogProb;
    std::weak_ptr<TokenTreeNode> mParent;
    std::vector<NodePtr> mChildren;
    ...
};

// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:48
struct TreeOutputs {
    std::vector<int> draftTokens;
    std::vector<int> positionIds;
    std::vector<std::vector<bool>> attentionMask;
    std::vector<std::vector<int>> retrieveIndices;
};
```

`TokenTreeNode` 里最重要的是四类信息：

- `mTokenId`：当前节点代表的 token；
- `mNodeId`：创建顺序，后面 `finalize(...)` 会用它恢复 draft token 的稳定顺序；
- `mDepth`：节点在草稿树里的层数，用来生成 `positionIds`；
- `mCumulativeLogProb`：从根到当前节点的累计分数，用来裁剪候选路径。

`TreeOutputs`是把内部维护的“树结构”整理成后面主模型验证能直接使用的四类数据，还需要在 `topkGenerate(...)` 把它转换成 `DraftInfo`给后续验证使用。

`TokenTree` 初始化时会创建一个虚拟根节点，并初始化第一层 mask：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:57
TokenTree(int topK, const int* d2tPtr = nullptr) : mTopK(topK), mD2tPtr(d2tPtr), mCounter(0) {
    mRoot = std::make_shared<TokenTreeNode>(-1, 0.0, nullptr, -1);
    mRoot->mDepth = -1;
    mMask.assign(topK, std::vector<bool>(topK, false));
    for (int i = 0; i < topK; i++) {
        mMask[i][i] = true;
    }
}
```

这里的 `d2tPtr` 来自 `eagle_d2t`。`Eagle` 模型输出的是 draft vocab 上的 id，`TokenTree::d2t(...)` 会把它映射回目标模型 token：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:243
int d2t(int token) {
    if (mD2tPtr) {
        return token + mD2tPtr[token];
    }
    return token;
}
```

第一轮 `topk` 通过 `init(...)` 建第一层节点：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:67
void init(const int* indices, const float* scores) {
    for (size_t i = 0; i < mTopK; i++) {
        auto node = std::make_shared<TokenTreeNode>(
            d2t(indices[i]), scores[i], mRoot, mCounter++
        );
        mRoot->addChild(node);
        mActives.push_back(node);
    }
}
```

后续每一层调用 `grow(...)`。它会对当前 `mActives` 中的每个叶子节点扩展 `topK` 个孩子，然后按 `mCumulativeLogProb` 排序，只保留全局最优的 `topK` 个节点作为下一轮 active leaf：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:77
void grow(const int* indices, const float* scores) {
    std::vector<NodePtr> candidates;
    std::map<NodePtr, int> parant2index;
    for (size_t i = 0; i < mActives.size(); ++i) {
        auto parent = mActives[i];
        parant2index[parent] = i;
        for (size_t j = 0; j < mTopK; j++) {
            auto child_node = std::make_shared<TokenTreeNode>(
                d2t(indices[i * mTopK + j]),
                scores[i * mTopK + j],
                parent,
                mCounter++
            );
            parent->addChild(child_node);
            candidates.push_back(child_node);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const NodePtr& a, const NodePtr& b) {
        return a->mCumulativeLogProb > b->mCumulativeLogProb;
    });
    ...
    mActives = std::move(newActives);
}
```

`getIds()` 和 `getMask()` 是给下一轮 `Eagle` 模型用的接口：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:215
const std::vector<std::vector<bool>>& getMask() const {
    return mMask;
}

// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:219
std::vector<int> getIds() const {
    std::vector<int> tokenIds;
    for (auto& active : mActives) {
        tokenIds.push_back(active->mTokenId);
    }
    return tokenIds;
}
```

在 `topkGenerate(...)` 里对应这段：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:146
inputEmbeds   = mLlm->embedding(tokenTree.getIds());
auto attentionMask = getMask(tokenTree.getMask(), seqLen);
outputs = eagleForwardRaw({inputEmbeds, inputHidden, attentionMask, mTreePosition, mLlm->logitsAllIdx});
...
tokenTree.grow(indices, scores);
```

最后 `finalize(sampleToken, mDraftLength)` 把内部树整理成验证输入：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:135
TreeOutputs finalize(int sampleToken, int maxDraftTokens) {
    auto allNodes = getAllNodes();
    std::sort(allNodes.begin(), allNodes.end(), [](const NodePtr& a, const NodePtr& b) {
        return a->mCumulativeLogProb > b->mCumulativeLogProb;
    });
    if (allNodes.size() > maxDraftTokens) {
        allNodes.resize(maxDraftTokens);
    }
    std::sort(allNodes.begin(), allNodes.end(), [](const NodePtr& a, const NodePtr& b) {
        return a->mNodeId < b->mNodeId;
    });
    ...
}
```

这里有两个排序：

- 先按累计分数选出最值得验证的 `maxDraftTokens` 个节点；
- 再按 `mNodeId` 恢复创建顺序，保证 `draftTokens` 的布局稳定。

`finalize(...)` 会先把已经由主模型采样出的 `sampleToken` 放到 `draftTokens[0]`，再追加选中的树节点：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:154
outputs.draftTokens.push_back(sampleToken);
outputs.positionIds.push_back(0);
for(size_t i = 0; i < allNodes.size(); ++i) {
    outputs.draftTokens.push_back(allNodes[i]->mTokenId);
    outputs.positionIds.push_back(allNodes[i]->mDepth + 1);
    nodeId2Idx[allNodes[i]->mNodeId] = i;
}
```

后面的 `attentionMask` 表示树上可见关系：每个节点可以看自己和祖先节点，不能看其它分支节点。`retrieveIndices` 则把每个叶子节点对应的根到叶路径整理出来：

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:162
outputs.attentionMask.assign(numDraftTokens, std::vector<bool>(numDraftTokens, false));
for (int i = 0; i < numDraftTokens; ++i) {
    outputs.attentionMask[i][i] = true;
    auto node = allNodes[i];
    auto parent = node->mParent.lock();
    if (parent && parent->mNodeId != -1 && nodeId2Idx.count(parent->mNodeId)) {
        int parentIdx = nodeId2Idx[parent->mNodeId];
        for (int j = 0; j < numDraftTokens; ++j) {
            if (outputs.attentionMask[parentIdx][j]) {
                outputs.attentionMask[i][j] = true;
            }
        }
    }
}

// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:201
for (const auto& leaf : leafNodes) {
    std::vector<int> path;
    auto curr = leaf;
    while(curr && curr->mNodeId != -1) {
        path.push_back(nodeId2Idx[curr->mNodeId] + 1);
        curr = curr->mParent.lock();
    }
    path.push_back(0);
    std::reverse(path.begin(), path.end());
    outputs.retrieveIndices.push_back(path);
}
```

`topkGenerate(...)` 拿到 `TreeOutputs` 后，会把 bool mask 转成主模型需要的 float mask，并把相对位置加上当前 `seqLen`：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:195
int inputLen = output.draftTokens.size();
DraftInfo info;
info.draftTokens = std::move(output.draftTokens);
info.retrieveIndices = std::move(output.retrieveIndices);
info.attentionMask = _Input({1, 1, inputLen, inputLen}, NCHW, halide_type_of<float>());
for (int i = 0; i < inputLen; i++) {
    for (int j = 0; j < inputLen; j++) {
        info.attentionMask->writeMap<float>()[i * inputLen + j] =
            output.attentionMask[i][j] ? 0.0 : std::numeric_limits<float>::lowest();
    }
}
info.positionIds = _Input({1, inputLen}, NCHW, halide_type_of<int>());
for (int i = 0; i < inputLen; i++) {
    info.positionIds->writeMap<int>()[i] = seqLen + output.positionIds[i];
}
```

所以 `TokenTree` 的接口可以按这个顺序理解：

| 接口 | 输入 | 输出 / 作用 |
|------|------|-------------|
| `init(indices, scores)` | 首轮 `topk` token 和分数 | 建第一层候选节点，初始化 `mActives` |
| `getIds()` | 当前 active leaf | 返回下一轮要送入 `Eagle` 模型的 token |
| `getMask()` | 当前 active leaf 可见关系 | 返回下一轮 `Eagle` 模型用的树形 mask |
| `grow(indices, scores)` | 每个 active leaf 的下一层 `topk` | 扩展候选树，并按累计分数裁剪到 `topK` 个 active leaf |
| `finalize(sampleToken, maxDraftTokens)` | 当前主模型采样 token 和草稿长度上限 | 生成主模型验证用的 `draftTokens / positionIds / attentionMask / retrieveIndices` |
| `toString(decoder)` | token 解码函数 | `EAGLE_DEBUG` 下打印树结构 |

### 4.4 tree decoding 验证

draft tree 建好以后，会进入 `treeDecoding(...)`：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:212
VARPS EagleGeneration::treeDecoding(const EagleGeneration::DraftInfo& drafInfo) {
    auto inputEmbeds   = mLlm->embedding(drafInfo.draftTokens);
    int inputLen = drafInfo.draftTokens.size();
    mLlm->mMeta->add = inputLen;
    auto outputs = mLlm->forwardRaw(inputEmbeds, drafInfo.attentionMask, drafInfo.positionIds);
    return outputs;
}
```

这一层做的事情很直接：把 draft tree 里的所有候选 token 一次性交给主模型，让主模型跑一次树形解码，得到树形输入对应的`logits`。

这里执行 `treeDecoding()` 的 KV 语义如下：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:214
int inputLen = drafInfo.draftTokens.size();
mLlm->mMeta->add = inputLen;
auto outputs = mLlm->forwardRaw(inputEmbeds, drafInfo.attentionMask, drafInfo.positionIds);
```

主模型在 tree decoding 阶段会先把“整棵 draft tree 扁平化后的 token 序列”都视为待验证输入，一次性把这些候选写进当前前向路径里。后面才需要 `updateDraft()` 时通过 `remove + reserve` 把接受的token的 KV 缓存保留。

### 4.5 接受token

主模型验证完 draft tree 后，会通过 `evaluatePosterior(...)` 选出最佳接受路径，主要是根据路径验证各个路径上的token是否都被接受，保留最长路径：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:220
EagleGeneration::AcceptInfo EagleGeneration::evaluatePosterior(const EagleGeneration::DraftInfo& drafInfo, VARP logits) {
    auto sampleTokens = MNN::Express::_ArgMax(logits, -1);
    std::vector<int> samples(drafInfo.draftTokens.size());
    ::memcpy(samples.data(), sampleTokens->readMap<int>(), samples.size() * sizeof(int));
    std::vector<int> bestCandidate;
    int nextSample = 0;
    for (auto indices : drafInfo.retrieveIndices) {
        std::vector<int> candidate;
        int next = -1;
        for (int i = 0; i < indices.size() - 1; i++) {
            int sampleIdx = indices[i];
            int draftIdx  = indices[i + 1];
            if (samples[sampleIdx] != drafInfo.draftTokens[draftIdx]) {
                break;
            }
            candidate.push_back(sampleIdx);
            next = draftIdx;
        }
        if (candidate.size() > bestCandidate.size()) {
            bestCandidate = candidate;
            nextSample = next;
        }
    }
}
```

然后在 `updateDraft(...)` 里把接受结果回写到主模型 KV 状态：

```cpp
// transformers/llm/engine/src/speculative_decoding/eagle.cpp:267
mLlm->updateContext(acceptLen, acceptLen);
mLlm->mMeta->remove = acceptInfo.sampleTokens.size();
mLlm->mMeta->n_reserve = acceptLen;
mLlm->mMeta->reserve = new int[mLlm->mMeta->n_reserve * 2];
for (size_t i = 0; i < acceptLen; i++) {
    mLlm->mMeta->reserve[2 * i] = acceptInfo.acceptIndices[i];
    mLlm->mMeta->reserve[2 * i + 1] = 1;
}
```

这里的逻辑是：

- `sampleTokens.size()` 表示这次 tree decoding 验证过的整棵候选树长度；
- `acceptLen` 表示最终真正接受了多少个 token；
- `remove + reserve` 保留了接受token位置的 KV 状态。

随后 `updateDraft(...)` 还会用接受路径对应的 hidden states 继续调用 `topkGenerate(...)`，进入下一轮 draft tree 生成。
