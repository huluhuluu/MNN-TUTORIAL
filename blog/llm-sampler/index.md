---
title: "LLM Sampler 采样逻辑"
date: 2026-06-10T12:00:00+08:00
lastmod: 2026-06-16T12:00:00+08:00
draft: false
description: "按当前仓库代码梳理 MNN LLM 中 sampler 配置、候选集过滤和 token 选择流程"
slug: "llm-sampler"
tags: ["mnn", "llm"]
categories: ["mnn"]

comments: true
math: true
---

# LLM Sampler 采样逻辑

- [LLM Sampler 采样逻辑](#llm-sampler-采样逻辑)
  - [1. Sampler 在推理链路里的位置](#1-sampler-在推理链路里的位置)
  - [2. 配置入口](#2-配置入口)
    - [2.1 `config.json` 字段](#21-configjson-字段)
    - [2.2 创建 `Sampler`](#22-创建-sampler)
  - [3. 采样主流程](#3-采样主流程)
    - [3.1 `Llm::sample()` 处理 logits 切片](#31-llmsample-处理-logits-切片)
    - [3.2 `Sampler::sample()` 先过滤再选择](#32-samplersample-先过滤再选择)
    - [3.3 `temperature` 在过滤和选择中的位置](#33-temperature-在过滤和选择中的位置)
  - [4. 候选集过滤逻辑](#4-候选集过滤逻辑)
    - [4.1 `topK`](#41-topk)
    - [4.2 `topP`](#42-topp)
    - [4.3 `minP`](#43-minp)
    - [4.4 `tfs` 和 `typical`](#44-tfs-和-typical)
    - [4.5 `penalty`](#45-penalty)
    - [4.6 `mixed`](#46-mixed)
  - [5. token 选择逻辑](#5-token-选择逻辑)
  - [6. 常用配置组合](#6-常用配置组合)
    - [6.1 greedy](#61-greedy)
    - [6.2 temperature](#62-temperature)
    - [6.3 topP](#63-topp)
    - [6.4 mixed](#64-mixed)
  - [7. 小结](#7-小结)

本文记录 MNN LLM 中通用 sampler 的代码路径，即文本生成阶段从 logits 里选择下一个 token 的逻辑。

## 1. Sampler 在推理链路里的位置

一次自回归生成中，sampler 所在位置可以整理成：

```text
输入 token / embedding
    ↓
`Llm::forwardVec(...)`
    ↓
模型输出 logits
    ↓
`Llm::sample(logits)`
    ↓
`Sampler::sample(logits)`
    ↓
得到下一个 token id
```

`Sampler` 不负责模型前向，也不负责 tokenizer decode。它只接收一段 logits，并根据配置把候选 token 缩小，再从候选集中选出一个 token。

## 2. 配置入口

### 2.1 `config.json` 字段

Sampler 相关字段集中在 `LlmConfig`：

```cpp
// transformers/llm/engine/src/llmconfig.hpp:521
std::string sampler_type() const {
    return config_.value("sampler_type", "greedy");
}
std::vector<std::string> mixed_samplers() const {
    return config_.value("mixed_samplers",
        std::vector<std::string>({"topK", "tfs", "typical", "topP", "min_p", "temperature"}));
}
float temperature() const { return config_.value("temperature", 1.0f); }
int topK() const { return config_.value("topK", 40); }
float topP() const { return config_.value("topP", 0.9f); }
float minP() const { return config_.value("minP", 0.1f); }
float tfsZ() const { return config_.value("tfsZ", 1.0f); }
float typical() const { return config_.value("typical", 1.0f); }
float penalty() const { return config_.value("penalty", 0.0f); }
int ngram() const { return config_.value("n_gram", 8); }
float ngram_factor() const { return config_.value("ngram_factor", 1.0f); }
std::string penalty_sampler() const {
    return config_.value("penalty_sampler", "greedy");
}
```

常见字段含义如下：

| 字段 | 默认值 | 作用 |
|------|------|------|
| `sampler_type` | `"greedy"` | 选择采样策略入口 |
| `temperature` | `1.0` | softmax 前缩放 logits，数值越低越偏确定 |
| `topK` | `40` | 只保留 logits 最大的 K 个候选 |
| `topP` | `0.9` | 按概率从高到低累加，直到累计概率超过 P |
| `minP` | `0.1` | 保留概率不低于阈值的候选，至少保留一个 |
| `tfsZ` | `1.0` | tail free sampling 的截断阈值 |
| `typical` | `1.0` | typical sampling 的累计概率阈值 |
| `penalty` | `0.0` | 重复 token 惩罚，`<= 1.0` 时实际不生效 |
| `n_gram` | `8` | ngram 重复检测长度 |
| `ngram_factor` | `1.0` | ngram 重复惩罚倍数，`<= 1.0` 时不启用 ngram 递增惩罚 |
| `penalty_sampler` | `"greedy"` | `sampler_type = "penalty"` 时最终选择 token 的方式 |
| `mixed_samplers` | `["topK", "tfs", "typical", "topP", "min_p", "temperature"]` | `sampler_type = "mixed"` 时的过滤顺序 |

这里有一个源码细节：`configSampler(...)` 判断的是 `"minP"`，但默认 `mixed_samplers()` 里写的是 `"min_p"`。如果配置里直接沿用默认列表，`subsetSampler(...)` 也只识别 `"minP"`，因此 `"min_p"` 不会触发 `minP(...)`。需要启用 `minP` 时，配置里写 `"minP"`。

### 2.2 创建 `Sampler`

`Llm::load()` 初始化模型、prompt、embedding 后，会创建 sampler：

```cpp
// transformers/llm/engine/src/llm.cpp:266
mSampler.reset(Sampler::createSampler(mContext, mConfig));
```

`createSampler(...)` 实例化一个 `Sampler` 类，用于持有 `LlmContext`、读取采样配置，并在生成阶段执行 token 选择：

```cpp
// transformers/llm/engine/src/sampler.cpp:154
Sampler* Sampler::createSampler(std::shared_ptr<LlmContext> context,
                                std::shared_ptr<LlmConfig> config) {
    return new Sampler(context, config);
}
```

`Sampler` 构造函数会读取 `max_all_tokens`、`max_new_tokens` 和 `sampler_type`，再根据类型填充 `SamplerConfig`：

```cpp
// transformers/llm/engine/src/sampler.cpp:158
Sampler::Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) {
    mContext = context;
    mConfig.max_all_tokens = config->max_all_tokens();
    mConfig.max_new_tokens = config->max_new_tokens();
    mConfig.type = config->sampler_type();
    mConfig.configSampler(mConfig.type, config);
}
```

多模态 Talker 分支会覆盖默认配置：

```cpp
// transformers/llm/engine/src/omni.cpp:1079
set_config("{\"sampler_type\": \"mixed\", \"temperature\": 0.9, \"topK\": 40, \"topP\": 0.8, \"penalty\": 1.05}");
mSampler.reset(Sampler::createSampler(mContext, mConfig));
```

## 3. 采样主流程

### 3.1 `Llm::sample()` 处理 logits 切片

`Llm::sample()` 是生成阶段进入 sampler 的统一入口：

```cpp
// transformers/llm/engine/src/llm.cpp:665
int Llm::sample(VARP logits, int offset, int size) {
    auto logitsShape = logits->getInfo()->dim;
    if (offset && size) {
        MNN_ASSERT(logits->getInfo()->size >= offset + size);
        logits = _Const(logits->readMap<float>() + offset,
                        {size},
                        NHWC,
                        halide_type_of<float>());
    }
    auto token_id = mSampler->sample(logits);
    return token_id;
}
```

普通 decode 通常直接传最后一段 logits。Speculative decoding 一类流程会传 `offset / size`，从更大的 logits buffer 里切出当前要采样的那一段。

### 3.2 `Sampler::sample()` 先过滤再选择

`Sampler::sample()` 把一次采样拆成候选集过滤和 token 选择两步：

```cpp
// transformers/llm/engine/src/sampler.cpp:486
int Sampler::sample(Express::VARP logits) {
    Timer _t;
    struct SubsetLogits subset = createSubsetLogits(logits);
    subset = subsetSampler(mConfig.type, subset);
    int res = handleSelect(subset);
    mContext->sample_us += _t.durationInUs();
    return res;
}
```

这里分两步：

1. `subsetSampler(...)` 根据 `sampler_type` 改写候选集合；
2. `handleSelect(...)` 在候选集合里选最终 token。

当前实现里，部分过滤器在确定候选子集时已经会用 temperature 计算概率；过滤完成后，如果 `select_type = "temperature"`，最终选择阶段还会对保留下来的 logits 再做一次 temperature softmax。

候选集合用 `SubsetLogits` 表示：

```cpp
// transformers/llm/engine/src/sampler.cpp:35
struct SubsetLogits{
    std::vector<int> index;
    MNN::Express::VARP logits;
    bool is_subset;
};
```

当 `is_subset = false` 时，`logits` 仍然对应完整 vocab；当 `is_subset = true` 时，`index` 保存候选集合里的局部位置到原始 token id 的映射。

### 3.3 `temperature` 在过滤和选择中的位置

MNN 这里的 temperature 有两类用途：

1. 在部分过滤器里，把 logits 转成概率，用这个概率决定候选子集；
2. 在最终选择时，把候选子集里的 logits 重新 softmax，然后随机采样。

temperature softmax 的实现很直接：先把 logits 乘上 `1 / temperature`，再做 softmax。

```cpp
// transformers/llm/engine/src/sampler.cpp:43
Express::VARP _TempratureSoftmax(Express::VARP logits, float temperature, int axis = -1) {
    return Express::_Softmax(logits * Express::_Scalar<float>(1.0f / temperature), axis);
}
```

`topP / minP / tfs` 都会通过 `packSoftmax(...)` 在过滤阶段使用 temperature：

```cpp
// transformers/llm/engine/src/sampler.cpp:139
int packSoftmax(Express::VARP logits, std::vector<IndexScore>& index_scores, float temperature) {
    auto prob_varp = _TempratureSoftmax(logits, temperature);
    auto probs = (float*)(prob_varp->readMap<float>());
    auto size = prob_varp->getInfo()->size;
    ...
}
```

`typical(...)` 没有走 `packSoftmax(...)`，但也会先调用 `_TempratureSoftmax(...)` 计算概率和 entropy：

```cpp
// transformers/llm/engine/src/sampler.cpp:378
float p = mConfig.typical, temperature = mConfig.temperature;
auto prob_varp = _TempratureSoftmax(superset.logits, temperature);
auto probs = (float*)(prob_varp->readMap<float>());
```

`topK(...)` 是例外。它确定子集时只看原始 logits 的排序，不使用 temperature：

```cpp
// transformers/llm/engine/src/sampler.cpp:260
int K = mConfig.topK;
auto scores = (float*)(superset.logits->readMap<float>());
...
m.score = scores[i];
```

过滤器返回的 `subset_logits` 保存的是原始 logits，不是概率。因此 `topP / minP / tfs / typical` 的 temperature 会影响“哪些 token 进入子集”，最终 `handleSelect(...)` 还会再次用 temperature 影响“在子集里抽中哪个 token”。

还有一个配置细节：单独使用 `sampler_type = "topK"` 时，`configTopK(...)` 只读取 `topK`，没有读取 `llmConfig->temperature()`。这时最终选择阶段仍然是 temperature select，但用的是 `SamplerConfig` 里的默认值 `0.8`，不是 `LlmConfig::temperature()` 的默认值 `1.0`，也不会读 JSON 里的 `temperature` 字段。

```cpp
// transformers/llm/engine/src/sampler.cpp:200
void Sampler::SamplerConfig::configTopK(std::shared_ptr<LlmConfig> llmConfig) {
    topK = llmConfig->topK();
    select_type = "temperature";
}
```

`topP / minP / tfs / typical / temperature` 会读取 `llmConfig->temperature()`：

```cpp
// transformers/llm/engine/src/sampler.cpp:204
void Sampler::SamplerConfig::configTopP(std::shared_ptr<LlmConfig> llmConfig) {
    topP = llmConfig->topP();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
```

## 4. 候选集过滤逻辑

### 4.1 `topK`

`topK(...)` 用一个小根堆保留分数最高的 K 个 token：

```cpp
// transformers/llm/engine/src/sampler.cpp:260
std::priority_queue<IndexScore, std::vector<IndexScore>, IndexScoreCmpGreater> heap;
for (int i = 0; i < size; i++) {
    IndexScore m;
    m.index = i;
    m.score = scores[i];
    if (heap.size() < K) {
        heap.push(m);
    } else {
        if (heap.top().score < m.score) {
            heap.pop();
            heap.push(m);
        }
    }
}
```

最后把 K 个候选写成新的 `SubsetLogits`。如果输入已经是子集，会调用 `transformIndex(superset, subset)` 把局部 index 映射回原始 token id。

`topK` 的子集大小只由 logits 排名和 `topK` 决定，temperature 不参与这一步。temperature 只会在后面的 `handleSelect(...)` 中参与最终随机选择；并且单独 `sampler_type = "topK"` 时，源码没有从 JSON 读取 `temperature`。

### 4.2 `topP`

`topP(...)` 先用 temperature softmax 得到概率，再按概率从高到低累加：

```cpp
// transformers/llm/engine/src/sampler.cpp:292
float p = mConfig.topP, temperature = mConfig.temperature;
std::vector<IndexScore> index_scores;
int size = packSoftmax(superset.logits, index_scores, temperature);
std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
...
while (cumulative < p && !index_scores.empty()) {
    std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
    IndexScore m = index_scores.back();
    index_scores.pop_back();
    index.push_back(m.index);
    subset_logits.push_back(scores[m.index]);
    cumulative += m.score;
}
```

注意这里 `subset_logits` 保存的是原始 logits，不是 softmax 之后的概率。`topP` 的子集由 temperature softmax 后的概率决定，最终选择时还会对保留下来的原始 logits 再做一次 temperature softmax。

### 4.3 `minP`

`minP(...)` 同样先按 temperature softmax 得到概率，然后按概率从高到低遍历。概率低于阈值后停止，但会保证已经至少保留一个候选：

```cpp
// transformers/llm/engine/src/sampler.cpp:316
for (int i = 0; i < size; ++i) {
    std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
    IndexScore m = index_scores.back();
    if (m.score < p && !index.empty()) break;
    index_scores.pop_back();
    index.push_back(m.index);
    subset_logits.push_back(scores[m.index]);
}
```

### 4.4 `tfs` 和 `typical`

`tfs(...)` 是 tail free sampling。它先用 temperature softmax 得到概率并排序，再计算相邻概率差分的变化量，最后按 `tfsZ` 截断尾部：

```cpp
// transformers/llm/engine/src/sampler.cpp:339
std::sort(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
...
derivatives[i] = std::fabs(first - second);
...
if (cumulative >= z && !index.empty()) break;
```

`typical(...)` 会先用 temperature softmax 得到概率，计算 entropy，再按每个 token 的 `abs(entropy + log(prob))` 排序，保留累计概率未超过 `typical` 的候选：

```cpp
// transformers/llm/engine/src/sampler.cpp:378
float entropy = 0.0f;
for (int i = 0; i < size; i++) entropy -= probs[i] * std::log(probs[i]);
...
m.score = std::fabs(entropy + std::log(probs[i]));
```

### 4.5 `penalty`

`penalty(...)` 根据 `mContext->history_tokens` 惩罚已经出现过的 token：

```cpp
// transformers/llm/engine/src/sampler.cpp:417
std::vector<int>& prev = mContext->history_tokens;
std::unordered_map<int, float> penalty_map;
...
for (int i = 0; i < prev.size(); ++i) {
    if (penalty_map.count(prev[i]) == 0) penalty_map[prev[i]] = penalty;
    ...
}
```

真正改 logits 的规则是：

```cpp
// transformers/llm/engine/src/sampler.cpp:451
scoresMap[it->first] = (scoresMap[it->first] >= 0.0f)
    ? (scoresMap[it->first] / it->second)
    : (scoresMap[it->first] * it->second);
```

正 logits 会除以惩罚系数，负 logits 会乘以惩罚系数，整体效果都是降低这些 token 被选中的概率。

`ngram_factor > 1.0` 时会额外检查最近 `n_gram - 1` 个 token。如果历史中存在相同前缀，会叠乘 `ngram_factor`；匹配到完整 ngram 时直接拉到 `max_penalty`。

### 4.6 `mixed`

`mixed` 会按 `mixed_samplers` 顺序串联多个过滤器：

```cpp
// transformers/llm/engine/src/sampler.cpp:458
struct SubsetLogits Sampler::mixed(struct SubsetLogits subset) {
    for (auto sampler : mConfig.mixedSamplers) {
        subset = subsetSampler(sampler, subset);
    }
    return subset;
}
```

配置时会把 `penalty` 从原位置移除，再插到最前面：

```cpp
// transformers/llm/engine/src/sampler.cpp:237
if (hasPenalty) {
    newSamplers.insert(newSamplers.begin(), "penalty");
}
mixedSamplers = newSamplers;
```

这样重复惩罚会先作用在完整 logits 上，再交给 `topK / topP / tfs / typical` 继续收缩候选集合。

## 5. token 选择逻辑

过滤结束后进入 `handleSelect(...)`：

```cpp
// transformers/llm/engine/src/sampler.cpp:477
int Sampler::handleSelect(struct SubsetLogits subset) {
    if (mConfig.select_type == "greedy") {
        return argmaxSelect(subset);
    } else if(mConfig.select_type =="temperature") {
        return reSoftmaxSelect(subset, mConfig.temperature);
    }
    return 0;
}
```

`greedy` 直接取最大 logits：

```cpp
// transformers/llm/engine/src/sampler.cpp:97
int argmaxSelect(struct SubsetLogits superset) {
    auto scores = (float*)(superset.logits->readMap<float>());
    ...
    return select(superset, token_id);
}
```

`temperature` 会对当前候选集合重新 softmax，再按累计概率随机采样：

```cpp
// transformers/llm/engine/src/sampler.cpp:134
int reSoftmaxSelect(struct SubsetLogits subset, float temperature) {
    int token_index_id = randomSelect(_TempratureSoftmax(subset.logits, temperature));
    return ((subset.is_subset) ? subset.index[token_index_id] : token_index_id);
}
```

`randomSelect(...)` 每次用 `std::random_device` 初始化随机源，然后在 `[0, 1)` 里取随机数并按概率累加选择：

```cpp
// transformers/llm/engine/src/sampler.cpp:115
std::random_device rd;
std::mt19937 generator(rd());
std::uniform_real_distribution<float> distribution(0.0, 1.0);
float target = distribution(generator);
```

因此 `temperature`、`topP`、`mixed` 等路径本身不是确定性的；`greedy` 是确定性的。对 `topP / minP / tfs / typical` 来说，temperature 同时影响候选子集和最终抽样；对 `topK` 来说，temperature 只影响最终抽样。

## 6. 常用配置组合

### 6.1 greedy

```json
{
  "sampler_type": "greedy"
}
```

选择最大 logits 对应的 token。适合需要稳定复现、调试输出或做确定性对照。

### 6.2 temperature

```json
{
  "sampler_type": "temperature",
  "temperature": 0.8
}
```

执行 temperature softmax 后随机采样，不额外缩小候选集合。

### 6.3 topP

```json
{
  "sampler_type": "topP",
  "topP": 0.9,
  "temperature": 0.8
}
```

先用 temperature softmax 得到概率，按概率累计保留候选，再在候选集合里重新做 temperature softmax 随机采样。

### 6.4 mixed

```json
{
  "sampler_type": "mixed",
  "mixed_samplers": ["penalty", "topK", "topP", "temperature"],
  "temperature": 0.9,
  "topK": 40,
  "topP": 0.8,
  "penalty": 1.05
}
```

这是更接近日常聊天的组合：先做重复惩罚，再用 `topK / topP` 收缩候选，最后按 temperature 随机选择。这里 `topK` 子集阶段不使用 temperature，`topP` 子集阶段会使用 temperature。

## 7. 小结

Sampler 逻辑可以按下面顺序理解：

```text
`config.json`
    ↓
`LlmConfig::sampler_type()` 等读取参数
    ↓
`Sampler::Sampler(...)` 初始化 `SamplerConfig`
    ↓
`Llm::sample(logits, offset, size)` 取出待采样 logits
    ↓
`Sampler::subsetSampler(...)` 过滤候选集合
    ↓
`Sampler::handleSelect(...)` 选择最终 token
```

其中 `topK / topP / minP / tfs / typical` 负责缩小候选集合，`penalty` 负责基于历史 token 改写 logits，`greedy / temperature` 负责最终选择 token。`topP / minP / tfs / typical` 在缩小候选集合时会先使用 temperature，最终 temperature select 还会再使用一次；`topK` 缩小候选集合时只看 logits 排名。`mixed` 用配置顺序组合多个过滤器，过滤完成后仍由 `handleSelect(...)` 决定最终 token。
