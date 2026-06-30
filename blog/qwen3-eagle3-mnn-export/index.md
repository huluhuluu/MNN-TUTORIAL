---
title: "Qwen3 Eagle3 MNN 导出流程"
date: 2026-06-29T12:00:00+08:00
lastmod: 2026-06-30T12:00:00+08:00
draft: false
description: "对照 llmexport.py 梳理 Qwen3-1.7B + Eagle3 导出为 MNN 的代码路径"
slug: "qwen3-eagle3-mnn-export"
tags: ["mnn", "llm", "qwen3", "eagle"]
categories: ["mnn"]

comments: true
math: true
---

# Qwen3 Eagle3 MNN 导出流程

本文记录 `transformers/llm/export/llmexport.py` 的导出路径，重点看下面这条命令如何把 `Qwen3-1.7B` 和 `Eagle3` draft model 导出成 `MNN` 文件。代码主线基于 `LlmExporter.export()`、`export_eagle()`、`export_language()` 和 `MNNConverter.export()`。

## 目录

- [目录](#目录)
- [1. 主入口](#1-主入口)
  - [1.1 导出命令](#11-导出命令)
  - [1.2 导出入口](#12-导出入口)
- [2. 初始化阶段](#2-初始化阶段)
- [3. 导出](#3-导出)
- [4. Eagle3 分支导出](#4-eagle3-分支导出)
- [5. 主 LLM 导出](#5-主-llm-导出)
- [6. MNN 转换和 HQQ 量化](#6-mnn-转换和-hqq-量化)
- [7. 配置文件和最终产物](#7-配置文件和最终产物)
- [8. 量化分支](#8-量化分支)
  - [8.1 普通量化](#81-普通量化)
  - [8.2 HQQ 量化](#82-hqq-量化)
  - [8.3 AWQ 量化](#83-awq-量化)
  - [8.4 SmoothQuant 量化](#84-smoothquant-量化)
- [小结](#小结)

## 1. 主入口
### 1.1 导出命令
导出依赖项目的py文件，需要虚拟环境:
```bash
# 创建独立环境，避免和系统 Python / 其它导出脚本依赖冲突
conda create -n mnn-export python=3.12 -y
conda activate mnn-export

# 使用 USTC 镜像安装导出脚本依赖
export UV_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
export MNN_ROOT=/path/to/MNN
cd ${MNN_ROOT}/transformers/llm/export
uv pip install -r requirements.txt
```
导出模型
```bash
# 导出 Qwen3-1.7B + Eagle3，参数需要根据自己情况指定
# 参数说明：
# --path: 基座模型路径
# --eagle_path: Eagle3 draft model checkpoint
# --dst_path: MNN 导出目标目录
# --export mnn: 触发 LlmExporter.export("mnn")
# --mnnconvert: 优先使用本地 MNNConvert 二进制
# --hqq: 权重量化走 HQQ 分支
conda activate mnn-export
python ${MNN_ROOT}/transformers/llm/export/llmexport.py \
    --path /data/HUGGINGFACE/Qwen3-1.7B \
    --eagle_path /path/to/qwen3-1p7b-eagle3/checkpoint \
    --dst_path /data/HUGGINGFACE/Qwen3-1.7B-eagle3-mnn-sglang-train \
    --export mnn \
    --mnnconvert ${MNN_ROOT}/build/MNNConvert \
    --hqq
```

### 1.2 导出入口
参数定义在入口代码文件 `transformers/llm/export/llmexport.py:676-723`位置，其中 `--eagle_path`、`--mnnconvert`、`--hqq` 是这条路径最关键的三个开关。

```python
# transformers/llm/export/llmexport.py:676-724
def build_args(parser):
    parser.add_argument('--path', type=str, required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='tokenizer path, default is `None` mean using `--path` value.')
    parser.add_argument('--eagle_path', type=str, default=None, help='eagle model path, default is `None`')
    parser.add_argument('--dst_path', type=str, default='./model', help='export onnx/mnn model to path, default is `./model`.')
    parser.add_argument('--export', type=str, default=None, help='export model to an onnx/mnn model.')
    parser.add_argument('--quant_bit', type=int, default=4, help='mnn quant bit, 4 or 8, default is 4.')
    parser.add_argument('--quant_block', type=int, default=64, help='mnn quant block, 0 mean channel-wise, default is 64.')
    parser.add_argument('--mnnconvert', type=str, default='../../../build/MNNConvert', help='local mnnconvert path, if invalid, using pymnn.')
    parser.add_argument('--hqq', action='store_true', help='Whether or not to use hqq quant.')
    # ...
```

主入口会根据模型路径选择 `EmbeddingExporter` 或 `LlmExporter`。本次路径是 `Qwen3-1.7B`，不会命中 `bge/gte/Qwen3-Embedding`，因此进入 `LlmExporter`。

```python
# transformers/llm/export/llmexport.py:737-755
def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    build_args(parser)
    args = parser.parse_args()

    model_path = args.path

    embedding_models = ['bge', 'gte', 'Qwen3-Embedding']
    if any(model in model_path for model in embedding_models):
        llm_exporter = EmbeddingExporter(args)
    else:
        llm_exporter = LlmExporter(args)

    if args.export is not None:
        llm_exporter.export(args.export)
```

主线可以概括成：

```text
`main()`
  -> `build_args(...)`
  -> `LlmExporter(args)`
    -> `load_model(--path)`
  -> llm_exporter.export(args.export)
    -> `export("mnn")`
    -> `export_eagle()`
    -> `export_language()`
    -> `export_tokenizer()`
    -> `export_config(mnn_config=True)`
    -> 删除中间 `onnx/`
```

## 2. 初始化阶段

`LlmExporter`对象的初始化只做两件事：整理路径参数，然后加载模型。

```python
# transformers/llm/export/llmexport.py:29-33
def __init__(self, args):
    super().__init__()
    self.init_from_args(args)
    self.load_model(args.path)
```

其中，`init_from_args()` 会把中间 ONNX 目录固定到 `dst_path/onnx`，并创建 `dst_path` 和 `onnx_path`。

```python
# transformers/llm/export/llmexport.py:34-51
def init_from_args(self, args):
    self.args = args
    self.max_new_tokens = 1024
    self.dst_name = 'llm'
    self.onnx_path = os.path.join(self.args.dst_path, 'onnx')
    if self.args.tokenizer_path is None:
        self.args.tokenizer_path = self.args.path
    if args.lm_quant_bit is None:
        self.args.lm_quant_bit = self.args.quant_bit
    if args.lm_quant_block is None:
        self.args.lm_quant_block = self.args.quant_block
    self.args.tie_word_embeddings = False
    if not os.path.exists(self.args.dst_path):
        os.makedirs(self.args.dst_path)
    if not os.path.exists(self.onnx_path):
        os.makedirs(self.onnx_path)
```

模型加载在 `load_model()` 中完成。这里会创建 `LlmModel`、`LlmTokenizer`，并写出后续导出需要的 `llm_config` 基础字段。

```python
# transformers/llm/export/llmexport.py:53-117
@spinner_run(f'load pretrained model ', True)
def load_model(self, model_path):
    self.model = LlmModel.from_pretrained(model_path, args=self.args)
    self.tokenizer = LlmTokenizer.from_pretrained(
        self.args.tokenizer_path,
        model_type=self.model.config.model_type
    )
    self.model.tokenizer = self.tokenizer
    self.config = self.model.config
    self.model_type = self.config.model_type
    # ...
    self.llm_config = {
        'model_type': self.config.model_type,
        'hidden_size' : self.config.hidden_size,
        'attention_mask': 'float',
        'attention_type': self.config.attention_type,
    }
    self.llm_config.update(self.model.get_config())
    # ...
    self.args.tie_word_embeddings = not self.args.seperate_embed and self.model.lm.lm.weight.equal(self.model.embed.embed.weight)
    self.visual = self.model.visual
    self.audio = self.model.audio
    self.talker = self.model.talker
    self.mtp = self.model.mtp
    self.scale_emb = self.model.scale_emb

    return model_path
```

`LlmModel.from_pretrained()` 的作用是把 HuggingFace 模型映射成导出器自己的模块结构。`ModelMapper.do_map()` 之后，embedding、rotary、decoder block、lm head 都被包装成导出友好的对象。

```python
# transformers/llm/export/utils/model.py:62-156
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, args=None, **kwargs):
    config = LlmConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    model_type = config.model_type
    model_class = cls.get_model_class(model_type)
    # ...
    original_model = model_class.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    # ...
    model = cls(config, args)

    ModelMapper.do_map(model, original_model, config.model_map['model'])
    # ...
    model.embed = Embedding(model.embed, config)
    model.rotary = Rotary(config)
    model.blocks = torch.nn.ModuleList([
        Decoder(block, i, config, model.rotary, config.model_map) for i, block in enumerate(model.blocks.children())
    ])
    model.lm = Lm(model.lm)
    # ...
    return model
```

从实现上看，导出器不直接对原始 HuggingFace 模型做 ONNX export，而是先映射到一套统一的 `LlmModel` 结构。后面的 `FakeLinear`、`FusedAttention`、KV cache 输出、`Eagle` hidden states 都依赖这层包装。

## 3. 导出

本次导出使用参数 `--export mnn` 使 `export_type == 'mnn'` 成立，因此 `export_mnn=True`，[主入口](#1-主入口)中指出的导出函数中将创建 `MNNConverter(self)`。

```python
# transformers/llm/export/llmexport.py:527-552
def export(self, export_type):
    if not self.args.skip_weight:
        if self.args.omni:
            self.omni_quant()
        if self.args.awq:
            self.awq_quant()
        if self.args.smooth:
            self.smooth_quant()
    export_mnn = export_type == 'mnn'
    self.mnn_converter = MNNConverter(self) if export_mnn else None
    self.export_talker() # 多模态分支
    self.export_vision()  # 多模态分支
    self.export_audio()  # 多模态分支
    self.export_eagle()
    self.export_language()
    self.export_mtp()
    self.export_tokenizer()
    self.export_config(export_mnn)
    if export_mnn:
        try:
            for file in glob.glob(f'{self.onnx_path}/*'):
                os.remove(file)
            os.rmdir(self.onnx_path)
        except Exception as e:
            print(f"remove onnx error: {e}")
```

这里将依次导出`export_eagle(), export_language(), self.export_tokenizer(), self.export_config(export_mnn)`。

## 4. Eagle3 分支导出

`export_eagle()` 的入口很短：没有 `--eagle_path` 就直接返回；有路径时，根据 `model_type` 选择 `Eagle` 类，导出两个 ONNX(主草稿模和给目标模式输出投影到hidden维度的线性层)，再转换成 MNN。

```python
# transformers/llm/export/llmexport.py:179-187
def export_eagle(self):
    if self.args.eagle_path is None:
        return
    from utils.eagle import Eagle
    self.eagle = Eagle.get_eagle(self.model_type)(self.args.eagle_path, self.model)
    eagle_onnx, eagle_fc_onnx = self.eagle.export(self.onnx_path)
    if self.mnn_converter:
        MNNConverter(self, None).export(eagle_onnx)
        MNNConverter(self, None).export(eagle_fc_onnx)
```

`Qwen3` 会被映射到 `LlamaEagle`。

```python
# transformers/llm/export/utils/eagle.py:127-135
@staticmethod
def get_eagle(model_type):
    eagles = {
        'llama': LlamaEagle,
        'qwen3': LlamaEagle,
    }
    if model_type in eagles:
        return eagles[model_type]
    return LlamaEagle
```

`Eagle.__init__()` 会读取 `eagle_path/config.json`，检查 `hidden_size` 和基座模型一致，然后构造 draft 模型需要的 `embed_tokens`、`fc`、`midlayer`、`norm`、`lm_head` 和 `d2t`。

```python
# transformers/llm/export/utils/eagle.py:18-106
class Eagle(torch.nn.Module):
    def __init__(self, eagle_path, base):
        super().__init__()
        config_file_path = eagle_path + "/config.json"
        self.eagle_config = PretrainedConfig.from_json_file(config_file_path)

        self.model_type = base.config.model_type
        self.eagle_path = eagle_path
        self.config = base.config
        # ...
        self.hidden_size = self.config.hidden_size
        if self.eagle_config.hidden_size != self.hidden_size:
            raise RuntimeError(f'eagle_config hidden_size not equal: {self.eagle_config.hidden_size}, {self.hidden_size}!')
        # ...
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
        # ...
        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        self.midlayer = nn.Module()
        self.midlayer.self_attn = Attention(None, 0, self.config, base.rotary, self.config.model_map)
        # ...
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size,bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        d2t = torch.zeros((self.draft_vocab_size), dtype=torch.int64)
        self.register_buffer("d2t", d2t)

        self.load()
        self.unloaded_ops = {}
```

`LlamaEagle.load()` 支持 `model.safetensors` 或 `pytorch_model.bin`。这也是 `--eagle_path` 目录必须具备的权重文件。

```python
# transformers/llm/export/utils/eagle.py:198-212
def load(self):
    safetensors_path = os.path.join(self.eagle_path, "model.safetensors")
    bin_path = os.path.join(self.eagle_path, "pytorch_model.bin")
    ea_layer_state_dict = None
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        ea_layer_state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        ea_layer_state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Eagle path '{self.eagle_path}' not found 'model.safetensors' or 'pytorch_model.bin'."
        )
    self.load_state_dict(ea_layer_state_dict, strict=False)
```

`Eagle.export()` 会生成三个和 speculative decoding 相关的产物：

- `eagle_d2t.mnn`：把 draft token 映射表保存成一个小 `MNN` 文件；
- `eagle_fc.onnx`：导出 `fc_hidden -> hidden_states`；
- `eagle.onnx`：导出 draft model 主体。

```python
# transformers/llm/export/utils/eagle.py:137-186
@spinner_run(f'export onnx model to ')
def export(self, onnx_path):
    import MNN.expr as expr
    torch_d2t = self.d2t.detach().to(torch.int32).contiguous().cpu()
    mnn_d2t = expr.const(torch_d2t.data_ptr(), torch_d2t.shape, expr.data_format.NHWC, expr.dtype.int)
    mnn_d2t.name = 'd2t'
    expr.save([mnn_d2t], f'{onnx_path}/../eagle_d2t.mnn')

    eagle_model = f'{onnx_path}/eagle.onnx'
    eagle_fc_model = f'{onnx_path}/eagle_fc.onnx'
    # ...
    with torch.no_grad():
        onnx_export(self.fc, (fc_hidden),
            eagle_fc_model,
            input_names=['fc_hidden'],
            output_names=['hidden_states'],
            dynamic_axes={ "fc_hidden" : { 1: "seq_len" } })
        onnx_export(
            self, (input_embed, hidden_states, attention_mask, position_ids, past_key_values, logits_index),
            eagle_model,
            input_names=[
                'input_embed', 'hidden_states',
                'attention_mask', 'position_ids',
                'past_key_values', 'logits_index'
            ],
            output_names=['logits', 'out_hidden_states', 'presents'],
            dynamic_axes={
                "input_embed" : { 0: "seq_len" },
                "hidden_states" : { 1: "seq_len" },
                "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                "position_ids" : { 1: "seq_len" },
                "past_key_values" : { 2: "history_len" }
                })
    return eagle_model, eagle_fc_model
```

主模型给 `Eagle3` 的 hidden states 在 `LlmModel.forward()` 里准备。只要 `args.eagle_path` 存在，它会在第 `2` 层、中间层、倒数第 `3` 层之前收集 `hidden_states`，最后拼接到 `final_layernorm` 输出，然后传给草稿模型计算，这里目标模型的单次前向计算流程大致如下。

```python
# transformers/llm/export/utils/model.py:165-228
def forward(self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            past_key_values: Optional[List[torch.Tensor]] = None,
            logits_index: torch.Tensor = torch.tensor([-1], dtype=torch.int32),
            deepstack_embeds: torch.Tensor = None
            ):
    hidden_states = input_ids
    presents = [None for i in range(len(self.blocks))]
    eagle_hidden_states = []
    rotary_pos_emb = self.rotary(position_ids)

    for i in range(len(self.blocks)):
        # 拷贝浅层/中层/深层 hidden states 给 Eagle3草稿模型
        if self.args and self.args.eagle_path and (i == len(self.blocks)-3 or i == len(self.blocks)//2 or i==2):
            eagle_hidden_states.append(hidden_states)
        # ...
        hidden_states, kv = self.blocks[i](hidden_states, rotary_pos_emb, layer_attention_mask, past_kv)
        presents[i] = kv

    # ...
    logits = self.lm(hidden_states)
    if presents[0] is not None and presents[0].shape == presents[-1].shape and None not in presents:
        presents = torch.stack(presents)

    if self.args and self.args.eagle_path is not None:
        final_layernorm = torch.cat(eagle_hidden_states, dim=-1)

    return logits, final_layernorm, presents, talker_embeds
```

从实现上看，这里的 `final_layernorm` 在 Eagle 路径下不再只是 lm head 前的 hidden state，而是三段 hidden states 的拼接结果。后面的 `config.json` 也会打开 `hidden_states`，让运行时知道主模型会输出这类信息。

## 5. 主 LLM 导出

`export_language()` 负责主模型导出。默认 `embed_bit=16`，所以会先写 `embeddings_bf16.bin`，再导出 `llm.onnx`，最后通过 `MNNConverter(self, self.unloaded_ops)` 转成 `llm.mnn`。

```python
# transformers/llm/export/llmexport.py:509-525
def export_language(self):
    if self.mnn_converter and self.args.tie_word_embeddings:
        pass
    else:
        self.export_embed()
    onnx_model = self.export_onnx()

    if self.args.onnx_slim:
        self.slim_onnx(onnx_model)
    if self.mnn_converter:
        tie_embeddings_info = MNNConverter(self, self.unloaded_ops).export(onnx_model)
        if tie_embeddings_info is not None:
            self.llm_config['tie_embeddings'] = tie_embeddings_info
    else:
        self.onnx_load_param(onnx_model)
```

主模型导出前会调用 `unload_param()`。这一步把每个 `torch.nn.Linear` 替换成 `FakeLinear`，真实权重保存在 `self.unloaded_ops` 里。

```python
# transformers/llm/export/llmexport.py:331-360
def unload_param(self):
    self.unloaded_ops = {}
    self.experts = []
    def build_faker(real, name):
        faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
        self.unloaded_ops[name] = real
        return faker
    with torch.no_grad():
        for i in range(len(self.model.blocks)):
            # ...
            for name, child in self.model.blocks[i].self_attn.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.model.blocks[i].self_attn, name, build_faker(child, f'/layers.{i}/self_attn/{name}/Linear'))
            for name, child in self.model.blocks[i].mlp.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.model.blocks[i].mlp, name, build_faker(child, f'/layers.{i}/mlp/{name}/Linear'))
                # ...
        self.model.lm.lm = build_faker(self.model.lm.lm, f'/lm/lm_head/Linear')
```

`FakeLinear` 的 ONNX symbolic 会产生 `LlmExporter::FakeLinear` 自定义 OP。它只保留 `in_features/out_features/has_bias/name`，不把大权重直接塞进 ONNX 图中。

```python
# transformers/llm/export/utils/custom_op.py:3-32
class FakeLinearOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, has_bias, name):
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "has_bias_i": has_bias,
            "name_s": name
        }
        # ...
        return g.op("LlmExporter::FakeLinear", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, has_bias, name):
        out_shape = list(input.shape)[:-1] + [out_features]
        return input.new_zeros(out_shape)

class FakeLinear(torch.nn.Module):
    def forward(self, x):
        return FakeLinearOp.apply(x, self.in_features, self.out_features, self.has_bias, self.name)
```

随后 `export_onnx()` 构造短序列 dummy input，导出主图。由于这次存在 `--eagle_path`，`output_names` 中的 `hidden_states` 对应上一节提到的 Eagle hidden states 拼接结果。

```python
# transformers/llm/export/llmexport.py:372-415
@spinner_run(f'export onnx model to ')
def export_onnx(self):
    self.unload_param()
    model = self.model
    seq_len = 3
    new_tokens = 0
    input_ids = torch.arange(seq_len, dtype=torch.long)
    attention_mask =  model.get_attention_mask(seq_len, new_tokens)
    position_ids = model.get_position_ids(seq_len, new_tokens, input_ids)
    onnx_model = f'{self.onnx_path}/{self.dst_name}.onnx'
    input_ids = model.embedding(input_ids)
    past_key_values = torch.zeros(self.past_kv_shape)
    logits_index = torch.tensor([-1], dtype=torch.int32)
    output_names = ['logits', 'hidden_states', 'presents']

    onnx_export(
        model, (input_ids, attention_mask, position_ids, past_key_values, logits_index),
        onnx_model,
        input_names=[
            'input_ids', 'attention_mask', 'position_ids', 'past_key_values', 'logits_index'
        ],
        output_names=output_names,
        dynamic_axes=self.model_dynamic_axes)
    return onnx_model
```

## 6. MNN 转换和 HQQ 量化

`MNNConverter` 初始化时会检查 `--mnnconvert` 指向的二进制是否存在。存在就走本地命令；不存在才 fallback 到 `MNN.tools.mnnconvert`。

```python
# transformers/llm/export/utils/mnn_converter.py:17-29
class MNNConverter:
    def __init__(self, exporter, weight_ops = None):
        self.weight_ops = weight_ops
        self.exporter = exporter
        self.args = exporter.args
        self.mnn_weight_offset = 0
        if os.path.exists(self.args.mnnconvert):
            self.mnnconvert = self.args.mnnconvert
        else:
            self.mnnconvert = None
        self.lm_weight = None
        self.tie_embeddings_info = None
```

`onnx2mnn()` 会拼出 `MNNConvert` 参数。`--hqq` 在这里会被追加到转换命令。

```python
# transformers/llm/export/utils/mnn_converter.py:52-76
@spinner_run(f'convert onnx model to ')
def onnx2mnn(self, onnx_path, mnn_path, args = [], transformer_fuse = True, group_conv_native = False, weight_sym = False, save_external_data = True):
    convert_args = [
        '',
        '-f',
        'ONNX',
        '--modelFile',
        str(onnx_path),
        '--MNNModel',
        str(mnn_path),
        '--allowCustomOp'
    ]
    if transformer_fuse:
        convert_args += ['--transformerFuse']
    if save_external_data:
        convert_args += ['--saveExternalData']
    if self.args.hqq:
        convert_args += ['--hqq']
    convert_args += args
    self.convert(convert_args)
    return mnn_path
```

`MNNConverter.export()` 有两条路径：

- `weight_ops is None`：直接用 `MNNConvert` 做 ONNX -> MNN，`Eagle` 的两个 ONNX 走这条路径；
- `weight_ops is not None`：先转 MNN，再转 JSON，调用 `rebuild()` 把 `FakeLinear` 重建成真正的 MNN op，并写外部权重，这里是目标模型的导出路径。

```python
# transformers/llm/export/utils/mnn_converter.py:118-158
def export(self, onnx_path, quant_bit = None, quant_block = None, transformer_fuse = True, group_conv_native = False, weight_sym = None):
    self.onnx_model_path = onnx_path
    self.mnn_name = os.path.basename(onnx_path).replace('.onnx', '.mnn')
    self.mnn_model_path = os.path.join(self.args.dst_path, self.mnn_name)
    self.mnn_weight_path = f'{self.mnn_model_path}.weight'
    if self.weight_ops is None:
        if quant_bit is None:
            quant_bit = self.args.quant_bit
        if quant_block is None:
            quant_block = self.args.quant_block
        # ...
        self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, quant_args, transformer_fuse=transformer_fuse, group_conv_native=group_conv_native, weight_sym=weight_sym)
    else:
        mnn_json = f'{self.mnn_model_path}.json'
        self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, transformer_fuse=transformer_fuse, group_conv_native=group_conv_native, weight_sym=weight_sym)
        self.mnn2json(self.mnn_model_path, mnn_json)
        self.rebuild(mnn_json)
        self.json2mnn(mnn_json, self.mnn_model_path)
        self.removeDupOps(self.mnn_model_path)
        self.mnn2json(self.mnn_model_path, mnn_json)
        # ...
    return self.tie_embeddings_info
```

主 LLM 的真实权重量化发生在 `rebuild()` 之后的 `rebuild_linear()`。它根据 `FakeLinear` 的 `name` 回到 `self.weight_ops[name]` 找真实 `Linear`，再调用 `build_weight()` 写入 `.mnn.weight`。

```python
# transformers/llm/export/utils/mnn_converter.py:365-378
def rebuild_op(self, op, graph):
    if "type" in op['main']:
        op_type = op['main']['type']
    else:
        op_type = op['type']
    if op_type == 'FakeLinear':
        return self.rebuild_linear(op, graph)
    if op_type == 'FusedAttention':
        return self.rebuild_attnention(op, graph)
    if op_type == "LayerNorm":
        return self.rebuild_layernorm(op, graph)
    if op_type == 'MoE':
        return self.rebuild_moe(op, graph)
    return None
```

`build_weight()` 中 `quant_bit` 默认是 `4`，`quant_block` 默认是 `64`。本次加了 `--hqq`，所以 `self.quant()` 会把 `hqq=True` 继续传给 `torch_quant()`。

```python
# transformers/llm/export/utils/mnn_converter.py:317-355
def build_weight(self, linear, quant_bit, quant_block, symmetric):
    ic, oc = linear.in_features, linear.out_features
    if quant_bit == 16:
        # ...
        alpha_len, q_min, shape_int32, header_len = 0, 0, False, 0
    else:
        q_min = 1
        assert(quant_bit in (1, 2, 4, 8))
        q_weight, alpha = self.quant(linear.weight.data, quant_bit, quant_block, symmetric)
        header_len, shape_int32 = self.write_header(ic, oc, quant_bit)
        weight_len = self.write_weight(q_weight) + header_len
        alpha_len = self.write_weight(alpha)
    # ...
    external = [self.mnn_weight_offset, weight_len, alpha_len, bias_length, 0]
    self.mnn_weight_offset += (weight_len + alpha_len + bias_length)
    return external, q_min, shape_int32, header_len
```

`HQQ` 分支在 `utils/torch_utils.py:29-101`。它构造 `HQQQuantizer`，得到 `W_q`、`scale`、`zero`，再按 `MNN` 权重文件需要的布局打包。

```python
# transformers/llm/export/utils/torch_utils.py:29-101
def quant(weight, quant_bit, quant_block, symmetric, awq, hqq):
    # ...
    if hqq:
        hqq_quantizer = HQQQuantizer(weight, quant_bit, block_size, symmetric, weight.dtype, weight.device)
        hqq_quantizer.quant()
        if not symmetric:
            q_weight = hqq_quantizer.W_q.flatten().to(torch.uint8)
            scale = hqq_quantizer.meta['scale'].flatten()
            zeros = scale * offset - scale * hqq_quantizer.meta['zero'].flatten()

            alpha = torch.stack([zeros.flatten(), scale.flatten()], axis=-1).flatten()
        else:
            q_weight = (hqq_quantizer.W_q.flatten() + offset).to(torch.uint8)
            scale = hqq_quantizer.meta['scale'].flatten()
            alpha = scale.flatten()
    else:
        # ...
        pass

    if quant_bit < 8 and 8 % quant_bit == 0:
        group_size = 8 // quant_bit
        q_weight = q_weight.reshape(-1, group_size)
        # ...
    return q_weight, alpha.float()
```

最后 `rebuild_linear()` 把一个 `FakeLinear` 替换成 `Reshape -> ConvertTensor -> Convolution -> ConvertTensor -> Reshape`。权重不放在 JSON 里，而是通过 `external` 指向 `.mnn.weight` 的偏移和长度。

```python
# transformers/llm/export/utils/mnn_converter.py:428-470
def rebuild_linear(self, op, graph):
    attrs = op['main']['attr']
    for attr in attrs:
        if attr['key'] == 'name':
            name = attr['s']
        elif attr['key'] == "in_features":
            ic = attr["i"]
        elif attr['key'] == "out_features":
            oc = attr["i"]
        elif attr['key'] == "has_bias":
            has_bias = attr["i"]
    linear = self.weight_ops[name]
    assert(linear.in_features == ic and
           linear.out_features == oc and
           (linear.bias is not None) == has_bias)

    is_lm = 'lm_head' in name
    quant_bit = self.args.lm_quant_bit if is_lm else self.args.quant_bit
    quant_block = self.args.lm_quant_block if is_lm else self.args.quant_block
    quant_sym = self.args.sym
    # ...
    external, q_min, shape_int32, header_len = self.build_weight(linear, quant_bit, quant_block, quant_sym)
    # ...
    if is_lm and self.args.tie_word_embeddings:
        weight_offset = external[0] + header_len
        alpha_offset = external[0] + external[1]
        alpha_size = external[2]
        self.tie_embeddings_info = [weight_offset, alpha_offset, alpha_size, quant_bit, quant_block]
```

```python
# transformers/llm/export/utils/mnn_converter.py:526-570
conv_op = {
    "name": conv_name,
    "inputIndexes": pre_convert_output,
    "outputIndexes": conv_output,
    "type": "Convolution",
    "main_type": "Convolution2D",
    "main": {
        'common': {
            'kernelX': 1, 'kernelY': 1,
            'outputCount': oc,
            'inputCount': ic,
        },
        "quanParameter": quanParameter,
        "external": external
    },
    "defaultDimentionFormat": "NHWC"
}
# ...
return [pre_reshape, pre_convert, conv_op, post_convert, post_reshape]
```

这一段解释了为什么主模型导出不是简单的 `torch.onnx.export -> MNNConvert`：它先用 `FakeLinear` 降低 ONNX 导出时的内存和文件压力，再在 `MNN` JSON 层恢复成更适合推理的 `Convolution2D` 权重布局。

## 7. 配置文件和最终产物

`export_config(True)` 会写三类配置：

- `export_args.json`：完整命令参数；
- `llm_config.json`：模型结构和 tokenizer/chat template 相关信息；
- `config.json`：运行时加载 `MNN` 模型需要的配置。

`--eagle_path` 会在 `config.json` 中追加 `speculative_type=eagle` 和 `hidden_states=true`。

```python
# transformers/llm/export/llmexport.py:255-297
@spinner_run(f'export config to ')
def export_config(self, mnn_config = False):
    with open(f'{self.args.dst_path}/export_args.json', 'w', encoding='utf-8') as f:
        json.dump(self.args.__dict__, f, ensure_ascii=False, indent=4)
    config_json = f'{self.args.dst_path}/llm_config.json'
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(self.llm_config, f, ensure_ascii=False, indent=4)
    if not mnn_config:
        return config_json
    with open(f'{self.args.dst_path}/config.json', 'w', encoding='utf-8') as f:
        config = {
            "llm_model": f"{self.dst_name}.mnn",
            "llm_weight": f"{self.dst_name}.mnn.weight",
            "backend_type": "cpu",
            "thread_num": 4,
            "precision": "low",
            "memory": "low",
            "sampler_type":'penalty',
            "penalty":1.1
        }
        # ...
        if self.args.eagle_path is not None:
            config['speculative_type'] = 'eagle'
            config['hidden_states'] = True
        json.dump(config, f, ensure_ascii=False, indent=4)
    return config_json
```

本次命令完成后，`dst_path` 下主要产物大概率是：

| 文件 | 来源 | 说明 |
|------|------|------|
| `llm.mnn` | `export_language()` | 主模型图 |
| `llm.mnn.weight` | `MNNConverter.rebuild()` | 主模型外部权重，默认 `int4 block=64`，启用 `HQQ` |
| `embeddings_bf16.bin` | `export_embed()` | embedding 权重，默认 `embed_bit=16` |
| `eagle.mnn` | `export_eagle()` | Eagle draft 主图 |
| `eagle.mnn.weight` | `MNNConvert --saveExternalData` | Eagle draft 权重 |
| `eagle_fc.mnn` | `export_eagle()` | Eagle hidden states 投影 |
| `eagle_fc.mnn.weight` | `MNNConvert --saveExternalData` | Eagle FC 权重 |
| `eagle_d2t.mnn` | `Eagle.export()` | draft token 到 target token 的映射 |
| `config.json` | `export_config(True)` | MNN runtime 配置，包含 `speculative_type=eagle` |
| `llm_config.json` | `export_config()` | 模型结构补充配置 |
| `export_args.json` | `export_config()` | 导出参数留档 |

由于 `export_type == "mnn"`，`export()` 最后会删除 `dst_path/onnx` 下的中间 ONNX 文件。如果需要调试 ONNX，需要临时改导出类型或注释清理逻辑。


## 8. 量化分支

这条命令显式打开的是 `--hqq`，但代码里实际有四条量化路径：默认 block 量化、`HQQ`、`AWQ`、`SmoothQuant`。先看入口 `LlmExporter.export()`，它会在创建 `MNNConverter` 之前先跑前置量化，再把处理后的模型继续送进 ONNX/MNN 导出。

```python
# transformers/llm/export/llmexport.py:527
def export(self, export_type):
    # skip_weight 只用于结构导出测试；真实导出会进入这里
    if not self.args.skip_weight:
        # Omni/AWQ/Smooth 是导出前对 torch model 做的预处理
        if self.args.omni:
            self.omni_quant()
        if self.args.awq:
            self.awq_quant()
        if self.args.smooth:
            self.smooth_quant()

    # --export mnn 时才创建 MNNConverter，后面负责 ONNX -> MNN 和权重重建
    export_mnn = export_type == 'mnn'
    self.mnn_converter = MNNConverter(self) if export_mnn else None
    # ...
    self.export_eagle()
    self.export_language()
    self.export_config(export_mnn)
```

分支关系先落到这一步：

| 开关 | 前置阶段 | 写权重阶段 | 主要作用 |
|------|----------|------------|----------|
| 默认 | 无 | `torch_quant(..., awq=False, hqq=False)` | min/max 普通权重量化 |
| `--hqq` | 无 | `torch_quant(..., hqq=True)` | 使用 `HQQQuantizer` 生成 `W_q/scale/zero` |
| `--awq` | `AwqQuantizer.quantize()` | `torch_quant(..., awq=True)` | 先搜索并应用 scale/clip，再用 AWQ 公式写权重 |
| `--smooth` | `SmoothQuantizer.quantize()` | `export_smooth_quant()` | 平滑 Norm/Linear，并把激活 scale 写回 MNN JSON |

再按“是否需要校准数据”和“处理对象是权重还是激活”来区分：

| 方法 | 是否需要校准样本 | 主要处理对象 | 在本导出器里的位置 |
|------|------------------|--------------|--------------------|
| 普通量化 | 不需要 | 权重 | `MNNConverter.quant()` -> `torch_utils.quant()` |
| `HQQ` | 不需要 | 权重 | `torch_utils.quant()` -> `HQQQuantizer._quantize()` |
| `AWQ` | 需要 | 权重，参考激活分布 | `LlmExporter.awq_quant()` -> `AwqQuantizer.quantize()` -> `torch_utils.quant()` |
| `SmoothQuant` | 需要 | 激活范围和权重共同迁移 | `LlmExporter.smooth_quant()` -> `SmoothQuantizer.quantize()` -> `MNNConverter.export_smooth_quant()` |

### 8.1 普通量化
普通权重量化的实际入口在 `MNNConverter.quant()`。主模型走 `FakeLinear` 重建路径时，每个 `Linear` 最终都会进到这里；`Eagle` 草稿分支如果走 `MNNConverter(self, None)`，则不会经过这套 Python 侧重建逻辑。

```python
# transformers/llm/export/utils/mnn_converter.py:273
def quant(self, weight, quant_bit, quant_block, symmetric):
    if self.exporter.args.skip_weight:
        # 结构测试模式只占位，不真正量化权重
        # ...
        return q_weight, alpha

    # awq / hqq 都作为标志传给 torch_quant
    q_weight, alpha = torch_quant(
        weight,
        quant_bit,
        quant_block,
        symmetric,
        self.args.awq,
        self.args.hqq,
    )
    return q_weight, alpha
```

`torch_quant()` 里先算 `block_size`，再按 `hqq -> symmetric -> asymmetric` 的顺序分支， 分别对应`hqq`量化、对称量化、非对称量化。

```python
# transformers/llm/export/utils/torch_utils.py:29
def quant(weight, quant_bit, quant_block, symmetric, awq, hqq):
    # quant_block=64 时按输入通道每 64 个一组；不能整除时会向下调整 block_size
    oc, ic = weight.shape
    block_size = ic if quant_block == 0 else quant_block
    while ic % block_size != 0:
        block_size /= 2
    block_size = int(block_size)
    block_num = ic // block_size

    offset = 1 << (quant_bit - 1)
    clip_max = offset - 1

    if hqq:
        # HQQ 分支：由 HQQQuantizer 给出 W_q、scale、zero
        hqq_quantizer = HQQQuantizer(weight, quant_bit, block_size, symmetric, weight.dtype, weight.device)
        hqq_quantizer.quant()
        # ...
    else:
        weight = weight.reshape(oc, block_num, block_size)
        if symmetric:
            # 对称量化：只记录 scale，不记录 zero point
            # ...
        else:
            # 普通非对称量化：每个 block 用 min/max 计算 scale 和 zero
            max_val, _ = torch.max(weight, axis=-1, keepdims=True)
            min_val, _ = torch.min(weight, axis=-1, keepdims=True)
            scale = (max_val - min_val) / (clip_max - clip_min)

            if awq:
                # AWQ 分支使用另一种 zero 计算方式
                q_weight = torch.round(weight / scale) - torch.round(min_val / scale) + clip_min
                zeros =  (torch.round(min_val / scale) - clip_min) * scale
            else:
                # 默认普通量化
                q_weight = torch.round((weight - min_val) / scale) + clip_min
                zeros =  min_val - scale * clip_min
            # ...

    # int4 会在这里把两个 4-bit 值打包到一个 uint8
    if quant_bit < 8 and 8 % quant_bit == 0:
        # ...
    return q_weight, alpha.float()
```
这里的 `oc, ic = weight.shape` 对应`Linear` 的权重维度 `[out_features, in_features]`，导出时也沿着这个约定: 代码里后面 `build_weight()` 取的是 `linear.in_features, linear.out_features`，再把 `linear.weight.data.flatten()` 写出去；真正把矩阵按导出格式重排的是 ONNX 重建那条路径里的 `linear.weight.data.transpose(1, 0)`，也就是先从 PyTorch 的 `[oc, ic]` 转成导出端更容易消费的 `[ic, oc]` 视角，再做扁平化写入。

如果使用分块量化，block沿着这里的`ic` 维度切分：`quant_block=64, W_shape=(512,4096)` 时，`512` 是 `oc`，`4096` 是 `ic`，每行会切成 `64` 个 block，每个 block 只在输入通道维度上做一组 `scale/zero`。

普通量化的核心就是当前 block 的 `min/max`。它不看校准数据，也不关心这组权重对应的激活分布。代价是如果某个 block 有离群值，整个 block 的步长会被拉大，普通值的精度会被压缩。

### 8.2 HQQ 量化

`HQQ` 的代码入口在 `torch_utils.quant()` 的 `if hqq:` 分支：先构造 `HQQQuantizer`，再从 `meta` 中拿到 `scale/zero`。

```python
# transformers/llm/export/utils/torch_utils.py:52
if hqq:
    # HQQ 分支不走普通 min/max 打包逻辑，而是交给 HQQQuantizer 优化 W_q
    hqq_quantizer = HQQQuantizer(weight, quant_bit, block_size, symmetric, weight.dtype, weight.device)
    hqq_quantizer.quant()
    if not symmetric:
        q_weight = hqq_quantizer.W_q.flatten().to(torch.uint8)
        scale = hqq_quantizer.meta['scale'].flatten()
        zeros = scale * offset - scale * hqq_quantizer.meta['zero'].flatten()
        alpha = torch.stack([zeros.flatten(), scale.flatten()], axis=-1).flatten()
    else:
        q_weight = (hqq_quantizer.W_q.flatten() + offset).to(torch.uint8)
        scale = hqq_quantizer.meta['scale'].flatten()
        alpha = scale.flatten()
```

`HQQQuantizer._quantize()` 中先按 `group_size` reshape 权重，再为每组初始化 `min/max`、`scale`、`zero`。非对称量化时，量化值范围是 `[0, 2^bit - 1]`；对称量化时，范围是 `[-2^(bit-1), 2^(bit-1)-1]`。

```python
# transformers/llm/export/utils/hqq_quantizer.py:24
@torch.inference_mode()
def _quantize(self, channel_wise: bool = True, axis: int = 1) -> tuple:
    W = self.weight.to(self.compute_dtype).float()
    shape = W.shape

    # 以 group_size 为单位分组；axis=1 时每行按输入通道方向切块
    if (self.group_size is not None) and channel_wise:
        W = W.reshape([-1, self.group_size]) if (axis == 1) else W.reshape([self.group_size, -1])

    _min = W.min(axis=axis, keepdim=True)[0]
    _max = W.max(axis=axis, keepdim=True)[0]

    if self.sym:
        max_v = 2**(self.bit-1) - 1
        min_v = -2**(self.bit-1)
        max_abs = torch.max(torch.abs(_min), torch.abs(_max))
        scale = max_v / max_abs
        zero = None
    else:
        max_v = round(2**self.bit - 1)
        min_v = 0
        denom = (_max - _min)
        scale = (max_v / denom)
        zero = torch.round(-_min * scale)

    W_q, scale, zero = self._optimize_weights(W, scale, zero, min_max=[min_v, max_v], axis=axis)
    # ...
```

关键差异在 `_optimize_weights()`。普通量化是直接 `round + clamp`，`HQQ` 会迭代更新 `scale/zero`，用重建误差 `|W_f - W_r|` 判断是否继续。这里更像是在固定 bit 数和 block 划分之后，寻找一组更适合当前权重分布的量化参数。

```python
# transformers/llm/export/utils/hqq_quantizer.py:112
@torch.inference_mode()
def _optimize_weights(self, W, scale, zero, min_max, axis=0, opt_params={"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20}, verbose=False) -> tuple:
    W_f = W.to(dtype=torch.float32, device=self.device)
    scale = scale.to(dtype=torch.float32, device=self.device)
    if not self.sym:
        zero = zero.to(dtype=torch.float32, device=self.device)

    best_error = torch.tensor(torch.inf, dtype=torch.float32, device=self.device)
    # 迭代优化量化后的重建值 W_r
    for i in range(iters):
        if not self.sym:
            self._optimize_weights_proximal_legacy_step(W_f, scale, zero, min_max, beta, lp_norm, axis, W_q, W_r, W_e)
        else:
            self._optimize_weights_proximal_scale_only(W_f, scale, min_max, beta, lp_norm, axis, W_q, W_r, W_e, W_prime)
        current_error = torch.abs(W_f - W_r).mean().float()
        if current_error < best_error:
            best_error = current_error
        else:
            break

    W_q = torch.round(W * scale + zero).clamp_(min_max[0], min_max[1]) if not self.sym else torch.round(W * scale).clamp_(min_max[0], min_max[1])
    return W_q, scale, zero
```

所以 `HQQ` 在这条导出路径里可以理解成“无校准数据的权重量化优化”。它仍然输出 `q_weight + alpha`，最终文件格式和普通量化一致，只是 `q_weight/scale/zero` 的求法不同。

### 8.3 AWQ 量化
`AWQ` 量化的入口在 `LlmExporter.awq_quant()`，具体算法可以参考[博客](https://zhuanlan.zhihu.com/p/697761176)。

导出参数`--awq` 有两个作用。第一层是在导出前调用 `AwqQuantizer.quantize()`，它会按 block 遍历模型层，搜索 scale 和 clip，然后直接改写当前 torch model 的权重分布。

```python
# transformers/llm/export/llmexport.py:417
def awq_quant(self):
    # 这里直接持有 self.model，后续 export_onnx/export_language 会使用被处理后的模型
    self.awq_quantizer = AwqQuantizer(self.model)
    self.awq_quantizer.quantize()
```

具体的，`AwqQuantizer.quantize()` 的主逻辑是：取每层 `Linear`，收集输入特征，搜索最优 scale，应用 scale，再可选应用 clip。

```python
# transformers/llm/export/utils/awq_quantizer.py:82
def quantize(self):
    for i in tqdm(range(len(self.modules)), desc="AWQ"):
        # 找到当前 block 内需要处理的 Linear
        named_linears = AwqQuantizer.get_named_linears(self.modules[i])
        named_linears = AwqQuantizer.exclude_layers_to_not_quantize(
            named_linears, self.modules_to_not_convert
        )

        # 用校准样本收集各 Linear 的输入特征
        input_feat = self._get_input_feat(self.modules[i], named_linears)

        # 搜索并应用 AWQ scale
        scales_list = [
            self._search_best_scale(self.modules[i], **layer)
            for layer in module_config
        ]
        AwqQuantizer.apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)

        # 可选搜索并应用 clip，默认 apply_clip=True
        if self.apply_clip:
            clip_list = self._search_best_clip(
                self.modules[i], named_linears, input_feat
            )
            AwqQuantizer.apply_clip(self.modules[i], clip_list)
```

`AWQ` 的算法直觉是：权重本身不是唯一要看的对象，哪些通道更重要要参考激活分布。`_search_best_scale()` 会同时看权重大小和输入激活均值，搜索一组 scale，让量化后的输出和原始输出更接近。

```python
# transformers/llm/export/utils/awq_quantizer.py:200
@torch.no_grad()
def _search_best_scale(self, module, prev_op, layers, inp, module2inspect=None, kwargs={}):
    # 权重按 group_size 组织，计算每个输入通道的相对权重强度
    weight = torch.cat([_m.weight for _m in layers], dim=0)
    org_shape = weight.shape
    weight = weight.view(-1, self.group_size)
    w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
    w_scale = w_scale.view(org_shape)
    w_mean = w_scale.mean(0)

    # 输入激活展平成 [token, hidden]，统计每个输入通道的激活强度
    inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
    # 后续会网格搜索 ratio，找量化误差最小的一组 scales
    # ...
```

`AWQ` 还会搜索 clipping。这里会跳过 `q/k` 这类和 attention score 相关的层，对其它 `Linear` 尝试缩小权重范围，比较量化输出和原始输出的均方误差。

```python
# transformers/llm/export/utils/awq_quantizer.py:386
@torch.no_grad()
def _search_best_clip(self, layer, named_linears, input_feat):
    clip_list = []
    avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

    for name in named_linears:
        # q/k 对 attention score 敏感，这里不做 clipping 搜索
        if any([_ in name for _ in avoid_clipping]):
            continue

        max_val = self._compute_best_clip(named_linears[name].weight, input_feat[name])
        clip_list.append((name, max_val))
    return clip_list
```

第二层是在 `torch_quant()` 写最终 `q_weight/alpha` 时，`awq=True` 会改变非对称量化里的 zero 计算公式。也就是说，`--awq` 不替代后面的 `int4/int8` 写权重逻辑，而是先用校准样本调整 torch 权重，再影响最终打包公式。

### 8.4 SmoothQuant 量化
`--smooth` 的位置更特殊。它也会在导出前跑一次校准，但它后面还会在 `MNNConverter.export()` 的 JSON 重建阶段把激活量化信息写回去。

```python
# transformers/llm/export/llmexport.py:453
def smooth_quant(self):
    total_lines = 128
    if self.args.calib_data:
        # 有自定义 calib_data 时，按文件行数决定最多使用多少条样本
        self.model.args.calib_data = self.args.calib_data
        # ...

    calib_samples = min(total_lines, 128)
    self.smooth_quantizer = SmoothQuantizer(
        model=self.model,
        max_calib_samples=calib_samples,
        act_bit=self.args.act_bit,
        act_sym=self.args.act_sym,
        generate_for_npu=self.args.generate_for_npu,
    )
    self.smooth_quantizer.quantize()
```

`SmoothQuantizer.quantize()` 分三步：先收集输入最大值用于平滑，再把 scale 应用到 `Norm + Linear`，最后收集静态激活 scale。

```python
# transformers/llm/export/utils/smooth_quantizer.py:437
def quantize(self):
    for i in tqdm(range(len(self.samples)), desc="collecting data and computing scales..."):
        # 先用校准样本统计每层输入范围
        self.module_kwargs, self.inps = self._get_first_input(sample)
        for idx in range(len(self.modules)):
            named_layers = SmoothQuantizer.get_all_leaf_modules(self.modules[idx])
            self._get_max_input(idx, self.modules[idx], named_layers)

    for idx in tqdm(range(len(self.modules)), desc="applying scales..."):
        # 把 activation scale 平滑进 Norm 和 Linear 权重
        self._apply_scale(idx, self.modules[idx])

    for i in tqdm(range(len(self.samples)), desc="collecting static activation scales..."):
        # 平滑后再统计静态激活量化参数
        self.module_kwargs, self.inps = self._get_first_input(sample)
        for idx in range(len(self.modules)):
            named_linears = SmoothQuantizer.get_all_leaf_modules(self.modules[idx])
            self._get_all_static_scales(idx, self.modules[idx], named_linears)

    self._extract_static_scales()
```

平滑本身发生在 `_apply_scale()`：`self_attn.q/k/v` 对应 `input_layernorm`，`mlp.gate/up` 对应 `post_attention_layernorm`。

```python
# transformers/llm/export/utils/smooth_quantizer.py:350
def _apply_scale(self, idx, module):
    # attention 前的 Norm 和 q/k/v 权重一起平滑
    attn_ln = module.input_layernorm
    qkv = [
            module.self_attn.q_proj,
            module.self_attn.k_proj,
            module.self_attn.v_proj,
        ]
    qkv_input_scales = self.act_scales[idx]["self_attn.q_proj"]
    SmoothQuantizer.smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, self.alpha)

    # MLP 前的 Norm 和 gate/up 权重一起平滑
    ffn_ln = module.post_attention_layernorm
    fcs = [module.mlp.gate_proj, module.mlp.up_proj]
    ffn_input_scales = self.act_scales[idx]["mlp.gate_proj"]
    SmoothQuantizer.smooth_ln_fcs(ffn_ln, fcs, ffn_input_scales, self.alpha)
```

`SmoothQuant` 的算法直觉是把一部分激活的量化难度迁移到权重上。`smooth_ln_fcs()` 里会用激活 scale 和权重 scale 计算一个新的 `scales`，然后让 `Norm` 除以它，让后面的 `Linear.weight` 乘以它。这样浮点计算等价关系大体保持，但激活范围会更适合量化。

```python
# transformers/llm/export/utils/smooth_quantizer.py:304
@staticmethod
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    # act_scales 来自校准样本，weight_scales 来自 Linear 权重
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # 前一个 Norm 除以 scales，后一个 Linear 权重乘以 scales
    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
```

`SmoothQuant` 的后处理入口在 `MNNConverter.export()`。只有主模型走 `weight_ops is not None` 的 rebuild 路径时，才会执行 `export_smooth_quant()`；`Eagle` 分支当前是 `MNNConverter(self, None)`，不会走这段 Python 侧 JSON 后处理。

```python
# transformers/llm/export/utils/mnn_converter.py:118
def export(self, onnx_path, quant_bit = None, quant_block = None, transformer_fuse = True, group_conv_native = False, weight_sym = None):
    if self.weight_ops is None:
        # Eagle 等直接 ONNX -> MNN 的路径，只把 quant 参数交给 MNNConvert
        self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, quant_args, transformer_fuse=transformer_fuse, group_conv_native=group_conv_native, weight_sym=weight_sym)
    else:
        # 主 LLM：ONNX -> MNN -> JSON -> rebuild -> MNN
        self.onnx2mnn(self.onnx_model_path, self.mnn_model_path, transformer_fuse=transformer_fuse, group_conv_native=group_conv_native, weight_sym=weight_sym)
        self.mnn2json(self.mnn_model_path, mnn_json)
        self.rebuild(mnn_json)
        self.json2mnn(mnn_json, self.mnn_model_path)
        # ...
        if self.args.smooth:
            self.export_smooth_quant(mnn_json)
    return self.tie_embeddings_info
```

```python
# transformers/llm/export/utils/mnn_converter.py:223
@spinner_run(f'export smooth quant scale to ')
def export_smooth_quant(self, mnn_json):
    # 把 SmoothQuant 统计到的激活 scale 写进 MNN JSON，再转回 MNN
    self.exporter.smooth_quantizer.apply(mnn_json)
    self.json2mnn(mnn_json, self.mnn_model_path)
    return self.mnn_model_path
```

所以这几个分支中：

- 普通量化：没有校准阶段，直接在 `build_weight()` 中按 `min/max` 生成 `q_weight + alpha`。
- `--hqq`：没有前置校准，但 `torch_quant()` 内部换成 `HQQQuantizer`，迭代优化每组权重的 `scale/zero`。
- `--awq`：先用校准数据判断重要通道，调整权重 scale/clip，再在写权重时启用 `awq=True` 的公式。
- `--smooth`：先把激活的量化压力迁移到权重和 Norm，再统计激活 scale，后续还要在 `MNN` JSON 上写入激活量化信息。
- 当前 `Eagle` 草稿模型走 `weight_ops is None`，主要由 `MNNConvert` 直接处理权重；这些 Python 侧 `rebuild()` 后处理主要作用在主 `LLM`。


## 小结

- 真实入口是 `llmexport.py:main()`，本次路径进入 `LlmExporter`，不是 `EmbeddingExporter`。
- `--eagle_path` 触发 `export_eagle()`，会额外导出 `eagle.mnn`、`eagle_fc.mnn` 和 `eagle_d2t.mnn`。
- 主模型 `forward()` 在 Eagle 路径下会拼接三处 hidden states，运行时通过 `config.json` 的 `hidden_states=true` 识别。
- 主 LLM 权重导出使用 `FakeLinear -> MNN JSON rebuild -> Convolution2D + external weight` 这条路径。
- `--hqq` 同时进入 `MNNConvert` 参数和 Python 侧 `torch_quant(..., hqq=True)`，主模型权重量化主要在 `MNNConverter.build_weight()` 触发。
- 下一步继续读代码，可以看 `utils/hqq_quantizer.py` 和 `transformers/llm/engine` 里 runtime 如何加载 `speculative_type=eagle`。
