---
title: "mnn-tutorial"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-06-16T12:00:00+08:00
draft: false
description: "MNN端侧推理部署--从环境配置到读懂代码"
slug: "mnn-tutorial"
tags: ["mnn"]
categories: ["mnn"]
build:
  list: never

comments: true
math: true
---

# mnn-tutorial

本tutorial介绍 [MNN](https://github.com/alibaba/MNN) 端侧推理部署的完整流程，包括环境配置&运行、核心概念讲解、代码流畅梳理。

## tutorial概览

| 阶段 | 内容 | 文章数 |
|------|------|--------|
| 环境配置 | 远程调试、交叉编译、设备运行 | 4 篇 |
| 核心概念 | Backend、工厂模式、核心类、BackendConfig 分支 | 4 篇 |
| MNN-LLM | 配置、加载、推理、导出、QNN Plugin | 8 篇 |

---

## 1. 环境配置

搭建端侧开发调试环境的完整指南。

| 文章 | 说明 | 状态 |
|------|------|------|
| [远程ADB环境配置](/p/remote-adb/) | 通过端口转发链路，让内网服务器直连本地手机 | ✅ 完成 |
| [交叉编译环境配置](/p/cross-compiler/) | Android NDK 配置、Clangd 配置、远程调试 | ✅ 完成 |
| [llm_demo 交叉编译与运行](/p/llm-demo-run/) | 以 `llm_demo` 为例，记录从编译到设备运行的最短路径 | ✅ 完成 |
| [llm_demo 在 QNN 后端的运行](/p/llm-demo-qnn-run/) | 记录 `llm_demo` 通过 `config_qnn.json` 和 QNN 产物在 Android 设备上运行 | ✅ 完成 |

---

## 2. 核心概念

深入理解 MNN 框架的设计理念与核心组件。

| 文章 | 说明 | 状态 |
|------|------|------|
| [核心类介绍](/p/introduce-core-class/) | VARP、Expr、Op 等关键类的设计 | ✅ 完成 |
| [Backend 介绍](/p/introduce-backend/) | CPU/OpenCL/Vulkan 等后端的作用和选择 | ✅ 完成 |
| [工厂模式介绍](/p/introduce-factory/) | MNN 中工厂模式的设计与应用 | ✅ 完成 |
| [BackendConfig 分支路径](/p/backend-config-branch/) | 对照源码梳理 `precision`、`memory`、`power` 各取值的执行分支 | ✅ 完成 |

---

## 3. MNN-LLM

端侧大语言模型部署实践。

| 文章 | 说明 | 状态 |
|------|------|------|
| [LLM 配置](/p/llm-config/) | 模型配置、量化配置 | ✅ 完成 |
| [LLM 加载流程](/p/llm-load/) | 模型文件到推理就绪的完整过程 | ✅ 完成 |
| [LLM 推理流程](/p/llm-infer/) | Token 处理、KV Cache 管理 | 🚧 WIP |
| [Eagle 推理流程](/p/llm-eagle-infer/) | speculative decoding 中 Eagle 路径的 draft、验证与 KV 回写 | ✅ 完成|
| [Qwen3 Eagle3 MNN 导出流程](/p/qwen3-eagle3-mnn-export/) | 对照 `llmexport.py` 梳理 Qwen3-1.7B + Eagle3 导出为 MNN 的代码路径 | ✅ 完成 |
| [LLM QNN 离线导出流程](/p/llm-qnn-export/) | `generate_llm_qnn.py`、`compilefornpu` 和 QNN context binary 生成链路 | ✅ 完成 |
| [LLM Sampler 采样逻辑](/p/llm-sampler/) | `Sampler` 从 logits 到下一个 token 的过滤和选择流程 | ✅ 完成 |
| [LLM QNN Plugin 推理流程](/p/llm-qnn-plugin-infer/) | `Plugin(type="QNN")` 从 shape 匹配到 `graphExecute()` 的运行时路径 | ✅ 完成 |

---

## 相关链接

- **MNN 官方仓库**：[alibaba/MNN](https://github.com/alibaba/MNN)
- **MNN 文档**：[mnn-docs](https://mnn-docs.readthedocs.io/)
- **问题反馈**：[GitHub Issues](https://github.com/huluhuluu/MNN-TUTORIAL/issues)
