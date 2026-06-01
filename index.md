---
title: "mnn-tutorial"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-05-27T12:00:00+08:00
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
| 核心概念 | Backend、工厂模式、核心类 | 3 篇 |
| MNN-LLM | 配置、加载、推理 | 4 篇 |

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
| [核心类介绍](/p/introduce-core-class/) | VARP、Expr、Op 等关键类的设计 | 🚧 WIP |
| [Backend 介绍](/p/introduce-backend/) | CPU/OpenCL/Vulkan 等后端的作用和选择 | ✅ 完成 |
| [工厂模式介绍](/p/introduce-factory/) | MNN 中工厂模式的设计与应用 | ✅ 完成 |

---

## 3. MNN-LLM

端侧大语言模型部署实践。

| 文章 | 说明 | 状态 |
|------|------|------|
| [LLM 配置](/p/llm-config/) | 模型配置、量化配置 | 🚧 WIP |
| [LLM 加载流程](/p/llm-load/) | 模型文件到推理就绪的完整过程 | 🚧 WIP |
| [LLM 推理流程](/p/llm-infer/) | Token 处理、KV Cache 管理 | 🚧 WIP |
| [Eagle 推理流程](/p/llm-eagle-infer/) | speculative decoding 中 Eagle 路径的 draft、验证与 KV 回写 | 🚧 WIP |

---

## 相关链接

- **MNN 官方仓库**：[alibaba/MNN](https://github.com/alibaba/MNN)
- **MNN 文档**：[mnn-docs](https://mnn-docs.readthedocs.io/)
- **问题反馈**：[GitHub Issues](https://github.com/huluhuluu/MNN-TUTORIAL/issues)
