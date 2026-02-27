---
title: "MNN端侧部署教程"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-02-26T23:00:00+08:00
draft: false
description: "MNN端侧推理部署--从环境配置到读懂代码"
slug: "mnn-tutorial"
tags: ["MNN", "端侧部署", "LLM"]
categories: ["MNN端侧部署"]
comments: true
---

# MNN端侧部署教程

本教程介绍 [MNN](https://github.com/alibaba/MNN) 端侧推理部署的完整流程，从环境配置到核心概念讲解，再到 LLM 部署实践。

## 教程概览

| 阶段 | 内容 | 文章数 |
|------|------|--------|
| 环境配置 | 远程调试、交叉编译 | 2 篇 |
| 核心概念 | Backend、工厂模式、核心类 | 3 篇 |
| MNN-LLM | 配置、加载、推理 | 3 篇 |

---

## 1. 环境配置

搭建端侧开发调试环境的完整指南。

| 文章 | 说明 | 状态 |
|------|------|------|
| [远程ADB环境配置](./blog/remote-adb.md) | 通过端口转发链路，让内网服务器直连本地手机 | ✅ 完成 |
| [交叉编译环境配置](./blog/cross-compiler.md) | Android NDK 配置、Clangd 配置、远程调试 | ✅ 完成 |

---

## 2. 核心概念

深入理解 MNN 框架的设计理念与核心组件。

| 文章 | 说明 | 状态 |
|------|------|------|
| [Backend 介绍](./blog/introduce-backend.md) | CPU/OpenCL/Vulkan 等后端的作用和选择 | ✅ 完成 |
| [工厂模式介绍](./blog/introduce-factory.md) | MNN 中工厂模式的设计与应用 | 📝 TODO |
| [核心类介绍](./blog/introduce-core-class.md) | VARP、Expr、Op 等关键类的设计 | 🚧 WIP |

---

## 3. MNN-LLM

端侧大语言模型部署实践。

| 文章 | 说明 | 状态 |
|------|------|------|
| [LLM 配置](./blog/llm-config.md) | 模型配置、量化配置 | 📝 TODO |
| [LLM 加载流程](./blog/llm-load.md) | 模型文件到推理就绪的完整过程 | 📝 TODO |
| [LLM 推理流程](./blog/llm-infer.md) | Token 处理、KV Cache 管理 | 📝 TODO |

---

## 相关链接

- **MNN 官方仓库**：[alibaba/MNN](https://github.com/alibaba/MNN)
- **MNN 文档**：[mnn-docs](https://mnn-docs.readthedocs.io/)
- **问题反馈**：[GitHub Issues](https://github.com/huluhuluu/MNN-TUTORIAL/issues)
