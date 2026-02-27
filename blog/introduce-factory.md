---
title: "MNN 工厂模式介绍"
date: 2026-02-26T23:00:00+08:00
lastmod: 2026-02-26T23:00:00+08:00
draft: true
description: "介绍MNN框架中工厂模式的设计与应用"
slug: "introduce-factory"
tags: ["MNN", "设计模式", "C++"]
categories: ["MNN端侧部署"]
comments: true
---

# MNN 介绍
## 1. 工厂模式

NN框架支持多种硬件后端和神经网络算子，通过经典的工厂模式（Factory Pattern）提供代码支撑。