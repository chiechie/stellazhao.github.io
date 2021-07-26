---
title: ml_platform_nots
author: chiechie
mathjax: true
date: 2021-07-26 11:35:00
tags: 
- 最佳实践
- 人工智能
- 时序预测
- 量化交易
- 读书笔记
categories: 
- 阅读
---

- 细分领域：时间序列预测
- 核心成果：针对时间序列预测问题，本文设计了一个端对端的ml系统具体来说，核心创新点包括： 
- 文章性质：工程类；系统设计类；

# 摘要：
- 背景： 时间序列预测的应用 **很重要**，相应的工业级别的解决方案 **研究不足**
- 核心：本文的**亮点/核心成果**，针对大规模时间序列预测问题，本文设计了一个端对端的ml系统
- 详细成果：（发现了什么、证明了什么、测试了什么）
    - 对大规模时序预测的四种建模方式提供了快速的分布式训练的方案（包括 基于FORTAIN的L-BFGS实现，数据并行方案），是基于spark实现的：
        - local：数据并行
        - global：最简单
        - local with 人工指定分组： 人工给出先验
        - global and local：人工不指定分组，给出商品的元信息，让模型去学
    - 提供了数据处理的high-leve API：用户只用关心dataflow的数据源，处理的算子（逻辑流），背后真正执行的DAG（数据流）会被系统重写（优化计算效率）
    - 抽象出了路由功能 和 集成功能（使用命令式编程的方式 就可以实现 1. 给每个算法or策略分配 sub group样本用来训练， 2 给每个sub group 分配多个算法，并指定集成的权重）
- 意义：在实践中有很大的价值

# 引言
- 工作的出发点？
- 想解决的问题是什么？
- 以及他们的主要发现和意义在哪里？
- 出发点：
    - 没有好用的时序预测的工具
- 想解决的问题：
    - 在工业界进行大规模时序预测的问题
- 难点：
    - 系统层面：又要 稳定 又要 灵活
        - 提升稳定性的代价：代码复杂度，缺少抽象导致 没法快速实验
        - 怎么解决？
            - 一般的思路：
                - 算法人员：使用一套语言（keras类似）快速实验和进行算法开发，在实验环境
                - 工程人员：将模型重新实现一遍，在生产环境上线
                - 适用场景：实验只需要单机训练；
                - 不适用场景：
            - 本论文的解决方案：将工程人员的工作沉淀为平台工具，如
                - 提供high-level的数据处理API
                - 后台支持分布式学习，对算法人员屏蔽
    - 使用层面：实验阶段 希望快速出结果 ，生产阶段希望自动化地配置各种任务，包括超参搜索
    - 算法层面：传统的时序算法只能解决历史数据充分，周期模式，促销模式比较stable的场景，实用性有限，需要将传统方法和深度nn方法结合
- 做了什么：
    - 端对端的ml平台，包括数据采集，预处理，分布式训练，集成预测，high-level的数据处理API
    - 针对多时序预测，后台提供四种模型训练方式
    - 数据处理的high-leve API，以及配套的DAG优化。

# 细节
- 背景
    - ![时序预测需求](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2Fsg4v0LRFEh.png?alt=media&token=ec469b3d-4912-4f27-9e1d-080f37bb27fa)
        - 黑色：观测值
        - 绿色：训练阶段，0.9 和 0.1 的分位数
        - 红色：未知数据上推断，0.9 和 0.1 的分位数
- 系统架构：
    - ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FdxvYHgGI4v.png?alt=media&token=ebc71e8e-6863-42d3-8b2c-e5f8159c2a8d)
    - 【重点】高级的数据处理API，让用户只用关心 数据源，算子组装  逻辑，平台会根据用户的声明式编程语言（declarative ），生成计算图（DAG）和 优化（合并相同计算逻辑，生成缓存节点）
        - ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FKE2kXC_eEL.png?alt=media&token=612d833e-359e-44d3-9f86-42d0c152d120)
- 性能评估：
    - 可伸缩性：
        - 可伸缩性(可扩展性)是一种对软件系统计算处理能力的设计指标，高可伸缩性代表一种弹性，在系统扩展成长过程中，软件能够保证旺盛的生命力，通过很少的改动甚至只是硬件设备的添置，就能实现整个系统处理能力的线性增长，实现高吞吐量和低延迟高性能。
        - ![Scalability with increasing data size/increasing data and cluster size/](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FHM7NBjHd7i.png?alt=media&token=75fe08c7-f683-4171-bba5-1e9fc9506e50)
        - ![increasing the time dimension of the
        feature matrices.](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2F77qGodt9K1.png?alt=media&token=d18c276a-4fe5-4b89-8316-519a5a493902)

- 几个问题？
    - 预测多个时序，使用全局模型和局部模型，系统后台分别是怎么设计的？为什么要这么设计（难点 或者 遇到的问题 是什么？怎么解决的）
    - 时序预测的后台能否拓展到其他场景，或者更general 的机器学习场景中？
    - 预处理 和 特征转换 阶段，专门针对时序数据做了哪些改进？提供了哪些算子？
    - 参数服务器是数据平行还是模型并行？
    - 
- 一些用户需求：
    - 预测某个类别（如电话 ）商品的销量
    - 历史数据缺失 比较常见：比如某段时间，商品卖脱销了
    - 有些商品会停产，使用混合策略预测：预测是否inactive + zero预测器，
    - 圣诞节前几周，销量可能会上升（跟商品自己无关）
    - 对高销量（看每周的平均销量 在所有商品中的排名）商品，使用混合策略，各自权重为0.5
        - baseline 模型
        - GLM模型
    - 对没有被路由到的商品，使用baseline模型兜底
    - 如果单个learner失败了，商品预测就出不来了，使用ensemble策略比较好
    - 在模型开发的整个生命周期：先尝试小范围的baseline模型，后面不断的对特定的商品子类（subset）添加 specialized learners，来提升准确率
    - 对于批量训练多个local模型的场景，怎么保证准确率？依据经验，将tolerance 设置的很低，将num_iteration设置为一个固定的值
    - 如果又想要在不同曲线之间迁移，又想保持各自的特点，可以这么做：将共享的向量弄小一点，或者 提供自由度：让每个曲线都可以选择自己的特征。
    - 开发模型阶段：临时性的需求，在小数据集中适用小规模实验来调试算法和 finetune模型，或者在集群上跑一个算法，来测试这个模型在大数据集上的效果（托管到我们平台），这种需求来自数据科学家，一般是高频手动，主要诉求是：易于执行 &快速出结果
    - 对系统的自动化需求：在生产阶段会用到集群，服务的可用性有严格的标准。自动化的模型选择，自动的模型更新（固定特征集，超参），自动化的模型集成策略。
    - 新发布一个模型之前要回测
    - 面对众多需要预测的商品时，使用二八原则，80%不重要的商品（例如历史数据稀疏），使用cheap解决方案（baseline 模型），20%重要的商品，使用expensive解决方案（nn，集成模型）
    
    
