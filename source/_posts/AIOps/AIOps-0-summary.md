---
title: chapter0 AIOps概览
author: chiechie
mathjax: true
date: 2021-05-06 16:03:30
tags:
- AIOps
- 异常检测
categories: 
- AIOps
---

## 目录

- [chapter0 概览](../AIOps-0-summary/)
- [chapter1 故障发现](../AIOps-1-event-generate/)
	- [chapter1.1 单指标异常检测](../AIOps-1_1-kpi-detector/)
	- [chapter1.3 故障预测](../AIOps-1_2-fault-prediction/)
	- [chapter1.4 指标异常关联](../AIOps-1_4-kpi-correlation/)
	- [chapter1.5 日志聚类](../AIOps-1_5-log-analysis/)
		- [chapter1.5.1 使用logmine加强版做日志聚类](../AIOps-1_5_1-log-analysis_logmine/)
		- [chapter1.5.2 美团日志聚类](../AIOps-1_5_2-log-analysis_meituan/)
- [chapter2 故障定位](../AIOps-2-event-analysis/)
	- [chapter2.1 微服务系统的故障定位](../AIOps-2_1-topo-rca/)
		- [chapter2.1.1 CauseInfer1](../AIOps-2_1_1-topo-rca-causeinfer-notes1/)
		- [chapter2.1.2 CauseInfer2](../AIOps-2_1_2-topo-rca-causeinfer-notes2/)
		- [chapter2.1.3 AIOps挑战赛2020-获奖方案分享](../AIOps-2_1_3-topo-rca-aiops2020/)
		- [chapter2.1.4 AIOps挑战赛2021-demo方案](../AIOps-2_1_4-topo-rca-aiops2021/)
		- [chapter2.1.5 N-Softbei2020比赛](../AIOps-2_1_5-topo-rca-cnsoftbei2020/)
		- [chapter2.1.6 MicroCause](../AIOps-2_1_6-topo-rca-MicroCause)
	- [chapter2.2 多维下钻根因定位](../AIOps-2_2-multi-dimensional-rca/)
	- [chapter2.3 调用链根因分析](../AIOps-2_3-trace_rca/)
	- [chapter2.4 时间序列关联性分析](../AIOps-2_4-metric_event_correlation/)
- chapter3 故障恢复


## 总结

智能运维（AIOps）全称是Artificial Intelligent Operation，即，使用AI技术辅助运维。现在业界和学术界研究的比较多的主要有9个场景，这10个场景按照监控的阶段可以分为故障发现阶段（6个场景）和故障分析（4个场景）两类。还有少部分人研究故障恢复阶段的场景，但是，这个场景对数据要求比较高，并且实施过程中，风险比较大，暂时不列。

故障发现阶段，根据监控的对象不同，可以细分为以下场景：

1. 单指标异常检测
2. 多指标异常检测
3. 故障预测
4. 指标异常关联
5. 日志聚类
6. 调用链异常检测

故障分析阶段, 根据需要定位的组件对象不同，也可以细分为以下几个场景：

1. 微服务系统的故障定位

2. 多维指标下钻分析

3. 调用链根因定位

4. 指标和事件的关联关系

   

从是否可以规模化的角度，AIOps场景又可以分为：基础监控和业务监控。基础监控的监控对象主要是物理设备，例如机器的健康度，网络的健康度。从监控对象（指标，日志等）到监控策略可以标准化一个或者多个固定的模版。

业务监控就需要跟具体的业务结合起来，具体来说，需要梳理相应的业务流程，每个流程需要监控哪些指标，分别使用哪些策略，这个解决方案是垂直领域沉淀出来的。

个大云厂商提供的AIOps服务大多属于基础层，做AIOps解决方案的中小型企业（比如必示，云智慧等），要么聚焦在基础层，要么在业务层做定制化方案，还比较少看到在某个细分领域，做成标准化方案或者标准化工具。

可能以后的发展趋势是，部分专注做解决方案的中小企业跟云厂商进行合作的模式。基础服务是免费模式，业务服务是收费模式，云厂商可以拿提成，中小企业可以享用云厂商的基建，而不用自建。
