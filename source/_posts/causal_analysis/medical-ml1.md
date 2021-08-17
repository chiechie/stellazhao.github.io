---
layout: categories
title: 因果推断1
date: 2021-08-17 13:03:39
mathjax: true
tags:
- 因果推断
- 因果分析
- 贝叶斯
- 复杂网络
categories: 
- 因果分析 
---





在诊断时，我们想要回答：病人5年内趋势的概率？虽然可以训练一个深度学习模型来预测，但是这个方法可能很危险，模型学习的样本，可能有幸存者偏差。

举个例子，样本中，有的病人可能一直在接受治疗，所以生命得以延长，而如果我们忽略了如此的样本生成过程，就会学到一个虚假的X到Y的映射，从而得到一个关于新病人的病情预测的错误结论（模型似乎在说，你不用治疗，你看其他人都没有死掉，你也不会死掉）

![image-20210817173930873](./image-20210817173930873.png)



有没有什么办法能够让机器预测的比人还准呢？



吸烟会引起癌症吗？

吸烟和肺癌的因果关系曾经引起过很大的争论。

传统的方式，来买书这个问题：随机控制试错。

此外，做随机实验不现实时（不能要求一个不抽烟的人抽烟，然后观测他得不得肺癌），

那么如果只根据观测数据就回答这些问题呢？可以先估计条件likelyhood（吸烟的人中得肺癌的概率，不吸烟的人中得肺癌的概率），但是有个风险，可能存在混杂因子，即可能存在某个因子，会导致人更容易抽烟以及得肺癌。（比如工厂工作的男性）





## 从结论到动机



在医疗保健中，一旦涉及机器学习领域，需要谨慎思考，毕竟任命观天。

因此，跟传统的机器需诶下不一样，我们不仅仅需要输入输出，还需要加入第三个变量--干遇，interventions，

并且，我们还要思考这三个变量之间的因果关系，其中一种问题就是，如何区分贝叶斯有向图模型中的不同的因果关系。

![image-20210817181259614](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/causal_analysis/medical-ml1/image-20210817181259614.png)

先看下因果分析中，最简单的一种问题：假设边的方向知道，那么边的强度是多少？

治疗方案T 可能是二值，或者是连续，或者是向量。

先假设是二值向量，有向图显示，T依赖病人的诊断结果X，同时最终效果取决于诊断情况和治疗情况。



## 潜在效果框架





潜在效果框架Potential outcomes framework

类似吸烟和肺癌的关系在政治学/统计学/经济学/生物统计学中，已经被研究过很年了。

然而，统计学派使用的传统的方法，不适用于高维设置，在很多现代医疗应用中。

如何使用机器学习算法来回答因果问题，在这个高维数据的情况下。



引入一种语言，接下来先考虑rubin-neyman因果模型，$Y_0(x)$表示x不接受治疗的效果，$Y_1(x)$表x接受治疗时的效果。

对个体i来说，条件平均治疗效果（conditional average treatment effect）为：
$$
C A T E\left(x_{i}\right)=\mathbb{E}_{Y_{1} \sim p\left(Y_{1} \mid x_{i}\right)}\left[Y_{1} \mid x_{i}\right]-\mathbb{E}_{Y_{0} \sim p\left(Y_{0} \mid x_{i}\right)}\left[Y_{0} \mid x_{i}\right]
$$


整个人群中，平均治疗效果(average treatment effect ):


$$
A T E=\mathbb{E}\left[Y_{1}-Y_{0}\right]=\mathbb{E}_{x \sim p(x)}[C A T E(x)]
$$


![image-20210817190359970](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/causal_analysis/medical-ml1/image-20210817190359970.png)



## 参考



1. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/ 
