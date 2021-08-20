---
layout: categories
title: 因果推断2
date: 2021-08-17 13:03:44
mathjax: true
tags:
- 因果推断
- 因果分析
- 贝叶斯
- 复杂网络
categories: 
- 因果分析 
---





## 总结



对于某个体（$x_i$），平均治疗效果：
$$
\operatorname{CATE}\left(x_{i}\right)=\mathbb{E}_{Y_{1} \sim p\left(Y_{1} \mid x_{i}\right)}\left[Y_{1} \mid x_{i}\right]-\mathbb{E}_{Y_{0} \sim p\left(Y_{0} \mid x_{i}\right)}\left[Y_{0} \mid x_{i}\right]
$$

> 也可以认为是，对于某个商品，模型上线前后效果的差别



总体的平均治疗效果：
$$
A T E=\mathbb{E}_{x \sim p(x)}[C A T E(x)]
$$

> 可以认为，对所有商品，模型上线前后效果的提升



反事实推断的两个基本方法：协变量调整（Covariate Adjustment）和Propensity scores



## 协变量调整



协变量调整就是说，显示地对治疗方案（T），混杂因子（x），结果（y）建模，构建一个回归模型，输入是<T,x>,输出是y

使用先行模型进行斜变量调整，

假设
$$
Y_{t}(x)=\beta x+\gamma t+\epsilon_{t}
$$

$$
\mathbb{E}\left[\epsilon_{t}\right]=0
$$

那么可以算出CATE：
$$
\begin{aligned} C A T E(x) &=\mathbb{E}_{p\left(Y_{1} \mid x\right)}\left[Y_{1}\right]-\mathbb{E}_{p\left(Y_{0} \mid x\right)}\left[Y_{0}\right] \\ &=\mathbb{E}_{\epsilon_{0}, \epsilon_{1}}\left[\beta x+\gamma+\epsilon_{1}-\beta x-\epsilon_{0}\right] \\ &=\gamma+\mathbb{E}\left[\epsilon_{1}\right]-\mathbb{E}\left[\epsilon_{0}\right] \\ &=\gamma \end{aligned}
$$
也可以计算出ATE:
$$
A T E=\mathbb{E}_{p(x)}[C A T E(x)]=\gamma
$$
对因果推断来说，希望很好地估计$\gamma$，而不是很好地预测Y，这一点跟ml不一样

更关注系数以及置信区间。

将协变量模型拓展到非线形模型上，假设数据生成过程是这样：
$$
Y_{t}(x)=\beta x+\gamma t+\delta x^{2}
$$


那么平均治疗效果ATE为：
$$
A T E=\mathbb{E}\left[Y_{1}-Y_{0}\right]=\gamma
$$


而，假设我们错误地建模成了线形关系：
$$
\hat{Y}_{t}(x)=\hat{\beta} x+\hat{\gamma} t
$$


那么对$\gamma$的估计就有可能非常离谱了
$$
\hat{\gamma}=\gamma+\delta \frac{\mathbb{E}[x t] \mathbb{E}\left[x^{2}\right]-\mathbb{E}\left[t^{2}\right] \mathbb{E}\left[x^{2} t\right]}{\mathbb{E}[x t]^{2}-\mathbb{E}\left[x^{2}\right] \mathbb{E}\left[t^{2}\right]}
$$


生物统计学家们通过小心谨慎地逐次添加非线性关系，来获得一个预测结果更精准的模型，有一下几种方法：

- 随机森林和贝叶斯树
- 高斯过程
- 神经网络



## 非线性模型


### 高斯过程



高斯过程，允许我们检测整体分布，以及比较两个诊疗方案的置信区间。

下图中，

如何确定给一个病人x开什么处方y呢？

可以计算CATE，如果大于某个值，就开治疗。

然而，这个策略可能是错的，因为没有考虑到我们对决策的置信度。

instead，我们希望有一个决策规则，可以量化我们的不确定度度，定义一个支持区域。

举个例子，如果P(CATE(x) > α) > 0.9,那么就治疗，负责就说，我们没充分的信息保证治疗方案有效。



使用高斯过程，既可以对两种治疗方案分开建模（左图），也可以构建一个联合分布（有图）

联合建模的好处是：两个治疗方案共享一套参数，参数估计需要的数据点更少，

![image-20210820142140615](./image-20210820142140615.png)








### 神经网络

也可以使用神经网络来学习非线性模型，





![image-20210820190705031](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/causal_analysis/medical-ml1/image-20210820190705031.png)



以下图为例，对输入施加多层非线性layer，然后使用一层treatment layer$\phi$，



注意，对不同的treatment，前面几层layer是共享的，从而可以学习联合表示。

接着，对不同的treetment使用个字的layer来获取不同的结果。



另外一个重要的要注意的是，当我们对输入convolve之后，再应用treatment，是因为treatment特征跟x一起在最前面出现的话，特征重要性由于没有x高，而被比下去了，因此信息就丢失了。





## 匹配

思路：找每个人的长期失联的兄弟，然后看他们身上的效果。

世纪钟，我们可以识别到另外一个$x_1$,跟$x_0$非常相似，但是不属于另外一个治疗组，接着可以评估x1的效果。

可以把这个值看成是对x0的反事实的估计，接着可以这个值来估计CATE和ATE：



1-NN Matching：








## 参考



1. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/ 
