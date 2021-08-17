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



观察到的效果：
$$
y_{i}=t_{1} Y_{1}\left(x_{i}\right)+\left(1-t_{1}\right) Y_{0}\left(x_{i}\right)
$$
t表示是否接受治疗，不可观测的反事实效果是：
$$
y_{i}=\left(1-t_{1}\right) Y_{1}\left(x_{i}\right)+t_{1} Y_{0}\left(x_{i}\right)
$$
因果推断的难点在于，我们永远只能观测到病人的其中一种情况：

![image-20210817205524554](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/causal_analysis/medical-ml1/image-20210817205524554.png)



上图中，有一些点不在曲线上，是因为我们允许一些随机性存在，然而虚线代表了可能的效果，圆圈代表现实观测值。

我们只能观测到每个病人在某个处方下的状态，而不知道反事实的状态。

为了填充反事实这个确实值，我们假设：



对比简单的平均诊断估计和真实平均诊断效果，

计算使用处方B的病人的平均血糖水平：

计算使用处方A的病人的平均血糖水平：

计算两者的差，得到0.75，说明A能够降低血糖



如果，我们现在开启上帝视角，知道了每个人在处方A/B下的血糖水平，那么，计算二者的均值，求差，得到平均诊断效果ATE为-0.75，

也就是说，朴素估计方法得到了错误的结论

##  Neyman-Rubin因果模型中的假设


直觉来说，反事实推断不可能，因为没有观测数据。

但是我们可以加上一些限制条件，使得估计反事实预测成为可能，这个限制条件也叫Neyman-Rubin规则，包括三个假设：

- Stable Unit Treatment Value Assumption (SUTVA).
- Ignorability假设
- Common support



Ignorability假设：即，T和Y0，Y1之间不存在不可观测的混杂因子（confounders），也就是说给定x时，最终效果（血压高低/肺癌）Y1，Y2跟治疗方案T条件独立：

$$ \left(Y_{0}, Y_{1}\right) \Perp  T \mid X $$

对应到因果图中就是，T到Y0和Y1没有边。

这个假设是必要的么？是！考虑一种igorability不满足的情况：存在某个隐变量h，同时会影响Y1，Y2和T。

怎么验证ignorability条件是否满足？需要跟一个该领域的专家交流，从而保证影响一生决策（T）和最终效果（Y1，Y2）的所有因素都已经考虑到了和观测到了。

即使没有观测到，影响大么？---敏感度分析（ sensitivity analysis）。

- Common support

The common support assumption says that there should always be some stochasticity in treatment deci- sions. That means that any group of patients/features should have nonzero probabilities for all considered treatments. The above statement could be formulated in the following way:he expression above goes by the name of propensity score: the probability of receiving the treatment for each individual. Thus we assume, that this probability is bounded between 0 and 1, while any violation of this condition is going to completely validate the conclusions drawn from the data.




## 几种因果推断的方法

因果推断的方法有很多，比如协变量调整(covariate adjustment )/ propensity score, doubly robust estimators, matching,


## 协变量调整

协变量调整是一个外推的自然的方法，这个方法的主要目标是，找到函数f(x, T), 输入病人的诊断情况，治疗方案T，输出可能的后果，可以认为是对条件概率的近似$p(Y_t|x, T = t).$

也就是说，协变量调整的方法就是构造一个回归模型，显示对治疗方案，cofounders和疗效三折的关系建模，

使用协变量调整方法估计出来的平均治疗效果（ATE）和CATE是：

$$ACE = \sum_i (f(x_i, 1) - f(x_i, 0))$$

$$CACE(x_i) =  f(x_i, 1) - f(x_i, 0)$$

那么，我们如何知道这个回归模型学到的变量之间的关系是有意义的呢？

举个例子，治疗决策可能只是影响疗效的众多因素中的一个，可能有其他因素对疗效的影响更大，这样就可能导致回归模型学到的治疗方案（t）跟疗效（y）的相关性很小。（自相关系数和偏相关系数的差别）

这就是因果推断和机器学习不一样的地方，本质上是二者关注的目标不一样，机器学习关注预测结果的准确性，因果推断更关注学习到的因果关系，在高维问题中，二者差别更加明显。

什么时候协变量调整（covariate adjustment ）会失败？当数据违反了common support/overlap的假设时，协变量方法会失败。
1. 没有充分的数据：回归模型没有充足的数据对y做外推。
2. 没有选对函数族：即使有了足够多的用来外推（extrapolate）的数据，还需要选对函数族。

> 假设x和y的映射函数实际上是二次的，而我们如果使用线形函数拟合，会得到错误估计。




## 参考



1. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/ 
