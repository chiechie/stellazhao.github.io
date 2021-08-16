---
title: 因果分析
author: chiechie
mathjax: true
date: 2021-03-04 23:23:02
tags:
- 因果推断
- 因果分析
- 贝叶斯
- 复杂网络
categories: 
- 数据结构
---



# 附录

1. 

学术界有两种对因果关系建模的方法：因果图和结构方程。
对于构建因果图，目前使用的最多的是PC算法。

通过观察N个独立同分布的样本，来挖掘随机变量的之间的因果关系。
每个样本都是来自M个随机变量的1组观测值。

有点类似机器学习中，使用随机森林去学习特征的相关性。特征就类似因果图中的节点，也就是随机变量。





## Neyman-Rubin

直觉来说，反事实推断不可能，因为没有观测数据。

但是我们可以加上一些限制条件，使得估计反事实预测成为可能。

Neyman-Rubin因果模型依赖的假设有：SUTVA，Ignorability，Common support



Ignorability假设不存在不可观测的混杂因子（confounders），也就是说给定x时，Y1，Y2跟治疗方案T条件独立：

$$ \left(Y_{0}, Y_{1}\right) \Perp  T \mid X $$

对应到因果图中就是，T到Y0和Y1没有边。

这个假设是必要的么？考虑一种igorability不满足的情况：存在某个隐变量h，同时会影响Y1，Y2和T。

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



## 附录



##  基本概念

- 因果图：因果模型的一种表达方式，因果图是一个DAG， G=（V，E），对于每一个节点v，给定了他的双亲pa（v），v跟所有非后代都是独立的。
- 冲撞点（collider）：一种V型结构的因果图. 两个因导致一个果。
- 马尔科夫条件：可用来区分因果关系和相关关系，被用于生成依赖关系集合，构造因果图框架。
- faithfulness假设：可以用来保证，V中的所有变量的独立关系，都可以通过D-seperation表达。
- PC算法的两个假设：马尔可夫条件和faithfulness。





# 因果推断和因果发现



因果问题分为两种：一种是因果发现；一种是因果推断

- 因果发现（causal discovery）：给定一组变量，找到他们之间的因果关系，这个在统计上是不可能的。
- 因果推断（causal inference）：给定两个变量，找到一个衡量它们之间因果关系的参数；

# 两种数据产生途径

数据有两种产生途径：

- 一种是通过有意控制、随机化的实验得到的:例如拨测,能够直接做因果推断
- 一种是通过观测数据得到的: 需要另外知道一些先验知识，才能做因果推断。


# 用数学语言表达因果模型

因果模型有三种表达方式：

- 反事实（counterfactuals）
- 因果图（causal graph）
- 结构方程模型（structural equation models）


两个变量的因果关系可以从随机化的实验中得到；但是很难从观察到的数据中得到。


## 反事实（Counterfactuals）

反事实（counterfactual）：我们可以观察 $\left\{\left(X_{i}, Y_{i}\right)\right\}$ ，但是我们不知道如果对于某一个数据点 $\left(X_{i}, Y_{i}\right)$ ，如果改变 X 的值，Y 会怎么变。

下图，从数据上看，X 和 Y 是正相关的，但其实对于每一个 样本来说，如果增加X，会引起 Y 的减小。

举一个例子。研究航空公司票价（X）对销量（Y）的影响，显然，对于某一个客户来说，增加票价（X 变大）会降低客户购买意愿，即使得销量将达（Y 变小）。但是实际中的情况是，在节假日人们出行意愿大导致销量高（Y 大），定价也会相应变高（X 大）。

![反事实举例](./img.png)

## 因果图（causal graph）

因果图是一个DAG，表明各变量之间的联合概率分布。

$p\left(y_{1}, \ldots, y_{k}\right)=\prod p\left(y_{j} \mid \operatorname{parents}\left(y_{j}\right)\right)$

下面举例说明，在给定一个因果图之后，如何做因果推断。
考虑下面一个因果图，目标是求$p(y \mid \operatorname{set} X=x)$

- 首先，从因果图中得到信息： 
$$p(x, y, z)=p(z) p(x \mid z) p(y \mid x, z)$$

- 接下来，构建一个新图$G_{*}$, 移除掉所有指向 X 的边，得到新的联合概率分布:

$$p_{*}(y, z)=p(z) p(y \mid x, z)$$


- 最后，该概率分布下的数值就是因果推断的结果:
  
  $$p(y \mid \text { set } X=x) \equiv p_{*}(y)=\int p_{*}(y, z) d z=\int p(z) p(y \mid x, z) d z$$


## 因果图 和 概率图 的区别

因果图的箭头代表了因果关系方向，但是概率图（贝叶斯网络）没有这个要求，他不要求箭头表达因果关系。。

举例说明，比如下雨（Rain）和湿草坪（Wet Lawn）这两个事件经常一起出现。
下面两个DAG都是概率图，但是只有左边才是因果图。

![right_causal_graph](right_causal_graph.png)

总存在一个 faithful 的分布使得在样本足够多的时候，产生足够大的 type I error。



## 如何应用贝叶斯网络来做决策？

- 首先需要将先验知识表达成一个因果图--贝叶斯网络，（准确来说贝叶斯网络的边不仅仅表达因果关系，他表达所有的信息传播的方向）。
- 做推断

1. 构造贝叶斯网络
  ![img.png](./img.png)
  
2. 推断

已知P(R,W,S,C）, 求P(r)

### 枚举法

1. 先条件概率分布

  ![img_1.png](img_1.png)

2. 
![img_2.png](img_2.png)


### 变量消除法

![img_3.png](img_3.png)

![img_4.png](img_4.png)

所有非query变量祖先的变量，都应该被消去，当然不是真的消去啦，是对该消去变量求和，然后变成一个新的因子f。

依次迭代，直到没有可以消除的变量。

![img_5.png](img_5.png)

### 贝叶斯网络中的依赖性

1. 每一个随机变量都是条件独立于他的非后代节点，给定他的父母节点时，
2. 如下，给定A，B时，C和D是独立的。

![img_6.png](img_6.png)

3. 每个随机变量独立其他任何变量，当给定他的马尔可夫毯（Markov Blanket）
    - 父亲，孩子，以及孩子的父母亲（配偶）

![Markov Blanket](img_7.png)
      

## D-分离

> D分离（D-Separation）是一种用来判断两个变量是否条件独立的图形化方法，相比于非图形化方法，更加直观，且计算简单。 

定义: 设 X，Y，Z 是 DAG 中不相交的节点集合， $\pi$ 为一条连接 X 中某节点到 Y 中某节点的路径 （不管方向）。 如果路径 $\pi$ 上某节点满足如下的条件：

1. 在路径$\pi$上，w 点处为 V结构 , 即$X \rightarrow w \leftarrow Y$，且 w 及其后代不在 Z中；(因为collider关系，w不可观测时,X,Y 相互独立)
2. 在路径$\pi$上，w 点处为 非V结构 ，且w在Z中 。

则称Z阻断了路径$\pi$。 

如果Z阻断了所有的X到Y的路径，就说Z集合D分离了X和Y，记作$(X \perp Y \mid Z)_{G}$.


# PC算法

PC算法输入N个m维的样本，输出一个有向无环图（DAG），
举个例子，如果要构建一个故障因果图，一个指标就是图中的一个节点，每个时间点的观测值，就是一个样本。
如果给定一个变量集合，S，A和B独立，即 A ⊥ B|S，那么A和B两个节点就没有边。

PC算法分为四步:

1) 构建一个全连接图，M个随机变量就对应M个节点
2) 条件独立性检验：对于每一组相邻的节点，给定置信度$\alpha$, 如果条件独立，那么两个变量之间的边就remove掉。
   条件变量S的大小每次迭代递增，直到没有多余的变量可以被加入S
3) 基于v-structure确定边的方向.
4) 确定剩下的边的方向，传播.

PC算法是一种发现因果关系的算法，在满足一定的假设的前提下（基于马尔科夫条件 和 D-separation），使用基于统计的方法（显著性检验-$G^2$），推导出因果关系（代表因果关系的DAG）。实现流程包括三步:

![图1-PC算法](pc-overview.png)

## 条件独立性检验

从全连接图开始，移走条件独立的边

![图2-确定框架](pc1.png)

Z从空集开始，不断增加

具体来说，可以用指标$G^2$（条件交叉熵）来检验给定Z时，X和Y的独立性，这里X，Y，Z是互斥的。

$G^{2}=2 m C E(X, Y \mid Z)$

- m：样本大小
- $ C E(X, Y \mid Z)$:给定Z时，X和Y的条件交叉熵，

> 显著性低于0.05，就说明条件分布和联合分布没有显著性差异，有因果关系
> 显著性超过0.05， 就说明条件分布和联合分布没有显著性差异，没有因果关系，是独立的。

如果$G^2$超过一个预先设定的显著性，比如0.05， 那么X和Y是条件独立的。

PC算法使用一个全连接的无向图，然后使用$G^2$描述条件独立性。


## 确定边的方向

当DAG的架子搭起来了， 接下来的工作就是决定因果关系的方向了。这里使用到了D-seperation的方法。 

![图3-确定方向](pc2.png)

## 传播方向

![图4-传播方向](pc3.png)


# 参考

1. https://www.youtube.com/watch?v=gRkUhg9Wb-I&t=3190s
2. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/
3. [2007-The Journal of MachineLearning Research-pc算法](https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf)
4. [pc算法-youtube](https://www.youtube.com/watch?v=o2A61bJ0UCw)
5. [因果图的基本概念-知乎](https://zhuanlan.zhihu.com/p/269625734)
6. [Page,3-paper](https://netman.aiops.org/wp-content/uploads/2020/06/%E5%AD%9F%E5%AA%9B.pdf)
7. [zhihu-关于因果推断公开课翻译](https://zhuanlan.zhihu.com/p/88173582)
8. [英文原文](http://www.stat.cmu.edu/~larry/=sml/Causation.pdf)
9. [Bayesian Networks-youtuybe](https://www.youtube.com/watch?v=TuGDMj43ehw)

