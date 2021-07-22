---
title:  chapter 5 自动化机器学习
author: chiechie
mathjax: true
date: 2021-04-29 08:32:24
tags:
- auto ml
- 人工智能
categories:
- 机器学习
---

## 总结

联想到之前构建的通用异常检测模型，整体流程大概是这样：定义多个检测器，曲线特征，并用一个随机森林去做集成。希望随机森林可以将不同的曲线路由到适合的检测器上去。

FBNet的思路跟上面的流程有点像，都是数据驱动去选择适配的组件。不同的地方在于，FBNet最终会减枝，即从9✖️20个可能的路径中，裁剪成一条路径。

FBNet要解决的任务更简单，给一个小任务找到适配的解决方案；通用异常检测更难，要给n个小任务找解决方案，并且用一套复合的解决方案。

NAS是auto-ml的一个子领域，相关的技术有随机搜索，强化学习，基于微分的方法，如DARTS 和 FBNet。

DARTS 和 FBNet是目前做NAS的the state of arts方法

下面简单介绍FBNet的思路

> 「贝叶斯优化」技术是求解黑盒最优化问题的一种方法，该技术可拓展到auto-ml的自动调参场景中。
>
> 接下来介绍下「贝叶斯优化」的思路
>
> 采集函数（acquisition function）下面有些地方也用效用函数代替






## FBNet的基本idea

1. 定义候选block, eg10个
2. 定义神经网络的层数,eg20，神经网络的每一层都是从9个候选block中选一个，搜索空间9^20
3. 构建一个很大的神经网络--supernet，每一层由9个候选block并联而成。supernet的待估参数包括9个候选block✖️20层，以及长度为9的权重向量✖️20。
	
	> 人工调参，可以看成是，指定supernet中的一条路径，然后灌入数据，来估计路径上20个blcock的参数。使用FBNet就不用人为指定其中某条路径，而是这个大网络supernet从训练数据中，学习出9个block✖️20（层）个参数，以及9✖️20（层）个权重。训练完之后，选择每一层权重最大的block并且连成一条路径，就是机器选定的20层的网络结构。
4. 模型要落地，还要考虑计算效率。具体的做法，事先算好候选block各自的latency(做一次推断平均耗时多久，block的权重随机给，跑多组数据求平均时延），然后将latency信息加入损失函数中，作为错误率之外的另外的一个惩罚项。



## 黑盒优化和自动调参

什么是「黑盒优化」？对一个黑盒函数求最优解。

求解 $ \max_{x \in \mathcal{X}} f(x)$，但是f(x)是个黑盒函数，我们对f内部运行机制一无所知，只知道输入输出。



调参的过程，少部分经验来自算法理论和工程实践，大部分来自试错。

模型越复杂，单次训练的时间就越长；

超参数越多，即x的维度就越高，搜索超参数的空间就越大，

两者都导致调参工作随着模型的复杂度增加而大幅增加。



那么有没有什么技术能将人力从跳参的工作中解放出来呢？即有没有「自动调参」的技术呢？

有，可以将该问题建模为一个黑盒优化问题，然后使用相关技术求解。

即将「自动调参」定义为这么一个优化问题：

求最优的一组超参数，使得组超参训练出来的模型的准确率最高
$$\max\limits_{x \in \mathcal{X}} f(x)$$

- x代表一组超参数的取值，是一个向量，每个元素代表一个超参的取值，向量长度等于超参的个数。

- f(x)代表某组超参对应的模型的准确率。

  

求解黑盒优化问题比较常用的方法有贝叶斯优化算法（Bayes Optimization），随机搜索，网络搜索, 以及贝叶斯优化的变形（SMAC和TPE）

## 随机搜索，网格搜索和贝叶斯优化

先看两种最简单的方法，随机搜索 和 网格搜索

![随机搜索和网格搜索](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/AI/grid_random_search.png)


- 网格搜索：如左图，需要指定搜索范围和间隔， 优点：考虑到了搜索空间内所有的参数组，缺点：存在组合爆炸的问题.
- 随机搜索：意思是，在超参空间进行随机采样，将采样得到的超参作为best guess。如右图，采用随机采样的方式得到新的超参。优点：容易理解。缺点：可能搜索不到最优的超参.


- 贝叶斯优化: BayesOptimazation(r，p, m，n_iters，算法名称)，优点：不需要指定搜索范围，可自动调节搜索过程中的步长

- 贝叶斯优化的变形-SMAC和TPE

  - SMAC：代理函数变成了回归随机森林
  - TPE：代理函数变成了高斯混合模型
    ![img.png](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/AI/bianxing.png)
- 实际中，随机搜索其实已经很有效了。


## 贝叶斯优化

贝叶斯优化算法（Bayes Optimization）大概思路是这样的：

为了求解该问题：
$x^{*}=\arg \max _{x \in \mathcal{X}} f(x)$

1. 收集样本$<x_i,f(x_i)>$，x是一组超参，f(x)表示这组超参对应的模型准确率。
2. 使用样本$<x_i,f(x_i)>$拟合一个高斯过程回归器（GaussianProgressRegressor）$\hat G$
3. 基于$\hat G$构造一个效用函数。效用函数是用来衡量当前的超参数组成的空间中，探索每个区域的潜在收益或者效用（注意我们的目标是，找到$x^\star$，使得f(x)取最大）。 最简单的acquisition function就是均值加上n倍方差（Upper condence bound算法）。
4. 搜索acquisition函数的最大值，即最值得探索的点（Next Best Guess）。
5. 将找到Best Guess（一组超参）带入原来算法，重新训练一遍，并记录下此时模型的准确率。
6. 将5得到的新样本数据加入样本集中，回到第二步，进行下一个迭代，循环往复，直到模型的准确率没有提升或者达到最大的迭代次数。



## 附录



### 贝叶斯优化和强化学习


贝叶斯优化跟强化学习有些许相似之处

![image-20210714194513883](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/AI/image-20210714194513883.png)





### 贝叶斯优化图

![贝叶斯优化](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/AI/img.png)

https://distill.pub/2020/bayesian-optimization/

上图中

- 横轴是超参，
- 上面图的纵轴：超参数取不同值时，模型的准确率。 
  - 黑色的线代表拟合出来的均值线，灰色区域代表拟合出来的置信区间
  - 红色的线代表超参和准确率的真实关系 红点代表下一次要探索的点
- 下面图的纵轴：超参数取不同值时，采集函数的值，取值越大，表示该区域越有可能存在最优超参。


### 代理函数

代理函数，就是对黑盒模型性能的评估函数。
原始的贝叶斯优化算法，使用的是高斯过程，后面一些变形就变成了其他的算法。


### 采集函数

怎么设计采集函数（acquisition function）？

需考虑2个因素--期望和不确定性。期望和不确定性都很大的区域，值得对重点探索，所以采集函数给予这些区域更高的分支


### 难点离散变量

贝叶斯优化本身只适用于连续变量，但是实际上很多模型地参数都是离散的，那么如何解决呢？ 直觉上来说，有两个思路:

1. 混合搜索（贝叶斯+网格搜索）：整数类型的参数采用网格搜索或者随机搜索，然后连续型的超参求子参数空间的优化问题。
2. 贝叶斯模型：在连续空间搜多到best guess之后，截断成整数，加工成样本，然后送入下一次迭代


### 贝叶斯优化和梯度算法的区别是什么？

贝叶斯优化和梯度算法是不是一类东西呀？感觉都是在求最优秀解。

两者都是优化算法。但是贝叶斯优化的野心更大，他不需要目标函数的具体表达式, 只要知道这个目标函数的输入输出就好了。
梯度算法解决的问题更小，必须知道目标函数的表达式以及梯度。



## 参考资料

1. [贝叶斯优化-通俗版-tobe-知乎](https://zhuanlan.zhihu.com/p/29779000)
2. [贝叶斯优化-技术版—Dai Zhongxiang-知乎](https://zhuanlan.zhihu.com/p/76269142)
3. [贝叶斯优化-可视化版-distill](https://distill.pub/2020/bayesian-optimization/)
4. [开源工具-Advisor-github](https://github.com/tobegit3hub/advisor)
5. [谷歌的调参工具](https://cloud.google.com/ai-platform/optimizer/docs/overview)
6. [超参数优---贝叶斯优化及其改进（PBT优化）csdn](https://blog.csdn.net/xys430381_1/article/details/103871212)
7. [微软新工具 NNI 使用指南之 Tuner 篇-jianshu](https://www.jianshu.com/p/3587b24f1a6d)
8. https://docs.qq.com/flowchart/DVFVYeEZpd1NMTEtX

1. [wangshusen-slide-github](https://github.com/wangshusen/DeepLearning)
2. [Differentiable Neural Architecture Search-youtube](https://www.youtube.com/watch?v=D9m9-CXw_HY)
