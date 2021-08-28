---
title: PageRank算法
author: chiechie
mathjax: true
date: 2021-05-22 18:07:11
tags: 
- 图算法
- 图
- 拓扑数据
categories: 
- 编程
---

> 最近需要对图数据进行分析，了解下PageRank的原理
> 
> BTW, PageRank还可以做社区划分？

## 总结一下

pagerank:
假设，网上冲浪的人都是漫无目的的人，他们在网页一个链接接一个链接点下去，但是，整个互联网的节点，最终流量可能的分布能达到一个稳态。
也就是说，随机分布于各网页的流量经过次数足够多的转移之后，会达到一个稳定的状态，这个也代表每个网页的信息度。

基于这个假设，可以构建一个模型，
$$V_t = M ^t * V_0$$​​

- M表示概率转移矩阵
- $V_n$​表示时刻t，网络流量分布

一般来说，可以拿到网页之间调用拓扑，经过转换，可以将这个拓扑变为概率转移矩阵M，最终算出稳定态的$V_t$​

给定点之间的连接关系，输出每个节点的分数，

补充下，最naive的方法存在终止点问题，也就是说，一个网站它不链接任何别的网站，或者一个微服务，它不调用其他任何微服务
按照上面的思路很可能出现一种情况，最终的流量全到这个自大狂网页那里去了，所以有一个改进的思路，假设冲浪着有一点点聪明，他并不是一味的接受灌输的链接，而是有一定概率，主动跳出去，到达一个新的站点，这个方式可以用下面的式子建模：

$$V_n = d * M * V_{n-1} + (1-d) * e $$

e = [1/n,...1/n]

d表示按照当前页面推荐的链接继续点下去的概率
1-d表示从当前网页跳出来，主动输入一个新网页重新开始的概率

## 代码

用代码验证一下，理解没有问题

https://github.com/chiechie/BasicAlgo/blob/main/pagerank.py

```shell
ground truth
 [[0.25419178], [0.13803151], [0.13803151], [0.20599017], [0.26375504]]
result
 [0.25419178 0.13803151 0.13803151 0.20599017 0.26375504]
```



---
title: 随机游走
author: chiechie
mathjax: true
date: 2021-05-23 12:52:46
tags:
- 因果推断
- 因果分析
- 贝叶斯
- 根因分析
- AIOps
categories: 
- 数据结构
---

## 介绍

已知因果图时，随机游走可以用来做因果推断。

原理是，基于因果关系，构建概率转移矩阵，通过模拟故障传播路径，得到故障根因。


## 随机游走算法-详细

- Step 1. 生成一个关系图G. $e_{ij} = 1$ 表示节点i是节点j的原因（之一）
- Step 2. 计算转移矩阵Q:
    1. 向前游走：从result节点到 the cause节点.理论上，跟异常节点越相关的节点，就越有可能是根因. 也就是说 $Q_{ij} = R(v_{abnormal}, v_j)$, $R(v_{abnormal}, v_j)$表示异常节点$v_{abnormal}$ 和 $ v_j$之间的相关系数, and $e_{ji} = 1$ 
    2. 向后游走：从cause节点到result节点. 为了避免算法陷入跟异常不相关或者低相关的节点，随机游走允许从cause节点跳出到result节点。
    如果 $e_{j i} \in E$且 $e_{ij} \notin E$, 那么 $$Q_{ji} =\rho R\left(v_{abnormal}, v_{i}\right),\rho \in[0,1]$$.
    3. 维持原状：如果一个节点，它的邻居们都跟异常节点的相关性很低，这个节点很有可能就是根因了，所以游走者应该停留在这里，
       $$Q_{i i}=\max \left[ 0, R\left(v_{abnormal}, v_{i}\right)- \max _{k: e_{k i} \in E} R\left(v_{abnormal}, v_{k}\right) \right]$$	
- Step 3. 对行做归一化，得到转移概率矩阵
  $$\bar{Q}_{i j}=\frac{Q_{i j}}{\sum_{j} Q_{i j}}$$
- Step 4. 在G上面随机游走，使转移概率$\bar{Q}$

采用类似pagerank的方法，得到每个节点的得分。

## 参考
4. [Page3,4 -paper](https://netman.aiops.org/wp-content/uploads/2020/06/%E5%AD%9F%E5%AA%9B.pdf)

1. [pagerank-csdn](https://blog.csdn.net/gamer_gyt/article/details/47443877)
2. [scikit-network-pagerank](https://scikit-network.readthedocs.io/en/latest/tutorials/ranking/pagerank.html)
3. [wiki-PageRank](https://zh.wikipedia.org/wiki/PageRank)
4. https://blog.csdn.net/google19890102/article/details/48660239
