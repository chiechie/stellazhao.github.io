---
title: 时间序列 数据合成
author: chiechie
mathjax: true
date: 2021-07-22 22:04:02
tags:
- 时间序列
categories:
- 时间序列
---



## 时间序列的特征生成

1. 对于统计模型和状态空间模型，需要用到时序中所有的数据点来拟合模型
2. 为了将机器学习技术应用到时间序列上面来，需要做特征生成
3. 特征生成：将时间序列数据中重要的信息表达为几个数值和类别标签。实际上是一个信息压缩的方法。
   例如,一个时序可以表达为均值和长度。
4. 传统的机器学习模式最初不是为时间序列数据设计的,因此它们不可以直接用于自动用于时间序列分析应用
5. 然而,使这些模型在时间数据发挥作用的一个办法是特征生成。
6. 例如,通过一系列特征集合来描述某个时间序列, 就可以使截面数据（cross-sectional ）使用的方法了。
7. 短时间序列做特征生成的例子,接下来看一下自动化的特征生成和特征选择，之后就具备了下游机器学习应用所需要的预处理的全部技能。
8. 时序的特征比如，周期性，斜率，均值。
9. 遍历性（ergodic）：
10. An *ergodic* time series is one in which every (reasonably large) sub‐ sample is equally representative of the series. This is a weaker label than stationarity, which requires that these subsamples have equal mean and variance. Ergodicity requires that each slice in time is “equal” in containing information about the time series, but not necessarily equal in its statistical measurements (mean and var‐ iance). You can find a helpful discussion on StackExchange.
11. 蒙特卡洛采样可以生成iid的样本，但是要生成有依赖关系的样本就不行了，可以使用马尔可夫链。











## 参考

1. [practical time series:prediction with statistic & machine learning](https://ipfs.io/ipfs/bafykbzaceajhmnmehz7amkjofpsxmsymk5u6ph4mrzytlp5zq7wf7ouhqkre2?filename=Aileen Nielsen - Practical Time Series Analysis_ Prediction with Statistics and Machine Learning-O’Reilly Media (2019).pdf)

