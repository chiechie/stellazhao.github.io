---
title: 时间序列4 时间序列预测应用之股价预测
author: chiechie
mathjax: true
date: 2021-06-05 00:06:21
tags: 
- 最佳实践
- 人工智能
- 时序预测
- 量化交易
categories: 
- 时间序列
---



## 总结

1. 使用机器学习算法去预测，整体流程如下:
   1. 搜集原始数据: 过去十年的成交数据
   2. 数据筛选: 选择交易日，固定时间段（９:30~15:30)的数据
   3. 对选好的数据进行采样，并不能保证每天都有整数个点
   4. 切分训练集 & 验证集
   5. 模型训练
       - 归一化训练集，<x， target>
       - 训练
   6. 模型评估
       - 归一化验证集
       - 预测&逆归一化&画图：

2. 不同品种的训练数据，分组训练 比 汇总训练 效果好。
	
	> 一定要汇总训练的话，要加入每个品种的静态特征，不然会学到混乱的模式。
	
3. 评估不仅仅要看mse，也要看correlation

4. 过拟合要check训练集 的比例是否远大于 验证集 和测试集。

5. 预测的准不一定能挣钱，预测的下游任务是交易策略，交易是否盈利要考虑滑点，交易费用，爆仓风险。

6. 同样一个模型（LSTM），在不同的数据上预测，效果相差很大，是有可能的，比如商品期货vs股指

   - 频率为1min的股指数据平滑，噪声少，而频率1min的商品期货数据很毛刺
   - 将商品期货聚合成5min之后，再使用过去1h的数据预测未来1h的数据，准确率有提升

   数据颗粒度太细时 噪声很大， 直接丢给模型，也可能造成模型学不好。

7. 直接用回归模型预测价格，可能有负数，就预测增长率，还可以建模成分类问题，换预测涨跌幅。

8. 怎么让模型记住更长期的信息？将特殊历史时刻的信息编码到特征中


![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2Fg6FAdLzkOj.png?alt=media&token=362ed41e-1773-4ff5-ac41-a3095f75bb86)


## 附录

 验证集的mse比训练集低，但相关系数却小于训练集，为什么？

mse和correlation评估的侧重点不一样。

mse小，但是correlation也小是有可能的，如下图：

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FNUSWvXNmWb.png?alt=media&token=8c03a9f3-60e1-424d-879e-3371c1516623)

几个距离：

- 欧式距离（mse）量化的 是两个实体（或者两个群体）之间的绝对物理距离
- 余弦距离（correlation）量化的是 两个实体（或者两个群体）相对原点的角度的距离
- pearson衡量的是 两个变量 之间的 线性相关性，具体做法是使用一个直线去拟合多组样本点<变量1，变量2> ，直线斜率就是相关性大小。


## 参考
1. [doc](https://github.com/Arturus/kaggle-web-traffic/blob/master/how_it_works.md)

