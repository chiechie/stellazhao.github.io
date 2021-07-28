---
title: 时间序列4-1 基于机器学习的时间序列预测方法
author: chiechie
mathjax: true
date: 2021-07-24 00:06:21
tags: 
- 最佳实践
- 人工智能
- 时序预测
- 量化交易
categories: 
- 时间序列
---



## 时间序列与深度学习

1. 时序预测中的一些难点：

   - 现实中需要对多条曲线建模，但是单个曲线去建模，人工成本高
   - 不同曲线模式不一样（最显而易见的，scale不一样），而学习一个global model对nn来说挑战很大（nn擅长局部建模）
   - 预测概率分布
2. 时序预测方法的分类：传统时间序列和深度神经网络。
3. 传统时间序列模型：arma/arima/sarima/ arima/state space/exponential smoothing平稳性检验。
4. 深度神经网络模型：卷积神经网络/全连接神经网络/循环神经网络/基于Seq2Seq的模型，以及是否基于attention。

   1. 基于全连接网络的模型：MQ-DNN
   2. 基于卷积的神经网络：TCN/CNN-QR/
   3. 基于隐藏状态的模型：DeepFactor/DeepStateSpaceModel
   4. 基于循环神经网络：SimpleRNN/LSTM/GRU，也可以加入attention层，变成基于重要性的神经网络。
   5. 基于Seq2Seq的模型：DeepAR/Transformer/Bert/GPT，其中Transformer/Bert/GPT是基于attention的。
5. 传统时间序列模型只对单个曲线做预测，缺点是要一个个曲线去做模型选择，并且每个样本曲线需要积累大量的历史数据。基于深度学习或机器学习的时序预测技术可以对多个曲线一起建模。
6. 神经网络中专门有一些算子来处理时间序列结构，比如卷积神经网络中的因果卷积（causal conv），空洞因果卷积（Dilated Causal Convolution），空洞卷积（dialted conv）。这些卷积原理上都大同小异，都假设相邻时间内，数据的相关性很强，卷起来可以得到更高层的语义信息。
7. 空洞卷积本来是图像领域的概念，希望扩大接受域（ reception field）而不增加计算量。
8. 因果卷积类似于自回归算子，对前置窗口中的数据做线性组合。可以说自回归模型是因果卷积的退化版。

> arima/sarima可以看成残差网络 + 因果卷积的特例

9. 膨胀因果卷积：空洞卷积 + 因果卷机的结合
10. TCN网络的结果：多个 膨胀因果卷积层的堆叠，加上了pooling（扔掉后置窗口）+ activation + Dropout（卷完之后的非线性转换+裁剪） + Resnet（残差连接避免梯度消失）
11. 其他领域如文本，图像，语音领域的技术都可以应用到时间序列，但是要注意时序场景中有几个特殊的地方
    1. 实践序列的下游的决策一般依赖输出概率分布，比如要做异常检测，就需要输出上下界。
    2. 预测target为整数时，不能使用常用的归一化方法。



## 附录




### 标准卷积

![Standard Convolution with a 3 x 3 kernel (and padding](https://pic3.zhimg.com/v2-d552433faa8363df84c53b905443a556_b.webp)



###  空洞卷积

![dilation](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/dilation.gif?raw=true)


### 膨胀因果卷积

![膨胀因果卷积](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FzEwTvSz-ee.png?alt=media&token=c46507da-4eca-4d43-970d-fd1180495d82)

- 

### CNN，RNN，基于attention的模型，对于时间序列建模

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2F_n2z_XQqI2.png?alt=media&token=facfccac-e8ac-4895-a84c-7add43cd165a)



### 多步预测-迭代法和直接法

- ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2Fcfl7jS1Uqc.png?alt=media&token=39859f26-511a-4d29-840f-8038bcaa824e)

  

###  wavenet

wavenet是什么？可视化长什么样？输入输出和参数个数分别是什么？

2. 

## 总结

1. 使用机器学习算法去预测，整体流程如下:
   1. 搜集原始数据: 过去十年的成交数据
   2. 数据筛选: 选择交易日，固定时间段（９:30~15:30)的数据
   3. 对选好的数据进行采样，并不能保证每天都有整数个点
   4. 切分训练集 & 验证集
   5. 归一化&模型训练
   6. 归一化验证集&预测&逆归一化&画图：

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

1. [如何理解空洞卷积（dilated convolution）？ - 刘诗昆的回答 - 知乎](https://www.zhihu.com/question/54149221/answer/323880412)
2. [conv_t](https://github.com/vdumoulin/conv_arithmetic)
