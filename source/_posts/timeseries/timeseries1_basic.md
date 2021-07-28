---
title: 时间序列1 时间序列技术介绍
author: chiechie
mathjax: true
date: 2021-07-21 15:23:49
tags: 
- 最佳实践
- 人工智能
- 时序预测
- 量化交易
categories: 
- 时间序列
---



## 时间序列相关技术概览

1. 时间序列结构比截面数据的信息量更多
2. 时序分析工具可以分析所有的序列类型数据，不一定是时间序列。
3. 对时间序列数据分析的过程包括：预处理，建模和评估
4. 对时间序列预处理包括，缺失值填充，时序数据合成。
5. 关于数据数据合成：蒙特卡洛采样可以生成iid的样本，但是要生成有依赖关系的样本就不行了，可以使用马尔可夫链。
6. 关于时间序列模型：常用的时间序列模型有统计模型，状态空间模型，机器学习/深度学习模型。
7. 传统的统计模型：arma/arima/sarima/ arima/state space/exponential smoothing平稳性检验。
8. 深度神经网络模型：卷积神经网络/全连接神经网络/循环神经网络/基于Seq2Seq的模型，以及是否基于attention。
   1. 基于全连接网络的模型：MQ-DNN
   2. 基于卷积的神经网络：TCN/CNN-QR/
   3. 基于隐藏状态的模型：DeepFactor/DeepStateSpaceModel
   4. 基于循环神经网络：SimpleRNN/LSTM/GRU，也可以加入attention层，变成基于重要性的神经网络。
   5. 基于Seq2Seq的模型：DeepAR/Transformer/Bert/GPT，其中Transformer/Bert/GPT是基于attention的。



## 时间序列预测

1. 时序预测中的一些难点：
   - 现实中需要对多条曲线建模，但是单个曲线去建模，人工成本高
   - 不同曲线模式不一样（最显而易见的，scale不一样），而学习一个global model对nn来说挑战很大（nn擅长局部建模）
   - 预测概率分布
2. 传统统计模型只对单个曲线做预测，缺点是要一个个曲线去做模型选择，并且每个样本曲线需要积累大量的历史数据。基于深度学习或机器学习的时序预测技术可以对多个曲线一起建模。





## 时序异常检测

1. 对时间序列进行异常检测，难度小于对时间序列进行预测。

2. 对时间序列进行异常检测的流程：
   1. 找出最匹配模式：用拟合程度来判断最符合的是哪个模式。
   2. 使用模式来预测未来范围：给一个区间。





## 参考

1. [practical time series:prediction with statistic & machine learning](https://ipfs.io/ipfs/bafykbzaceajhmnmehz7amkjofpsxmsymk5u6ph4mrzytlp5zq7wf7ouhqkre2?filename=Aileen Nielsen - Practical Time Series Analysis_ Prediction with Statistics and Machine Learning-O’Reilly Media (2019).pdf)

