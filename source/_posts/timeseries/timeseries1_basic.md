---
title: 时间序列预测和深度学习技术
author: chiechie
mathjax: true
date: 2021-07-21 15:23:49
tags:
categories:
---



## 总结 

1. 神经网络中专门有一些算子来处理时间序列结构，比如卷积神经网络中的因果卷积（causal conv），空洞因果卷积（Dilated Causal Convolution），空洞卷积（dialted conv）。这些卷积原理上都大同小异，都假设相邻时间内，数据的相关性很强，卷起来可以得到更高层的语义信息。

2. 空洞卷积本来是图像领域的概念，希望扩大接受域（ reception field）而不增加计算量。

3. 因果卷积类似于自回归算子，对前置窗口中的数据做线性组合。可以说自回归模型是因果卷积的退化版。

   > arima/sarima可以看成残差网络 + 因果卷积的特例

4. 膨胀因果卷积：空洞卷积 + 因果卷机的结合

5. TCN网络的结果：多个 膨胀因果卷积层的堆叠，加上了pooling（扔掉后置窗口）+ activation + Dropout（卷完之后的非线性转换+裁剪） + Resnet（残差连接避免梯度消失）

7. 时序预测方法的分类：传统时间序列和深度神经网络。

8. 传统时间序列模型：arma/arima/sarima/ arima/state space/exponential smoothing。平稳性检验。

9. 深度神经网络模型：卷积神经网络/全连接神经网络/循环神经网络/基于Seq2Seq的模型，以及是否基于attention。

   1. 基于全连接网络的模型：MQ-DNN
   2. 基于卷积的神经网络：TCN/CNN-QR/
   3. 基于隐藏状态的模型：DeepFactor/DeepStateSpaceModel
   4. 基于循环神经网络：SimpleRNN/LSTM/GRU，也可以加入attention层，变成基于重要性的神经网络。
   5. 基于Seq2Seq的模型：DeepAR/Transformer/Bert/GPT，其中Transformer/Bert/GPT是基于attention的。

10. 传统时间序列模型只对单个曲线做预测，缺点是要一个个曲线去做模型选择，并且每个样本曲线需要积累大量的历史数据。基于深度学习或机器学习的时序预测技术可以对多个曲线一起建模。

11. 其他领域如文本，图像，语音领域的技术都可以应用到时间序列，但是要注意时序场景中有几个特殊的地方
    1. 实践序列的下游的决策一般依赖输出概率分布，比如要做异常检测，就需要输出上下界。
    2. 预测target为整数时，不能使用常用的归一化方法。

12. 时序预测中的一些难点：

  - 现实中需要对多条曲线建模，但是单个曲线去建模，人工成本高

  - 不同曲线模式不一样（最显而易见的，scale不一样），而学习一个global model对nn来说挑战很大（nn擅长局部建模）
  - 预测概率分布



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

## 参考

1. [如何理解空洞卷积（dilated convolution）？ - 刘诗昆的回答 - 知乎](https://www.zhihu.com/question/54149221/answer/323880412)

2. https://github.com/vdumoulin/conv_arithmetic
