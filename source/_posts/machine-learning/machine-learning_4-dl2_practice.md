---
title:  chapter 4.2 backpropagation和训练小技巧
author: chiechie
mathjax: true
date: 2021-04-20 10:17:31
tags:
- 深度学习
- low level
- 最佳实践
categories: 
- 机器学习
---



## 反向传播（backpropagation）

> ** The problem with Backpropagation is that it is a** [**leaky abstraction**](https://en.wikipedia.org/wiki/Leaky_abstraction)**.**---Andrej Karpathy

![img](https://kratzert.github.io/images/bn_backpass/BNcircuit.png)

1. 关于激活函数--sigmoid，另外一个容易忽视的事实是，其梯度在z=0.5时，达到最大--0.25，这意味着，每次梯度信号穿越过sigmoid门时，会衰减为1/4或者更多，如果使用基本 SGD，这会使网络的低层比高层的参数更新慢得多。

3. 长话短说： 如果您在网络中使用 sigmoids 或者 tanh 作为激活函数，那么您应该警惕，初始化不会导致想训练过程完全饱和（ fully saturated）。

4. 还有一个有趣的非线性函数ReLU，它从下面将神经元阈值为零。使用 ReLU 的完全连接层的向前和向后传递在核心包括:









1. 损失函数是关于训练数据和网络权重的函数，其中训练数据是常数，权重是变量，我们可以控制的。因此，虽然算出损失关于训练数据的梯度很简单，（同样bp），但是我们也不会算这个梯度，而是算损失关于权重的梯度，从而使用这个梯度去更新权重。
2. 损失函数关于样本的梯度也不是说完全没有用，他可以用来做可视化，解释模型当前学到了什么。
3. 导数是什么？随着某个变量的变化，一个函数的变化量。表示函数对于当前变量值的敏感性。
4. 考虑一个多层嵌套的函数f，对其应用链式法则，就可以到达终极变量的导数

- f(x,y,z)=(x+y)z可以被分解为:q=x+y 和f=qz

- 分开来看，求偏导很简单，∂f/∂q=z, ∂f/∂z=q，∂q/∂x=1，∂q/∂y=1.

- 虽然，我们对中间变量的倒数不感兴趣，使用链式法则可以沿着中间变量的导数得到f对于终极变量的导数，举个例子：

  ∂f/∂x=∂f/∂q.∂q/∂x

  也就是两个中间变量相关的偏导数的乘积

- 看一个激活函数sigmoid的例子，它的导数很好求
$$ \sigma(x)=\frac{1}{1+e^{-x}}  \rightarrow \frac{d \sigma(x)}{d x}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}=\left(\frac{1+e^{-x}-1}{1+e^{-x}}\right)\left(\frac{1}{1+e^{-x}}\right)=(1-\sigma(x)) \sigma(x) $$

- $$f(w, x)=\frac{1}{1+e^{-\left(w_{0} x_{0}+w_{1} x_{1}+w_{2}\right)}}$$, 

## layer的输入输出

```python
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors),
                    input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu',
                   input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
```

![Image for post](https://miro.medium.com/max/2073/1*Y117iNR_CnBtBh8MWVtUDg.png)

- Conv1D：　model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))：
    - 输入：(timesteps,  num_series) 
    - 输出：(timesteps - kerner_size+1, num_kernels)
    - 参数个数：(kernel_size * num_sensors  + 1) * num_kernels
- MaxPooling1D: model_m.add(MaxPooling1D(3))
    - 输入：(timesteps,  num_series)
    - 输出：(timesteps//3,  num_series)
    - 备注：书的页数（channel）没变，但是每页的字数变成了1/3
    - ![Image for post](https://miro.medium.com/max/3058/1*W34PwVsbTm_3EbJozaWWdA.jpeg)
- GlobalAveragePooling1D：model_m.add(GlobalAveragePooling1D())
    - 输入：　(timesteps,  num_series) 或者（feature_values, feature_detectors）
    - 输出：　（１，num_series）或者（１, feature_detectors）
    - 备注： 书的页数（channel）没变，但是每页的字数变成了1
- Dropout: model_m.add(Dropout(0.5))： 形状不变
    输入：（１，num_series）
    输出：（１，num_series）
- LSTM：model.add(LSTM(units=128,  dropout=0.5, return_sequences=True, input_shape=input_shape))
    - 输入： (timestep, series)
    - 输出：(timestep, units)
- Dense： model_m.add(Dense(num_classes, activation='softmax'))
    - 输入：（１，num_series）
    - 输出：（１，num_classes　×　２）
- BatchNormalization：
    - 输入:  (timestep, series)
    - 输出：(timestep, series) 
    - 参数个数：４ * series (channel) 
        - $$y=\gamma\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\beta $$


## layer如何设置初始化权重？

```python
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
```

![3种初始化方法](./img.png) 


## 训练过程太长？

保存训练结果

```python
# 1. 设置loss，优化算法
model.compile(loss=..., optimizer=...,
              metrics=['accuracy'])

EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'

# 2. 定义回调函数
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
# 3.在fit中传入回调函数，模型会一边训练一边存储
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# 4. 从缓存路径中加载模型
# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
```


## 参考
1. Hands-On Machine Learning with Scikit-Learn and TensorFlow, P334
2. [keras-model_checkpoint-官网文档](https://keras.io/api/callbacks/model_checkpoint/)
3. [concatenate](https://keras.io/api/layers/merging_layers/concatenate/)
4. http://cs231n.github.io/optimization-2/#intuitive

5. https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
