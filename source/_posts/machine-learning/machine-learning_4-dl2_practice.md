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



## 反向传播

> ** The problem with Backpropagation is that it is a** [**leaky abstraction**](https://en.wikipedia.org/wiki/Leaky_abstraction)**.**---Andrej Karpathy

![img](https://kratzert.github.io/images/bn_backpass/BNcircuit.png)

1. 关于激活函数--sigmoid，另外一个容易忽视的事实是，其梯度在z=0.5时，达到最大--0.25，这意味着，每次梯度信号穿越过sigmoid门时，会衰减为1/4或者更多，如果使用基本 SGD，这会使网络的低层比高层的参数更新慢得多。![img](https://miro.medium.com/max/1400/1*gkXI7LYwyGPLU5dn6Jb6Bg.png)

2.  如果网络中使用 sigmoids 或者 tanh 作为激活函数，那么您应该警惕，初始化不会导致想训练过程完全饱和（ fully saturated）。

3. 非线性函数ReLU：当输入小于0时，导数也为0，此时没有梯度信号能通过激活函数，这个现象叫“dead ReLU” 问题。如果初始权重没选好导致relu的输出为0，那么这个relu神经元再也不会被激活了，就永久保持死亡状态。在训练的过程中发现，大部分神经元（neuron，可能有40%）都是死亡状态。

    

![img](https://miro.medium.com/max/1400/1*g0yxlK8kEBw8uA1f82XQdA.png)



1. 注意：构建的神经网络中有 ReLUs单元时，应该始终对dead ReLUs保持警惕。在训练过程中，如果学习率设置的过于aggressive，常出现relu神经元死亡。
2. RNNs中的梯度爆炸（Exploding gradients in RNNs）,假设有一个简化的 RNN，不接受输入 x，只递归计算隐藏状态(等价地，输入 x 总是可以为零) ，RNN 是展开了T的时间步长。注意看反向通道时，会看到梯度信号逆着时间传播，并且是通过将隐藏状态乘以同一个矩阵循环矩阵**Whh**传播的，总是被同一个矩阵乘以(递归矩阵 Whh) ，中间穿插着非线性函数的反向传播。
3. 当你用一个数 a 乘以另一个数 b (即 a * b * b * b * b * b * b * b)会发生什么。.)？如果 | b | < 1，这个序列趋近零; 如果 | b | > 1，这个序列趋近无穷。同样的事情也发生在 RNN 的反向传播过程中，不过 b 是一个矩阵，而不仅仅是一个数字，这个时候要算它的最大特征值。
4. **使用RNN时要注意**: 警惕梯度截断（gradient clipping），或者使用LSTM.
5. 额外的发现（Spotted in the Wild: DQN Clipping）：DQN中， 使用 **target_q_t**表示$ [reward * \gamma *argmax_a Q(s’,a)]$，还有一个变量**q_acted**, which is **Q(s,a)** of the action that was taken. 二者相剑得到一个变量**delta,** 可以用使用l2损失最小化这个变量， **tf.reduce_mean(tf.square()).** 
6. 损失函数是关于训练数据和网络权重的函数，其中训练数据是常数，权重是变量。因此，虽然损失关于训练数据的梯度很容易算，但是不去算，因为跟目标（更新权重）不一致。算损失关于权重的梯度，从而使用这个梯度去更新权重。
7. 损失函数关于样本的梯度虽然对更新参数没有用，但是可以用来解释模型当前学习到了什么。
8. 导数是什么？随着某个变量的变化，一个函数的变化量。表示函数对于当前变量值的敏感性。
9. 考虑一个多层嵌套的函数f，对其应用链式法则，就可以到达终极变量的导数

- f(x,y,z)=(x+y)z可以被分解为:q=x+y 和f=qz

- 分开来看，求偏导很简单，∂f/∂q=z, ∂f/∂z=q，∂q/∂x=1，∂q/∂y=1.

- 虽然，我们对中间变量的倒数不感兴趣，使用链式法则可以沿着中间变量的导数得到f对于终极变量的导数，举个例子：

  ∂f/∂x=∂f/∂q.∂q/∂x

  也就是两个中间变量相关的偏导数的乘积

- 看一个激活函数sigmoid的例子，它的导数很好求
$$ \sigma(x)=\frac{1}{1+e^{-x}}  \rightarrow \frac{d \sigma(x)}{d x}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}=\left(\frac{1+e^{-x}-1}{1+e^{-x}}\right)\left(\frac{1}{1+e^{-x}}\right)=(1-\sigma(x)) \sigma(x) $$

- $$f(w, x)=\frac{1}{1+e^{-\left(w_{0} x_{0}+w_{1} x_{1}+w_{2}\right)}}$$, 

>  The derivative on each variable tells you the sensitivity of the whole expression on its value.*

- 导数是函数相对于某个自变量的敏感度；梯度是偏导数组成的向量。
- Backpropagation can thus be thought of as gates communicating to each other (through the gradient signal) whether they want their outputs to increase or decrease (and how strongly), so as to make the final output value higher.

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
