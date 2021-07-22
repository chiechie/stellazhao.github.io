---
title:  chapter 4.2 backpropagation背后的直觉
author: chiechie
mathjax: true
date: 2021-03-08 10:17:31
tags:
- 深度学习
- low level
- 最佳实践
categories: 
- 机器学习
---



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
$$ \sigma(x)=\frac{1}{1+e^{-x}} $$

$$ \rightarrow \frac{d \sigma(x)}{d x}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}=\left(\frac{1+e^{-x}-1}{1+e^{-x}}\right)\left(\frac{1}{1+e^{-x}}\right)=(1-\sigma(x)) \sigma(x) $$

- $$f(w, x)=\frac{1}{1+e^{-\left(w_{0} x_{0}+w_{1} x_{1}+w_{2}\right)}}$$, 

```python
# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
dfdy = 1.0 * dfdq # dq/dy = 1
```

![image-20200206174811146](/Users/stellazhao/Desktop/EasyMLBOOK/_image/image-20200206174811146.png)

### 4 Modularity: Sigmoid example

The gates we introduced above are relatively arbitrary. Any kind of differentiable function can act as a gate, and we can group multiple gates into a single gate, or decompose a function into multiple gates whenever it is convenient. Lets look at another expression that illustrates this point:

$$

$$
as we will see later in the class, this expression describes a 2-dimensional neuron (with inputs **x** and weights **w**) that uses the sigmoid activation *function. But for now lets think of this very simply as just a function from inputs *w,x*to a single number. The function is made up of multiple gates. In addition to the ones described already above (add, mul, max), there are four more:

![image-20200206180231348](/Users/stellazhao/Desktop/EasyMLBOOK/_image/image-20200206180231348.png)

Where the functions $f_{c}, f_{a}$ translate the input by a constant of c and scale the input by a constant of a, respectively. These are technically special cases of addition and multiplication, but we introduce them as (new) unary gates here since we do need the gradients for the constants.c,a. The full circuit then looks as follows:

![image-20200206180550084](/Users/stellazhao/Desktop/EasyMLBOOK/_image/image-20200206180550084.png)

> Example circuit for a 2D neuron with a sigmoid activation function. The inputs are [x0,x1] and the (learnable) weights of the neuron are [w0,w1,w2]. As we will see later, the neuron computes a dot product with the input and then its activation is softly squashed by the sigmoid function to be in range from 0 to 1.

In the example above, we see a long chain of function applications that operates on the result of the dot product between **w,x**. The function that these operations implement is called the *sigmoid function* $\sigma (x)$. It turns out that the derivative of the sigmoid function with respect to its input simplifies if you perform the derivation (after a fun tricky part where we add and subtract a 1 in the numerator):




As we see, the gradient turns out to simplify and becomes surprisingly simple. For example, the sigmoid expression receives the input 1.0 and computes the output 0.73 during the forward pass. The derivation above shows that the*local*gradient would simply be (1 - 0.73) * 0.73 ~= 0.2, as the circuit computed before (see the image above), except this way it would be done with a single, simple and efficient expression (and with less numerical issues). Therefore, in any real practical application it would be very useful to group these operations into a single gate. Lets see the backprop for this neuron in code:

```python
w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot)) # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit
```

**Implementation protip: staged backpropagation**. 

As shown in the code above, in practice it is always helpful to break down the forward pass into stages that are easily backpropped through. For example here we created an intermediate variable`dot`which holds the output of the dot product between`w`and`x`. During backward pass we then successively compute (in reverse order) the corresponding variables (e.g.`ddot`, and ultimately`dw, dx`) that hold the gradients of those variables.

The point of this section is that the details of how the backpropagation is performed, and which parts of the forward function we think of as gates, is a matter of convenience. It helps to be aware of which parts of the expression have easy local gradients, so that they can be chained together with the least amount of code and effort.



### 5. Backprop in practice: Staged computation

Lets see this with another example. Suppose that we have a function of the form:

$$
f(x, y)=\frac{x+\sigma(y)}{\sigma(x)+(x+y)^{2}}
$$
To be clear, this function is completely useless and it’s not clear why you would ever want to compute its gradient, except for the fact that it is a good example of backpropagation in practice. It is very important to stress that if you were to launch into performing the differentiation with respect to eitherxxoryy, you would end up with very large and complex expressions. However, it turns out that doing so is completely unnecessary because we don’t need to have an explicit function written down that evaluates the gradient. We only have to know how to compute it. Here is how we would structure the forward pass of such expression:

```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done!                                 #(8)
```

Phew, by the end of the expression we have computed the forward pass. Notice that we have structured the code in such way that it contains multiple intermediate variables, each of which are only simple expressions for which we already know the local gradients. Therefore, computing the backprop pass is easy: We’ll go backwards and for every variable along the way in the forward pass (`sigy, num, sigx, xpy, xpysqr, den, invden`) we will have the same variable, but one that begins with a`d`, which will hold the gradient of the output of the circuit with respect to that variable. Additionally, note that every single piece in our backprop will involve computing the local gradient of that expression, and chaining it with the gradient on that expression with a multiplication. For each row, we also highlight which part of the forward pass it refers to:

```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

Notice a few things:

**Cache forward pass variables**. To compute the backward pass it is very helpful to have some of the variables that were used in the forward pass. In practice you want to structure your code so that you cache these variables, and so that they are available during backpropagation. If this is too difficult, it is possible (but wasteful) to recompute them.

**Gradients add up at forks**. The forward expression involves the variables**x,y**multiple times, so when we perform backpropagation we must be careful to use`+=`instead of`=`to accumulate the gradient on these variables (otherwise we would overwrite it). This follows the*multivariable chain rule*in Calculus, which states that if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add.

### 6 Patterns in backward flow

It is interesting to note that in many cases the backward-flowing gradient can be interpreted on an intuitive level. For example, the three most commonly used gates in neural networks (*add,mul,max*), all have very simple interpretations in terms of how they act during backpropagation. Consider this example circuit:

![image-20200206182203095](/Users/stellazhao/Desktop/EasyMLBOOK/_image/image-20200206182203095.png)

> An example circuit demonstrating the intuition behind the operations that backpropagation performs during the backward pass in order to compute the gradients on the inputs. Sum operation distributes gradients equally to all its inputs. Max operation routes the gradient to the higher input. Multiply gate takes the input activations, swaps them and multiplies by its gradient.

Looking at the diagram above as an example, we can see that:

The **add gate** always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass. This follows from the fact that the local gradient for the add operation is simply +1.0, so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged). In the example circuit above, note that the + gate routed the gradient of 2.00 to both of its inputs, equally and unchanged.

The **max gate** routes the gradient. Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass). This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values. In the example circuit above, the max operation routed the gradient of 2.00 to the**z**variable, which had a higher value than **w**, and the gradient on **w** remains zero.

The **multiply gate** is a little less easy to interpret. Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule. In the example above, the gradient on **x** is -8.00, which is -4.00 x 2.00.

***Unintuitive effects and their consequences***.

Notice that if one of the inputs to the **multiply gate** is very small and the other is very big, then the multiply gate will do something slightly unintuitive: it will assign a relatively huge gradient to the small input and a tiny gradient to the large input. Note that in linear classifiers where the weights are **dot producted** $$
w^{T} x_{i}
$$(multiplied) with the inputs, this implies that the scale of the data has an effect on the magnitude of the gradient for the weights. For example, if you multiplied all input data examples $x_i$ by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you’d have to lower the **learning rate** by that factor to compensate. This is why preprocessing matters a lot, sometimes in subtle ways! And having intuitive understanding for how the gradients flow can help you debug some of these cases.

### 7 Gradients for vectorized operations

The above sections were concerned with single variables, but all concepts extend in a straight-forward manner to matrix and vector operations. However, one must pay closer attention to dimensions and transpose operations.

**Matrix-Matrix multiply gradient**. Possibly the most tricky operation is the matrix-matrix multiplication (which generalizes all matrix-vector and vector-vector) multiply operations:

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```

*Tip: use dimension analysis!*

Note that you do not need to remember the expressions for `dW` and `dX`because they are easy to re-derive based on dimensions. For instance, <u>we know that the gradient on the weights `dW` must be of the same size as `W` after it is computed</u>, and that it <u>must depend on matrix multiplication of `X` and `dD`</u>(as is the case when both`X,W`are single numbers and not matrices). There is always exactly one way of achieving this so that the dimensions work out.

 For example,`X`is of size [10 x 3] and`dD`of size [5 x 3], so if we want`dW`and`W`has shape [5 x 10], then the only way of achieving this is with`dD.dot(X.T)`, as shown above.

**Work with small, explicit examples**. Some people may find it difficult at first to derive the gradient updates for some vectorized expressions. Our recommendation is to explicitly write out a minimal vectorized example, derive the gradient on paper and then generalize the pattern to its efficient, vectorized form.

Erik Learned-Miller has also written up a longer related document on taking matrix/vector derivatives which you might find helpful.[Find it here](http://cs231n.stanford.edu/vecDerivs.pdf).



### 8. Summary

- We developed intuition for what the gradients mean, how they flow backwards in the circuit, and how they communicate which part of the circuit should increase or decrease and with what force to make the final output higher.
- We discussed the importance of **staged computation** for practical implementations of backpropagation. You always want to break up your function into modules for which you can easily derive local gradients, and then chain them with chain rule. Crucially, you almost never want to write out these expressions on paper and differentiate them symbolically in full, because you never need an explicit mathematical equation for the gradient of the input variables. Hence, decompose your expressions into stages such that you can differentiate every stage independently (the stages will be matrix vector multiplies, or max operations, or sum operations, etc.) and then backprop through the variables one step at a time.

In the next section we will start to define Neural Networks, and backpropagation will allow us to efficiently compute the gradients on the connections of the neural network, with respect to a loss function. In other words, we’re now ready to train Neural Nets, and the most conceptually difficult part of this class is behind us! ConvNets will then be a small step away.

### References

- [Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)



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
