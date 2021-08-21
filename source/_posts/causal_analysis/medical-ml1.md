---
title: 医疗中的因果分析
date: 2021-08-17 13:03:39
mathjax: true
tags:
- 因果推断
- 因果分析
- 贝叶斯
- 复杂网络
categories: 
- 因果分析 
---





## 背景



在诊断时，我们想要回答：病人5年内趋势的概率？虽然可以训练一个深度学习模型来预测，但是这个方法可能很危险，模型学习的样本，可能有幸存者偏差。

举个例子，样本中，有的病人可能一直在接受治疗，所以生命得以延长，而如果我们忽略了如此的样本生成过程，就会学到一个虚假的X到Y的映射，从而得到一个关于新病人的病情预测的错误结论（模型似乎在说，你不用治疗，你看其他人都没有死掉，你也不会死掉）

![](./image-20210817173930873.png)

有没有什么办法能够让机器预测的比人还准呢？

吸烟会引起癌症吗？

吸烟和肺癌的因果关系曾经引起过很大的争论。

解决这个问题的传统方法：随机控制试错。

此外，当随机实验不现实时（不能要求一个不抽烟的人抽烟，然后观测他得不得肺癌），可以仅根据观测数据就回答这些问题那？可以，先估计条件likelyhood（吸烟的人中得肺癌的概率，不吸烟的人中得肺癌的概率）。

但是有个风险，可能存在混杂因子，即可能存在某个因素，会导致人更容易抽烟以及得肺癌。（比如工厂工作的男性）



## 从结论到动机



在医疗保健中，一旦涉及机器学习领域，需要谨慎思考，毕竟人命关天。

跟传统的机器学习只需要输入输出不一样，还需要考虑第三个变量--干预，interventions，

并且，还要思考这三个变量之间的因果关系

![最简单的因果图](./image-20210817181259614.png)

因果分析中最简单的一种问题：假设边的方向知道，那么边的强度是多少？

治疗方案T 可能是二值，或者是连续，或者是向量。

先假设是二值向量，有向图显示，T依赖病人的诊断结果X，而最终效果Y取决于诊断情况X和治疗方案T。



## 潜在效果框架



潜在效果框架Potential outcomes framework

在政治学/统计学/经济学/生物统计学中，类似吸烟和肺癌的关系这种问题已经被研究过很年了，大部分采用的是统计学派的方法。

然而在现代医疗应用中经常出现的高维问题，传统的统计方法就失效了。

不过倒是可以引入机器学习来回答高维因果问题。


引入一种语言，接下来先考虑

### Neyman-Rubin因果模型

rubin-neyman因果模型是这样的：$Y_0(x)$表示x不接受治疗的效果，$Y_1(x)$表x接受治疗时的效果。

对于某个体（$x_i$），平均治疗效果（conditional average treatment effect）为：
$$
C A T E\left(x_{i}\right)=\mathbb{E}_{Y_{1} \sim p\left(Y_{1} \mid x_{i}\right)}\left[Y_{1} \mid x_{i}\right]-\mathbb{E}_{Y_{0} \sim p\left(Y_{0} \mid x_{i}\right)}\left[Y_{0} \mid x_{i}\right]
$$
> 也可以认为是，对于某个商品，模型上线前后效果的差别



对整个人群，平均治疗效果(average treatment effect)为:

$$ A T E=\mathbb{E}\left[Y_{1}-Y_{0}\right]=\mathbb{E}_{x \sim p(x)}[C A T E(x)]$$

> 总体的平均治疗效果，类似推荐场景中ABtest，可以认为，对所有商品，模型上线前后效果的提升




![image-20210817190359970](./image-20210817190359970.png)
治疗效果：
$$
y_{i}=t_{1} Y_{1}\left(x_{i}\right)+\left(1-t_{1}\right) Y_{0}\left(x_{i}\right)
$$
t表示是否接受治疗，反事实治疗效果是：
$$
y_{i}=\left(1-t_{1}\right) Y_{1}\left(x_{i}\right)+t_{1} Y_{0}\left(x_{i}\right)
$$



反事实推断的两个基本方法：协变量调整（Covariate Adjustment）和Propensity scores


因果推断的难点在于，我们永远只能观测到病人的其中一种情况，就会导致朴素估计方法得到了错误的结论

![image-20210817205524554](./image-20210817205524554.png)


上图中，有一些点不在曲线上，是因为我们允许一些随机性存在，然而虚线代表了可能的效果，圆圈代表现实观测值。我们只能观测到每个病人在某个处方下的状态，而不知道反事实的状态。为了填充反事实这个确实值，我们假设：

- 对比简单的平均诊断估计和真实平均诊断效果，
- 计算使用处方B的病人的平均血糖水平：
- 计算使用处方A的病人的平均血糖水平：
- 计算两者的差，得到0.75，说明A能够降低血糖
- 如果开启上帝视角，知道了每个人在处方A/B下的血糖水平，那么，计算二者的均值，求差，得到平均诊断效果ATE为-0.75，
- 也就是说，朴素估计方法得到了错误的结论



###  Neyman-Rubin因果模型中的假设



直觉来说，反事实推断不可能，因为没有观测数据。

但是我们可以加上一些限制条件，使得估计反事实预测成为可能，这个限制条件也叫Neyman-Rubin规则，包括三个假设：

- Stable Unit Treatment Value Assumption (SUTVA).
- Ignorability假设
- Common support



Ignorability假设：即，T和Y0，Y1之间不存在不可观测的混杂因子（confounders），也就是说给定x时，最终效果（血压高低/肺癌）Y1，Y2跟治疗方案T条件独立：

$$ \left(Y_{0}, Y_{1}\right) \Perp  T \mid X $$

对应到因果图中就是，T到Y0和Y1没有边。

这个假设是必要的么？是！考虑一种igorability不满足的情况：存在某个隐变量h，同时会影响Y1，Y2和T。

怎么验证ignorability条件是否满足？需要跟该领域的专家交流，从而保证影响一生决策（T）和最终效果（Y1，Y2）的所有因素都已经考虑到了和观测到了。

即使存在某个混杂因子没有被观测到，影响大么？---使用敏感度分析（ sensitivity analysis）进行度量。



Common support假设，诊断决策始终存在随机性，即，对于任意一组病患，被分配任意一个诊断方案都是有可能的。

这个可能性用 propensity score表示，即取值大于0.



因果推断的方法有很多，比如协变量调整(covariate adjustment )/ propensity score, doubly robust estimators, matching,




## 协变量调整



协变量调整是一个外推的自然的方法，这个方法的主要目标是，找到函数f(x, T), 输入病人的诊断情况，治疗方案T，输出可能的后果，可以认为是对条件概率的近似$p(Y_t|x, T = t).$

也就是说，协变量调整的方法就是构造一个回归模型，显示对治疗方案，cofounders和疗效三折的关系建模，

使用协变量调整方法估计出来的平均治疗效果（ATE）和CATE是：

$$ACE = \sum_i (f(x_i, 1) - f(x_i, 0))$$

$$CACE(x_i) =  f(x_i, 1) - f(x_i, 0)$$

那么，我们如何知道这个回归模型学到的变量之间的关系是有意义的呢？

举个例子，治疗决策可能只是影响疗效的众多因素中的一个，可能有其他因素对疗效的影响更大，这样就可能导致回归模型学到的治疗方案（t）跟疗效（y）的相关性很小。（自相关系数和偏相关系数的差别）

这就是因果推断和机器学习不一样的地方，本质上是二者关注的目标不一样，机器学习关注预测结果的准确性，因果推断更关注学习到的因果关系，在高维问题中，二者差别更加明显。

什么时候协变量调整（covariate adjustment ）会失败？当数据违反了common support/overlap的假设时，协变量方法会失败。
1. 没有充分的数据：回归模型没有充足的数据对y做外推。
2. 没有选对函数族：即使有了足够多的用来外推（extrapolate）的数据，还需要选对函数族。

> 假设x和y的映射函数实际上是二次的，而我们如果使用线形函数拟合，会得到错误估计。







协变量调整就是说，显示地对治疗方案（T），混杂因子（x），结果（y）建模，构建一个回归模型，输入是<T,x>,输出是y

使用先行模型进行斜变量调整，

假设
$$
Y_{t}(x)=\beta x+\gamma t+\epsilon_{t}
$$

$$
\mathbb{E}\left[\epsilon_{t}\right]=0
$$

那么可以算出CATE：
$$
\begin{aligned} C A T E(x) &=\mathbb{E}_{p\left(Y_{1} \mid x\right)}\left[Y_{1}\right]-\mathbb{E}_{p\left(Y_{0} \mid x\right)}\left[Y_{0}\right] \\ &=\mathbb{E}_{\epsilon_{0}, \epsilon_{1}}\left[\beta x+\gamma+\epsilon_{1}-\beta x-\epsilon_{0}\right] \\ &=\gamma+\mathbb{E}\left[\epsilon_{1}\right]-\mathbb{E}\left[\epsilon_{0}\right] \\ &=\gamma \end{aligned}
$$
也可以计算出ATE:
$$
A T E=\mathbb{E}_{p(x)}[C A T E(x)]=\gamma
$$
对因果推断来说，希望很好地估计$\gamma$，而不是很好地预测Y，这一点跟ml不一样

更关注系数以及置信区间。

将协变量模型拓展到非线形模型上，假设数据生成过程是这样：
$$
Y_{t}(x)=\beta x+\gamma t+\delta x^{2}
$$


那么平均治疗效果ATE为：
$$
A T E=\mathbb{E}\left[Y_{1}-Y_{0}\right]=\gamma
$$


而，假设我们错误地建模成了线形关系：
$$
\hat{Y}_{t}(x)=\hat{\beta} x+\hat{\gamma} t
$$


那么对$\gamma$的估计就有可能非常离谱了
$$
\hat{\gamma}=\gamma+\delta \frac{\mathbb{E}[x t] \mathbb{E}\left[x^{2}\right]-\mathbb{E}\left[t^{2}\right] \mathbb{E}\left[x^{2} t\right]}{\mathbb{E}[x t]^{2}-\mathbb{E}\left[x^{2}\right] \mathbb{E}\left[t^{2}\right]}
$$


生物统计学家们通过小心谨慎地逐次添加非线性关系，来获得一个预测结果更精准的模型，有一下几种方法：

- 随机森林和贝叶斯树
- 高斯过程
- 神经网络



## 非线性模型


### 高斯过程



高斯过程，允许我们检测整体分布，以及比较两个诊疗方案的置信区间。

下图中，

如何确定给一个病人x开什么处方y呢？

可以计算CATE，如果大于某个值，就开治疗。

然而，这个策略可能是错的，因为没有考虑到我们对决策的置信度。

instead，我们希望有一个决策规则，可以量化我们的不确定度度，定义一个支持区域。

举个例子，如果P(CATE(x) > α) > 0.9,那么就治疗，负责就说，我们没充分的信息保证治疗方案有效。



使用高斯过程，既可以对两种治疗方案分开建模（左图），也可以构建一个联合分布（有图）

联合建模的好处是：两个治疗方案共享一套参数，参数估计需要的数据点更少，

![image-20210820142140615](./image-20210820142140615.png)








### 神经网络

也可以使用神经网络来学习非线性模型，





![image-20210820190705031](/Users/stellazhao/research_space/chiechie.github.io/source/_posts/causal_analysis/medical-ml1/image-20210820190705031.png)



以下图为例，对输入施加多层非线性layer，然后使用一层treatment layer$\phi$，



注意，对不同的treatment，前面几层layer是共享的，从而可以学习联合表示。

接着，对不同的treetment使用个字的layer来获取不同的结果。



另外一个重要的要注意的是，当我们对输入convolve之后，再应用treatment，是因为treatment特征跟x一起在最前面出现的话，特征重要性由于没有x高，而被比下去了，因此信息就丢失了。





## 匹配

思路：找每个人的长期失联的兄弟，然后看他们身上的效果。

世纪钟，我们可以识别到另外一个$x_1$,跟$x_0$非常相似，但是不属于另外一个治疗组，接着可以评估x1的效果。

可以把这个值看成是对x0的反事实的估计，接着可以这个值来估计CATE和ATE：



1-NN Matching：

定义一个距离函数d，那么对每个个体i，找到距离最近的（也就是表现最相似的）反事实邻居：
$j(i)=\underset{j: t_{j} \neq t_{i}}{\operatorname{argmin}} d\left(x_{j}, x_{i}\right)$



$t_i = 0$表示个体i是控制组，$t_i = 1$表示个体i是实验组，可以用一个公式表达两种情况下的CATE：


$$
\widehat{C A T E}\left(x_{i}\right)=\left(2 t_{i}-1\right)\left(y_{i}-y_{j(i)}\right)
$$
对个体i取期望，可以得到整体人群中的疗效
$$
\widehat{A T E}=\frac{1}{n} \sum_{i=1}^{n} \widehat{C A T E}\left(x_{i}\right)
$$

总结下匹配的优缺点

- 可解释，特别在小样本时候
- 非参数
- 重度依赖潜在指标
- 可能被一些无关特征干扰
- 匹配在实践中使用并不多，在stanford医疗中心，有一个人使用该方法查找电子医疗记录相似（健康历史，背景等）的另外一个病人，观测其治疗效果如何，然后来指导当前病人的诊断决策。
- 1-NN和机器学习里面的K-NN有一样的问题，例如欧氏距离在高维空间不好用，同时，极度依赖样本，找不到反事实样本就不行了。


## 协变量调整和匹配


$$
\hat{Y}_{1}(x)=y_{N N_{1}(x)}, \hat{Y}_{0}(x)=y_{N N_{0}(x)}
$$

where $y_{N N_{t}(x)}$ is th nearest-neighbor of x among units with treatment assignment t = 0,1


Covariate Adjustment and Matching



## Propensity score re-weighting

Propensity score re-weighting (henceforth PSR)是另外一个估计ATE的工具.

主要思路：将观察实验实验转化为一个伪随机试验，通过调整样本权重。


这个跟统计学中的重要性采样很相似（Importance Sampling）

PSR对于处理非均衡数据集很有帮助。

如果说，我们看到p(x|t = 0) and p(x|t = 1)相差很大，我们希望增加一个权重函数w0，w1，
到我们的样本点中，从而保证p(x|t = 0)w0(x) ≈ p(x|t = 1)w1(x)
通过下图可以看到PSR的原理，希望对被蓝色的样本点包围的红色样本点给更多权重，原因是，希望对特征空间覆盖度更高。

![image-20210821121130529](./image-20210821121130529.png)



Propensity score 算法

1. PS算法使用PSR来估计ATE

2. 首先ps的定义是：$p(T=t \mid x)$, 表示给定病人x的诊断数据后，接收到处方t的概率。我们可以使用很多机器学习算法来估计这个取值：

   1. 搜集样本：
      $$
      \left(x_{1}, t_{1}, y_{1}\right), \ldots,\left(x_{n}, t_{n}, y_{n}\right)
      $$

   2. 构建监督模型：f(x)->t, 从而估计ps$p(T=t \mid x)$

   3. 计算整体平均治疗效果：
      $$
      \widehat{A T E}=\frac{1}{n} \sum_{i, t_{i}=1} \frac{y_{i}}{\hat{p}\left(t_{i}=1 \mid x_{i}\right)}-\frac{1}{n} \sum_{i, t_{i}=0} \frac{y_{i}}{\hat{p}\left(t_{i}=0 \mid x_{i}\right)}
      $$



推导过程如下



![image-20210821135001230](./image-20210821135001230.png)

计算ATE时，对每个观测效果yi, 使用逆propensity score来加权，这个就叫re-weighting。

在随机试验中，$p(T=t \mid x)=0.5$， ATE表达式为：
$$
\widehat{A T E}=\frac{2}{n} \sum_{i, t_{i}=1} y_{i}-\frac{2}{n} \sum_{i, t_{i}=0} y_{i}
$$
关于psr的其他说明：

- 通常ps不知道，需要顾及
- 如果在时间上没有什么ovelap，ps就没有什么信息量，还容易造成误会
- 权重能够导致大方差和大的误差，对于小的ps
- 我们仅仅只能计算ATE



## 其他方法



## 自然实验

主要思路是，寻找观测数据的，desired treatment happened to be given to some members of the population and not given to other members.

- As an example, suppose we want to study how stress during pregnancy affects later child development. We can’t conduct a randomized controlled trial, so instead we can look for a natural experiment in which otherwise similar populations were split into “treatment” (i.e. stress during pregnancy) and “control” (i.e. no extra stress during pregnancy).
- The Cuban missile crisis of October 1962 caused increased levels of stress because people were afraid a nuclear war would break out. We could compare children who were in utero during the crisis with children from immediately before and after.



## 实验变量

- An instrumental variable is a variable which affects treatment assignment but not the outcome.
- We could use instrumental variables to answer the question: are private schools better than public schools? Again, we can’t conduct a RCT here because we can’t force people which school to go to. However, we could randomly give out vouchers to some students, giving them an opportunity to attend private schools. In this case, the voucher assignment is the instrumental variable.



## 结论

1. We discussed two approaches to use machine learning for causal inference:

2. 1. Predict outcome given features and treatment, then use resulting model to impute counterfactuals (covariate adjustment)
   2. Predict treatment using features (propensity score), then use to reweight outcome or stratify the data

3. It is also important to think through causal graphs to see whether problem is setup appropriately and whether assumptions hold before doing any reductions to Machine Learning.






## 参考



1. https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-s897-machine-learning-for-healthcare-spring-2019/ 

2. References

   [SJS17] Shalit, Johansson, and Sontag. Estimating individual treatment effect: Generalization. Health Affairs, 2017.

