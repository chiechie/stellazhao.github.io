---
title: chapter 2.2 树模型2-随机森林,Adaboost和GBDT
author: chiechie
mathjax: true
date: 2021-04-17 10:39:45
tags:
- 机器学习
- 树模型
- 人工智能
categories:
- 机器学习
---

> boosting tree的代表是gbdt，以及一系列变形xgboost,catboost,下面重点介绍一下xgboost

# 何为bagging和boosting？

1. Bagging是一种数据采样的方法，Blending是一种模型混合的方法.
2. Blending的方法有几个：投票(uniform aggregation)/线性/stacking。
    - 如果每个base estimator一样重要，就对他们采用uniform的aggregation方式。
    - 如果有的base estimator特别重要，可以将estimator当成特征转换器使用，然后将不同estimator的输出做一个线性映射，得到最终的结果，这个也叫线性blending。
    - 如果想让这个aggregation更加复杂，也就是说，想要这个模型实现，在不同的condition下，不同的estimator发挥的重要性不一样，这个时候就要基于不同estimator的输出构建一个非线性映射。这个也叫stacking。
    - 投票追求的是公平稳定，线性/stacking追求高准确率。

    - 也可以一边构建base estimator，一边决定模型Blending的方法.
    - 不同的blending方法中，有的方法解决overfitting，有的适合解决underfitting。
4. 三种blending方法的代表算法分别有：
    - 投票（uniform aggregation）的代表是Bagging Tree/随机森林
    - 线性Blending的代表是AdaBoost/GradientBoost：
      - AdaBoost通过改变样本权重的方式得到不一样的base estimator，一边根据他们的表现决定给每个estimator多少票。
      - GradientBoost在构建新的base estimator时，不是像AdaBoost那样重新对样本赋予权重，而是，直接去最小化残差，然后根据学到的结果，赋予当前base estimator一个权重。
    - stacking Blending的代表是决策树，即在不同条件下找到最优的base estimator。在决策树里面，不同的路径使用的是不同的叶子（在决策树的场景中，一个叶子就是一个base estimator）。
    - 其中，boosting-like的算法最受欢迎，即AdaBoost和GradientBoost，还有xgboost和catboost
   
   ![img_2.png](./img_2.png)
5. 为什么对模型做aggregation之后，效果变好了？相当于对特征做了非线性变换，整体表达能力更强；多个estimator求共识，相当于做了正则化，模型效果更稳定。
5. boosting是多个base estimtor进行aggregate的一种方法，并且每个estimator都有一个自己的权重。boosting相对于uniform的方式，多了n个待估参数，复杂度更高，如何求解这个高维的优化问题？衍生出了两种算法，第一种是像AdaBoost，使用一个reweighted的样本去构建一个新的estimator，并使用其预测准确率作为权重。
    GradientBoost则是基于前面所有的estimator的学习成果，做增量学习。



# 随机森林，Adaboost和 GBDT

## 随机森林

1. bootstrap是什么？是一种抽样的方法，通过多次有放回的抽样，得到多份独立的样本集，获得统计量。
2. Bagging是什么？通过bootstrap的方式抽样得到多份样本。Bagging的特点是，可以降低整体的模型预测的不稳定性，通过让多个base estimator投票或者取均值的方式。
3. decision tree非常不稳定，训练数据稍微有一点变化，分支条件就会改变，从而整个树都长得不一样了。
4. bagging trees 是什么？针对单个决策树方差过大的问题，bagging trees 利用bootstrap，提取B份独立的样本集，分别估计出预测值$\hat{f}^{1}(x), \hat{f}^{2}(x), \ldots, \hat{f}^{B}(x)$ ，然后将结果取平均，就可以得到一个方差更小的预测模型，通过多次实验获得多个regression tree或者classification tree，对每一个input，会产生多个输出，regression tree的输出是K个树的输出取average。classification tree是采用投票的方法，大多数相同的那一类就是输出的那一类 。
5. Bagging + decision tree综合起来构建随机森林。
    ![img_1.png](./img_1.png)
6. Bagging的部分可以并行化，从而可以更加有效地处理更大的训练数据。
7. 构建base estimator时，除了对样本采样，还可以对特征采样，随机森林的原始论文作者建议，在构建cart的每个分支条件时，都随机采样一个特征子空间，在这个子空间中找一个最优的分割点。
8. 随机森林论文的作者还建议，可个对特征作随机组合，即构建投影矩阵，将原始的训练数据映射到一个低维的特征空间。
9. random forest是bagging tree的加强版本，因为他考虑到了tree之间的de correlates，使得组合后结果的variance更小。构造树过程如下：consider 每一个split的时候，会从p个predictors中**随机**的挑$m = \sqrt p$）个作为split candidate，
10. boosting跟bagging tree不一样的地方在于，trees are grown sequentially: each tree is grown using information from previously grown trees. Boosting不涉及每个bootstrap采样。



## Adaboost

![adaboost](./img.png)

1. AdaBoost通过改变样本权重的方式得到不一样的base estimator，一边根据他们的表现决定给每个estimator多少票。

2. adaboosting的思路是，先训练出一个base estimator，根据预测结果的准确率，对样本的权重进行调整，预测不准的样本调高权重，预测准的样本调低权重，然后让下一个base estimmtor来学习。这样，原来base estimator搞不定的样本，在后续的学习中，得到更多的关注，最终的预测模型，是所有base estimator预测结果的线性组合，权重表示对应base estimator的预测准确率。


## GBDT 

1. GBDT是将一个优化问题，拆解成一系列子优化问题，每个子优化问题要解决的问题是，在当前学习成果的基础上查漏补缺，从而更加逼近target variable。即当前的base estimator的优化目标是最小化target和截止当前的预测值之间的差，也叫residual。
   ![img_4.png](img_4.png)
   h是当前的base estimator，$\eta$表示权重
2. 怎么求这个优化问题？先求里层的优化问题，再求外层的优化问题。里面的优化问题是将err对当前prediction做一阶泰勒展开，其中一阶项（h，待学习的目标）的系数就是error相对当前prediction的一偏导数，也就是梯度。这里有一个问题，如果我已经知道要做regression，也就是当前的目标函数是关于h的二次函数，我直接求最优解不就好了，即令损失函数关于h的导数为0？其实是可以的，但是GBDT的野心很大，想推导出，任意形式的err对应的最优解h，随意下面的一阶近似方法是更具备通用性的。（想象一下如果err是absolute loss或者像交叉熵这种非convex，就不适用了。btw，xgboost是使用err的二阶泰勒展开去作为err的近似）
  
   > 回忆一下一阶泰勒展开：
   > $$f(x + \Delta x) = f(x) + \Delta x \partial f_x $$
   
   ![img_5.png](img_5.png)
3. 里层的优化问题求解方法类似拉格朗日乘子法，先将h的大小限制到一定的范围内。然后经过各种变形之后，发现直接拟合residual就好。
   ![img_7.png](img_7.png)
4. 如何使用GBDT做回归？将上面的error 设置为squared loss。
6. AdaBoost是要根据不同样本的权重来找一一个拟合的最好的小g; GBDT是找一个能拟合当前残差最好的小g。


## xgboost

1. 除了Adaboost，GradientBoost，Boosting tree的另外一个代表是xgboost,简称为XGB

2. 相对GBDT用err的一阶泰勒展开做近似，XGB做了更逼近的近似--二阶近似，并且在err中加入了正则项，限制了base estimator的复杂度。
3. 为什么要使用二阶近似？
   1. 先回答为什么要用近似（不管是一阶还是二阶）？为了简化优化目标：让各种稀奇古怪的loss function都统一表达为关于f的二次函数，然后就可以使用同样的优化算法求最优的f了，不管loss长什么样都可以用一招方法。
   2. 再回答为什么是二阶而不是一阶？二阶更精确啊。



# 附录

## xgboost的公式推导

### 损失函数

先看看，xgboost怎么定义ensemble tree的损失函数：

$$\begin{array}{l}{\mathcal{L}(\phi)=\sum\limits_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum\limits_{k} \Omega\left(f_{k}\right)}\end{array}$$
$$ \\{\Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}} $$

- $\hat{y}_{i}=\phi\left(\mathbf{x}_{i}\right)=\sum^{K} f_{k}\left(\mathbf{x}_{i}\right), \quad f_{k} \in \mathcal{F}$
- l是一个可微convex函数
- k表示第k棵树
- i表示第i个样本
- T is the number of leaves in the tree.
- $f_k$表示第k棵树的预测函数，
- q代表树的结构，w代表叶子节点的权重
- $\Omega(f_k)$表示第k棵树的复杂度


### reformulate为一系列子优化问题

上面的优化问题的解，近似求一系列优化问题的解，如下

$\min \mathcal{L}^{(t)}=\min\limits_{f_t}\sum\limits_{i=1}^{n} l\left(\hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right), y_{i}\right)+\Omega\left(f_{t}\right)$

$\mathcal{L}^{(t)}$表示第t棵树的训练目标

### 将子优化问题做二阶近似

跟GBDT一样，将l进一步简化，使用二阶展开

$\mathcal{L}^{(t)} \simeq\sum\limits_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)$

这里

- $g_{i}=\partial_{\hat{y}^{(t-1) }}l\left(y_{i}, \hat{y}^{(t-1)}\right)$，即l对$y^{(t-1)}$的一阶导数

- $h_{i}=\partial_{\hat y^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)$，即l对$y^{(t-1)}$的二阶导数

进一步remove常数项，得到优化目标为：

$min \sum\limits_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)$

### 基于样本集重新定义损失函数

1. 更进一步，上面的优化目标可以转换为：
   $$\min\limits_{\omega, q} \sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} $$
   即：
   $$\min\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T $$
   
   - $I_{j}=\left\{i | q\left(\mathbf{x}_{i}\right)=j\right\}$为叶子j对应的特征子空间 
   
   
   也就是说，在构造第t个base estimator时，需满找到一个使得上面函数取得最小值的的base estimator，即找到最优$w^{\star}$​和$q^{\star}$​​对应的决策树。
   
2. 如何找到满足以上优化目标的决策树？拆解成两层优化问题，内层的优化问题确定叶子的预测值w，外层的优化问题确定分支q。

   $$\min_{q} （\min_{w}\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right] +  \gamma T） $$

3. 最优化叶子预测值w：对于某个给定分支(q)的决策树, $I_{j}$​​​​​是确定，

   $$\min_{w}\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]$$

   里面的优化问题是一个二次优化问题，直接令导数取0，即可得到最优解:
   $$w_{j}^{*}=-(\sum_{i \in I_{j}} g_{i}) /(\sum_{i \in I_{j}} h_{i}+\lambda)$$​​​​​​​

4. 最优化叶子分支q：将$w_{j}^{*}$​​带入上面的目标函数，优化问题转换为：

   $\min\limits_q  -\frac{1}{2} \sum\limits_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T$​  

可以使用构造决策树的方式--递归构造子树，使用该指标的下降来评价splitting的质量（cart使用entropy或者gini index对应的信息增益来衡量），当前的「信息增益」可以表达
  $$\mathcal{L}_{s p l i t}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma$$​​​  

其中，$I=I_{L} \cup I_{R}$

### 切分点查找算法

最朴素的算法，遍历所有特征的所有的切分值，找出最佳的。

目前大部分单机的机器学习库都是采用的这种算法。

![image-20200107161207967](./image-20200107161207967.png)


## 最佳实践

- number of trees:太大会造成过拟合，一般用cross validation去挑最好的参数
- shrinkage parameter:迭代步长，通常取0.01或者0.001，较小的lambda会需要很大的B来取得较好的效果。
- number of split d:通常d=1效果就比较好了。d表示的interation depth，就是交叉项的深度。

# 参考资料
1. [Random Forest Algorithm-Hsuan-Tien Lin](https://www.youtube.com/watch?v=ATM3sH0D45s&list=RDCMUC9Wi1Ias8t4u1OosYnHhi0Q&index=9)

2. [Adaptive Boosting linxuantian](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/208_present.pdf)
3. [youtube-gbdt](https://www.youtube.com/watch?v=2xudPOBz-vs)
4. [xgboost](https://arxiv.org/pdf/1603.02754.pdf)
5. [An Introduction to Statistical Learning](https://static1.squarespace.com/static/5ff2adbe3fe4fe33db902812/t/6062a083acbfe82c7195b27d/1617076404560/ISLR%2BSeventh%2BPrinting.pdf)
7. [Gradient Boosted Decision Tree :: Gradient Boosting](https://www.youtube.com/watch?v=F_EuNXhS9js&list=RDCMUC9Wi1Ias8t4u1OosYnHhi0Q&index=4)
5. [Blending and Bagging :: Bagging (Bootstrap Aggregation)- Tien Lin](https://www.youtube.com/watch?v=3T1mdvzRAF0&list=RDCMUC9Wi1Ias8t4u1OosYnHhi0Q&index=6)
7. [xgboost](https://arxiv.org/pdf/1603.02754.pdf)

