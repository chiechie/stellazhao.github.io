---
title:  chapter 2.1 树模型-决策树介绍
author: chiechie
mathjax: true
date: 2021-04-16 20:56:02
tags: 
- 人工智能
- 树模型
- 决策树
categories:
- 机器学习
---

# TLDR

1. 决策树是一个预测算法。
2. 决策树构造过程：递归构造一个大树，然后剪纸。构造大树的过程，每次只构造一个树桩，即，找到最佳的切分条件，使得切分前后信息增益最大。对生成的两个子树继续分裂，直到某个子树中样本集纯度很高，就标记为叶子结点。叶子节点的值就是落入其中的样本label取平均值或者众数。
3. 决策树的预测过程：将样本按照分支条件路由到某个叶子结点，叶子结点的值就是预测值。
1. 决策树也是一个常见的机器学习模型，基本思想是，根据特征构造出树结构的预测模型。
2. 一个决策树包含一个根节点，若干内部节点以及叶子节点。
3. 叶子节点对应最终的决策结果，其他节点针对数据的某个属性（特征）进行判断和分支：在该节点中，会根据样本的某个特征的值进行判断，依据判断结果将样本划分到某个子树之中，
4. 通过决策树，可以从根节点出发，把一个具体的样本最终分配到某个叶子节点上，实现预测的目的。
5. 因为每个节点上的分支操作是非线性的，因此决策树可以实现比较复杂的非线性映射，
6. 决策树算法的目的是，根据训练数据，学习处一颗泛化能力较强的树，能够很精准的把位置样本分到对应的叶子节点上。
7. 树模型的优势在于可解释性，但是很难兼顾准确性和泛化性，
8. 常见的决策树算法包括：分类回归树（CART），ID3算法，C4.5算法，决策树桩（Decision Stump）。总结起来，这些算法的流程都包括分支和剪枝两个步骤。
9. 分支的目的是，找到一个划分条件（一般是某个特征满足某个条件），使得划分之后能够提升样本的纯度。
10. 剪枝要解决的问题是减少过拟合，有两种剪枝的技术：预剪枝和后剪枝。
    1. 预剪枝是指，在决策树生成过程中，对每个节点在划分前使用验证集来估计，该划分能否带来决策树泛化性能提升，如果没有提升则将当前节点标记为叶子结点；
    2. 后剪枝是指，从训练集中生成一颗完整的决策树，然后自下而上的考察去掉每个节点是否提升了泛化能力（将该节点及其子树合并成一个叶子节点），没有提升的话就进行剪枝。

11. 有些情况下数据中隐含的模式比较复杂，那么单个决策树的表达性能不够，通常会使用集成学习来提升树的表达能力。将多棵树进行组合，有不同的方式组合，如bagging/boosting等。
12. adaboosting的思路是，先训练出一个base estimator，根据预测结果的准确率，对样本的权重进行调整，预测不准的样本调高权重，预测准的样本调低权重，然后让下一个base estimmtor来学习。这样，原来base estimator搞不定的样本，在后续的学习中，得到更多的关注，最终的预测模型，是所有base estimator预测结果的线性组合，权重表示对应base estimator的预测准确率。
13. boosting在抵抗过拟合方面效果很好，随着训练的推进的，即使在训练机已经把误差降到0，更多的迭代还可以提高测试机上的准确率，可以用间隔定理（Margine Theory）来解释这个现象：随着迭代的推进，训练样本上的分类置信度（即每个样本点上的间隔）仍在不断变大。




# 附录



## 信息增益



- 信息增益: 衡量切分前后，样本纯度的提升or混乱度的下降。

```python
IG = information before splitting (parent) — information after splitting (children)
```

- 具体的，有两个衡量纯度/混乱度的指标：Entropy 和 Gini Impurity
  - 基尼系数（**gini index**）: $$I_{G}=1-\sum_{j=1}^{c} p_{j}^{2}$$
    - $p_j$: 落入该节点的样本中，第j类样本的占比
    - 如果所有样本都属于某一类c，gini系数最小，为0。
  - 熵（entropy）：$$I_{H}=-\sum_{j=1}^{c} p_{j} \log _{2}\left(p_{j}\right)$$
    - $p_j$: 落入该节点的样本中，第j类样本的占比
    - 如果所有样本都属于某一类c，熵最小，为0。





## 决策树

1. 决策树是一个预测算法。
2. 决策树构造原理，用递归的方法来找到一颗大树，然乎剪纸。决策树预测的原理，将新数据按照决策树的分支条件路由到叶子结点，叶子结点中的值就是预测值。
3. 构建一个决策树需要确定分支条件以及每个分支的预测值。构建分支条件即把特征空间划分为多个distinct and non-overlapping regions,$R_1,\dots,R_J$；第二步，对每个region定义一个response variable，作为该region的值。
4. 构建好了一个决策树，想要使用决策树做预测，步骤也分为两步：第一步是按照分支条件将新样本路由到指定的叶子结点，第二步将叶子结点对应的responsible varible作为该样本的预测值。
5. 怎么得到分支条件呢？有三类构建决策树的算法：ID3，C4.5和CART。前两者可以构造多叉树，CART只能构造二叉树。 
  因为CART效果最好，现在通常就用它（例如sklean）。
6. ID3，C4.5和CART三类算法的大致思路一样，分为两步：第一步是将feature space切成多个boxes。为什么不是切成多个球？因为球没法填充整个feature space.
7. 如何找到最优的切割boxes的方式，如果去遍历每一组partition of feature space，计算量太大了，通常采用greedy的方法。
8. 构建决策树的过程中需要确定的参数：分支的个数，条件分支的条件，终止条件，叶子结点的值。

## 决策树算法-CART

1. CART代表分类树和回归树，分别使用均方误差和entropy/ginix index计算信息增益。CART的原理就是，先递归构造一颗大树$T_0$，然后剪枝。

1. CART的特色是构建的一个binary tree，每次分支条件都是将一个空间分为2个子空间，每个叶子结点的值，即response variable都是一个常数，response variable是这么的到的：
  - 如果target variable是一个连续变量，求落入该region的训练集的response均值，即${y_n}$的均值，其实这个均值对应的是最小化suqared error。
  构建一个决策树，主要是要确定partition，或者说分支条件，以及每个落入每个partition（或者说）中对应的预测值。
  - 如果target var是一个离散变量，求众数对应的那个类别。
2. 怎么确定分支条件？找一个decision stump，使用纯度（purify）来衡量分支的质量，如果左边的data set 和右边的dataset 纯度 都很高--即其中的大部分样本的label很接近--就说切分的很好。对应到计算上面，就是找一个让平均不纯度最小的切分方式（decision stump）。
3. 如何确定分支/切割的不纯度？
    - 如果target variable是一个连续变量，使用均方误差（squared loss）来描述impurity，（跟样本子集的均值比），
    - 如果target variable是一个离散变量，使用不一致样本比例来描述impurity，（跟样本子集的众数币），多分类的时候，不纯度常用giniindex，或者熵。
    - 如果是多分类，经常使用Gini index来刻画不纯度。
4. 什么时候会停下来？当满足下面的条件时，也叫fully grown tree：
    - 落入某个分支的样本的target都一样，不纯度取到最小了，
    - 落入某个分支的样本的x都一样，没有decision stupms了。
5. 为何要剪枝(pruning)? 高方差，过拟合。

6. 

### 回归树（regression tree）

regression tree的cost function是均放误差RSS加上正则项

$$\min\limits_{T\in T_0} \sum\limits_{m=1}^{|T|}\sum\limits_{x_i\in R_m}(y_i - \hat y_{R_m})^2+\alpha|T|$$

- $|T|$是叶子节点的个数。
- m表示第m个叶子
- $R_m$表示第m个partition region 
- $y_i$表示第i个样本的真实值
- $y_{R_m}$表示第m个partition region的预测值

使用递归的方法构建一个决策树，确定分支个叶子。主要是要确定partition，或者说分支条件，以及每个落入每个partition（或者说）中对应的预测值。

![regression tree 构造流程](./img.png)

### 分类树（classification tree）

classification tree切分节点时，参考信息增益，其他流程和构建回归树是一样的

如果response var是imbalance, 全部预测为label占比更多的类，怎么办？



## 决策树算法-ID3

ID3,  was the first of three Decision Tree implementations developed by Ross Quinlan

It builds a decision tree for the given data in a top-down fashion. each node of the tree, one feature is tested based on 最大熵降, and the results are used to split the sample set. This process is recursively done until the set in a given sub-tree is homogeneous (i.e. it contains samples belonging to the same category). The ID3 algorithm uses a greedy search. 

Disadvantages:

- Data may be over-fitted or over-classified, if a small sample is tested.
- Only one attribute at a time is tested for making a decision.
- Does not handle numeric attributes and missing values.

## 决策树算法-C4.5

Improved version on ID 3 . The new features (versus ID3) are:

- (i) accepts both continuous and discrete features; 
- (ii) handles incomplete data points;
- (iii) solves over-fitting problem by (very clever) bottom-up technique usually known as "pruning"; and 
- (iv) different weights can be applied the features that comprise the training data.

Disadvantages

- Over fitting happens when model picks up data with uncommon features value, especially when data is noisy.


# 参考
1. [决策树算法-linxuantian](https://www.youtube.com/watch?v=s9Um2O7N7YM)

2. [决策树-linxuantian](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/209_present.pdf)

3. [An Introduction to Statistical Learning](https://static1.squarespace.com/static/5ff2adbe3fe4fe33db902812/t/6062a083acbfe82c7195b27d/1617076404560/ISLR%2BSeventh%2BPrinting.pdf)

4. [Why-is-entropy-used-instead-of-the-Gini-index](https://www.quora.com/Why-is-entropy-used-instead-of-the-Gini-index)

5. [github-id3的实现1](https://github.com/dozercodes/DecisionTree)

6. [github-id3的实现2](https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/decision_tree_functions.py)

7. [wiki-Information_gain_in_decision_trees](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)

8. [sklearn-decisiontree](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py)

9. [quora-ID3-C4-5-and-CART的区别？](https://www.quora.com/What-are-the-differences-between-ID3-C4-5-and-CART)

1. [youtube-gbdt](https://www.youtube.com/watch?v=2xudPOBz-vs)
2. [xgboost]( https://arxiv.org/pdf/1603.02754.pdf)
