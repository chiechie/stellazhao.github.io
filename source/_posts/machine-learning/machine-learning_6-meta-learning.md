---
title: 机器学习6 元学习
author: chiechie
mathjax: true
date: 2021-07-21 14:47:47
tags:
categories:
---




## 总结

1. 小样本学习（few shot learning）是元学习（meta learning）的一个子类，目标是learn to learn，大致思路是这样的：

- 【在大样本中学习】在领域1中使用大样本中学习得到距离函数，
- 【在小样本中预测】在领域2中使用该距离函数找出最接近的样本，其类别就是预测结果。

>  few-shot learning 叫few-shot prediction会更贴切。

2. 领域1中的样本叫training set，领域2中的样本集叫support set，新预测的实例叫query sample。
3. support set有两个属性：k-way和n-shot，𝑘表示support set中的类别数，𝑛表示support set中每一类的样本数。
4. k和n对预测准确率的影响：k越大准确率越低，因为任务变复杂了；n越大准确率越高，因为可以学习的样本数变多了。
5. 在**学习**阶段，构造一个孪生网络，将样本从原始空间映射到特征空间，并且不同类别样本在特征空间中距离很远，相同类别距离很近
6. 在**预测**阶段，输入一个𝑘-way 𝑛-shot的support set和一个query，然后输出query的类别。使用孪生网络中的部分参数，计算query和support中每个样本的（特征的）距离，返回最近的样本的类别。
7. 小样本学习的思路是这样的：首先从一个大的训练集上学习一个相似度函数，然后将这个相似度函数应用到新的query样本上，先比较query和support中每个样本的距离，然后返回最接近的那个样本
8. 具体的建模方法有两类：构建孪生网络（siamese-network）和直接学习图片的embedding表示，后者效果更好。
9. 孪生网络有两个变体，分别是基于Pairwise loss的孪生网络和基于tripplet loss的孪生网络。



## 附录

### training set/support set/query sample三者关系

k-way和n-shot

![training set/support set/query sample](/Users/stellazhao/research_space/EasyMLBOOK/_image/image-20200505125818526.png)

- support set有两个属性：k-way和n-shot
- 𝑘-way:  support set中的类别数，图中为6，
- 𝑛-shot: support set中每一类的样本数，图中为1


### k和n对预测准确率的影响

![k和n对预测准确率的影响（](/Users/stellazhao/research_space/EasyMLBOOK/_image/image-20200505130507134.png)



#### 基于Pairwise loss的孪生网络

![基于Pairwise loss的孪生网络](/Users/stellazhao/research_space/EasyMLBOOK/_image/image-20200505131406968.png)


#### 基于tripplet loss的孪生网络

![基于tripplet loss的孪生网络](/Users/stellazhao/research_space/EasyMLBOOK/_image/image-20200505132048021.png)


## 参考

1. [slide](https://github.com/wangshusen/DeepLearning/blob/master/Slides/16_Meta_1.pdf)
2. [youtube](https://www.youtube.com/watch?v=Er8xH_k0Vj4)
3.  Bromley et al. Signature verification using a Siamese time delay neural network. In *NIPS*. 1994.
4. Koch, Zemel, & Salakhutdinov. Siamese neural networks for one-shot image recognition. In *ICML*, 2015.
5. Schroff, Kalenichenko, & Philbin. Facenet: A unified embedding for face recognition and clustering. In *CVPR*, 2015.
