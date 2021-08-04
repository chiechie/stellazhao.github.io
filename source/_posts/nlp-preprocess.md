---
title: 文本的预处理
author: chiechie
mathjax: true
date: 2021-08-03 20:42:05
tags:
- 文本
categories:
- NLP
---



## 总结

- NLP 的第一步是语料预处理，语料预处理的流程为：分词(tokenization),  词干提取（Stemming），词性还原（Lemmatisation）。

![词干提取和词形还原在 NLP 中在什么位置](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-weizhi.png)

- 中英语处理的区别：

  1. **英文单词有多种形态**，所以需要做词干提取和词性还原，
  2. 中文有多种分词方式，**并且有多种粒度**



## 分词（Tokenization）


- 目前中文分词没有统一的标准，也没有公认的规范。

- 3种典型的分词方法：基于词典匹配/基于统计/基于深度学习。
  - 词典匹配： 将待分词的中文文本根据一定规则切分和调整，然后跟词典中的词语进行匹配，匹配成功则按照词典的词分词，匹配失败通过调整或者重新选择，如此反复循环即可。代表方法有基于正向最大匹配和基于逆向最大匹配及双向匹配法。
  - 基于统计：基本思路是对汉字进行标注训练，不仅考虑了词语出现的频率，还考虑上下文，具备较好的学习能力，因此其对歧义词和未登录词的识别都具有良好的效果。常用的是算法是HMM、CRF、SVM等算法，比如stanford、Hanlp分词工具是基于CRF算法。以CRF为例，
  - 基于深度学习：本质上是序列标注，所以有通用性，命名实体识别等都可以使用该模型，例如有人尝试使用双向LSTM+CRF实现分词器，其分词器字符准确率可高达97.5%。

- 中文分词工具：

  - [Hanlp](https://github.com/hankcs/HanLP)
  - [Stanford 分词](https://github.com/stanfordnlp/CoreNLP)
  - [ansj 分词器](https://github.com/NLPchina/ansj_seg)
  - [哈工大 LTP](https://github.com/HIT-SCIR/ltp)
  - [KCWS分词器](https://github.com/koth/kcws)
  - [jieba](https://github.com/yanyiwu/cppjieba)
  - [IK](https://github.com/wks/ik-analyzer)
  - [清华大学THULAC](https://github.com/thunlp/THULAC)
  - [ICTCLAS](https://github.com/thunlp/THULAC)


## 词干提取（Stemming）

词干提取是去除单词的前后缀得到词根的过程。大家常见的前后词缀有「名词的复数」、「进行式」、「过去分词」



cities，children，teeth 这些词需要词干提取转换为 city，child，tooth这些基本形态。



![词干提取](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-cigan.png)




##  词形还原 – Lemmatisation

词形还原是基于词典，将单词的复杂形态转变成最基础的形态，而不仅仅是简单地将前后缀去掉，比如「drove」会转换为「drive」，dive就是原形。


does，done，doing，did 通过词性还原恢复成 do。

![词形还原](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-cixing.png)



1. 

## 参考

1. https://easyai.tech/ai-definition/stemming-lemmatisation/#weizhi
