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



## 词干提取 – Stemming

词干提取是去除单词的前后缀得到词根的过程。大家常见的前后词缀有「名词的复数」、「进行式」、「过去分词」…

![词干提取](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-cigan.png)



##  词形还原 – Lemmatisation

词形还原是基于词典，将单词的复杂形态转变成最基础的形态，而不仅仅是简单地将前后缀去掉，比如「drove」会转换为「drive」。



![词形还原](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-cixing.png)



## **为什么要做词干提取和词形还原？**

词干提取和词形还原的目的就是将含义相同，但是长相不同的词，统一起来，这样方便后续的处理和分析。



## 词干提取和词形还原对比

相同点：

1. 目标一致：词干提取和词形还原都是将长相不同，但是含义相同的词统一起来，方便后续的处理和分析。
2. 结果部分可能相同：如“dogs”的词干为“dog”，其原形也为“dog”。
3. 主流实现方法类似：均是利用语言中存在的规则或利用词典映射提取词干或获得词的原形。
4. 应用领域相似。主要应用于信息检索和文本、自然语言处理等方面，二者均是这些应用的基本步骤。



不同点：

1. 在复杂性上，词干提取更简单粗暴，只用删减前后缀。词形还原则更复杂，为了返回词的原形，不仅要进行词缀的转化，还要进行**词性识别**，区分相同词形但原形不同的词的差别。词性标注的准确率也直接影响词形还原的准确率，因此，词形还原更为复杂。
2. 

## 参考

1. https://easyai.tech/ai-definition/stemming-lemmatisation/#weizhi
