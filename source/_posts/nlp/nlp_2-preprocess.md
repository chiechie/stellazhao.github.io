---
title: 自然语言处理2 预处理
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

- 分词的方法大致分为3类：：基于词典匹配/基于统计/基于深度学习。
  - 基于词典匹配：将待分词的中文文本根据一定规则切分和调整，然后跟词典中的词语进行匹配，匹配成功则按照词典的词分词，匹配失败通过调整或者重新选择，如此反复循环即可。代表方法有基于正向最大匹配和基于逆向最大匹配及双向匹配法
  - 基于统计：这类目前常用的是算法是**HMM、CRF、[SVM](https://easyai.tech/ai-definition/svm/)、深度学习**等算法，比如stanford、Hanlp分词工具是基于CRF算法。以CRF为例，基本思路是对汉字进行标注训练，不仅考虑了词语出现的频率，还考虑上下文，具备较好的学习能力，因此其对歧义词和未登录词的识别都具有良好的效果。
  - 基于深度学习：本质上是序列标注，所以有通用性，命名实体识别等都可以使用该模型，例如有人尝试使用双向LSTM+CRF实现分词器，其分词器字符准确率可高达97.5%。常见的分词器都是使用机器学习算法和词典相结合，一方面能够提高分词准确率，另一方面能够改善领域适应性。

- 中文分词工具：

  - [Hanlp](https://github.com/hankcs/HanLP)
  - [Stanford 分词](https://github.com/stanfordnlp/CoreNLP)
  - [ansj 分词器](https://github.com/NLPchina/ansj_seg)
  - [哈工大 LTP](https://github.com/HIT-SCIR/ltp)
  - [KCWS分词器](https://github.com/koth/kcws)
  - [jieba](https://github.com/yanyiwu/cppjieba)
  - [IK](https://github.com/wks/ik-analyzer)
  - [清华大学THULAC](https://github.com/thunlp/THULAC)
  - ICTCLAS

## 词干提取和词形还原

### 词干提取 – Stemming

词干提取是去除单词的前后缀得到词根的过程。大家常见的前后词缀有「名词的复数」、「进行式」、「过去分词」

cities，children，teeth 这些词需要词干提取转换为 city，child，tooth这些基本形态。



![词干提取](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-cigan.png)




##  

###  词形还原 – Lemmatisation

词形还原是基于词典，将单词的复杂形态转变成最基础的形态，而不仅仅是简单地将前后缀去掉，比如「drove」会转换为「drive」，dive就是原形。


does，done，doing，did 通过词性还原恢复成 do。

![词形还原](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-15-cixing.png)



### **为什么要做词干提取和词形还原？**

词干提取和词形还原的目的就是将含义相同，但是长相不同的词，统一起来，这样方便后续的处理和分析。

### 词干提取和词形还原对比

相同点：

1. 目标一致：词干提取和词形还原都是将长相不同，但是含义相同的词统一起来，方便后续的处理和分析。
2. 结果部分可能相同：如“dogs”的词干为“dog”，其原形也为“dog”。
3. 主流实现方法类似：均是利用语言中存在的规则或利用词典映射提取词干或获得词的原形。
4. 应用领域相似。主要应用于信息检索和文本、自然语言处理等方面，二者均是这些应用的基本步骤。



不同点：

1. 在复杂性上，词干提取更简单粗暴，只用删减前后缀。词形还原则更复杂，为了返回词的原形，不仅要进行词缀的转化，还要进行**词性识别**，区分相同词形但原形不同的词的差别。词性标注的准确率也直接影响词形还原的准确率，因此，词形还原更为复杂。





## 命名实体识别（NER）

命名实体识别（Named Entity Recognition，简称NER），是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。简单的讲，就是识别自然文本中的实体指称的边界和类别。

NER一直是NLP领域中的研究热点，从早期基于词典和规则的方法，到传统机器学习的方法，到近年来基于深度学习的方法，NER研究进展的大概趋势大致如下：

- 阶段 1：早期的方法，如：基于规则的方法、基于字典的方法

- 阶段 2：传统机器学习，如：HMM、MEMM、CRF

- 阶段 3：深度学习的方法，如：[RNN](https://easyai.tech/ai-definition/rnn/) – CRF、[CNN](https://easyai.tech/ai-definition/cnn/) – CRF

- 阶段 4：近期新出现的一些方法，如：注意力模型、迁移学习、半监督学习的方法

  

早期的命名实体识别方法基本都是基于规则的。之后由于基于大规模的语料库的统计方法在自然语言处理各个方面取得不错的效果之后，一大批机器学习的方法也出现在命名实体类识别任务。宗成庆老师在统计自然语言处理一书粗略的将这些基于机器学习的命名实体识别方法划分为以下几类：

- **有监督的学习方法**：这一类方法需要利用大规模的已标注语料对模型进行参数训练。目前常用的方法包括隐马尔可夫模型、语言模型、最大熵模型、支持向量机、决策树和条件随机场（CRF）等。CRF是命名实体识别中最成功的方法。

- **半监督的学习方法**：这一类方法利用标注的小数据集（种子数据）自举学习。

- **无监督的学习方法**：这一类方法利用词汇资源（如WordNet）等进行上下文聚类。

- **混合方法**：几种模型相结合或利用统计方法和人工总结的知识库。

值得一提的是，由于深度学习在自然语言的广泛应用，基于深度学习的命名实体识别方法也展现出不错的效果，此类方法基本还是把命名实体识别当做**序列标注**任务来做，经典的方法是[LSTM](https://easyai.tech/ai-definition/lstm/)+CRF、BiLSTM+CRF。



### NER 数据集

https://github.com/thunlp/THULAC

| **数据集**                 | **简要说明**                                                 | **访问地址**                                                 |
| :------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 电子病例测评               | CCKS2017开放的中文的电子病例测评相关的数据                   | [测评1](https://biendata.com/competition/CCKS2017_1/) \| [测评2](https://biendata.com/competition/CCKS2017_2/) |
| 音乐领域                   | CCKS2018开放的音乐领域的实体识别任务                         | CCKS                                                         |
| 位置、组织、人…            | 这是来自GMB语料库的摘录，用于训练分类器以预测命名实体，例如姓名，位置等。 | [kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) |
| 口语                       | NLPCC2018开放的任务型对话系统中的口语理解评测                | [NLPCC](http://tcci.ccf.org.cn/conference/2018/taskdata.php) |
| 人名、地名、机构、专有名词 | 一家公司提供的数据集,包含人名、地名、机构名、专有名词        | [boson](https://bosonnlp.com/dev/resource)                   |

 

### NER工具推荐

| **工具**                                        | **简介**                                                     | **访问地址**                                                 |
| :---------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Stanford NER                                    | 斯坦福大学开发的基于条件随机场的命名实体识别系统，该系统参数是基于CoNLL、MUC-6、MUC-7和ACE命名实体语料训练出来的。 | [官网](https://nlp.stanford.edu/software/CRF-NER.shtml) \| [GitHub 地址](https://github.com/Lynten/stanford-corenlp) |
| MALLET                                          | 麻省大学开发的一个统计自然语言处理的开源包，其序列标注工具的应用中能够实现命名实体识别。 | [官网](http://mallet.cs.umass.edu/)                          |
| Hanlp                                           | HanLP是一系列模型与算法组成的NLP工具包，由大快搜索主导并完全开源，目标是普及自然语言处理在生产环境中的应用。支持命名实体识别。 | [官网](http://hanlp.linrunsoft.com/) \| [GitHub 地址](https://github.com/hankcs/pyhanlp) |
| [NLTK](https://easyai.tech/ai-definition/nltk/) | NLTK是一个高效的Python构建的平台,用来处理人类自然语言数据。  | [官网](http://www.nltk.org/) \| [GitHub 地址](https://github.com/nltk/nltk) |
| SpaCy                                           | 工业级的自然语言处理工具，遗憾的是不支持中文。               | [官网](https://spacy.io/) \| [GitHub 地址](https://github.com/explosion/spaCy) |
| Crfsuite                                        | 可以载入自己的数据集去训练CRF实体识别模型。                  | [文档](https://sklearn-crfsuite.readthedocs.io/en/latest/?badge=latest ) \| [GitHub 地址](https://github.com/yuquanle/StudyForNLP/blob/master/NLPbasic/NER.ipynb) |



## 参考

1. https://easyai.tech/ai-definition/stemming-lemmatisation/#weizhi
2. https://easyai.tech/ai-definition/ner/

3. https://easyai.tech/ai-definition/stemming-lemmatisation/#weizhi

