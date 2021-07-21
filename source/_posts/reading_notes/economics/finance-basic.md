---
title: 金融基础
author: chiechie
mathjax: true
date: 2021-07-15 22:50:07
tags:
categories:
---



##  总结

1. 资产是什么？连续产生现金流。比如公司，股票债券，知识，名声。

2. 如何对资产定价？未来所有现金流折现。

3. 如何计算net PV？减去初始时刻的投资。

4. 按什么折现？无风险利率。

	> ...公认的risk free rate是指美帝的treasury security rate,也就是美国国债的利率。通常而言，根据maturity长短不同，我们有主要三种treasury securities：T-Bill（一年以下），T-Note（两年到十年），和T-Bond（十年以上）。

5. 复利？不仅仅一年才计息，银行账户每天计利，抵押物每个月计利息，债券半年计利息。

6. 有效年利率和年利率
   1. 有效年利率（EAR effective annual interest rate）是指把各种不同的复利周期（如一年复利12次，或者一年复利4次）换算为以年为一个复利周期时，利率是多少。
   2. 年利率(annual interest rate)是指银行公布的利率，一般需要和公布的复利频率结合起来用，才能知道每一个复利期真正的利率，单独看用处不大。

7. 通胀（inflaction）：1美元购买力的变化。（对应物理世界）

8. 计算资产的NPV时，实物现金流使用实际利率折现；名义现金流使用名义利率折现。

9. 固定收益证券，未来现金流固定的资产。

10. 固定收益市场的参与者：证券发行人/承销商/投资人。

11. 债券的即期收益率，到期收益率，远期收益率有什么区别？

    1. **YTM主要用于quotation，spot rate主要用于pricing，而forward rate多用于modeling。**

    2. 即期利率与远期利率的区别在于某利率起作用的起点
       即期利率通常在即期交易前1-2天报价
       远期利率是某一未来时间到另一未来时间的利率.远期利率是用即期利率根据无套利原则推算的

       ![img](https://pic2.zhimg.com/80/8c16de723d1089665bd6d38d94b66738_1440w.jpg?source=1940ef5c)

    3. YTM是一种内部收益率（Internal rate of return），
       是净现值NPV=0(即成本与收益现值相等)时的收益率。

    4. yield curve也更多是作为市场的一个indicator，而不是分析工具。

12. 股票定价模型：
    1. 分红折现模型v1--折现现金流模型（DCF）：假设分红是常数
    2. 分红折现模型v2--gorden增长模型：分红以g的增长率增长

13. 期货定价

    ![image-20210716000554865](/Users/shihuanzhao/research_space/chiechie.github.io/source/_posts/finance-basi/image-20210716000554865.png)

## 参考

1. https://ocw.mit.edu/courses/sloan-school-of-management/15-401-finance-theory-i-fall-2008/video-lectures-and-slides/MIT15_401F08_lec02.pdf
2. https://www.zhihu.com/question/22562103/answer/21833989
3. https://ocw.mit.edu/courses/sloan-school-of-management/15-401-finance-theory-i-fall-2008/video-lectures-and-slides/MIT15_401F08_lec04.pdf
4. [债券的即期收益率，到期收益率，远期收益率有什么区别？]((https://www.zhihu.com/roundtable/money))
5. https://ocw.mit.edu/courses/sloan-school-of-management/15-401-finance-theory-i-fall-2008/video-lectures-and-slides/MIT15_401F08_lec07.pdf
