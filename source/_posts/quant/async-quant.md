---
title: 构建一个双均线策略
author: chiechie
mathjax: true
date: 2021-08-22 23:30:15
categories:
- 量化
tags:
- 计算机原理
- 线程
- 进程
- 异步
- 量化
---



做中高频率的量化交易时，需要较高的响应速度，灵活的通信方式，常常会使用到异步编程的范式。

众多方法中, 基于协程实现的ayncIO在python语言中使用的最多。

下面实现一个基于ayncIO的双均线策略的实现的流程：

- 订阅实时行情：使用websocket订阅行情
- 实时触发交易：在订阅行情的回调函数中加入策略函数，即下单逻辑。回调函数是一个异步的调用，即是在做行情推送这个事情的同时，还要监控有没有交易信号，然后执行交易策略。


待补充

---



1. https://github.com/paulran/aioquant/blob/cfa4344f6787dd380339add6261bcce23a7e7e8e/aioquant/trade.py#L22
2. https://github.com/paulran/aioquant/tree/cfa4344f6787dd380339add6261bcce23a7e7e8e/example/demo
3. [chiechie的github--使用binance的API实现一个可实时交易的双均线策略](https://github.com/chiechie/quantML/blob/master/test/test_binance.py)