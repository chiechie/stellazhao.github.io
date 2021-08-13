---
title: 并发和并行
author: chiechie
mathjax: true
date: 2021-06-27 12:16:07
tags:
- 编程
categories:
- 编程
---



1. 并发（Concurrency）和并行（Parallelism）都是一次做多件事，但是并发是方案层面的概念，并行是执行层面的概念。并发的方案不一定要并行执行。
2. 并发编程和并行编程的区别和联系，类似并发和并行的区别和联系。
3. 举个例子的，笔记本电脑有4个CPU核心，通常有超过100个进程同时运行，就说这些进程是并发处理。但是，实际上CPU同时在做的事情不超过4件，这就叫并行（Parallelsm），另外96个事件处于pending状态，等待被唤醒。
4. Python语言中有一个asyncio包，使用事件循环驱动的协程实现并发。

	> Python 3.4把Tulip添加到标准库中时，把它重命名为asyncio。








## 参考

1. https://weread.qq.com/web/reader/ab832620715c017eab864a6kb6d32b90216b6d767d2f0dc
2. https://github.com/fluentpython/example-code
