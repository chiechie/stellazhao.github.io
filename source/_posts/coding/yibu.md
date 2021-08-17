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
1. yield本质上是一个控制流程的工具。
2. 生成器函数作用不大，但是进行一系列功能改进之后， 得到了Python协程用处大了。
1. yield from：Python 生成器 中的 yield 有两个含义：产出和让步。yield item 这行代码会产出一个值，提供给 next(...) 的调用方;此外，还会作出让步，暂停执行 生成器，让调用方继续工作，直到需要使用另一个值时再调用 next()。调用方会从生成器中拉取值。
2. 从句法上看，协程与生成器类似，都是定义体中包含 yield 关键字的 函数。
3. 在协程中，yield 通常出现在表达式的右边(例如，datum = yield)，协程可能会从调用方 接收数据，不过调用方把数据提供给协程使用的是 .send(datum) 方 法，而不是 next(...) 函数。
1. 并发（Concurrency）： tasks start，run，and complete in overlapping time periods
2. 并行（Parallelism）：tasks run simulateously。

1. 并发（Concurrency）和并行（Parallelism）都是一次做多件事，但是并发是方案层面的概念，并行是执行层面的概念。并发的方案不一定要并行执行。
2. 并发编程和并行编程的区别和联系，类似并发和并行的区别和联系。
3. 举个例子的，笔记本电脑有4个CPU核心，通常有超过100个进程同时运行，就说这些进程是并发处理。但是，实际上CPU同时在做的事情不超过4件，这就叫并行（Parallelsm），另外96个事件处于pending状态，等待被唤醒。
4. Python语言中有一个asyncio包，使用事件循环驱动的协程实现并发。

	> Python 3.4把Tulip添加到标准库中时，把它重命名为asyncio。

5. 
6. Python3中有一个内置的package---asyncio，asyncio提供了一个基础环境，可以使用协程（corotuine）来编写单线程（single-threaded）的并发（concurrent）代码，多路復用（multiplexing）I/O通过sockets和其他资源，运行客户端和服务端。
7. 同步(sync)/异步(async), 
同步(sync)：不同task按照顺序进行。
异步(async)：两个task可以同时进行。
一个象棋高手，同时跟100个人下棋
同步：一个下完一步棋，接着跟另外一个人下。
异步：一个高手同时跟100个人下棋。
8. GIL: 全局解释器锁，存在该锁时，同一个时刻只有一个线程执行。
这样就起不到多线程的加速效果。
如何避开GIL的问题呢？可以使用multiprossing，基于多进程的，所以GIL就不会限制并行加速的效果。
9. 同步和异步，
10. CPU密集型程序，使用多进程（multi-processing）
11. IO密集型程序：IO延迟很短的话，可以使用多线程。IO延迟很慢的话，使用asyncio
12. 通过yield可以生成一个xiecheng。





## 参考

1. https://weread.qq.com/web/reader/ab832620715c017eab864a6kb6d32b90216b6d767d2f0dc
2. https://github.com/fluentpython/example-code
3. https://www.youtube.com/watch?v=zgKYPBj3x9E&list=PLfQqWeOCIH4ClkdjqvTuc9hnCnagSlBHE&index=2
