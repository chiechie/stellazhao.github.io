---
title: python中的异步编程
author: chiechie
mathjax: true
date: 2021-08-22 22:30:12
categories:
- 编程
tags:
- 编程
- 协程
- 并发
- 计算机原理
- 线程
- 进程
- 异步
---




## 异步执行和同步执行

同步（synchronous）和异步（Asynchronous）：
同步就是一些事务按照顺序执行，做完第一个，接着做第二个，适用于比较简单的任务。
异步就是说事务穿插着进行，做第一个的途中，去做一下其他的，交替进行，适用于比较复杂的任务，很像实际工作构成中总免不了对上下游的依赖。

Asyncio是python自带的一个异步编程工具，利用该工具，通过编写简单的单线程逻辑，就可以实现异步逻辑。
简单点来说，借助于由async, await构成的coroutine, 用同步的方式，编写异步的代码！


- 可以使用多个协程(coroutines)来定义多个子任务，相当于多个worker，函数头前面加关键字async，子任务之间的相互依赖关系使用关键字await
- 然后使用1个协程(coroutine)作为主任务，相当于一个master将这些子任务编排起来，使用asyncio.gather
- 最后使用asyncio.run()将主协程启动起来。


```python
import asyncio


async def foo():
    print('Running in foo')
    await asyncio.sleep(0)
    print('Explicit context switch to foo again')


async def bar():
    print('Explicit context to bar')
    await asyncio.sleep(0)
    print('Implicit context switch back to bar')


async def main():
    tasks = [foo(), bar()]
    await asyncio.gather(*tasks)


asyncio.run(main())

```

输出为：
```markdown
$ python 1b-cooperatively-scheduled.py
gr1 started work: at 0.0 seconds
gr2 started work: at 0.0 seconds
Let's do some stuff while the coroutines are blocked, at 0.0 seconds
Done!
gr1 ended work: at 2.0 seconds
gr2 Ended work: at 2.0 seconds
```




## 参考
1. https://yeray.dev/python/asyncio/asyncio-for-the-working-python-developer
2. [asyncio资料汇总](https://github.com/timofurrer/awesome-asyncio)
3. https://zhuanlan.zhihu.com/p/237067072

