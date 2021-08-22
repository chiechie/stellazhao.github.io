---
title: 日志处理
author: chiechie
mathjax: true
date: 2021-08-22 11:10:38
categories: 
- 编程
tags:
- 日志处理
---

现在面临一个需求：多个文件打印日志。


解决思路：将文本分为两类，1个主脚本，多个被调用的子脚本。

在多个子脚本中注册各自的logger
```python
import logging

logger = logging.getLogger(__name__)


def my_function():
    logger.info('something')
```




在主脚本中注册rooted logger，并且配置日志的全局变量

```python
import logging
import mypkg

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

mypkg.my_function()  # produces output to stderr
```




## 参考
1. https://stackoverflow.com/questions/40000929/python-logging-for-a-module-shared-by-different-scripts