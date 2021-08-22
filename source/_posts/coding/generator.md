---
title: 生成器
author: chiechie
mathjax: true
date: 2021-08-22 11:11:23
categories: 
- 编程
tags:
- 日志处理
---



在处理大数据量的问题时，经常会遇到内存瓶颈问题，一种常用的解决思路是将批处理转化为流处理(on the fly).
通过改变计算逻辑，来换取原本消耗的大量内存。


具体到python里，最常用到的就是生成器(generator), 以下面的代码为例子：

处理完一个chunk数据，就载入下一个chunk的数据来覆盖原有的内存，不给内存带来额外的负担。

内存中始终占用的大小为chunk_size，而不是所有的原始数据。

```python

# 构造一个生成小分区dataframe的生成器
def dataframe_generator(dataframe, chunk_size):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, dataframe.shape[0], chunk_size):
        yield i, dataframe.loc[i:i + chunk_size, :]


def run_paralle_process(dataframe, predefined_varibles, workers=16):
    model_list = []
    chunk_size = dataframe.shape[0] // workers
    for chunk, dataframe_single in dataframe_generator(dataframe, chunk_size=chunk_size):
        logger.info(f"begin deal with chunk = {chunk}, which has {dataframe_single.shape[0]} logs")
        model = fit_single(dataframe_single,
                           dataframe_schema={},
                           max_dist_list="0.6,0.5",
                           min_members=2,
                           predefined_varibles=predefined_varibles,
                           delimeter=delimeter)

        model_list.append(copy.deepcopy(model))
        logger.info(f"end deal with chunk = {chunk}, which has {dataframe_single.shape[0]} logs")
        chunk += 1
    ...
```



符合类似思路的还有sgd。


## 参考
1. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
