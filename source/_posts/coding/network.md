---
title: web服务和web应用
author: chiechie
mathjax: true
date: 2021-08-22 14:32:29
categories:
- 编程
tags:
- 编程
- 协程
- 并发
---



## 客户端跟web应用通信的过程

• 用户请求一个web服务的过程：用户操作使用浏览器发送请求；浏览器转发到web服务器，web服务器将请求标准化，转发给web应用，wenb应用处理并返回结果为web服务器，web服务器返回给浏览器，浏览器将结果解析并展示给用户
• Django是web应用的开发框架， WSGI/Gunicorn/asgi是web服务工具
• 在web服务层或者协议层需要一个web服务工具，传统有wsgi和gunicorn（不支持http2），现在多用asgi（支持http2）



## websocket


• 首先有两类通信模型：客服模式（被动回复型）和接线员+客服模式（主动回复型，也叫回调）
• 在遇到请求个数变多时：
• 客服模式可拓展为轮询或者轮询+阻塞；
• 接线员+客服模式即为，接线员和客服分工，接线员负责验证用户身份，以及把请求分配给客服，客服先回复给接线员，接线员再转发给用户，可以解决客服不够用，客服工作负荷大的（请求一次，就要验证一下身份）问题。


## Web服务器

• Web服务器在将请求转交给web应用程序之前，需要先将http报文转换为WSGI规定的格式
• Web服务器比较底层，一般直接用，简单说web服务主要做，http协议解释，请求转发，请求高并发，高可用这些咯。
• Gunicorn是一个 UNIX 下的 WSGI HTTP 服务器，在管理 worker 上，使用了 pre-fork 模型，即一个 master 进程管理多个 worker 进程，所有请求和响应均由 Worker 处理。Master 进程是一个简单的 loop, 监听 worker 不同进程信号并且作出响应，比如提升/降低 worker 数量，如果 worker 挂了，则重启失败的 worker。
• WSGI全称Python Web Server Gateway Interface，指定了web服务器和Python web应用或web框架之间的标准接口，以提高web应用在一系列web服务器间的移植性。
• WSGI是一套接口标准协议/规范；
• 通信（作用）区间是Web服务器和Python Web应用程序之间；
• 目的是制定标准，以保证不同Web服务器可以和不同的Python程序之间相互通信
• web应用处理请求的具体流程：
• 用户操作操作浏览器发送请求；
• 请求转发至对应的web服务器
• web服务器将请求转交给web应用程序，web应用程序处理请求
• web应用将请求结果返回给web服务器，由web服务器返回用户响应结果
• 浏览器收到响应，向用户展示
• 可以看到，请求时Web服务器需要和web应用程序进行通信，但是web服务器有很多种，Python web应用开发框架也对应多种啊，所以WSGI应运而生，定义了一套通信标准。试想一下，如果不统一标准的话，就会存在Web框架和Web服务器数据无法匹配的情况，那么开发就会受到限制，这显然不合理的。
• 既然定义了标准，那么WSGI的标准或规范是？
• web服务器在将请求转交给web应用程序之前，需要先将http报文转换为WSGI规定的格式。
• WSGI规定，Web程序必须有一个可调用对象，且该可调用对象接收两个参数，返回一个可迭代对象：
• environ：字典，包含请求的所有信息
• start_response：在可调用对象中调用的函数，用来发起响应，参数包括状态码，headers等


## 参考

1. https://zhuanlan.zhihu.com/p/95942024
2. https://www.zhihu.com/question/20215561/answer/40316953
3. https://zhuanlan.zhihu.com/p/102716258