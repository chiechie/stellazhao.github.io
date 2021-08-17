---
title: 计算机网络
author: chiechie
date: 2021-02-28 20:19:10
categories: 
- 编程
tags:
- 计算机网络
- 计算机原理
- 读书笔记
---



1. 什么是协议（protocol）？一堆定义好的标准集合没，每台电脑都要遵守，才能够正常通信
2. 什么是计算机网络？覆盖所有相关事物：电脑之间如何通信
3. 网络的作用是，保证每台电脑都能倾听到彼此，并且每台电脑说的话，其他电脑也能听得懂

4. 什么是TCP/IP模型？
   ![图1-TCP/IP网络模型](img_1.png)
   TCP/IP是一个5层的模型 ，每一层都有自己的协议，除了TCP还有，其他的模型如OSI。

5. TCP/IP模型中包含哪些东西？

- 物理层：即一些物理设备，跟电脑相互联系的，是通过网线（cable）连接的。物理层的几个核心元素是：cable，connectors，发送信号

![图2-物理层-使用网线将物理设备连接起来](img_2.png)
- 数据层：也叫网络接口层（networklayer），这里是第一次涉及到协议（才开始跟内容有关，物理层仅跟cable有关）
    数据连接层，负责定义如何解读这些信号。这一步之后，两边才能交流（communicate，不然就是鸡同鸭讲）。数据连接层有很多协议，最出名的叫"以太网"，尽管现在无线技术越来越popular，
  在物理层之上，以太网标准定义了一个协议，负责从同一个网络的一个节点获取数据，accross a single link。

- 网络层，也叫internetwork层
    允许不用的网络进行通信，当然，依靠的是路由器
  这里涉及到一个的概念
  internetwork ：一系列的网络，通过路由器，彼此相连接。
  最有名的internetwork叫Internet。
  这一层常用的协议叫：IP即 Internet Protocol
  IP是Internet和大部分更小的网络的核心，他的作用是将数据从一个节点传给另一个节点。
  network software通常被分类为：client 和 server
  client应用初始化一个request， 并且通过netwrok发送给server应用 ，
  server将response通过network发送回来

- 传输层，邮件的传输格式和网页的传输格式不一样，这个区别就是在这一层定义的。
  会议一下，
  网络层在两个node之间传输数据， 一个node，可以运行多个client应用或者server应用，
  传输层sort out哪一个client和server 应该去获取数据。
    第四层传输层用最常见的协议--TCP，Transmission Control Protocol。
  虽然TCP和IP经常被放到一起说，但是要注意，TCP和IP网络模型的不同层的协议。
  除了TCP，其他的传输层协议，也会利用IP先将数据送达，
  例如UDP，user datagram protocol
  UDP和TCP的区别在于，TCP机制可确保数据安全被送达，但是UDP并不要求。
  
    TCP和UDP是用来确保数据被一个node上正确的程序所接受。
    对比以下，IP只能保证数据传达到正确的node。
    类似送快递，IP可以保证包括从武汉商店送到深圳的腾讯大厦，
    TCP可以保证包裹 送至 对应的员工手中。
  
- 应用层，这一层的协议是跟具体的应用类型相关的
  比如网页类型的应用要准手http协议
  邮件类型的应用要准讯smtp协议。

再用送快递打个比方

- 物理层，就是一个一运送快递的卡车和道路
- 数据连接层：卡车将包裹从一个
- 网络层：识别出 从A到B要走哪条路
- 传输层：告诉A中哪个程序来收包裹。
- 应用层：就是包裹的内容。


![图3-用送快递类比TCP/IP网络模型的运行机制](img_4.png)

- TCP-FIN-WAIT1-客户端视角： 「客户端」 发出关闭指令后， 到服务器响应，间隔的时间。
- TCP-CLOSE-WAIT-服务端视角：「服务端」从接受”关闭指令“，到执行完成，间隔的时间。
- TCP-FIN-WAIT2-客户端视角： 「服务器」从 响应命令 到 完成命令，间隔的时间。

![图1-握手原理](dl-framework/img.png)

6. 为什么要分层？为了让职能分化（specialization）？让不同的人做不同的事？

## request

1. python中的requests库，可以允许对网页上发送http请求，从而获取信息。
2. 在http模型中，客户端向服务器端请求数据，通过一种方法，如get/post/put/delete
- get: 查询服务端的内容
- post：生产一些新的内容并返回
- put：替换
- delete：删除
3. http模型只是一个无状态的一次性问答的模式，随后出现了一种更高级的模式--Ajax请求，可以异步发送数据到服务器，在发送第一次请求之后。
4. websocks是一种更高级的通信模式，可以双向通信，也就是说可以持续说话，而不用等待某一方先发送一个请求，再回应。fullduplex。websocks是http模型的升级版。websocks的易用性更好，只用发送一次头文件。



## 私网IP 和 公网IP 分别是什么？

- 私网IP，也叫private ip address，或者内网IP。是一个局域网内部，用于识别彼此身份的一个id。
- 公网IP，也叫public ip address，或者外网IP。是在因特网中，用户识别设备身份的一个id。

> 早期，当联网设备只有一台电脑时，将你的电脑连向一个modem，会被分配一个ip，就是public ip，这个ip就是你的电脑在整个互联网的一个id
> 随着家中联网设备变多（电视机，手机，平板），需要有一个路由的功能，加上上面的modem的功能，就是现在的路由器。他可以将public ip映射为家庭环境中的多个id，分配给多个设备。注意家庭中联网设备构成了一个局域网（LAN）。

- (https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/)https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/

## modem 和 路由器 有什么区别？

- 一台电脑如要跟其他电脑通信，需先hook up on 一个modem，然后modem给他分配一个IP，也就是外网ip。
- 路由器是modem的升级版。 路由有两个作用：一个是连接家里的设备，另外一个是连接外面的世界。
  家里的多个设备组成了一个局域网，即使不跟外面通信，他们之间是可以通信的，通信的过程中如何识别彼此呢？
  这就是路由器的第二个作用，给局域网中的每个设备分配一个IP--也就是内网IP，这个是局域网的成员才能读懂的代码。

  
## 端口转发是什么？

由于外网只知道内网设备的外网地址，如何能让外网访问内网某个设备的某个端口的服务呢？
一个办法就是将局域网中的所有设备和服务，先注册到路由器中，路由器就可以根据外网的请求的端口号码，将之转移到内网提供相应服务的设备上。
这个就是端口转发做的事情。

## 网络中的三种设备-集线器，交换机 和 路由器

集线器和交换机是用来创建一个个小网络（局域网），路由器将这些小网络连接起来，变成因特网。

- 集线器（hub）： 把内网中的网络设备连接起来，只能广播。
- 交换机（switch）：维护一个交换机表，记录了连接的网络设备的mac地址，支持网络设备间p2p通信
- 路由器（router）：网关（gateway），网络出入口


## 参考
1. https://www.youtube.com/watch?v=8ARodQ4Wlf4
1. [私有ip和公共ip，以及端口forward](https://www.youtube.com/watch?v=92b-jjBURkw&list=LL70j7MlBEzFqOk5aIHKzmxQ)
1. [How to Enable SSH in Ubuntu---Install openssh-server？](https://www.youtube.com/watch?v=92b-jjBURkw&list=LL70j7MlBEzFqOk5aIHKzmxQ)
2. [how-do-you-run-a-ssh-server-on-mac-os-x](https://superuser.com/questions/104929/how-do-you-run-a-ssh-server-on-mac-os-x)
1. [The Bits and Bytes of Computer Networking](https://www.coursera.org/learn/computer-networking/home/welcome)
2. [西安交通大学公开课-计算机网络原理](https://open.163.com/newview/movie/free?pid=ME74DFHFC&mid=ME74E6NLA)
2. [TCP 握手和挥手图解（有限状态机）-CSDN博客](https://blog.csdn.net/xy010902100449/article/details/48274635)
