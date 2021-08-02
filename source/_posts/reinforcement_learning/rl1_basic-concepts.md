---
title: 强化学习1 基本概念
author: chiechie
mathjax: true
date: 2021-04-19 20:27:02
tags:
- 强化学习
- 人工智能
categories:
- 强化学习
---



> 相对监督学习，强化学习的神奇之处在于，可能可以超越人类。而监督学习只能模仿，不能超越。
>
> 

# 总结

1. 强化学习是一个序列决策过程，它试图让agent找到一个聪明的策略，使得agent在跟环境互动的过程中，获得最大的长期价值。强化学习中的一些基本概念：
   2. agent和环境：agent就是一个机器人，是我们希望去设计出来的一个AI，环境负责跟AI进行互动，对agent的动作评价，以及生成新的状态。agent接受当前时刻的状态，返回动作agent，环境接受行动，返回下个时刻的的状态，以及奖励。
   2. 状态(state)：当前环境，在围棋例子中，表示当前的棋局。
   3. 状态转移(state transaction)：状态转移可以是确定的也可以是随机的（随机性来源于环境）
   4. 行动(action)表示当前agent的行动空间，在围棋例子中，棋盘上有361个落子位置，对应有361个action
   5. 动作空间(action space)是指所有可能动作的集合，在围棋例子中，动作空间是 A = {1, 2, 3, · · · , 361}。
   6. 策略函数(policy)：表示agent根据观测到的状态做出决策. 策略函数常常被定义为一个条件概率密度函数，输入当前state，输出动作空间每个动作的概率得分。强化学习的目的，就是希望让agent学习到一个聪明的策略函数。
   7. 即时收益（reward)和长期收益（discounted return, aka，cumulative discounted future reward）
      1. Reward对于某个<s，a>，是在agent执行一个动作之后，环境返回的一个激励信号，是一个数值记为$R_t$，取值越大说明越赞赏当前的行动.
      2. Return对于某个<s，a>，即时收益的折扣累积就是长期收益, 记为$$U_{t}=R_{t}+ \gamma R_{t+1}+\gamma ^{2} R_{t+2}  \cdots$$​
      3. reward和return都是随机变量：reward的随机性来自当前状态和当前行动。return的随机性来自状态转移和 policy。
3. 强化学习的方法通常分为两类，基于模型的方法（model-based）和 无模型的方法（model-free）。model-based是强化学习中一种技术，核心思想就是对环境建模，使用计算机构建仿真环境。比如构建一个模拟环境，更新状态和reward。model-based是为了解决样本获取成本高的问题。例如，无人驾驶/无人机，如果让机器在真实交通环境中行驶，并且通过于真实环境互动来获取数据，那必然以发生多次严重车祸作为获取样本的巨大代价。在机器人场景中应用较多。
3. 跟model-based相对的概念是无模型（model-free）方法，它直接通过跟环境交互来获取样本，假设现实中获取样本的成本几乎为0。但是现在的强化学习论文大部分都采用model-free的方法，可见离落地还有一段距离。无模型的方法又大至可分为价值学习和策略学习。
4. 强化学习中，控制agent有两种方法：最大化策略函数和最大化动作价值。
5. 最大化策略函数的方法，也叫策略学习方法，即使用一个神经网络来近似策略函数。更新策略梯度的方法分别有reinforce和actor-critic。
6. 最大化动作价值的方法，也叫价值学习方法，即使用神经网络拟合最优动作价值函数。常用的价值学习方法例如DQN网络，通常采用时序差分的方法去估计网络参数。价值学习高级技巧:对TD算法改进;对神经网络结构改进.
7. 策略梯度中的baseline:策略梯度（Policy gradient） 方法中常用 baseline 来降低方差,加速收敛。Baseline还可应用于REINFORCE 和 A2C 方法中。
8. 连续控制问题即使用随机策略做连续控制。
9. 如何引导agent做出好的策略？定义合适的收益函数，并且对收益求期望--即价值，然后以最大化价值作为目标来训练agent。
10. 强化学习中的价值函数主要有两个作用：一是用来判断agent的策略的好坏，一是用来判断当前局势（state）的好坏。强化学习中有3个价值函数: 行动价值函数，最优行动价值函数，状态价值函数。行动价值函数和最优行动价值函数都是用来判断agengt的动作好坏，状态价值函数用来判断当前局势好坏。
11. 强化学习的应用：神经网络结构搜索；自动生成SQL语句；推荐系统；网约车调度。






#  附录


## 强化学习的两个方向: 价值学习和策略学习

价值学习:

1. 价值学习 (Value-Based Learning) ，是指 以最大化价值函数（3个价值函数都可）为目标去训练agent。

2. 价值学习的大致思路是这样的，每次agent观测到一个$s_t$,就把它输入价值函数，让价值函数来对所有动作做评价（分数越高越好）

3. 那么问题来了，如何去学习价值函数呢？可以使用一个神经网络来近似价值函数。
4. 最有名的价值学习方法是DQN, 还有Q-learning



策略学习

1. 策略学习指的是学习策略函数$\pi$, 然后agent可利用策略函数计算不同state下行动的得分，然后随机选一个执行。
2. 那么问题来了，如何去学习策略函数呢？可以使用一个神经网络来近似策略函数，然后使用策略梯度来更新网络参数。
3. 策略梯度算法的代表有REINFORCE



## 价值学习

价值学习： value-based，目的是学习最优行动价值函数。

- Deep Q network: 近似最优行动价值函数
- 时间差分（TD）算法:TD（Temporal Difference）算法：
  - SARSA算法：基于表格的方法和基于神经网络的方法
  - Q-learning算法
  - Multi-step TD target

- 策略学习：policy-based，目的是让agent直接学会最优策略。 actor critic
  - 使用策略网络来近似策略函数， 使用策略梯度更新网络参数
    -  ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2Frx6kfw7dc6.png?alt=media&token=22e1d520-3194-42b5-b624-e52034b62b4d)
  - ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FNC2bv9ZwlF.png?alt=media&token=8fb33ced-8383-42fe-8fee-4742d9abadc4)
  - ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FV4DSavxJZ8.png?alt=media&token=408e1eb5-24f9-4fd9-bbf5-a7d20a53f7fb)
  - 计算策略梯度和行动价值函数
    - 方法1-跟环境互动获取长期的收益![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2F8F3Gq0YTEB.png?alt=media&token=d6cd369f-ea76-41b0-84e1-66953c0d4e56)
    - 方法2-构造价值网络来计算action-value
- 价值学习和策略学习结合： actor critic ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FQX2HjRc5gn.png?alt=media&token=18a877c8-f337-4c15-a9f5-7b3f410c8475)

## 三个价值函数？

强化学习中的价值函数主要有两个作用：一是用来判断agent的策略的好坏，一是用来判断当前局势（state）的好坏。强化学习中有3个价值函数: 行动价值函数，最优行动价值函数，状态价值函数

1. 动作价值函数（action-value function）表示在当前状态s下，采取行动a的收益如何, 在采取某个策略$\pi$时$Q_{\pi}\left(s_{t}, a_{t}\right)=E_{\pi}\left(U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right)$

2. 最优动作价值函数(optimal  action-value function)表示采取最优策略$\pi^{\star}$时，给当前的<s, a>打分

   $Q^{*}\left(s_{t}, a_t\right)=\max\limits_{\pi} Q_{\pi}\left(s_{t}, a_t\right)$

3. 状态价值函数(state value function):在后续采取某个策略$\pi$的情况下，当前局势的好坏（是快赢了还是快输了）。用于给当前state打分。如果对s求期望$\mathbb{E}_{S}\left[V_{\pi}(S)\right]$，就是对策略$\pi$打分。$V_{\pi}\left(s_{t}\right)=\operatorname{E_A}\left(Q_{\pi}\left(s_{t}, A\right)\right), A \sim \pi(.\mid s_t)$

   - 如果动作（action）是离散变量：$V_{\pi}\left(s_{t}\right) = \sum\limits_{a}Q_{\pi}(s_t,a) \pi\left(a \mid s_{t}\right)$
   - 如果动作（action）是连续变量： $V_{\pi}\left(s_{t}\right) = \int_{a} Q_{\pi} \left(s_{t}, a\right) \pi\left(a \mid s_{t} \right) da$

4. 总结一下，动作价值函数和最优动作价值函数是来给<state,action>打分，状态价值函数是在给当前<state，>打分，不涉评价动作。

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FDl41z9c-9y.png?alt=media&token=d5e65193-4372-4d43-b8c4-85237c20b61d)

## 控制agent的两种方法-基于策略和基于最优动作价值函数

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FtnK44wspcQ.png?alt=media&token=259a4682-aa14-4b7d-8f55-e88d29cdb319)

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Frf_learning%2FJf9FJZ0nSH.png?alt=media&token=8fb09202-0693-4658-9c45-d2bec3f8642c)



## Q&A

### 价值学习，策略学习，动作裁判学习 这三种方法 存在的必要？

- 既然有了policy Function 或者 Q, 都可以告诉agent怎么操作。那有什么必要去做 actor-critic 算法呢？单独学两个神经网络不就可以了？
- 问题就是不容易学呀！所以才会有Actor-Critic这种复杂的方法。

### 强化学习 适合哪些应用场景？图像分类 适合用吗？为什么 auto-ml适合用？

- 适合 序列决策 问题， 图像分类没必要 用
- 训练一个大的神经网络类似玩一局游戏 耗时很久，每一个迭代 类似 游戏里面的 一小步，可以拿到即时reward，这样就可以使用TD算法来学习 value-function。

### 在线学习 和 强化学习的 区别？

- online learning 不假定模型输出会影响未来输入，只看单步损失
- reinforcement learning 中模型输出会影响未来输入，必须考虑输出的后果

### on-policy 和 off-policy 的 区别？

- 相同点：都是用来更新价值网络的学习算法
- 不同点是：on-policy是按照当前的policy来估计q-value；off-policy是按照最优的policy来估计q-value


### 监督学习 和 强化学习的区别

1. 监督学习要求数据 是独立同分布的，学习过程必须要有老师手把手教，标准答案（label）是什么，
2. 强化学习数据不用iid，没有导师（supervisor），不会被告知正确的action是什么，只有一个奖励信号，并且有延迟的。需要自己去试错，找到具长期reward最大的action
3. 强化学习的本质优势在于，不要求反馈信号相对策略参数可导，甚至连反馈信号跟策略的表达式都不需要有，而这也是真实应用中的情况，通常的解决方案是构建模拟环境，以期低成本地搜集海量样本。



### 强化学习的特点

- 强化学习中随机性的两个来源：action可以是随机的，状态转移可以是随机的
- 试错机制
- 延迟reward
- 时间会产生影响
- agent的行为会影响后续收到的数据（agent的action改变了环境） 
  
  > 类似我们的模型自动更新，担心时间会对模型造成影响。比如模型推荐了商品a，我们也只能收到关于商品a的反馈。
  



## 参考资料 

1. [wangshusen-slide](https://github.com/wangshusen/DRL/blob/master/Slides/1_Basics_1.pdf)
1. [on-policy和off-policy的区别-stackoverflow](https://stats.stackexchange.com/questions/184657/what-is-the-difference-between-off-policy-and-on-policy-learning)
2. [深度强化学习 在实际应用中少吗？难点在那里？](https://www.zhihu.com/question/290530992)
3. [宋一松SYS](https://weibo.com/titaniumviii?refer_flag=0000015010_&from=feed&loc=nickname)
4. [强化学习和在线学习的区别-zhihu](https://www.zhihu.com/question/64526936)
6. [zhoubolei-github-课程资料](https://github.com/zhoubolei/introRL) ｜[slide](https://github.com/zhoubolei/introRL/blob/master/lecture1.pdf)
7. [zhoubolei视频](https://www.bilibili.com/video/BV1LE411G7Xj)
8. [wangshusen-强化学习中文教材](https://github.com/wangshusen/DRL/blob/master/Notes_CN/DRL.pdf)
9. [wangshusen-视频](https://youtu.be/vmkRMvhCW5c)
10. [深度学习课件](https://github.com/wangshusen/DeepLearning)
11. [深度强化学习-notes](https://github.com/wangshusen/DRL)
12. [使用强化学习炒股](https://github.com/wangshub/RL-Stock)
13. [强化学习应用于金融问题的文章](https://zhuanlan.zhihu.com/p/267998242)
14. [Gym-强化学习开发框架](https://gym.openai.com/)：开发强化学习算法的工具箱，有很多第三方环境

