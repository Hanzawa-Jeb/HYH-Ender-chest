# 马尔科夫决策过程(Markov Decision Process)
## ***Algorithm Description***
- MDP与MRP的最大区别就在于MRP中并没有 ***“决策”*** 这一概念，MRP中是 **(S, P, r, γ)** 而MDP中是 **（S, A, P, r，γ）**,在MRP中我们只能评估这个环境转移的好坏而不能做出决策，但是在MDP中我们可以改变相对应的策略。
- 同时应该注意MDP中的state transition matrix变成了***P(S|s, a)*** 也就是在s状态下采取action a 到达state S的概率。
- MDP中的***奖励是同时基于s与a的***，这两个都需要加入到考虑当中。
- 有关于policy: 多用π表示策略 如果是***deterministic policy***，那么只有一个动作的概率为1，如果是***stochastic policy***，那么每一个状态下都有一个有关于动作的概率分布
- 给定了一个MDP与确定的policy π，那么MDP就会变成一个MRP， 我们就可以开始评估这一个策略的优劣性
- 状态价值函数与动作价值函数都可以通过***Bellman Expectation Equation得到***

