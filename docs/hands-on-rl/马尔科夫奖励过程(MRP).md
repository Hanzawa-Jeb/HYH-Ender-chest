# 马尔科夫奖励过程(Markov Reward Process)
## ***Algorithm Description***
- MRP中没有action的概念，只用state transition matrix来表示不同状态之间的转换关系
- 一个MRP由***S, P, r，γ构成（state space, state transition matrix, reward function, discount factor）***
- MRP中的reward只要***到达了某一个状态就可以获得***，而与初始状态与动作并没有关系
- chain在这个算法之中代表了访问状态的序列，在这里通过***反向迭代器***来实现reward的逆向运算
- state value就是从这一个状态出发而得到的***期望回报***，注意这里求的是期望回报哦！也就是说从这一个state出发的***所有可能的episode的return***的概率加权平均
- ***贝尔曼方程(Bellman Equation)*** 就是一种使用下一个状态的state value来表示这一个状态的state value的方程，这一种方程可以通过***closed-form solution 也就是数学中的解析解***来解决，例如解一个线性矩阵方程，也可以使用迭代方法来解决问题，例如我们后面将会提到的***Dynamic Programming(Policy Iteration/Value Iteration), Monte-Carlo Method, Temporal-Difference Method(Robbins Monro Algorithm)*** 来解决。
## ***Source Code***
```python
import numpy as np  
np.random.seed(0)  
# P is the state transition matrix  
P = [  
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],  
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],  
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]  
]  
P = np.array(P)     #transfer the python list into a numpy array  
  
rewards = [-1, -2, -2, 10, 1, 0]    #the rewards of MRP are only determined by the state achieved.  
gamma = 0.5     #the discount factor gamma  
  
def compute_return(start, chain, discount_f):   #this is the function that computes the return of a certain episode  
    G = 0  
    for i in reversed(range(start, len(chain))):      #construct a reversed iterator  
        G = discount_f * G + rewards[chain[i] - 1]  
    return G  
  
chain = [1, 2, 3, 6]    #chain stands for the state visiting history  
start_index = 0     #stands for the position(in chain) of the state from where we calculate the return from  
G = compute_return(start_index, chain, gamma)  
print("从第1个state计算而得的return是%.2f" % G)
```
## ***Syntax Reminder***
- `numpy.array`底层是使用C语言实现的， 所以与原生python的列表相比，具有更高的运算性能，且占用的内存空间是连续的，而且内部存储的数据类型也必须是相同的。如果有一个list类型的变量`lst`，那么如果要将其的类型转换为numpy array就可以`array = np.array(lst)`
- `reversed`用于返回一个反方向的迭代器，但是reversed并不会改变原来的序列，只会生成一个全新的迭代器而已
```python
# 使用 reversed()
my_list = [1, 2, 3, 4]
reversed_iterator = reversed(my_list)  # 返回迭代器
reversed_list = list(reversed_iterator)  # 转换为列表

# 使用切片
reversed_list_slice = my_list[::-1]  # 直接生成新列表
```
