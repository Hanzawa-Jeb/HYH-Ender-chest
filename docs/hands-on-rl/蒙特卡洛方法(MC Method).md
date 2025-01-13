# 蒙特卡洛方法(Monte-Carlo Method)
## ***Algorithm Description***
- 运用概率统计进行数值预测
- 通过采集多个从***相同的state/state action pair出发的trajectory的return*** 来预计相对应的state value或者action value
- 对每一个action进行***访问次数统计***与***访问return之和***的统计，计算相对应的state的state value。在每一个state被访问的次数趋近于正无穷的时候预测到的state value会趋近于真正的state value（当然也可以采用***Incremental Method***来对state value进行增量式均值估计）
- 评估的是action value，可以采用exploring starts来保证全部pair均被访问或者使用`soft policy(e.g. ε-greedy)`来使用单一episode进行访问
## ***Source Code***
- 这段代码仅仅展示随机采样的部分
```python
import numpy as np  
  
def sample(MDP, Pi, timestep_max, number):  #sampling function, calculating the return with given timestep.  
    #number is the sampling time, and timestep_max is the max step of every single episode    S, A, P, R, gamma = MDP     #MDP is an iterable variable storing the essence of MDP  
    episodes = []  
    for _ in range(number):  
        episode = []  
        timestep = 0  
        s = S[np.random.randint(4)]     #randomly choose a state to be the state starting from  
        while s != "s5" and timestep < timestep_max:    #s5 is the ending state and timestep_max is the max timestep  
            timestep += 1  
            rand, temp = np.random.rand(), 0  
            for a_opt in A:     #A is the action space  
                temp += Pi.get(join(s, a_opt))  
                if temp > rand:  
                    a = a_opt  
                    r = R.get(join(s, a), 0)  
                    break  
            rand, temp = np.random.rand(), 0  
            #getting s_next with random process  
            for s_opt in S:  
                temp += P.get(join(join(s, a), s_opt), 0)  
                if temp > rand:  
                    s_next = s_opt  
                    break  
            episode.append((s, a, r, s_next))  
            s = s_next  
        episodes.append(episode)  
    return episodes  
  
episodes = sample(MDP, Pi_1, 20, 5)
```
## ***Syntax Reminder***
- Python中元组可以存储一系列类型不同的变量，而且可以使用与普通列表访问相同的方法进行访问，但是元组内的存储内容是不可改变的
- Python中的解包操作`Unpacking`就是将一个***可迭代变量中的值分别赋值给相应的变量***，注意可迭代变量中的元素个数与赋值变量的***个数必须相同*** 。
```python
f = (1, 2, 3, 4, 5)
a, b, c, d, e = f    #这样的操作是可行的，f中的值会按顺序赋值给前面的变量
```
- `np.random.randint()`后面如果有两个数字，那么是前闭后开，如果后面只有一个数字，那么就是以0为下界，以这个数为上界进行左闭右开的随机数采样
- `np.random.rand()`可生成`[0, 1)`之间的任意数
- `dict.get(key, default = prompt)`就是从字典中取出键值为key的数据，但是与直接`[]`取值不同的是如果这里没有找到相对应的键那么则会返回默认的返回值prompt
- `str.join(iterable variable)`的作用是将str插入到迭代变量中每两项的中间再以字符串形式返回
- `map(function, iterable)`函数的作用是将可迭代变量中的每一项均使用function进行迭代，例程如下:
```python
# 定义一个函数，计算平方
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
result = map(square, numbers)  # 对每个元素应用 square 函数
print(list(result))  # 输出: [1, 4, 9, 16, 25]
```