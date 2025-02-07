# Temporal Difference Method
## ***Algorithm Description***
- ***比较`Sarsa`与`Q-Learning`后者会更加激进，因为其采用的是最优的动作价值***
- 注意这里的`CliffWalkingEnv`与前面的有所不同，不需要提供奖励函数与`State Transition Function`，只需要`step()`来提供Agent交互的接口。
- 采用`SoftPolicy:ε-greedy`算法来实现充分的`Exploratory Demand`。
- 从不同的`initial state`出发进行遍历以确保充分的探索性。
- 可以不用显式地维护策略，只要在生成episode的时候根据`ε-greedy`的策略就可以了，而不用维护一个特定的`policy table`
- `Q-Learning`也可以是On-policy的，只要轨迹数据是实时采集生成的。但是如果`episode`中的`s, a, r, s'`都是通过`πb`事先存储在`buffer zone`的话那么就属于`off-policy`的版本。
## ***Source Code***
- ***Sarsa Method***
```python
import matplotlib.pyplot as plt  
import numpy as np  
from tqdm import tqdm  
  
class CliffWalkingEnv:  
    def __init__(self, ncol, nrow):  
        self.nrow = nrow  
        self.ncol = ncol  
        self.x = 0      # x coordinate of the agent  
        self.y = self.nrow - 1      # y coordinate of the agent  
  
    def step(self, action):     # update the location of the agent  
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # up, down, left, right  
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))  
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))  
        next_state = self.y * self.ncol + self.x  
        reward = -1  
        done = False  
        if self.y == self.nrow - 1 and self.x > 0:  
            done = True  
            if self.x != self.ncol - 1:  
                reward = -100   #fall  
        return next_state, reward, done  
  
    def reset(self):  
        self.x = 0  
        self.y = self.nrow - 1  
        return self.y * self.ncol + self.x  
  
class Sarsa:  
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action = 4):   # n_action stores the number of actions  
        self.Q_table = np.zeros([ncol * nrow, n_action])  
        self.n_action = n_action  
        self.alpha = alpha      #alpha here is the learning rate (coefficient of the TD error)  
        self.gamma = gamma      #discount factor  
        self.epsilon = epsilon  #parameter of the ε-greedy algorithm  
  
    def take_action(self, state):   # choose the next_step action with ε-greedy algorithm  
        if np.random.rand() < self.epsilon:  
            action = np.random.randint(self.n_action)  
        else:  
            action = np.argmax(self.Q_table[state])  
        return action  
  
    def best_action(self, state):  
        Q_max = np.max(self.Q_table[state])  
        a = [0 for _ in range(self.n_action)]  
        for i in range(self.n_action):  
            if self.Q_table[state][i] == Q_max:  
                a[i] = 1  
        return a  
  
    def update(self, s0, a0, r, s1, a1):  
        td_error = r + self.gamma * self.Q_table[s1][a1] - self.Q_table[s0][a0]  
        self.Q_table[s0][a0] += self.alpha * td_error  
  
ncol = 12  
nrow = 4  
env = CliffWalkingEnv(ncol, nrow)  
np.random.seed(0)  
epsilon = 0.1  
alpha = 0.1  
gamma = 0.9  
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)  
num_episodes = 500  
  
return_list = []    #log the return of every single trajectory  
for i in range(10):  
    with tqdm(total = int(num_episodes / 10), desc = 'Iteration %d' % i) as pbar:  
        for i_episode in range(int(num_episodes / 10)):  
            episode_return = 0  
            state = env.reset()  
            action = agent.take_action(state)  
            done = False  
            while not done:  
                next_state, reward, done = env.step(action)  
                next_action = agent.take_action(next_state)  
                episode_return += reward  
                agent.update(state, action, reward, next_state, next_action)  
                state = next_state  
                action = next_action  
            return_list.append(episode_return)  
            if (i_episode + 1) % 10 == 0:  
                pbar.set_postfix({'episode' : '%d' % (num_episodes / 10 * i + i_episode + 1), 'return' : '%.3f' % np.mean(return_list[-10:])})  
            pbar.update(1)  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episodes')  
plt.ylabel('Returns')  
plt.title('Sarsa on {}' . format('CliffWalking'))  
plt.show()
```
- ***n-step-Sarsa Method***
```python
import matplotlib.pyplot as plt  
import numpy as np  
from tqdm import tqdm  
  
class CliffWalkingEnv:  
    def __init__(self, ncol, nrow):  
        self.nrow = nrow  
        self.ncol = ncol  
        self.x = 0      # x coordinate of the agent  
        self.y = self.nrow - 1      # y coordinate of the agent  
  
    def step(self, action):     # update the location of the agent  
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # up, down, left, right  
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))  
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))  
        next_state = self.y * self.ncol + self.x  
        reward = -1  
        done = False  
        if self.y == self.nrow - 1 and self.x > 0:  
            done = True  
            if self.x != self.ncol - 1:  
                reward = -100   #fall  
        return next_state, reward, done  
  
    def reset(self):  
        self.x = 0  
        self.y = self.nrow - 1  
        return self.y * self.ncol + self.x  
  
class nstep_Sarsa:  
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action = 4): #n is the step number of n-step-sarsa  
        self.Q_table = np.zeros((nrow * ncol, n_action))                    #Q_table[s] will visit all the action of state s  
        self.n_action = n_action  
        self.epsilon = epsilon  
        self.alpha = alpha  
        self.gamma = gamma  
        self.n = n  
        self.state_list = []    #log the previous state  
        self.action_list = []   #log the previous action  
        self.reward_list = []   #log the previous reward  
  
    def take_action(self, state):  
        if np.random.rand() < self.epsilon:  
            action = np.random.randint(self.n_action)  
        else:  
            action = np.argmax(self.Q_table[state])  
        return action  
  
    def best_action(self, state):  
        Q_max = np.max(self.Q_table[state])  
        a = [0 for _ in range(self.n_action)]  
        for i in range(self.n_action):  
            if self.Q_table[state][i] == Q_max:  
                a[i] = 1  
        return a  
  
    def update(self, s0, a0, r, s1, a1, done):  
        self.state_list.append(s0)  #this is a queue, always keeping the length of n  
        self.action_list.append(a0)  
        self.reward_list.append(r)  
        if len(self.state_list) == self.n:  
            G = self.Q_table[s1][a1]  
            for i in reversed(range(self.n)):  
                G = self.gamma * G + self.reward_list[i]  
                if done and i > 0:  
                    s = self.state_list[i]  
                    a = self.action_list[i]  
                    self.Q_table[s][a] += self.alpha * (G - self.Q_table[s][a])  
            s = self.state_list.pop(0)  
            a = self.action_list.pop(0)  
            self.reward_list.pop(0)  
            self.Q_table[s][a] += self.alpha * (G - self.Q_table[s][a])  
        if done:  
            self.state_list = []  
            self.action_list = []  
            self.reward_list = []  
  
np.random.seed(0)  
n_step = 5  
alpha = 0.1  
epsilon = 0.1  
gamma = 0.9  
ncol = 12  
nrow = 4  
env = CliffWalkingEnv(ncol, nrow)  
agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)  
num_episodes = 500  
  
return_list = []  
for i in range(10):  
    with tqdm(total = int(num_episodes / 10), desc = 'Iteration %d' % i) as pbar:  
        for i_episode in range(int(num_episodes / 10)):  
            episode_return = 0  
            state = env.reset()  
            action = agent.take_action(state)  
            done = False  
            while not done:  
                next_state, reward, done = env.step(action)  
                next_action = agent.take_action(next_state)  
                episode_return += reward  
                agent.update(state, action, reward, next_state, next_action, done)  
                state = next_state  
                action = next_action  
            return_list.append(episode_return)  
            if (i_episode + 1) % 10 == 0:  
                pbar.set_description(desc = 'Episode %d' % i_episode)  
            pbar.update(1)  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episodes')  
plt.ylabel('Return')  
plt.title('5step-Sarsa')  
plt.show()
```
- ***Q-Learning Method***
```python
import numpy as np  
from tqdm import tqdm  
import matplotlib.pyplot as plt  
  
class CliffWalkingEnv:  
    def __init__(self, ncol, nrow):  
        self.nrow = nrow  
        self.ncol = ncol  
        self.x = 0                                          # initial x coordinate of the agent  
        self.y = self.nrow - 1                              # initial y coordinate of the agent  
  
    def step(self, action):                                 # update the location of the agent  
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]         # up, down, left, right  
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))  
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))  
        next_state = self.y * self.ncol + self.x  
        reward = -1  
        done = False  
        if self.y == self.nrow - 1 and self.x > 0:  
            done = True  
            if self.x != self.ncol - 1:  
                reward = -100   #fall  
        return next_state, reward, done  
  
    def reset(self):        #going back to the starting point  
        self.x = 0  
        self.y = self.nrow - 1  
        return self.y * self.ncol + self.x  
  
class QLearning:  
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action = 4):  
        self.Q_table = np.zeros((nrow * ncol, n_action))  
        self.epsilon = epsilon  
        self.alpha = alpha  
        self.gamma = gamma  
        self.n_action = n_action  
  
    def take_action(self, state):  
        if np.random.rand() < self.epsilon:  
            action = np.random.randint(self.n_action)  
        else:  
            action = np.argmax(self.Q_table[state])  
        return action  
  
    def best_action(self, state):       #only for printing  
        Q_max = np.max(self.Q_table[state])  
        a = [0 for _ in range(self.n_action)]  
        for i in range(self.n_action):  
            if self.Q_table[state][i] == Q_max: #useful in the final policy generation stage  
                a[i] = 1  
        return a  
  
    def update(self, s0, a0, r, s1):    #s, a, r, s are for parameters of Q-Learning  
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0][a0]  
        self.Q_table[s0][a0] += self.alpha * td_error  
  
def print_agent(agent, env, action_meaning, disaster=[], end=[]):  
    for i in range(env.nrow):  
        for j in range(env.ncol):  
            if (i * env.ncol + j) in disaster:  
                print('****', end=' ')  
            elif (i * env.ncol + j) in end:  
                print('EEEE', end=' ')  
            else:  
                a = agent.best_action(i * env.ncol + j)  
                pi_str = ''  
                for k in range(len(action_meaning)):  
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'  
                print(pi_str, end=' ')  
        print()  
  
np.random.seed(0)  
epsilon = 0.1  
alpha = 0.1  
gamma = 0.9  
ncol = 12  
nrow = 4  
env = CliffWalkingEnv(ncol, nrow)  
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)  
num_episodes = 500  
  
return_list = []    #log the return of every single episode  
for i in range(10):  
    with tqdm(total = int(num_episodes / 10), desc = 'Iteration %d' % i) as pbar:  
        for i_episode in range(num_episodes):  
            episode_return = 0  
            state = env.reset()  
            done = False  
            while not done:  
                action = agent.take_action(state)  
                next_state, reward, done = env.step(action)  
                episode_return += reward        #discount factor not considered here  
                agent.update(state, action, reward, next_state)  
                state = next_state  
            return_list.append(episode_return)  
            if (i_episode + 1) % 10 == 0:  
                pbar.set_postfix({'episode': (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})  
            pbar.update(1)  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episodes')  
plt.ylabel('Return')  
plt.title('Q-Learning on {}' . format("Cliff Walking"))  
plt.show()  
  
action_meaning = ['^', 'v', '<', '>']  
print("the policy generated by Q-learning Algorithm is :")  
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
```
## ***Syntax Reminder***
- state与next_state都是由环境中的`step()`来决定的
- 在`numpy`中，可以通过`[a][b]或者[a, b]`的方式来访问同一个`numpy array`中的同一个坐标，但是普通的Python多维数组坐标是不能这么访问的
- `with`语句在`Python`中用于处理需要显式释放的资源，例如文件处理，网络连接等等
- `np.zeros((a, b))`会生成一个`a * b`规模的二维数组，同理也可以通过小括号中的其他参数来指定零矩阵的大小规模
- `Python`的`Constructor`中如果没有将变量显式地赋值给示例的话，在`__init__()`方法之外并不能直接访问到这些变量，例如
```python
class exp:
	def __init__(self, a, b):
		self.a = a
```
- 那么在上述代码之中，我们在实例化了一个类之后只能访问到a而不能访问到b
- 对于一个`numpy array a`而言， `a.max()`与`max(a)`是不同的。如果是多维数组，那么`a.max()`会返回整个数组的最大值，但是如果a是一个多维数组，`Python`会依据字典序返回最大的列表。所以他们有***必然返回一个值***与***在多维数组情况下返回降一维的列表***的区别。