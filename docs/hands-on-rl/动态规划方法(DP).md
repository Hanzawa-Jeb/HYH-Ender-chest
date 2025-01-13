# 动态规划方法(Dynamic Programming Method)
## ***Algorithm Description***
- 共有两种方法，策略迭代与价值迭代，这两种方法都是***Model-Based***的。
- 策略迭代中有`Policy Evaluation & Policy Improvement`，策略评估通过动态规划方法来计算每一个`state-value`，后面再进行策略提升。
- `Value Iteration`直接使用动态规划算法来实现最终的最优状态价值函数。
- 请注意`Policy Iteration`中是先评估policy(需要很多很多步迭代)，再对策略进行相对应的改进，但是`Value Iteration`中是先对Policy进行优化，再***仅仅对value进行单次迭代***以更新state value的值。
## ***Source Code***
- ***Policy Iteration Case:***
```python
import copy  
  
class CliffWalkingEnv:  
    def __init__(self, ncol = 12, nrow = 4):  
        self.ncol = ncol    #the column number of the grid world  
        self.nrow = nrow    #the row number of the grid world  
        self.P = self.createP()     #P[state][action] will be the state transition matrix containing prob, reward and next state, the current state  
  
    def createP(self):  
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]      #initialize the P table  
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]     #represents the four action  
        for i in range(self.nrow):  
            for j in range(self.ncol):  
                for a in range(4):  
                    if i == self.nrow - 1 and j > 0:  
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]     #indicates the prob of every action taken at state s  
                        continue  
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))  
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))  
                    next_state = next_y * self.ncol + next_x  
                    reward = -1  
                    done = False  
                    if next_y == self.nrow - 1 and next_x > 0:  
                        done = True     #if the ending state is reached, or we have fell down, then the done is True  
                        if next_x != self.ncol - 1:  
                            reward = -100  
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]  
        return P  
  
class PolicyIteration:  
    def __init__(self, env, theta, gamma):      #theta is the threshold of ending iteration  
        self.env = env  
        self.v = [0] * self.env.ncol * self.env.nrow    #v is the initial value of the state  
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(self.env.ncol * self.env.nrow)]      #initialize the policy  
        self.theta = theta  
        self.gamma = gamma  
  
    def policy_evaluation(self):  
        cnt = 1  
        while 1:  
            max_diff = 0  
            new_v = [0] * self.env.nrow * self.env.ncol  
            for s in range(self.env.nrow * self.env.ncol):  
                qsa_list = []       #calculates the Q(s, a) of all actions from state s  
                for a in range(4):  
                    qsa = 0  
                    for res in self.env.P[s][a]:  
                        p, next_state, r, done = res  
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))  
                    qsa_list.append(self.pi[s][a] * qsa)  
                new_v[s] = sum(qsa_list)  
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))  
            self.v = new_v  
            if max_diff < self.theta:  
                break  
            cnt += 1  
        print("%d-th iteration satisfies the threshold requirements" % cnt)  
  
    def policy_improvement(self):   #the parameter should be a PolicyIteration class  
        for s in range(self.env.nrow * self.env.ncol):  
            qsa_list = []       #stores the state-action value of each state-action pair  
            for a in range(4):  
                qsa = 0  
                for res in self.env.P[s][a]:  
                    p, next_state, r, done = res  
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))  
                qsa_list.append(qsa)  
            maxq = max(qsa_list)  
            cntq = qsa_list.count(maxq)     #get the number of maxq  
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]  
        print("Policy Improvement finished.")  
        return self.pi  
  
    def policy_iteration(self):  
        while 1:  
            self.policy_evaluation()  
            old_pi = copy.deepcopy(self.pi)  
            new_pi = self.policy_improvement()  
            if old_pi == new_pi:  
                break  
  
def print_agent(agent, action_meaning, disaster = [], end = []):  
    print("State Value:")  
    for i in range(agent.env.nrow):  
        for j in range(agent.env.ncol):  
            print("%6.6s" % ('%.3f' % agent.v[i * agent.env.ncol + j]), end = ' ')  #outputs the state value  
        print()  
  
    print("Policy:")  
    for i in range(agent.env.nrow):  
        for j in range(agent.env.ncol):  
            if (i * agent.env.ncol + j) in disaster:  
                print("****", end = ' ')  
            elif (i * agent.env.ncol + j) in end:  
                print("EEEE", end = ' ')  
            else:  
                a = agent.pi[i * agent.env.ncol + j]  
                pi_str = ''  
                for k in range(len(action_meaning)):  
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'  
                print(pi_str)  
        print()  
  
env = CliffWalkingEnv()  
action_meaning = ['^', 'v', '<', '>']  
theta = 0.001  
gamma = 0.9  
agent = PolicyIteration(env, theta, gamma)  
agent.policy_iteration()  
print_agent(agent, action_meaning, list(range(37, 47)), [47])
```
## ***Syntax Reminder***
- python中类的`__init__(self, var1, var2)`语句中最好要在初始化中与类变量中采用相同的名字，虽然不同其实也不会影响运行的结果，但是为了代码的可读性我们一般都会这么做。类的
```python
class example:
	def __init__(self, var1, var2):
		self.var1 = var1
		self.var2 = var2    #一般约定俗成我们都会使用相同的名字进行类的初始化工作 

class example2:
	def __init__(self, a, b):
		self.c = a
		self.d = b    #虽然这样也可以运行，但是我们一般不会这么做
```
- `[[] for j in range(4)]`在python中将会生成一个包含四个空列表的列表`[[],[],[],[]]`
- 注意这里P的初始化过程中***P是按照一行一行进行存储的***，所以在访问索引的时候要`i * ncol`来访问地址
- `print(end = str)`中，如果不写，那么end默认等于`'\n'`，如果写了那么就会默认把这个打在前面的字符串的末尾