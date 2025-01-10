- Source Code:
```python
import numpy as np  
import matplotlib.pyplot as plt  
  
class BernoulliBandit:  
    def __init__(self, K): #K is the number of bandit arms  
        self.probs = np.random.uniform(size = K)    #this is a numpy array  
  
        self.best_idx = np.argmax(self.probs)   #this is the index of the highest prob  
        self.best_prob = self.probs[self.best_idx]  
        self.K = K  
  
    def step(self, k):  #the k here is the k that the user inputs  
        if np.random.rand() > self.probs[k]:  
            return 0  
        else:  
            return 1  
  
np.random.seed(1)  
K = 10  
bandit_10_arm = BernoulliBandit(K)  #Instantiation of the class BernoulliBandit  
print("现在已经随机生成了一个%d臂老虎机" % bandit_10_arm.K)  
print("获奖概率最大的拉杆是%d, 其获奖概率为%.4f" % (int(bandit_10_arm.best_idx), float(bandit_10_arm.best_prob)))  
  
class Solver:  
    def __init__(self, bandit):  
        self.bandit = bandit  
        self.counts = np.zeros(self.bandit.K)   #trial times for every single lever  
        self.regret = 0.    #cumulative regret of the current step  
        self.actions = []   #log the history steps of the actions taken  
        self.regrets = []   #log the cumulative regret of every step in th trajectory  
  
    def update_regret(self, k):     #calculate the current regret and save, k is the bandit arm we chose  
        self.regret += self.bandit.best_prob - self.bandit.probs[k]     #this is in a stochastic form rather than discrete form  
        self.regrets.append(self.regret)  
  
    def run_one_step(self):  
        raise NotImplementedError   #means that the function here is not yet implemented by the coder  
  
    def run(self, num_steps):  
        for _ in range(num_steps):  
            k = self.run_one_step()  
            self.counts[k] += 1  
            self.actions.append(k)  
            self.update_regret(k)  
  
class EpsilonGreedy(Solver):    #inherit the class Solver  
    def __init__(self, bandit, epsilon = 0.01, init_prob = 1.0):    #the epsilon here is the one in epsilon-greedy algorithm.  
        super(EpsilonGreedy, self).__init__(bandit)  
        self.epsilon = epsilon  
        self.estimates = np.array([init_prob] * self.bandit.K)  
  
    def run_one_step(self):     #super class method override  
        if np.random.random() < self.epsilon:       #exploration  
            k = np.random.randint(0, self.bandit.K)     #randomly choose any k  
        else:  
            k = np.argmax(self.estimates)   #exploitation  
        r = self.bandit.step(k)     #get the reward of the current step  
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])    #update the estimate of bandit arm k  
        return k  
  
def plot_results(solvers, solver_names):    #uses matplotlib to show the graph  
    for idx, solver in enumerate(solvers):  
        time_list = range(len(solver.regrets))  
        plt.plot(time_list, solver.regrets, label = solver_names[idx])  
    plt.xlabel('Time Steps')  
    plt.ylabel('Cumulative Regrets')  
    plt.title('%d-armed bandit' % solvers[0].bandit.K)  
    plt.legend()  
    plt.show()  
  
np.random.seed(1)  
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon = 0.01)  
epsilon_greedy_solver.run(5000)  
print('epsilon-greedy算法的cumulative regret为：', epsilon_greedy_solver.regret)  
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"]
```
- `numpy.random.uniform(lower, higher, size = k)`意思是在lower与higher之间生成k个随机数，如果lower与higher不写那就默认是在0-1之间，返回的类型是numpy数组
- `np.random.rand()`用于随机生成0-1之间的随机浮点数，如果括号里面有数字则可以指定返回的数据规模与数组格式
- `np.random.seed(n)`用于生成唯一确定的随机值序列，在代码中的开头确定一次，后面调用的全部`np.random`相关函数都会依据完全相同的随机值序列进行赋值，例如
```python
np.random.seed(1)
a = np.random.rand(3)
b = np.random.rand(5)
```
- 那么a与b的前三项应该是完全相同的
- `np.zeros(5)`生成全为0的数组，类型为`np.ndarray`
- 对于一个完整的trajectory, 我们可以计算第t步的累计懊悔(cumulative regret)，每一次的懊悔都是**当前拉杆动作**与***最优拉杆***的期望奖励之差
- `raise NotImplementedError`的意思是在这里显示“未实现错误”，意思是提醒使用的人员这里的功能还暂时没有被实现
- `Python Class Inheritance`只用在子类的class定义过程中在小括号里面写上父类的名称就可以了
- 如果在子类中定义了一个与父类中同名的method, 那么会因为`method override`而使在子类中子类的方法覆盖了父类的方法
- `np.random.randint(low, high)`意思是生成`[low, high)`之间的随机整数
- `python for`语句中的`enumerate` 会将index赋值给第一个迭代变量，将value赋值给第二个迭代变量。注意这样赋值的关系与迭代变量的名字并没有关系
```python
for a, b in enumerate(iteratable instance):
	print("the index is %d, the value is %d\n" % (a, b))
```
- 如果对npdarray取argmax但是有多个最大值，那么将会返回***第一个最大值的索引***
- 可以在函数的定义或者类的定义中写下某些内部参数的默认值，但是如果我在后面实例化的调用过程中具体写出了这一个值，那么就会进行覆盖。
- 在这里的代码中，***cumulative regret***仅用于评估算法的优劣性而***不参与决策***！在这里决策是通过self.estimate进行评估的
- 解决MAB问题中的UCB(***Upper Confidence Bound***)算法意思就是优先访问之前访问的较少的臂（也就是不确定性较大的臂）**在一个arm被访问次数较少时，UCB值会更大，算法倾向于访问这一支arm**
- 解决MAB问题中的***Thompson Sampling***方法是先假设一个概率分布，并且通过一次一次的遍历来不断更新这一个概率分布的特性。随后我们会***在beta分布当中随机抽样***，在***采样次数较少的时候***，beta分布的图形会更宽，也就是说我们有更高的概率取到一个较高的数值，这个采样值就是我们在这一步***所认为的先验reward = 1***的概率，然后我们再取采样值最高的拉杆即可
- ***为什么Thompson Sampling***是有效的？因为在一个遍历次数较少的臂上，采样采到大值的概率更大，也就有更大的概率取这一个值
- 有关***先验概率***，这就是我们在观察到具体数据之前对一个量的初始信念或者假设，可以在后面的过程中对参数进行更新