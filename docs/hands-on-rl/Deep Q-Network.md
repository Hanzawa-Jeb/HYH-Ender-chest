# ***Algorithm Description***
- DQN与Q-Learning相比较下，DQN一次性需要更新很多参数，所以会涉及到概率分布的问题，也就会涉及到求期望，所以需要采用***experience replay***技术来保证***uniform distribution*** ，但是Q-Learning一次只会更新一个特定的Q(s, a)，故不需要考虑概率分布与期望的问题。Q-Learning也可以采用***experience replay来保证更高的经验利用率。***
- DQN与普通的Q-learning相比较下，DQN使用了梯度来对参数进行更新而Q-Learning则是使用普通的Robbins-Monro方法。
- 我们在实现的时候会用到`Target Network`技术，也就是在估计`TD-Target`的时候采用一个不一样的目标网络来***选择动作值最大的动作(Selection)*** 与 ***评估选择的动作的价值（Evaluation）***。在一定的迭代次数之后我们会让`Target Network`与`Q-Net`同步。（如果是在DDPG中可以采用软同步，也就是通过加权平均的方式限制同步的速度快慢），***这样可以一定程度上缓解Bootstrapping造成的Overestimation***
- 如果是`Double DQN`，我们会在选择动作的时候采用原Network而在评估时采用`Target Network`，这样可以缓解Overestimation。
# ***Source Code***
```python
import random  
import gym  
import numpy as np  
import collections  
from tqdm import tqdm  
import torch  
import torch.nn.functional as F  
import matplotlib.pyplot as plt  
import rl_utils  
  
class ReplayBuffer:  
    ''' 经验回放池 '''    def __init__(self, capacity):  
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出  
  
    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer  
        self.buffer.append((state, action, reward, next_state, done))  
  
    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size  
        transitions = random.sample(self.buffer, batch_size)  
        state, action, reward, next_state, done = zip(*transitions)  
        return np.array(state), action, reward, np.array(next_state), done  
  
    def size(self):  # 目前buffer中数据的数量  
        return len(self.buffer)  
  
class Qnet(torch.nn.Module):  
    ''' 只有一层隐藏层的Q网络 '''    def __init__(self, state_dim, hidden_dim, action_dim):  
        super(Qnet, self).__init__()  
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数  
        return self.fc2(x)  
  
class DQN:  
    ''' DQN算法 '''    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,  
                 epsilon, target_update, device):  
        self.action_dim = action_dim  
        self.q_net = Qnet(state_dim, hidden_dim,  
                          self.action_dim).to(device)  # Q网络  
        # 目标网络  
        self.target_q_net = Qnet(state_dim, hidden_dim,  
                                 self.action_dim).to(device)  
        # 使用Adam优化器  
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),  
                                          lr=learning_rate)  
        self.gamma = gamma  # 折扣因子  
        self.epsilon = epsilon  # epsilon-贪婪策略  
        self.target_update = target_update  # 目标网络更新频率  
        self.count = 0  # 计数器,记录更新次数  
        self.device = device  
  
    def take_action(self, state):  # epsilon-贪婪策略采取动作  
        if np.random.random() < self.epsilon:  
            action = np.random.randint(self.action_dim)  
        else:  
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)  # 修改为 np.array            action = self.q_net(state).argmax().item()  
        return action  
  
    def update(self, transition_dict):  
        states = torch.tensor(np.array(transition_dict['states']),  
                              dtype=torch.float).to(self.device)  # 修改为 np.array        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(  
            self.device)  
        rewards = torch.tensor(transition_dict['rewards'],  
                               dtype=torch.float).view(-1, 1).to(self.device)  
        next_states = torch.tensor(np.array(transition_dict['next_states']),  
                                   dtype=torch.float).to(self.device)  # 修改为 np.array        dones = torch.tensor(transition_dict['dones'],  
                             dtype=torch.float).view(-1, 1).to(self.device)  
  
        q_values = self.q_net(states).gather(1, actions)  # Q值  
        # 下个状态的最大Q值  
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(  
            -1, 1)  
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones  
                                                                )  # TD误差目标  
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数  
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0  
        dqn_loss.backward()  # 反向传播更新参数  
        self.optimizer.step()  
  
        if self.count % self.target_update == 0:  
            self.target_q_net.load_state_dict(  
                self.q_net.state_dict())  # 更新目标网络  
        self.count += 1  
  
lr = 2e-3  
num_episodes = 500  
hidden_dim = 128  
gamma = 0.98  
epsilon = 0.01  
target_update = 10  
buffer_size = 10000  
minimal_size = 500  
batch_size = 64  
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(  
    "cpu")  
  
env_name = 'CartPole-v1'  # 使用 CartPole-v1env = gym.make(env_name)  
random.seed(0)  
np.random.seed(0)  
env.reset(seed=0)  # 设置环境随机种子  
env.action_space.seed(0)  # 设置动作空间随机种子  
torch.manual_seed(0)  
replay_buffer = ReplayBuffer(buffer_size)  
state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.n  
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,  
            target_update, device)  
  
return_list = []  
for i in range(10):  
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  
        for i_episode in range(int(num_episodes / 10)):  
            episode_return = 0  
            state, _ = env.reset()  # 重置环境并获取初始状态  
            done = False  
            while not done:  
                action = agent.take_action(state)  
                next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作  
                done = terminated or truncated  # 判断是否结束  
                replay_buffer.add(state, action, reward, next_state, done)  
                state = next_state  
                episode_return += reward  
                # 当buffer数据的数量超过一定值后,才进行Q网络训练  
                if replay_buffer.size() > minimal_size:  
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)  
                    transition_dict = {  
                        'states': b_s,  
                        'actions': b_a,  
                        'next_states': b_ns,  
                        'rewards': b_r,  
                        'dones': b_d  
                    }  
                    agent.update(transition_dict)  
            return_list.append(episode_return)  
            if (i_episode + 1) % 10 == 0:  
                pbar.set_postfix({  
                    'episode':  
                    '%d' % (num_episodes / 10 * i + i_episode + 1),  
                    'return':  
                    '%.3f' % np.mean(return_list[-10:])  
                })  
            pbar.update(1)  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episodes')  
plt.ylabel('Returns')  
plt.title('DQN on {}'.format(env_name))  
plt.show()  
  
mv_return = rl_utils.moving_average(return_list, 9)  
plt.plot(episodes_list, mv_return)  
plt.xlabel('Episodes')  
plt.ylabel('Returns')  
plt.title('DQN on {}'.format(env_name))  
plt.show()
```
# ***Syntax Reminder***
- `collections`是Python中的一个标准库，内部有许多现成的数据结构。比如说`deque`就是一个***双端队列*** `(Double-Ended Queue)`，可以从队列的两端进行数据的加入或者删除，参数中可以指定最大的长度`q = collections.deque(maxlen = a)`，可以通过`q.appendleft(1)`或者`q.append(1)`分别在队列q的左侧和右侧添加元素
- `random.sample(list, k)`实现了无放回的抽样，但是实际上并没有从序列中取出元素，这里注意k必须要小于等于sample的长度，要不然会不可避免地出现重复的取样
- `zip(*iter_var)`可以将可迭代变量`iter_var`解包给前面的各个变量，如果没有`*`的话则是打包，而不是解包。
- 可以使用`super().__init__()`来调用父类的`Initialization Method` 
- `self.fc1 = torch.nn.Linear(state_dim, hidden_dim)`的作用是创建了一个全连接层，***输入维度为state_dim, 输出维度为hidden_dim***
- `forward(self, x)`规定了数据是如何通过网络的各层的，参数x是输入的数据（在这里是***state***）
- `x = F.relu(self.fc1(x))`的作用是将x通过第一个全连接层
- `state = torch.tensor([state], dtype = torch.float).to(self.device)` 的作用是将全部state先包装为一个torch.float类型的torch张量，再将这一张量移动到指定的设备上
- `self.q_net(state)`的作用是将state输入到网络`self.q_net`中，得到每一个动作的Q值
- `.argmax()`返回Q值最大的索引
- `.item()`将张量中的值转换为Python标量`(int)`