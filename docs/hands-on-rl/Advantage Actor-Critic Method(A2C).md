# Advantage Actor-Critic
## ***Algorithm Description***
- AC方法与REINFORCE算法均***使用神经网络来维护Policy Net***，区别在于如何获得Action Value
- 最经典的`Actor-Critic`方法（QAC）中维护价值网络使用的是***基于Temporal-Difference的梯度下降方法（最小化TD-Error）***，而普通的Sarsa则是使用`Robbins-Monro`算法来对Action Value进行更新。这是因为Sarsa针对的是有噪音的表格环境，而QAC解决的是更加复杂与高维的状态空间与动作空间。
- A2C与QAC的区别在与A2C采用Advantage Action Value来评估动作的价值，在对策略网络对参数求梯度时，A2C采用的是Advantage Action Value ,这样可以在不改变梯度期望（可用数学证明）的情况下有效地降低方差。
- 在PolicyGradient方法中，虽然一般的Objective Function就是return的加权平均，但是我们在实现的过程中一般都会使用$E(log(\pi_\theta(a|s_t))A(a|s_t))$，***这种操作的目的是尽可能地方便计算并防止一系列问题。***
- ***注意，这里的loss_function并不是传统机器学习中衡量两个量之间距离的函数，而是一个用于最小化的函数。*** 所以如果我们想要最大化一个值，那么我们就要在`loss.backward()`的时候在loss前面加上一个负号。
## ***Source Code***
```python
import gym  
import torch  
import torch.nn.functional as F  
import numpy as np  
import matplotlib.pyplot as plt  
  
import rl_utils  
  
class PolicyNet(torch.nn.Module):  
    def __init__(self, state_dim, hidden_dim, action_dim):  
        super().__init__()  
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        return F.softmax(self.fc2(x), dim=1)  
  
class ValueNet(torch.nn.Module):  
    def __init__(self, state_dim, hidden_dim):  
        super().__init__()  
        self.net = torch.nn.Sequential(  
            torch.nn.Linear(state_dim, hidden_dim), torch.nn.ReLU(),  
            torch.nn.Linear(hidden_dim, 1)  
        )  
  
    def forward(self, x):  
        return self.net(x)  
  
class ActorCritic:  
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):  
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)  
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)  
        self.gamma = gamma  
        self.device = device  
  
    def take_action(self, state):  
        state = torch.tensor([state], dtype=torch.float).to(self.device)  
        probs = self.actor(state)  
        action_dist = torch.distributions.Categorical(probs)  
        action = action_dist.sample()  
        return action.item()  
  
    def update(self, transition_dict):  
        states = torch.tensor(transition_dict['states'],  
                              dtype=torch.float).to(self.device)  
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(  
            self.device)  
        rewards = torch.tensor(transition_dict['rewards'],  
                               dtype=torch.float).view(-1, 1).to(self.device)  
        next_states = torch.tensor(transition_dict['next_states'],  
                                   dtype=torch.float).to(self.device)  
        dones = torch.tensor(transition_dict['dones'],  
                             dtype=torch.float).view(-1, 1).to(self.device)  
  
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  
        td_delta = td_target - self.critic(states)  
        log_probs = torch.log(self.actor(states).gather(1, actions))  
        actor_loss = torch.mean(-log_probs * td_delta.detach())  
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  
  
        self.actor_optimizer.zero_grad()  
        actor_loss.backward()  
        self.actor_optimizer.step()  
  
        self.critic_optimizer.zero_grad()  
        critic_loss.backward()  
        self.critic_optimizer.step()  
  
actor_lr = 1e-3  
critic_lr = 1e-2  
num_episodes = 1000  
hidden_dim = 128  
gamma = 0.98  
device = torch.device('cuda')  
  
env_name = 'CartPole-v0'  
env = gym.make(env_name)  
env.reset(seed = 0)  
torch.manual_seed(0)  
state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.n  
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)  #Agent Initialization  
  
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)                      #Training  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episode')  
plt.ylabel('Return')  
plt.title('A2C on {}'.format(env_name))  
plt.show()  
  
mv_return = rl_utils.moving_average(return_list, 9)  
plt.plot(episodes_list, mv_return)  
plt.xlabel('Episode')  
plt.ylabel('Return')  
plt.title('A2C on {}'.format(env_name))  
plt.show()
```
## ***Syntax Reminder***
- `torch.nn.functional.relu()`与`torch.nn.ReLU()`在功能上是等价的。
- 设置优化器往往都使用类似的语法，其中self.actor是一个模型，可以通过`self.actor.parameters()`提取出模型内全部可以优化的参数。这个优化器会在后续负责更新PolicyNet`(此处为self.actor内部的可优化参数)`
```python
self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor.lr)
```

