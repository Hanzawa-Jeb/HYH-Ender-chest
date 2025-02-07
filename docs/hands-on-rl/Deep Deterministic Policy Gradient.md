# DDPG
## ***Algorithm Description***
- DDPG也是一种Actor-Critic方法，只不过更新的方式不同。
- DDPG的Critic用于估计基于决定性策略，同样需要维护TargetNetwork与普通NetWork DDPG的Critic需要最小化TD Error，同样通过TargetNetwork来稳定TD Target的计算，稳定更新过程，同时通过软更新来维护TargetNet。
- DDPG的Actor也需要维护一个TargetNetwork，actor的作用是根据状态选择合适的动作（确定性的）。
- Actor网络用于最大化Q值，也就是在做***实际上的梯度参数更新***，而Actor目标函数用于计算$a = \mu^\prime(s)$，可以稳定动作输出，确保训练稳定。***这个a并不是用于行为的产生的，而是用于critic网络中的梯度下降方向确定，因为critic的TD目标必须由action来决定***。
- $\tau$的作用是决定两个TargetNetwork更新的幅度大小（Target与原网络的***加权平均的权重）***
- 我们会使用`Ornstein-Uhlenbeck noise`来使动作具有更好的探索性。这个噪声是由参数$\sigma$来决定的。
- 这里在`update()`中用了***全部经历过的样本用于更新，而不是MBGD的方法***，但是可以考虑我们在训练过程中只给update输入特定数量的数据从而达到MBGD的实现。
- ***目标Actor与目标Critic网络仅仅用于计算td_target***以确保Critic网络的更新具有更好的准确性与稳定性，但是这两个目标网络并***不会直接影响Actor网络的更新***
- ***Actor网络的优化都是基于最新的模型的，而不是目标网络***。连续动作空间的RL算法（DDPG, TD3）中，Actor通过最大化Critic估计的Q值来优化策略，而且都是使用最新的Actor与Critic网络来进行策略优化，而不是目标网络。
## ***Source Code***
```python
import random  
import gym  
import numpy as np  
from tqdm import tqdm  
import torch  
from torch import nn  
import torch.nn.functional as F  
import matplotlib.pyplot as plt  
import rl_utils  
  
class PolicyNet(torch.nn.Module):  
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(state_dim, hidden_dim), torch.nn.ReLU(),  
            nn.Linear(hidden_dim, action_dim)  
        )  
        self.action_bound = action_bound  
  
    def forward(self, x):  
        return torch.tanh(self.net(x)) * self.action_bound  
  
class ValueNet(torch.nn.Module):  
    def __init__(self, state_dim, hidden_dim, action_dim):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(state_dim + action_dim, hidden_dim), torch.nn.ReLU(),  
            nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),  
            nn.Linear(hidden_dim, 1)  
        )  
  
    def forward(self, state, action):  
        cat_x = torch.cat([state, action], dim = 1)  
        return self.net(cat_x)  
  
class DDPG:  
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):  
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)  
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)  
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)  
        self.target_critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)  
        self.target_actor.load_state_dict(self.actor.state_dict())  
        self.target_critic.load_state_dict(self.critic.state_dict())  
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)  
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)  
        self.state_dim = state_dim  
        self.hidden_dim = hidden_dim  
        self.action_dim = action_dim  
        self.action_bound = action_bound  
        self.sigma = sigma  
        self.actor_lr = actor_lr  
        self.critic_lr = critic_lr  
        self.tau = tau  
        self.gamma = gamma  
        self.device = device  
  
    def take_action(self, state):  
        state = torch.tensor([state], dtype = torch.float).to(self.device)  #为什么要转换成torch上的向量？？  
        action = self.actor(state).item()  
        action = action + self.sigma * np.random.randn(self.action_dim) #add noise to action  
        return action  
  
    def soft_update(self, net, target_net):  
        for param_target, param in zip(target_net.parameters(), net.parameters()):  
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)   #为什么要.data？  
  
    def update(self, transition_dict):  
        states = torch.tensor(transition_dict['states'], dtype = torch.float).to(self.device)  
        actions = torch.tensor(transition_dict['actions'], dtype = torch.float).view(-1, 1).to(self.device) #why dtype?  
        rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device)  
        next_states = torch.tensor(transition_dict['next_states'], dtype = torch.float).to(self.device)  
        dones = torch.tensor(transition_dict['dones'], dtype = torch.float).view(-1, 1).to(self.device)  
  
        next_q_values = self.target_critic(next_states, self.target_actor(next_states)) #q(s, a)  
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)  #the values here are all estimated by target_net  
        critic_loss = torch.mean(F.mse_loss(q_targets, self.critic(states, actions)))  
        self.critic_optimizer.zero_grad()  
        critic_loss.backward()  
        self.critic_optimizer.step()  
  
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))  
        self.actor_optimizer.zero_grad()    #???  
        actor_loss.backward()  
        self.actor_optimizer.step()  
  
        self.soft_update(self.actor, self.target_actor)  
        self.soft_update(self.critic, self.target_critic)  
  
actor_lr = 3e-4  
critic_lr = 3e-3  
num_episodes = 200  
hidden_dim = 64  
gamma = 0.98  
tau = 0.005  
buffer_size = 10000  
minimal_size = 1000  
batch_size = 64  
sigma = 0.01    #gauss noise parameter  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
env_name = 'Pendulum-v1'  
env = gym.make(env_name)  
random.seed(0)  
np.random.seed(0)  
env.seed(0)  
torch.manual_seed(0)  
replay_buffer = rl_utils.ReplayBuffer(buffer_size)  
state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.shape[0]  
action_bound = env.action_space.high[0]  
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)  
  
return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episodes')  
plt.ylabel('Returns')  
plt.title("DDPG on {}".format(env_name))  
plt.show()  
  
mv_return = rl_utils.moving_average(return_list, 9)  
plt.plot(episodes_list, mv_return)  
plt.xlabel('Episodes')  
plt.ylabel('MV Returns')  
plt.title("DDPG on {}".format(env_name))  
plt.show()
```
## ***Implementation Reminder***
- 在连续动作空间的Actor中，我们可以在输出层使用tanh激活函数，值域为`[-1, 1]`，方便通过比例来调节成环境可以接受的动作范围。
- `torch.cat([list1, list2], dim = 0/1)`的作用是拼接两个张量，如果dim = 0则说明行的个数发生变化，如果dim = 1则说明列的个数发生变化。
- `nn2.load_state_dict(nn1.state_dict())`的作用是将nn1的初始参数全部赋值给nn2。
- 一般默认的线性层权重初始化方法都是`Xavier(Glorot) Initialization`，偏置一般被默认为0.
- target网络一般都不需要由自己特有的优化器，因为只要通过加权平均就可以达到参数更新的效果。
- ***为什么PPO中actions没有dtype = torch.float而DDPG中有？*** 因为PPO中的action是确定的离散整数值，每一个动作都有相对应的整数值（并不是概率分布，而是通过Categorical变量sample而得来的量）
- ***为什么要.data***->因为这样可以将模型的数据从计算图中剥离出来，在这里soft-update更新的过程中可以避免计算图被牵扯。
```python
critic_loss = torch.mean(F.mse_loss(q_targets, self.critic(states,actions)))
```
- 为什么这里的q_targets不用`.detach()`？：因为`optimizer`中更新的参数是critic_network的参数而不是targe_critic的参数，target_critic的参数是固定的
- q_target完全由两个target network来决定，所以不用担心梯度图中会包括其中的参数。