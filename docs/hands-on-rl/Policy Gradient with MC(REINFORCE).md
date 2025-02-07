# ***Algorithm Description***
- `REINFORCE`算法使用`Monte-Carlo`方法对value进行估计，而`Actor-Critic`方法则采用`TD`方法对值进行迭代
- `∇θ​J(θ)=Es∼ρπ,a∼πθ​​[∇θ​logπθ​(a∣s)⋅Qπ(s,a)]` 这里的s要遵循策略下的访问概率，而a也应该满足该参数下的策略。所以我们只要随机挑选一个起始点并走过足够长的步数，就可以通过SGD的方法来实现梯度上升（访问到的s与a都是遵从当下的状态空间与动作空间概率分布的）
- 估计Action Value需要Discounted Case下进行计算。
# ***Source Code***
```python
import gym  
import torch  
import torch.nn.functional as F  
import numpy as np  
import matplotlib.pyplot as plt  
from tqdm import tqdm  
import rl_utils  
  
class PolicyNet(torch.nn.Module):  
    def __init__(self, state_dim, hidden_dim, action_dim):  
        super(PolicyNet, self).__init__()  
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  
  
    def forward(self, x):  
        x = F.relu(self.fc1(x))  
        return F.softmax(self.fc2(x), dim=1)  
  
class REINFORCE:  
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):  
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器  
        self.gamma = gamma  # 折扣因子  
        self.device = device  
  
    def take_action(self, state):  # 根据动作概率分布随机采样  
        state = torch.tensor([state], dtype=torch.float).to(self.device)  
        probs = self.policy_net(state)  
        action_dist = torch.distributions.Categorical(probs)  
        action = action_dist.sample()  
        return action.item()  
  
    def update(self, transition_dict):  
        reward_list = transition_dict['rewards']  
        state_list = transition_dict['states']  
        action_list = transition_dict['actions']  
  
        G = 0  
        self.optimizer.zero_grad()  
        for i in reversed(range(len(reward_list))):  # 从最后一步算起  
            reward = reward_list[i]  
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)  
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)  
            log_prob = torch.log(self.policy_net(state).gather(1, action))  
            G = self.gamma * G + reward  
            loss = -log_prob * G  # 每一步的损失函数  
            loss.backward()  # 反向传播计算梯度  
        self.optimizer.step()  # 梯度下降  
  
learning_rate = 1e-3  
num_episodes = 1000  
hidden_dim = 128  
gamma = 0.98  
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
  
env_name = "CartPole-v1"  # 使用最新版本的 CartPole 环境  
env = gym.make(env_name)  
env.reset(seed = 0)  
torch.manual_seed(0)  
state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.n  
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)  
  
return_list = []  
for i in range(10):  
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  
        for i_episode in range(int(num_episodes / 10)):  
            episode_return = 0  
            transition_dict = {  
                'states': [],  
                'actions': [],  
                'next_states': [],  
                'rewards': [],  
                'dones': []  
            }  
            state, _ = env.reset()  # 新版本 gym 的 reset() 返回两个值  
            done = False  
            while not done:  
                action = agent.take_action(state)  
                next_state, reward, terminated, truncated, _ = env.step(action)  # 新版本 gym 的 step() 返回五个值  
                done = terminated or truncated  # 判断是否结束  
                transition_dict['states'].append(state)  
                transition_dict['actions'].append(action)  
                transition_dict['next_states'].append(next_state)  
                transition_dict['rewards'].append(reward)  
                transition_dict['dones'].append(done)  
                state = next_state  
                episode_return += reward  
            return_list.append(episode_return)  
            agent.update(transition_dict)  
            if (i_episode + 1) % 10 == 0:  
                pbar.set_postfix({  
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),  
                    'return': '%.3f' % np.mean(return_list[-10:])  
                })  
            pbar.update(1)  
  
episodes_list = list(range(len(return_list)))  
plt.plot(episodes_list, return_list)  
plt.xlabel('Episodes')  
plt.ylabel('Returns')  
plt.title('REINFORCE on {}'.format(env_name))  
plt.show()  
  
mv_return = rl_utils.moving_average(return_list, 9)  
plt.plot(episodes_list, mv_return)  
plt.xlabel('Episodes')  
plt.ylabel('Returns')  
plt.title('REINFORCE on {}'.format(env_name))
plt.show()
```
# ***Syntax Reminder***
- `super()`中不需要指定任何参数
- `state = torch.tensor([state], dtype = torch.float).to(self.device)`的作用是将state列表转换为`torch.tensor`且转移到指定的设备上运行，注意这里的`[state]`将state包装为了一个单元素的列表，确保张量的维度是正确的
- ！！***一定要注意，神经网络的输入必须要是一个二维的张量，所以这里需要用 `[state]`来确保符合神经网络的输入数据格式要求***
- 在`PolicyNet`的实例化过程中，我们需要将`state_dim, hidden_dim, action_dim`进行全部指定
- `PolicyNet`中需要自己定义全连接层 ***(注意，如果全部都是全连接层的话那么就可以被称为MLP)***  ，通过`self.fc1 = torch.nn.Linear(input_dimension, output_dimension)`就可以定义一个全连接层，***通过前后的参数来决定输入输出的数据规模***
- 在定义了传参层后，我们在现阶段可以显式地表现出传参的过程，例如使用`x = F.relu(self.fc1(x))`这样就可以将参数通过第一个全连接层并产生结果
- 如果我们需要的神经网络的输出是***概率分布***，那么我们就可以在输出层中采用`softmax()`函数，这也是在`torch.nn.functional`中已经包括的内容。如果我们神经网络的输出是标量值或者是`DDPG`这样不需要概率分布的情况，那么就不需要使用Softmax
- 如果不是需要输出概率分布的情况，那么我们一般采用直接输出或者采用`Sigmoid`或者`tanh`激活函数。
- `probs`是通过`policy_net`生成的一个张量，意思是当前状态下采取每一个动作的几率`(policy_net是一个从状态到动作空间的概率映射)`
- `action_dist = torch.distributions.Categorical(probs)`的作用是创建一个分类分布，这是一个`Categorical`类型的实例，可以从中进行`sample`操作（即进行取样），这一个`sample`过程是根据概率来进行采样的
- 在`update method`中，首先会输入`transition_dict`，内部会存储历史的参数，将这些参数分别存储到`reward_list, state_list, action_list`中。
- 先对优化器进行参数清零`self.optimizer.zero_grad()`，反向遍历累积`return`参数
- `state`与`action`分别将当前步的状态与动作转化为`torch`张量
- `log_prob`的作用是计算每一个动作的对数概率，需要使用的代码为
- `gather(dim, index)`意思是在维度dim上根据index依次取值。
```python
log_prob = torch.log(self.policy_net(state).gather(1, action))
```
- 将`loss定义为-log(π(a|s) * G)`，通过`loss.backward()`与`self.optimizer.step()`将差量进行反向传播。
- `.detach()`的作用是将一个向量复制，而且这个向量并不会参与到梯度的计算过程中，而是作为一个定值。
- .view()强制将一个非二维向量转换为二维向量。
