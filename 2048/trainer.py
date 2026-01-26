import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
import time
import copy

from env import Env
from model import ActorCritic, QNetwork


class DQNTrainer:
    def __init__(self, env, episode=200_000, batch_size=1024, epsilon=1.0, buffer_size=6000, lr=3e-4):
        self.env: Env = env
        self.episode = episode
        self.epsilon_end = 0.000001
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork().to(self.device)       # 当前网络
        self.target_net = QNetwork().to(self.device)  # 目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将目标网络和当前网络初始化一致，避免网络不一致导致的训练波动
        self.best_net = QNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)           # 经验回放缓冲区
        self.batch_size = batch_size
        self.gamma = 0.99
        self.epsilon = epsilon
        self.update_target_freq = 500  # 目标网络更新频率
        self.step_count = 0
        self.eval_episodes = 2  # 评估时的episode数量
        self.best_avg_sum = 0

        def lr_lambda(current_step: int):
            warmup_steps = 10000
            total_steps = 20_000_000
            if current_step<warmup_steps:
                return float(current_step+1) / float(warmup_steps)
            else:
                # Warmup 后：余弦衰减（从 1 衰减到 0）
                progress = float(current_step-warmup_steps) / float(total_steps-warmup_steps)
                return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        wandb.init(
            project="my-2048-dqn",      # 项目名
            name="dqn-cosine",
            config={
                "learning_rate": lr,
                "gamma": self.gamma,
                "bs": self.batch_size,
                "update_target_freq": self.update_target_freq,
            }
        )
        
    def choose_action(self, state, valid_actions) -> int:
        if np.random.rand() < self.epsilon:
            # 探索：在 valid_actions 中均匀随机选择
            return np.random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                values = self.q_net(state_tensor)
                values = values.squeeze(0)            # 1, 4 -> 4
                action = torch.argmax(values)
            return int(action.item())

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))        # 缓存state->action->new state信息，不会即时用于训练

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
      
        # 从缓冲区随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()    # 用当前agent的q_net预测奖励收益

        # 计算目标Q值（使用目标网络），rewards是即时奖励，next q则是延时奖励，是下一个状态可获得预期最大的奖励
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]         # 下一个状态可获得预期最大的奖励
            target_q = rewards + self.gamma * next_q * (1 - dones)  # 如果已经done，则没有下一个状态，没有延时奖励

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q)                    # q_net学习新的奖励估计
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        self.scheduler.step()

        wandb.log({
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]['lr'],
        }, step=self.step_count)

        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

    def evaluate(self, episode):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []
        sums = []
        maxs = []

        for _ in range(self.eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            while True:
                valid_actions = self.env.get_actions()
                action = self.choose_action(state, valid_actions)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            total_rewards.append(episode_reward)
            sums.append(self.env.get_sum())
            maxs.append(self.env.get_max())

        # print(f"ep: {episode} | sums: {sums} | maxs: {maxs}")
        # print("-------------------------------")
        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards), np.mean(sums), np.max(maxs)

    def train(self):
        for episode in tqdm(range(self.episode), desc="Training"):
            state = self.env.reset()
            while True:
                valid_actions = self.env.get_actions()
                action = self.choose_action(state, valid_actions)
                next_state, reward, done = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)     # 缓存游戏过程
                self.update()

                self.step_count += 1
                state = copy.deepcopy(next_state)
                if done:
                    break

            # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # 每10个episode评估一次模型
            if episode % 10 == 0:
                avg_reward, avg_sum, max_num = self.evaluate(episode)
                wandb.log({
                    "avg_reward": avg_reward,
                    "avg_sum": avg_sum,
                    "max_num": max_num,
                    "epsilon": self.epsilon,
                }, step=self.step_count)

                if avg_sum > self.best_avg_sum:
                    self.best_avg_sum = avg_sum
                    # 深拷贝当前最优模型的参数
                    self.best_net.load_state_dict({k: v.clone() for k, v in self.q_net.state_dict().items()})
                    torch.save(self.q_net.state_dict(), "2048_DQN.pth")
                    


"""
PPO method
"""
class PPOTrainer:
    def __init__(self, env, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=10, batch_size=256, max_steps=5000, num_ep=32, total_timesteps=1000000):
        self.env: Env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.num_episodes = num_ep
        self.total_timesteps = total_timesteps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

        self.data_buffer = {}

    def collect_trajectories(self):
        """Collect trajectories for PPO update"""
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        for _ in range(self.num_episodes):
            state = self.env.reset()
            while True:
                # 获取有效动作
                valid_actions = self.env.get_actions()

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)   # 16, 4, 4 -> 1, 16, 4, 4
                with torch.no_grad():
                    probs, value = self.agent(state_tensor)
                    probs = probs.squeeze(0)            # 1, 4 -> 4
                    probs = probs / probs.sum()
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_state, reward, done = self.env.step(int(action.item()))

                states.append(state_tensor.squeeze(0))  # 16, 4, 4
                actions.append(action)                  # 1
                log_probs.append(log_prob)              # 1
                rewards.append(reward)
                dones.append(done)
                values.append(value)                    # 1

                state = next_state
                if done:
                    break

        self.data_buffer = {
            'states': torch.stack(states),                      # T, 16, 4, 4
            'actions': torch.stack(actions).squeeze(-1),        # T
            'old_log_probs': torch.stack(log_probs).squeeze(-1),# T
            'rewards': torch.FloatTensor(rewards),              # T
            'dones': torch.FloatTensor(dones),                  # T
            'values': torch.stack(values).squeeze(-1),          # T
        }

    def update(self):
        states = self.data_buffer['states']
        actions = self.data_buffer['actions']
        old_log_probs = self.data_buffer['old_log_probs']
        advantages = self.data_buffer['advantages']
        returns = self.data_buffer['returns']

        # Normalize advantages
        advantages = (advantages) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            # Mini-batch sampling
            T = states.shape[0]
            indices = np.arange(T)
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                idx = indices[start:start + self.batch_size]
                s_batch = states[idx]
                a_batch = actions[idx]
                old_lp_batch = old_log_probs[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]

                probs, values = self.agent(s_batch)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(a_batch)
                entropy = dist.entropy()

                ratios = torch.exp(log_probs - old_lp_batch)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, ret_batch)
                entropy_loss = entropy.mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()

    def compute_gae(self):
        """Compute GAE advantages for recurrent PPO"""
        rewards = self.data_buffer['rewards']
        values = self.data_buffer['values']
        dones = self.data_buffer['dones']

        # Compute returns and advantages
        T = len(rewards)
        gae = 0
        returns = torch.zeros_like(rewards, device=self.device)
        advantages = torch.zeros_like(rewards, device=self.device)

        # Start from the end
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        self.data_buffer['returns'] = returns
        self.data_buffer['advantages'] = advantages

    def train(self):
        timesteps = 0
        max_steps = 0
        while timesteps < self.total_timesteps:
            self.collect_trajectories()
            self.compute_gae()
            T = self.data_buffer['states'].shape[0]
            steps = T / self.num_episodes
            timesteps += T
            self.update()
            print(f"Timesteps: {timesteps}, steps: {steps}, Avg Reward: {np.mean(self.data_buffer['returns'].detach().cpu().numpy()):.2f}")

            # 坚持最久的一次训练结果保存下来
            if steps > max_steps:
                torch.save(self.agent.state_dict(), "2048_PPO.pth")
                max_steps = steps
