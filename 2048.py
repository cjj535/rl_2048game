"""
2048 的 Docstring
当前的reward的定义是每次行为增加的分数，但是2048是一个偏奖励累积的游戏，越到分数高（局面复杂的时候），新增分数越困难
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import swanlab
import os

# 设置随机数种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

reward_threshold = 1e5

# 定义Q网络，输入是state，输出是每个action的收益
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
  
    def forward(self, x):
        return self.fc(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim).to(DEVICE)       # 当前网络
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)  # 目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将目标网络和当前网络初始化一致，避免网络不一致导致的训练波动
        self.best_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)           # 经验回放缓冲区
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.update_target_freq = 100  # 目标网络更新频率
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 2  # 评估时的episode数量

    def choose_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            # 探索：在 valid_actions 中均匀随机选择
            return np.random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            q_values = q_values.cpu().numpy()

            # 只考虑 valid_actions 中的动作
            valid_q = {a: q_values[a] for a in valid_actions}
            best_action = max(valid_q, key=valid_q.get)
            return best_action

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))        # 缓存state->action->new state信息，不会即时用于训练

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
      
        # 从缓冲区随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()    # 用当前agent的q_net预测奖励收益

        # 计算目标Q值（使用目标网络），rewards是即时奖励，next q则是延时奖励，是下一个状态可获得预期最大的奖励
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]     # 下一个状态可获得预期最大的奖励
            target_q = rewards + self.gamma * next_q * (1 - dones)  # 如果已经done，则没有下一个状态，没有延时奖励

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q)            # q_net学习新的
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")
      
    def evaluate(self, env: "Env"):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []
        sums = []

        for _ in range(self.eval_episodes):
            state = env.reset()
            episode_reward = 0
            while True:
                valid_actions = env.get_valid_actions()
                action = self.choose_action(state, valid_actions)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done or episode_reward > reward_threshold:
                    break
            total_rewards.append(episode_reward)
            sums.append(env.get_sum())

        print("eval sums: ", sums)
        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)


import random
from typing import List, Tuple

class Env:
    H: int = 4
    W: int = 4
    actions: List[int] = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
    rng_num: List[int] = [2, 4]
    rng_prob: List[float] = [0.67, 0.33]  # 67% chance for 2, 33% for 4

    def __init__(self):
        self.state: List[List[int]] = []
        self.reset()

    @staticmethod
    def _flatten_state(state_2d):
        """将 4x4 二维列表展平为长度为 16 的一维列表"""
        return [cell for row in state_2d for cell in row]

    def reset(self) -> List[List[int]]:
        """重置游戏状态，并在两个随机位置生成初始数字（通常是一个2或4）"""
        self.state = [[0 for _ in range(self.W)] for _ in range(self.H)]
        self._add_random_tile()
        return self._flatten_state(self.state)

    def _add_random_tile(self) -> int:
        """在空白位置随机添加一个新数字（2 或 4），返回该数字"""
        free_blocks = [(i, j) for i in range(self.H) for j in range(self.W) if self.state[i][j] == 0]
        if not free_blocks:
            return 0  # no space

        i, j = random.choice(free_blocks)
        val = random.choices(self.rng_num, weights=self.rng_prob)[0]
        self.state[i][j] = val
        return val

    def check_game_over(self) -> bool:
        """检查是否游戏结束（无空格且无法合并）"""
        # 检查是否有空格
        for i in range(self.H):
            for j in range(self.W):
                if self.state[i][j] == 0:
                    return False

        # 检查是否有相邻相同数字（可合并）
        for i in range(self.H):
            for j in range(self.W):
                if (i + 1 < self.H and self.state[i][j] == self.state[i + 1][j]) or \
                   (j + 1 < self.W and self.state[i][j] == self.state[i][j + 1]):
                    return False
        return True

    def _slide_and_merge(self, row: List[int]) -> Tuple[List[int], bool, int]:
        """
        对一行进行滑动和合并。
        返回:
            new_row: 合并后的行
            changed: 是否发生变化
            merge_score: 本次合并产生的总分（即所有新块的值之和）
        """
        non_zero = [x for x in row if x != 0]
        new_row = []
        i = 0
        merge_score = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                new_row.append(merged_val)
                merge_score += merged_val  # 关键：累加合并结果值
                i += 2
            else:
                new_row.append(non_zero[i])
                i += 1
        new_row.extend([0] * (self.W - len(new_row)))
        changed = (new_row != row)
        return new_row, changed, merge_score

    def get_valid_actions(self) -> List[int]:
        """
        返回所有能对当前状态产生变化的动作（0: left, 1: right, 2: up, 3: down）
        """
        valid_actions = []

        # 尝试每个动作
        for action in self.actions:
            changed = False
            if action == 0:  # left
                for i in range(self.H):
                    _, row_changed, _ = self._slide_and_merge(self.state[i])
                    if row_changed:
                        changed = True
                        break

            elif action == 1:  # right
                for i in range(self.H):
                    reversed_row = self.state[i][::-1]
                    _, row_changed, _ = self._slide_and_merge(reversed_row)
                    if row_changed:
                        changed = True
                        break

            elif action == 2:  # up
                for j in range(self.W):
                    col = [self.state[i][j] for i in range(self.H)]
                    _, col_changed, _ = self._slide_and_merge(col)
                    if col_changed:
                        changed = True
                        break

            elif action == 3:  # down
                for j in range(self.W):
                    col = [self.state[i][j] for i in range(self.H)][::-1]
                    _, col_changed, _ = self._slide_and_merge(col)
                    if col_changed:
                        changed = True
                        break

            if changed:
                valid_actions.append(action)

        return valid_actions

    def step(self, action: int):
        """
        执行一步动作
        action: 0=left, 1=right, 2=up, 3=down
        返回: (state, reward, done, info, extra) —— 兼容 Gym 风格
        """
        total_reward = 0
        changed = False

        if action == 0:  # left
            for i in range(self.H):
                new_row, row_changed, score = self._slide_and_merge(self.state[i])
                self.state[i] = new_row
                changed = changed or row_changed
                total_reward += score

        elif action == 1:  # right
            for i in range(self.H):
                reversed_row = self.state[i][::-1]
                new_row, row_changed, score = self._slide_and_merge(reversed_row)
                self.state[i] = new_row[::-1]
                changed = changed or row_changed
                total_reward += score

        elif action == 2:  # up
            for j in range(self.W):
                col = [self.state[i][j] for i in range(self.H)]
                new_col, col_changed, score = self._slide_and_merge(col)
                for i in range(self.H):
                    self.state[i][j] = new_col[i]
                changed = changed or col_changed
                total_reward += score

        elif action == 3:  # down
            for j in range(self.W):
                col = [self.state[i][j] for i in range(self.H)][::-1]
                new_col, col_changed, score = self._slide_and_merge(col)
                new_col = new_col[::-1]
                for i in range(self.H):
                    self.state[i][j] = new_col[i]
                changed = changed or col_changed
                total_reward += score

        # 如果有变化，添加新块（注意：add_random_tile 不影响 reward）
        if changed:
            self._add_random_tile()

        done = self.check_game_over()

        return self._flatten_state(self.state), total_reward, done, None, None

    def get_sum(self) -> int:
        """可选：计算当前总分（所有方块之和）"""
        return sum(sum(row) for row in self.state)

    def render(self):
        """打印当前状态（便于调试）"""
        print("\n".join(" ".join(f"{cell:4}" if cell else "   ." for cell in row) for row in self.state))
        print("-" * 20)


# 训练过程
env = Env()
state_dim = env.H * env.W
action_dim = 4
agent = DQNAgent(state_dim, action_dim)


# 初始化SwanLab日志记录器
swanlab.init(
    project="RL-All-In-One",
    experiment_name="DQN-2048",
    config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "update_target_freq": agent.update_target_freq,
        "replay_buffer_size": agent.replay_buffer.maxlen,
        "learning_rate": agent.optimizer.param_groups[0]['lr'],
        "episode": 600,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
    },
    description="2048 game"
)

# ========== 训练阶段 ==========

agent.epsilon = swanlab.config["epsilon_start"]

for episode in range(swanlab.config["episode"]):            # 游戏轮数
    state = env.reset()
    last_reward = 0
  
    while True:
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)     # 缓存游戏过程
        agent.train()

        last_reward += reward              # 单轮游戏奖励叠加
        state = next_state
        if done or last_reward > reward_threshold:      # 游戏结束，或者奖励满足要求，这里是步数足够
            break
  
    # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
    agent.epsilon = max(swanlab.config["epsilon_end"], agent.epsilon * swanlab.config["epsilon_decay"])  
  
    # 每10个episode评估一次模型
    if episode % 10 == 0:
        eval_env = Env()
        avg_reward = agent.evaluate(eval_env)
        del eval_env
      
        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            # 深拷贝当前最优模型的参数
            agent.best_net.load_state_dict({k: v.clone() for k, v in agent.q_net.state_dict().items()})
            agent.save_model(path=f"./output/2048_best_model.pth")
            print(f"New best model saved with average reward: {avg_reward}")

    print(f"Episode: {episode}, Train Reward: {last_reward}, Best Eval Avg Reward: {agent.best_avg_reward}")
  
    swanlab.log(
        {
            "train/reward": last_reward,
            "eval/best_avg_reward": agent.best_avg_reward,
            "train/epsilon": agent.epsilon
        },
        step=episode,
    )

# 测试并录制视频
# agent.epsilon = 0  # 关闭探索策略
# test_env = gym.make('CartPole-v1', render_mode='rgb_array')
# test_env = RecordVideo(test_env, "./dqn_videos", episode_trigger=lambda x: True)  # 保存所有测试回合
# agent.q_net.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型

# for episode in range(3):  # 录制3个测试回合
#     state = test_env.reset()[0]
#     total_reward = 0
#     steps = 0
  
#     while True:
#         action = agent.choose_action(state)
#         next_state, reward, done, _, _ = test_env.step(action)
#         total_reward += reward
#         state = next_state
#         steps += 1
      
#         # 限制每个episode最多1500步,约30秒,防止录制时间过长
#         if done or steps >= 1500:
#             break
  
#     print(f"Test Episode: {episode}, Reward: {total_reward}")

# test_env.close()
