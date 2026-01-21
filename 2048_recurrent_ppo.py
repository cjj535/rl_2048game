"""
2048 的 Docstring
当前的reward的定义是每次行为增加的分数，但是2048是一个偏奖励累积的游戏，越到分数高（局面复杂的时候），新增分数越困难
PPO中reward使用了GAE，但这种奖励估计方法可能对2048游戏来说很难估计准确，可能需要改用类似alphaGo的训练方式
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import random
import math
import matplotlib.pyplot as plt
import argparse

# 设置随机数种子
def deterministic():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


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
        return self.state
    
    def set_state(self, state) -> List[List[int]]:
        """设置游戏状态"""
        self.state = [[state[i][j] for j in range(self.W)] for i in range(self.H)]

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

    def _slide_and_merge(self, row: List[int]) -> Tuple[List[int], bool, float]:
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
                merge_score += 1
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

    def get_empty_sum(self):
        return sum(cell == 0 for row in self.state for cell in row)

    def step(self, action: int):
        """
        执行一步动作
        action: 0=left, 1=right, 2=up, 3=down
        返回: (state, reward, done, info, extra) —— 兼容 Gym 风格
        """
        total_reward = 0

        if action == 0:  # left
            for i in range(self.H):
                new_row, _, score = self._slide_and_merge(self.state[i])
                self.state[i] = new_row
                total_reward += score

        elif action == 1:  # right
            for i in range(self.H):
                reversed_row = self.state[i][::-1]
                new_row, _, score = self._slide_and_merge(reversed_row)
                self.state[i] = new_row[::-1]
                total_reward += score

        elif action == 2:  # up
            for j in range(self.W):
                col = [self.state[i][j] for i in range(self.H)]
                new_col, _, score = self._slide_and_merge(col)
                for i in range(self.H):
                    self.state[i][j] = new_col[i]
                total_reward += score

        elif action == 3:  # down
            for j in range(self.W):
                col = [self.state[i][j] for i in range(self.H)][::-1]
                new_col, _, score = self._slide_and_merge(col)
                new_col = new_col[::-1]
                for i in range(self.H):
                    self.state[i][j] = new_col[i]
                total_reward += score

        # 如果有变化，添加新块（注意：add_random_tile 不影响 reward）
        self._add_random_tile()

        done = 1 if self.check_game_over() else 0
        total_reward -= done * 50

        return self.state, total_reward, done, None, None

    def get_sum(self) -> int:
        return sum(sum(row) for row in self.state)

    def get_max(self) -> int:
        return max(max(row) for row in self.state)

    def render(self):
        """打印当前状态（便于调试）"""
        print("\n".join(" ".join(f"{cell:4}" if cell else "   ." for cell in row) for row in self.state))
        print("-" * 20)


class ActorCritic(nn.Module):
    def __init__(self, action_dim=4, hidden_size=1024, gru_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.gru_layers = gru_layers

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),           # [B, 256]
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        # log2 转换, 避免 log(0)
        B, T, _, _ = x.shape            # 1, T, 4, 4
        x = x.view(B * T, 1, 4, 4)

        x = torch.where(x == 0, torch.zeros_like(x), torch.log2(x) / 16.0)

        features = self.encoder(x)
        features = features.view(B, T, -1)      # B, T, 256

        # RNN forward
        if hidden is not None:
            gru_out, hidden = self.gru(features, hidden)
        else:
            gru_out, hidden = self.gru(features)

        logits = self.actor(gru_out)            # B, T, 4
        value = self.critic(gru_out)            # B, T, 1

        return logits, value, hidden


class EpisodeBuffer:
    def __init__(self):
        self.episodes = []  # list of episodes

    def add_episode(self, states, actions, log_probs, rewards, dones, values, masks):
        """
        Each input: list of T elements (T = episode length)
        """
        self.episodes.append({
            'states': torch.stack(states),                          # T, 4, 4
            'actions': torch.stack(actions).squeeze(-1),            # T
            'old_log_probs': torch.stack(log_probs).squeeze(-1),    # T
            'values': torch.stack(values).squeeze(-1),              # T
            'rewards': torch.FloatTensor(rewards),                  # T
            'dones': torch.FloatTensor(dones),                      # T
            'masks': torch.stack(masks),                            # T, 4
        })

    def clear(self):
        self.episodes = []


def compute_gae(episodes, gamma=0.99, lam=0.95):
    """Compute GAE advantages for recurrent PPO"""
    for ep in episodes:
        rewards = ep['rewards']
        values = ep['values']
        dones = ep['dones']

        # Compute returns and advantages
        T = len(rewards)
        gae = 0
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Start from the end
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        ep['returns'] = returns
        ep['advantages'] = advantages


class PPOTrainer:
    def __init__(self, env, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=10, batch_size=256, max_steps=5000):
        self.env: Env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_steps = max_steps

        self.buffer = EpisodeBuffer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

    def collect_episodes(self, num_episodes=16):
        self.buffer.clear()

        for _ in range(num_episodes):
            state_list, act_list, logp_list, reward_list, done_list, val_list, mask_list = [], [], [], [], [], [], []
            state = self.env.reset()
            hidden = None

            while True:
                # 获取有效动作
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break
                mask = torch.zeros(4, dtype=torch.float32, device=self.device)
                mask[valid_actions] = 1.0

                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)   # 4, 4 -> 1, 1, 4, 4
                with torch.no_grad():
                    probs, value, hidden = self.agent(state_tensor, hidden)
                    probs = probs.squeeze(0)            # 1, 4 -> 4
                    value = value.squeeze(0)            # 1, 1 -> 1
                    # 应用 mask
                    probs = probs * mask + 1e-8
                    probs = probs / probs.sum()
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_state, reward, done, _, _ = self.env.step(action.item())

                # Store
                state_list.append(state_tensor.squeeze(0).squeeze(0))   # 1, 1, 4, 4 -> 4, 4
                act_list.append(action)                                 # 1
                logp_list.append(log_prob)                              # 1
                reward_list.append(reward)
                done_list.append(done)
                val_list.append(value.squeeze(-1))                      # 1
                mask_list.append(mask)                                  # 4

                state = next_state
                if done:
                    break

            self.buffer.add_episode(state_list, act_list, logp_list, reward_list, done_list, val_list, mask_list)

    def train_recurrent_ppo(self):
        episodes = self.buffer.episodes
        compute_gae(episodes)

        for _ in range(self.epochs):
            shuffled_episodes = random.sample(episodes, len(episodes))

            for ep in shuffled_episodes:
                T = len(ep['states'])
                if T == 0:
                    continue

                # Forward pass through entire episode
                valid_actions_masks = ep['masks']               # T, 4
                states = ep['states']                           # T, 4, 4
                actions = ep['actions']                         # T
                advantages = ep['advantages']                   # T
                old_log_probs = ep['old_log_probs']             # T
                returns = ep['returns']                         # T

                # print("states:", states.shape)
                # print("actions:", actions.shape)
                # print("log_probs:", old_log_probs.shape)
                # print("advantages:", advantages.shape)
                # print("returns:", returns.shape)
                # print("mask:", valid_actions_masks.shape)
                # exit(0)

                states = states.unsqueeze(0)                    # T, 4, 4 -> 1, T, 4, 4
                logits, values, _ = self.agent(states, None)
                logits = logits.squeeze(0)                      # 1, T, 4 -> T, 4
                values = values.squeeze(0).squeeze(-1)          # 1, T, 1 -> T

                logits = logits * valid_actions_masks + 1e-8    # T, 4
                probs = logits / logits.sum(dim=-1, keepdim=True)   # T, 4
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(actions)               # T
                entropy = dist.entropy().mean()

                # PPO loss
                ratios = torch.exp(log_prob - old_log_probs)    # T
                surr1 = ratios * advantages                     # T
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages  # T
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (values - returns).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()

    def train(self, total_timesteps=100000):
        num_episodes = 16
        timesteps = 0
        max_steps = 0
        while timesteps < total_timesteps:
            self.collect_episodes(num_episodes=num_episodes)
            total_steps = sum(len(subdict['states']) for subdict in self.buffer.episodes)
            steps = total_steps / len(self.buffer.episodes)
            timesteps += total_steps
            self.train_recurrent_ppo()
            print(f"Timesteps: {timesteps}, steps: {steps}")

            # 平均坚持最久的一次训练结果保存下来
            if steps > max_steps:
                torch.save(self.agent.state_dict(), "recurrent_ppo_2048.pth")
                max_steps = steps


class VizEval:
    def __init__(self, env):
        self.env: Env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = ActorCritic().to(self.device)
        self.agent.load_state_dict(torch.load("ppo_2048.pth", map_location='cpu'))
        self.agent.eval()

    def viz(self, state, hidden=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        logits, score, _ = self.agent(state_tensor, hidden)
        logits = logits.squeeze(0)
        score = score.squeeze(0).squeeze(-1)
        print("state score: ", score)
        print("next action: ", logits)

        board = np.array(state, dtype=np.float32)

        # 安全地计算 log2：0 保持为 0，正数取 log2
        board = np.where(board == 0, 0.0, np.log2(board))

        # 绘制热力图
        plt.figure(figsize=(4, 4))
        plt.imshow(board, cmap='viridis', interpolation='nearest')

        for i in range(4):
            for j in range(4):
                plt.text(j, i, str(board[i, j]) if board[i, j] != 0 else '',
                        ha="center", va="center", color="white", fontsize=14)

        # 设置坐标轴
        plt.xticks([])
        plt.yticks([])
        plt.title("2048 Board Heatmap")
        plt.tight_layout()
        plt.show()

    def play(self, state):
        self.env.set_state(state)
        hidden = None
        while True:
            # 获取有效动作
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break
            mask = torch.zeros(4, dtype=torch.float32, device=self.device)
            mask[valid_actions] = 1.0

            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)   # 4, 4 -> 1, 1, 4, 4
            with torch.no_grad():
                probs, value, hidden = self.agent(state_tensor, hidden)
                probs = probs.squeeze(0)                # 1, 4 -> 4
                value = value.squeeze(0).squeeze(-1)    # 1, 1, 1 -> 1
                # 应用 mask
                probs = probs * mask + 1e-8
                probs = probs / probs.sum()
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            self.env.render()
            print("state score: ", value.cpu().numpy())
            print("next action: ", probs.cpu().numpy())
            print("mask: ", mask.cpu().numpy())
            input("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            next_state, _, done, _, _ = self.env.step(int(action.item()))
            state = next_state
            if done:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "test"],
        help="mode"
    )
    args = parser.parse_args()

    deterministic()
    mode = args.mode
    if mode == "train":
        trainer = PPOTrainer(env=Env(), lr=1e-4, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=10)
        trainer.train(total_timesteps=16*1024*20)
    else:
        np.set_printoptions(precision=2, suppress=True)
        init_state = [
            [8, 256, 64, 2],
            [4, 128, 16, 16],
            [2, 16, 32, 4],
            [0, 4, 0, 2]
        ]
        viz = VizEval(env=Env())
        viz.play(init_state)

if __name__ == "__main__":
    main()
