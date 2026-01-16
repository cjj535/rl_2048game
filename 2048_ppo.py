"""
2048 的 Docstring
当前的reward的定义是每次行为增加的分数，但是2048是一个偏奖励累积的游戏，越到分数高（局面复杂的时候），新增分数越困难
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import swanlab
import os
from typing import List, Tuple
import random


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

        done = 1 if self.check_game_over() else 0

        return self.state, total_reward, done, None, None

    def get_sum(self) -> int:
        """可选：计算当前总分（所有方块之和）"""
        return sum(sum(row) for row in self.state)

    def render(self):
        """打印当前状态（便于调试）"""
        print("\n".join(" ".join(f"{cell:4}" if cell else "   ." for cell in row) for row in self.state))
        print("-" * 20)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=16, action_dim=4, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state, valid_actions_mask=None):
        logits = self.actor(state)
        if valid_actions_mask is not None:
            # 将无效动作概率设为极小值
            logits = logits * valid_actions_mask + 1e-8
            probs = logits / logits.sum(dim=-1, keepdim=True)
        else:
            probs = logits
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()                  # 按策略网络提供的概率采样
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(self, state, action, valid_actions_mask=None):
        logits = self.actor(state)
        if valid_actions_mask is not None:
            logits = logits * valid_actions_mask + 1e-8
            probs = logits / logits.sum(dim=-1, keepdim=True)
        else:
            probs = logits
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state).squeeze(-1)
        return log_prob, entropy, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    """
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# 动作映射表：每种变换对应的动作重映射
AUGMENTATIONS = [
    # (变换函数, 动作映射)
    (lambda x: x,                     [0, 1, 2, 3], [0, 1, 2, 3]),  # 原始, left, right, up, down
    (lambda x: np.rot90(x, 1),        [3, 2, 0, 1], [2, 3, 1, 0]),  # 逆时针90°
    (lambda x: np.rot90(x, 2),        [1, 0, 3, 2], [1, 0, 3, 2]),  # 逆时针180°
    (lambda x: np.rot90(x, 3),        [2, 3, 1, 0], [3, 2, 0, 1]),  # 逆时针270°
    (lambda x: np.fliplr(x),          [1, 0, 2, 3], [1, 0, 2, 3]),  # 水平翻转
    (lambda x: np.flipud(x),          [0, 1, 3, 2], [0, 1, 3, 2]),  # 垂直翻转
    (lambda x: x.T,                   [2, 3, 0, 1], [2, 3, 0, 1]),  # 转置（主对角线）
    (lambda x: np.fliplr(np.rot90(x)),[3, 2, 1, 0], [3, 2, 1, 0]),  # 副对角线
]

def augment_sample(
    state: np.ndarray, 
    action: int,
    log_prob: float,
    advantage: float,
    return_: float,
    mask: np.ndarray,
) -> List[Tuple[np.ndarray, int, float, float, float, np.ndarray]]:
    """
    对单个样本进行 8 种对称增强，返回增强后的样本列表。
    每个样本: (new_state, new_action, log_prob, advantage, return_)
    """
    augmented = []
    for transform, action_map, inv_action_map in AUGMENTATIONS:
        new_state = transform(state)
        new_action = action_map[action]
        new_mask = mask[inv_action_map]
        # 注意：log_prob, advantage, return 不变
        augmented.append((new_state, new_action, log_prob, advantage, return_, new_mask))

    return augmented


class PPOTrainer:
    def __init__(self, env_fn, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=10, batch_size=256, max_steps=5000):
        self.env_fn = env_fn
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_steps = max_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def collect_trajectories(self, num_episodes=16):
        """Collect trajectories for PPO update"""
        states, actions, log_probs, rewards, dones, values, masks = [], [], [], [], [], [], []

        for _ in range(num_episodes):
            env = self.env_fn()
            state = env.reset()
            step = 0
            while step < self.max_steps:
                # 获取有效动作
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                mask = torch.zeros(4, device=self.device)
                mask[valid_actions] = 1.0

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                state_tensor = state_tensor.view(1, -1)
                with torch.no_grad():
                    probs, value = self.policy(state_tensor)
                    # 应用 mask
                    probs = probs * mask + 1e-8
                    probs = probs / probs.sum()
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_state, reward, done, _, _ = env.step(action.item())

                states.append(np.array(state))
                actions.append(action.item())
                log_probs.append(log_prob.item())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())
                masks.append(mask.cpu().numpy())

                state = next_state
                step += 1
                if done:
                    break

        # 计算 GAE
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'masks': masks,
        }

    def apply_augmentation_to_batch(self, data):
        """对整个批次应用数据增强"""
        aug_states, aug_actions, aug_log_probs, aug_advs, aug_returns, aug_masks = [], [], [], [], [], []
        
        states = data['states']
        actions = data['actions']
        log_probs = data['old_log_probs']
        advantages = data['advantages']
        returns = data['returns']
        masks = data['masks']

        for i in range(len(states)):
            aug_samples = augment_sample(
                states[i],
                actions[i],
                log_probs[i],
                advantages[i],
                returns[i],
                masks[i],
            )
            for s, a, lp, adv, ret, mask in aug_samples:
                aug_states.append(s)
                aug_actions.append(a)
                aug_log_probs.append(lp)
                aug_advs.append(adv)
                aug_returns.append(ret)
                aug_masks.append(mask)

        return {
            'states': aug_states,
            'actions': aug_actions,
            'old_log_probs': aug_log_probs,
            'advantages': aug_advs,
            'returns': aug_returns,
            'masks': aug_masks,
        }

    def update(self, data):
        # 通过旋转、翻转等操作进行数据增强
        data = self.apply_augmentation_to_batch(data)

        states = torch.from_numpy(
            np.stack(data['states'])
            .reshape(len(data['states']), -1)
            .astype(np.float32)
        )
        actions = torch.tensor(data['actions'], dtype=torch.long)
        old_log_probs = torch.tensor(data['old_log_probs'], dtype=torch.float32)
        advantages = torch.tensor(data['advantages'], dtype=torch.float32)
        returns = torch.tensor(data['returns'], dtype=torch.float32)
        masks = torch.from_numpy(
            np.stack(data['masks'])
            .astype(np.float32)
        )

        # Normalize advantages
        advantages = (advantages) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            # Mini-batch sampling
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                s_batch = states[idx].to(self.device)
                a_batch = actions[idx].to(self.device)
                old_lp_batch = old_log_probs[idx].to(self.device)
                adv_batch = advantages[idx].to(self.device)
                ret_batch = returns[idx].to(self.device)
                m_batch = masks[idx].to(self.device)

                log_probs, entropy, values = self.policy.evaluate(s_batch, a_batch, m_batch)
                ratios = torch.exp(log_probs - old_lp_batch)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, ret_batch)
                entropy_loss = entropy.mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

    def train(self, total_timesteps=100000):
        timesteps = 0
        update_steps = 0
        while timesteps < total_timesteps:
            data = self.collect_trajectories()
            timesteps += len(data['states'])
            self.update(data)
            print(f"Timesteps: {timesteps}, Avg Reward: {np.mean(data['returns']):.2f}")

            update_steps += 1
            if update_steps % 2 == 0:
                torch.save(self.policy.state_dict(), "ppo_2048.pth")
        torch.save(self.policy.state_dict(), "ppo_2048.pth")

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    def make_env():
        return Env()

    deterministic()

    trainer = PPOTrainer(env_fn=make_env, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=10)
    trainer.train(total_timesteps=16*8*4096*10)
