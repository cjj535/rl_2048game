import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import random

from env import Env
from model import ActorCritic, QNetwork
from trainer import PPOTrainer, DQNTrainer


class VizEval:
    def __init__(self, env, method="PPO"):
        self.env: Env = env
        self.method = method

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.method == "PPO":
            self.agent = ActorCritic().to(self.device)
            self.agent.load_state_dict(torch.load("2048_PPO.pth", map_location='cpu'))
        elif self.method == "DQN":
            self.agent = QNetwork().to(self.device)
            self.agent.load_state_dict(torch.load("2048_DQN.pth", map_location='cpu'))
        self.agent.eval()

    def viz(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, score = self.agent(state_tensor)
        logits = logits.squeeze(0)
        score = score.squeeze(0)
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
        while True:
            # 获取有效动作
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break
            mask = torch.zeros(4, device=self.device)
            mask[valid_actions] = 1.0

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)   # 16, 4, 4 -> 1, 16, 4, 4
            if self.method == "PPO":
                with torch.no_grad():
                    probs, value = self.agent(state_tensor)
                    probs = probs.squeeze(0)            # 1, 4 -> 4
                    value = value.squeeze(0)            # 1, 1 -> 1
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
                next_state, _, done = self.env.step(int(action.item()))
                state = next_state
                if done:
                    break
            elif self.method == "DQN":
                with torch.no_grad():
                    values = self.agent(state_tensor)
                    values = values.squeeze(0)            # 1, 4 -> 4
                    values = values * mask + 1e-8
                    values = values / values.sum()
                    action = torch.argmax(values)

                self.env.render()
                print("next action: ", values.cpu().numpy())
                print("mask: ", mask.cpu().numpy())
                input("+++++++++++++++++++++++++++++++++++++++++++++++++++")
                next_state, _, done = self.env.step(int(action.item()))
                state = next_state
                if done:
                    break

# 设置随机数种子
def deterministic():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "test"],
        help="mode"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["PPO", "DQN", "RecurrentPPO"],
        help="mode"
    )
    args = parser.parse_args()

    deterministic()
    mode = args.mode
    method = args.method
    if mode == "train":
        if method == "PPO":
            trainer = PPOTrainer(env=Env())
            trainer.train()
        elif method == "DQN":
            trainer = DQNTrainer(env=Env())
            trainer.train()
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
