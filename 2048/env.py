"""
2048 game
reward定义
    1、合并成功一次，增加合并的分数
    2、达到目标加固定分数，达到2048加2048/128=16分
    3、不对game over做惩罚
    4、不能改变局面的动作设置为不可执行的动作，不必对此动作进行惩罚
"""
import numpy as np
from typing import List, Tuple
import random
import math
import copy


class Env:
    H: int = 4
    W: int = 4
    C: int = 16
    actions: List[int] = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
    rng_num: List[int] = [2, 4]
    rng_prob: List[float] = [0.67, 0.33]  # 67% chance for 2, 33% for 4

    def __init__(self, H=4, W=4, C=16):
        self.H = H
        self.W = W
        self.C = C
        self.state: List[List[int]] = []
        self.max_num = 0
        self.invalid_step = 0
        self.invalid_thresold = 16
        self.reset()

    @staticmethod
    def _flatten_state(state_2d):
        """将 4x4 二维列表展平为长度为 16 的一维列表"""
        return [cell for row in state_2d for cell in row]

    def reset(self) -> np.ndarray:
        """重置游戏状态，并在两个随机位置生成初始数字（通常是一个2或4）"""
        self.state = [[0 for _ in range(self.W)] for _ in range(self.H)]
        self._add_random_tile()
        self.max_num = max(max(row) for row in self.state)
        self.invalid_step = 0
        return self._create_onehot()
    
    def set_state(self, state) -> np.ndarray:
        """设置游戏状态"""
        assert len(state) == self.H and all(len(row) == self.W for row in state)
        self.state = [[state[i][j] for j in range(self.W)] for i in range(self.H)]
        return self._create_onehot()

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
        """检查是否游戏结束"""
        # 检查无效步数是否累积超过限制
        if self.invalid_step > self.invalid_thresold:
            return True

        # 检查是否超出可表示范围
        if self.max_num > 32768:
            return True

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

    def get_actions(self) -> List[int]:
        return self.actions

    def get_valid_actions(self) -> List[int]:
        """
        返回所有能对当前状态产生变化的动作（0: left, 1: right, 2: up, 3: down）
        """
        valid_actions = []

        def _try_slide(row: List[int]) -> bool:
            """
            对一行进行滑动和合并。
            返回:
                changed: 是否发生变化
            """
            non_zero = [x for x in row if x != 0]
            new_row = []
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged_val = non_zero[i] * 2
                    new_row.append(merged_val)
                    i += 2
                else:
                    new_row.append(non_zero[i])
                    i += 1
            new_row.extend([0] * (self.W - len(new_row)))
            changed = (new_row != row)
            return changed

        # 尝试每个动作
        for action in self.actions:
            changed = False
            if action == 0:  # left
                for i in range(self.H):
                    row_changed = _try_slide(self.state[i])
                    if row_changed:
                        changed = True
                        break

            elif action == 1:  # right
                for i in range(self.H):
                    reversed_row = self.state[i][::-1]
                    row_changed = _try_slide(reversed_row)
                    if row_changed:
                        changed = True
                        break

            elif action == 2:  # up
                for j in range(self.W):
                    col = [self.state[i][j] for i in range(self.H)]
                    col_changed = _try_slide(col)
                    if col_changed:
                        changed = True
                        break

            elif action == 3:  # down
                for j in range(self.W):
                    col = [self.state[i][j] for i in range(self.H)][::-1]
                    col_changed = _try_slide(col)
                    if col_changed:
                        changed = True
                        break

            if changed:
                valid_actions.append(action)

        return valid_actions

    def _slide_and_merge(self, row: List[int]) -> Tuple[List[int], float]:
        """
        对一行进行滑动和合并。
        返回:
            new_row: 合并后的行
            merge_score: 本次合并产生的总分
        """
        non_zero = [x for x in row if x != 0]
        new_row = []
        i = 0
        merge_score = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                new_row.append(merged_val)
                merge_score += math.log2(merged_val)
                if merged_val > self.max_num:
                    self.max_num = merged_val
                    if self.max_num >= 2048:
                        merge_score += (self.max_num/128)
                i += 2
            else:
                new_row.append(non_zero[i])
                i += 1
        new_row.extend([0] * (self.W - len(new_row)))
        return new_row, merge_score

    def _create_onehot(self):
        onehot_state = np.zeros((self.C, self.H, self.W))
        for h in range(self.H):
            for w in range(self.W):
                c = int(math.log2(self.state[h][w])) if self.state[h][w] > 0 else 0
                onehot_state[c, h, w] = 1       # 255 or 1
        return onehot_state

    def step(self, action: int) -> Tuple[np.ndarray, float, int]:
        """
        执行一步动作
        action: 0=left, 1=right, 2=up, 3=down
        返回: (state, reward, done)
        """
        total_reward = 0
        origin_state = copy.deepcopy(self.state)

        if action == 0:  # left
            for i in range(self.H):
                new_row, score = self._slide_and_merge(self.state[i])
                self.state[i] = new_row
                total_reward += score

        elif action == 1:  # right
            for i in range(self.H):
                reversed_row = self.state[i][::-1]
                new_row, score = self._slide_and_merge(reversed_row)
                self.state[i] = new_row[::-1]
                total_reward += score

        elif action == 2:  # up
            for j in range(self.W):
                col = [self.state[i][j] for i in range(self.H)]
                new_col, score = self._slide_and_merge(col)
                for i in range(self.H):
                    self.state[i][j] = new_col[i]
                total_reward += score

        elif action == 3:  # down
            for j in range(self.W):
                col = [self.state[i][j] for i in range(self.H)][::-1]
                new_col, score = self._slide_and_merge(col)
                new_col = new_col[::-1]
                for i in range(self.H):
                    self.state[i][j] = new_col[i]
                total_reward += score

        # 合并生效，需要随机添加新块（add_random_tile不影响reward）
        self._add_random_tile()

        if origin_state == self.state:
            self.invalid_step += 1
            total_reward -= 5

        done = 1 if self.check_game_over() else 0

        return self._create_onehot(), total_reward, done

    def get_sum(self) -> int:
        return sum(sum(row) for row in self.state)

    def get_max(self) -> int:
        return self.max_num
    
    def get_invalid_steps(self) -> int:
        return self.invalid_step

    def get_empty_sum(self):
        return sum(cell == 0 for row in self.state for cell in row)

    def render(self):
        """打印当前状态（便于调试）"""
        print("\n".join(" ".join(f"{cell:4}" if cell else "   ." for cell in row) for row in self.state))
        print("-" * 20)
