import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, channel_size: int = 128):
        super().__init__()
        self.channel_size = channel_size

        # 第一层：从 input_x 提取 (1,2) 和 (2,1) 卷积
        self.conv1_1 = nn.Conv2d(16, channel_size, kernel_size=(1, 2), padding=0)
        self.conv1_2 = nn.Conv2d(16, channel_size, kernel_size=(2, 1), padding=0)

        # 第二层：对 conv1_1 的输出再做 (1,2) 和 (2,1)
        self.conv2_1_from_1 = nn.Conv2d(channel_size, channel_size, kernel_size=(1, 2), padding=0)
        self.conv2_2_from_1 = nn.Conv2d(channel_size, channel_size, kernel_size=(2, 1), padding=0)

        # 第二层：对 conv1_2 的输出再做 (1,2) 和 (2,1)
        self.conv2_1_from_2 = nn.Conv2d(channel_size, channel_size, kernel_size=(1, 2), padding=0)
        self.conv2_2_from_2 = nn.Conv2d(channel_size, channel_size, kernel_size=(2, 1), padding=0)

        # 直接从 input_x 做更深的单层卷积（对应 conv1_r, conv2_r）
        self.conv1_r = nn.Conv2d(16, channel_size, kernel_size=(1, 2), padding=0)
        self.conv2_r = nn.Conv2d(16, channel_size, kernel_size=(2, 1), padding=0)

        # 全连接层
        self.fc1 = nn.Linear(self._get_flattened_size(), 512)
        self.fc_out = nn.Linear(512, 4)  # 4 个动作

    def _get_flattened_size(self):
        """
        计算拼接后的特征维度（根据原始 TF 代码的 reshape 尺寸）
        原始输入：假设为 (B, 16, 4, 4) —— one-hot 16通道，4x4网格
        """
        # 模拟前向计算各分支输出尺寸
        B = 1
        x = torch.zeros(B, 16, 4, 4)

        c1 = F.leaky_relu(self.conv1_1(x))  # (B, C, 4, 3)
        c2 = F.leaky_relu(self.conv1_2(x))  # (B, C, 3, 4)

        c3 = F.leaky_relu(self.conv2_1_from_1(c1))  # (1,2) on (4,3) → (4,2) → flatten: 4*2*C = 8C
        c4 = F.leaky_relu(self.conv2_2_from_1(c1))  # (2,1) on (4,3) → (3,3) → 3*3*C = 9C
        c5 = F.leaky_relu(self.conv2_1_from_2(c2))  # (1,2) on (3,4) → (3,3) → 9C
        c6 = F.leaky_relu(self.conv2_2_from_2(c2))  # (2,1) on (3,4) → (2,4) → 2*4*C = 8C

        cr1 = F.leaky_relu(self.conv1_r(x))  # (1,2) → (4,3) → 但原代码 reshape to 12*C → 4*3=12
        cr2 = F.leaky_relu(self.conv2_r(x))  # (2,1) → (3,4) → 3*4=12 → 12*C

        total = (
            c3.view(B, -1).size(1) +
            c4.view(B, -1).size(1) +
            c5.view(B, -1).size(1) +
            c6.view(B, -1).size(1) +
            cr1.view(B, -1).size(1) +
            cr2.view(B, -1).size(1)
        )
        return total

    def forward(self, x):
        # x: (B, 16, 4, 4) —— one-hot encoded board
        _, C, H, W = x.shape
        assert C==16 and H==4 and W==4

        # First-level convs
        conv1 = F.leaky_relu(self.conv1_1(x))  # (B, C, 4, 3)
        conv2 = F.leaky_relu(self.conv1_2(x))  # (B, C, 3, 4)

        # Second-level from conv1
        conv3 = F.leaky_relu(self.conv2_1_from_1(conv1))  # (B, C, 4, 2) → 8*C
        conv4 = F.leaky_relu(self.conv2_2_from_1(conv1))  # (B, C, 3, 3) → 9*C

        # Second-level from conv2
        conv5 = F.leaky_relu(self.conv2_1_from_2(conv2))  # (B, C, 3, 3) → 9*C
        conv6 = F.leaky_relu(self.conv2_2_from_2(conv2))  # (B, C, 2, 4) → 8*C

        # Direct deeper single convs (like residual but not really)
        conv1_r = F.leaky_relu(self.conv1_r(x))  # (B, C, 4, 3) → 12*C
        conv2_r = F.leaky_relu(self.conv2_r(x))  # (B, C, 3, 4) → 12*C

        # Flatten each
        conv3 = conv3.view(conv3.size(0), -1)  # [B, 8*C]
        conv4 = conv4.view(conv4.size(0), -1)  # [B, 9*C]
        conv5 = conv5.view(conv5.size(0), -1)  # [B, 9*C]
        conv6 = conv6.view(conv6.size(0), -1)  # [B, 8*C]
        conv1_r = conv1_r.view(conv1_r.size(0), -1)  # [B, 12*C]
        conv2_r = conv2_r.view(conv2_r.size(0), -1)  # [B, 12*C]

        # Concatenate all
        conv_res = torch.cat([conv1_r, conv2_r, conv3, conv4, conv5, conv6], dim=1)  # [B, (12+12+8+9+9+8)*C] = [B, 58*C]

        # Fully connected layers
        layer1 = F.leaky_relu(self.fc1(conv_res))
        logits = self.fc_out(layer1)  # [B, 4]

        return logits


class ActorCritic(nn.Module):
    def __init__(self, action_dim=4, input_channel=16):
        super().__init__()

        # 输入: (batch, 16, 4, 4)
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channel, 32, kernel_size=1, stride=1, padding=0, bias=False
            ),                                                                  # (32, 4, 4)
            nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),             # (256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),            # (256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                                       # 256*4*4 = 4096
        )

        # 共享特征
        self.shared = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: [batch, 16, 4, 4]
        B, C, H, W = x.shape
        assert C == 16 and H == 4 and W == 4
        features = self.shared(self.conv(x))
        logits = self.actor(features)
        value = self.critic(features).view(B)

        return logits, value
