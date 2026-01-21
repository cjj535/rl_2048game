import torch.nn as nn


class QNetwork(nn.Module):
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

        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

  
    def forward(self, x):
        # x: [batch, 16, 4, 4]
        _, C, H, W = x.shape
        assert C == 16 and H == 4 and W == 4
        logits = self.fc(self.conv(x))
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
        _, C, H, W = x.shape
        assert C == 16 and H == 4 and W == 4
        features = self.shared(self.conv(x))
        logits = self.actor(features)
        value = self.critic(features)

        return logits, value
