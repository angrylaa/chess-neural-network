# src/policy_model.py

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class PolicyNet(nn.Module):
    """
    Policy network mapping [B, 18, 8, 8] → [B, action_size] logits.

    - in_channels: 18 (12 piece planes + 6 context planes)
    - channels: internal conv width
    - num_blocks: residual depth
    - action_size: number of UCI moves in vocabulary (len(move2idx))
    """

    def __init__(
        self,
        action_size: int,
        in_channels: int = 18,
        channels: int = 128,
        num_blocks: int = 5,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Flatten(),             # [B, C]
            nn.Linear(channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, action_size),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 18, 8, 8]
        returns: [B, action_size] logits
        """
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.policy_head(x)
        return x