# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))
