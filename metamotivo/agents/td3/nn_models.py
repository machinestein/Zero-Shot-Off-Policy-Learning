# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp

import torch
from torch import nn

from ...base import BaseConfig
from ...nn_models import TruncatedNormal, linear


class CriticArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    model: str = "simple"  # not used at the moment
    hidden_layers: int = 2
    num_parallel: int = 2
    layer_norm: bool = False
    ensemble_mode: str = "batch"  # not used at the moment

    def build(self, obs_space, action_dim):
        return SimpleCritic(obs_space, action_dim, self)


class SimpleCritic(nn.Module):
    """A critic with a simple MLP architecture, relu non-linearities, and fully parallel layers"""

    def __init__(self, obs_space, action_dim, cfg: CriticArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 1, "obs_space must be 1D box"
        obs_dim = obs_space.shape[0]
        seq = [linear(obs_dim + action_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        if self.cfg.layer_norm:
            seq += [nn.LayerNorm(cfg.hidden_dim)]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
            if self.cfg.layer_norm:
                seq += [nn.LayerNorm(cfg.hidden_dim)]
        seq += [linear(cfg.hidden_dim, 1, cfg.num_parallel)]
        self.Qs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        if self.cfg.num_parallel > 1:
            obs = obs.expand(self.cfg.num_parallel, -1, -1)
            action = action.expand(self.cfg.num_parallel, -1, -1)
        return self.Qs(torch.cat([obs, action], dim=-1))


class ActorArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    model: str = "simple"  # not used at the moment
    hidden_layers: int = 2

    def build(self, obs_space, action_dim):
        return SimpleActor(obs_space, action_dim, self)


class SimpleActor(nn.Module):
    """An actor with a simple MLP architecture and relu non-linearities"""

    def __init__(self, obs_space, action_dim, cfg: ActorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 1, "obs_space must be 1D box"
        obs_dim = obs_space.shape[0]

        seq = [linear(obs_dim, cfg.hidden_dim), nn.ReLU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, std):
        mu = torch.tanh(self.policy(obs))
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


class NoiseConditionedActorArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    model: str = "simple"  # not used at the moment
    hidden_layers: int = 2

    def build(self, obs_space, action_dim):
        return SimpleNoiseConditionedActor(obs_space, action_dim, cfg=self)


class SimpleNoiseConditionedActor(nn.Module):
    """An actor with a simple MLP architecture and relu non-linearities"""

    def __init__(self, obs_space, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()
        self.cfg: NoiseConditionedActorArchiConfig = cfg

        assert len(obs_space.shape) == 1, "obs_space must be 1D box"
        obs_dim = obs_space.shape[0]
        seq = [linear(obs_dim + action_dim, cfg.hidden_dim), nn.ReLU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, noise):
        embedding = torch.cat([obs, noise], dim=-1)
        actions = torch.tanh(self.policy(embedding))
        return actions


class SimpleVectorFieldArchiConfig(BaseConfig):
    # name: tp.Literal["vector_field"] = "vector_field"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1

    def build(self, obs_space, action_dim: int) -> "VectorField":
        return VectorField(obs_space, action_dim, self)


class VectorField(nn.Module):
    def __init__(self, obs_space, action_dim, cfg: SimpleVectorFieldArchiConfig) -> None:
        super().__init__()
        self.cfg: SimpleVectorFieldArchiConfig = cfg

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        # plus 1 is for time
        seq = [linear(obs_dim + action_dim + 1, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = torch.cat([obs, action, t], dim=-1)
        return self.net(embedding)
