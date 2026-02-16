# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn

from ...base import BaseConfig
from ...nn_models import Norm, layernorm, linear, parallel_orthogonal_


class ICVFForwardArchiConfig(BaseConfig):
    name: tp.Literal["ICVFForwardArchi"] = "ICVFForwardArchi"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    norm: bool = True

    def build(self, obs_space, z_dim: int) -> torch.nn.Module:
        """Note: Forward model is also used for critics"""
        return ICVFForwardMap(obs_space, z_dim, self)


class ICVFForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        cfg: ICVFForwardArchiConfig,
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        self.cfg = cfg
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        seq = [linear(z_dim, cfg.hidden_dim, cfg.num_parallel), layernorm(cfg.hidden_dim, cfg.num_parallel), nn.Tanh()]
        for _ in range(cfg.embedding_layers - 2):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        self.embed_z = nn.Sequential(*seq)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, z_dim, cfg.num_parallel)]
        if cfg.norm:
            seq += [Norm()]
        self.Ts = nn.Sequential(*seq)

        self.A = nn.Parameter(torch.empty(cfg.num_parallel, z_dim, z_dim))
        self.B = nn.Parameter(torch.empty(cfg.num_parallel, z_dim, z_dim))
        parallel_orthogonal_(self.A)
        parallel_orthogonal_(self.B)

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        # obs is already embedded
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(z)  # num_parallel x bs x h_dim // 2
        Ts = self.Ts(z_embedding)  # num_parallel x bs x z_dim
        Fs = obs * Ts
        Fs = torch.matmul(Fs, self.A)
        Fs = torch.matmul(Fs, self.B)
        Fs = Fs * Ts
        return Fs
