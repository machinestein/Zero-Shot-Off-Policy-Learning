# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import gymnasium
import numpy as np
import torch

from ....nn_models import NoiseConditionedActorArchiConfig
from ...td3.nn_models import SimpleVectorFieldArchiConfig
from ..model import SFModel, SFModelArchiConfig, SFModelConfig


class SFFlowBCModelArchiConfig(SFModelArchiConfig):
    actor: NoiseConditionedActorArchiConfig = NoiseConditionedActorArchiConfig()
    actor_vf: SimpleVectorFieldArchiConfig = SimpleVectorFieldArchiConfig()


class SFFlowBCModelConfig(SFModelConfig):
    name: tp.Literal["SFFlowBCModel"] = "SFFlowBCModel"
    archi: SFFlowBCModelArchiConfig = SFFlowBCModelArchiConfig()

    @property
    def object_class(self):
        return SFFlowBCModel


class SFFlowBCModelMixin:
    def __init__(self, obs_space, action_dim, cfg: SFModelConfig):
        obs_space = (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.archi.L_dim,), dtype=np.float32)
            if cfg.actor_encode_obs
            else self._sf_encoder.output_space
        )
        self._actor_vf = cfg.archi.actor_vf.build(obs_space, action_dim)
        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        obs = self._sf_encoder(self._normalize(obs))
        obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        return self._actor(obs, z, noises)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean  # not used
        return self.actor(obs, z)


class SFFlowBCModel(SFFlowBCModelMixin, SFModel):
    config_class = SFFlowBCModelConfig

    def __init__(self, obs_space, action_dim, cfg: SFModelConfig):
        SFModel.__init__(self, obs_space, action_dim, cfg)
        SFFlowBCModelMixin.__init__(self, obs_space, action_dim, cfg)
