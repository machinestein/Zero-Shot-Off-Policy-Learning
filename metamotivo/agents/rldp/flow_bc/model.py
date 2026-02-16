# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import gymnasium
import numpy as np
import pydantic
import torch
from torch.amp import autocast

from ....base_model import load_model
from ....nn_models import NoiseConditionedActorArchiConfig
from ...td3.nn_models import SimpleVectorFieldArchiConfig
from ..model import RLDPModel, RLDPModelArchiConfig, RLDPModelConfig


class RLDPFlowBCModelArchiConfig(RLDPModelArchiConfig):
    # noise conditioned actor
    actor: NoiseConditionedActorArchiConfig = pydantic.Field(NoiseConditionedActorArchiConfig(), discriminator="name")
    # vector field
    actor_vf: SimpleVectorFieldArchiConfig = SimpleVectorFieldArchiConfig()


class RLDPFlowBCModelConfig(RLDPModelConfig):
    name: tp.Literal["RLDPFlowBCModel"] = "RLDPFlowBCModel"
    archi: RLDPFlowBCModelArchiConfig = RLDPFlowBCModelArchiConfig()

    @property
    def object_class(self):
        return RLDPFlowBCModel


class RLDPFlowBCModel(RLDPModel):
    def __init__(self, obs_space, action_dim, cfg: RLDPFlowBCModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        # For IDEs
        self.cfg: RLDPFlowBCModelConfig = cfg

        obs_space = (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.archi.L_dim,), dtype=np.float32)
            if self.cfg.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = self.cfg.archi.actor_vf.build(obs_space, action_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
            noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
            actions = self._actor(obs, z, noises)
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean  # not used
        return self.actor(obs, z)

    @classmethod
    def load(
        cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None
    ) -> "RLDPFlowBCModel":
        return load_model(path, device, strict=strict, config_class=RLDPFlowBCModelConfig, build_kwargs=build_kwargs)
