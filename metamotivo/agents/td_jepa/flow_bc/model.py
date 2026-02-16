# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch

from ....base_model import load_model
from ....nn_models import NoiseConditionedActorArchiConfig
from ...td3.nn_models import SimpleVectorFieldArchiConfig
from ..model import TDJEPAModel, TDJEPAModelArchiConfig, TDJEPAModelConfig


class TDJEPAFlowBCModelArchiConfig(TDJEPAModelArchiConfig):
    # noise conditioned actor
    actor: NoiseConditionedActorArchiConfig = NoiseConditionedActorArchiConfig()
    # vector field
    actor_vf: SimpleVectorFieldArchiConfig = SimpleVectorFieldArchiConfig()


class TDJEPAFlowBCModelConfig(TDJEPAModelConfig):
    name: tp.Literal["TDJEPAFlowBCModel"] = "TDJEPAFlowBCModel"
    archi: TDJEPAFlowBCModelArchiConfig = TDJEPAFlowBCModelArchiConfig()

    def build(self, obs_space, action_dim):
        return TDJEPAFlowBCModel(obs_space, action_dim, self)

    @property
    def object_class(self):
        return TDJEPAFlowBCModel


class TDJEPAFlowBCModel(TDJEPAModel):
    def __init__(self, obs_space, action_dim, cfg: TDJEPAFlowBCModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        # For IDEs
        self.cfg: TDJEPAFlowBCModelConfig = cfg

        self._actor_vf = self.cfg.archi.actor_vf.build(self._actor_input_space, action_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        actor_in = self.phi(obs) if self.cfg.actor_use_full_encoder else self._phi_rgb_encoder(self._normalize(obs))
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        actions = self._actor(actor_in, z, noises)
        return actions

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        del mean  # not used
        return self.actor(obs, z)

    @classmethod
    def load(
        cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None
    ) -> "TDJEPAFlowBCModel":
        return load_model(path, device, strict=strict, config_class=TDJEPAFlowBCModelConfig, build_kwargs=build_kwargs)
