# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch

from ....base_model import load_model
from ..model import TD3Model, TD3ModelArchiConfig, TD3ModelConfig
from ..nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig


class TD3FlowBCModelArchiConfig(TD3ModelArchiConfig):
    # noise conditioned actor
    actor: NoiseConditionedActorArchiConfig = (
        NoiseConditionedActorArchiConfig()
    )  # pydantic.Field(NoiseConditionedActorArchiConfig(), discriminator="name")
    # vector field
    actor_vf: SimpleVectorFieldArchiConfig = (
        SimpleVectorFieldArchiConfig()
    )  # pydantic.Field(SimpleVectorFieldArchiConfig(), discriminator="name")


class TD3FlowBCModelConfig(TD3ModelConfig):
    name: tp.Literal["TD3FlowBCModel"] = "TD3FlowBCModel"
    archi: TD3FlowBCModelArchiConfig = TD3FlowBCModelArchiConfig()

    @property
    def object_class(self):
        return TD3FlowBCModel


class TD3FlowBCModel(TD3Model):
    def __init__(self, obs_space, action_dim, cfg: TD3FlowBCModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        # For IDEs
        self.cfg: TD3FlowBCModelConfig = cfg

        self._actor_vf = self.cfg.archi.actor_vf.build(self._encoder.output_space, action_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        noises = torch.randn((obs.shape[0], self.action_dim), device=self.device, dtype=torch.float32)
        actions = self._actor(self._encoder(self._normalize(obs)), noises)
        return actions

    def act(self, obs: torch.Tensor, z: None = None, mean: bool = True) -> torch.Tensor:
        del z  # not used
        del mean  # not used
        return self.actor(obs)

    @classmethod
    def load(
        cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None
    ) -> "TD3FlowBCModel":
        return load_model(path, device, strict=strict, config_class=TD3FlowBCModelConfig, build_kwargs=build_kwargs)
