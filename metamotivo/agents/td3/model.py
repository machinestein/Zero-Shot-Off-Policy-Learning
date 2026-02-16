# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp

import pydantic
import torch

from metamotivo.base import BaseConfig
from metamotivo.base_model import BaseModel, BaseModelConfig, load_model, save_model

from ...nn_models import AugmentatorArchiConfig, DrQEncoderArchiConfig, IdentityNNConfig, eval_mode
from ...normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig
from .nn_models import ActorArchiConfig, CriticArchiConfig


class TD3ModelArchiConfig(BaseConfig):
    actor: ActorArchiConfig = ActorArchiConfig()
    critic: CriticArchiConfig = CriticArchiConfig()
    rgb_encoder: IdentityNNConfig | DrQEncoderArchiConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")
    augmentator: IdentityNNConfig | AugmentatorArchiConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")


class TD3ModelConfig(BaseModelConfig):
    name: tp.Literal["TD3Model"] = "TD3Model"
    archi: TD3ModelArchiConfig = TD3ModelArchiConfig()
    actor_std: float = 0.2
    obs_normalizer: AVAILABLE_NORMALIZERS = pydantic.Field(IdentityNormalizerConfig(), discriminator="name")

    def build(self, obs_space, action_dim) -> "TD3Model":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return TD3Model


class TD3Model(BaseModel):
    def __init__(self, obs_space, action_dim, cfg: TD3ModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: TD3ModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device

        # create networks
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)
        self._encoder = arch.rgb_encoder.build(obs_space)
        self._augmentator = arch.augmentator.build(obs_space)
        obs_space = self._encoder.output_space
        self._critic = arch.critic.build(obs_space, action_dim)
        self._actor = arch.actor.build(obs_space, action_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        self._target_critic = copy.deepcopy(self._critic)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor, action: torch.Tensor):
        obs = self._encoder(self._normalize(obs))
        return self._critic(obs, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, std: float):
        obs = self._encoder(self._normalize(obs))
        return self._actor(obs, std)

    def act(self, obs: torch.Tensor, z: None = None, mean: bool = True) -> torch.Tensor:
        del z  # not used
        dist = self.actor(obs, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None) -> "TD3Model":
        return load_model(path, device, strict=strict, config_class=TD3ModelConfig, build_kwargs=build_kwargs)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
