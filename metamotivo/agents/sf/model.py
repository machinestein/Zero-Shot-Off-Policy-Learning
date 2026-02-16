# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import typing as tp

import pydantic
import torch
import torch.nn.functional as F

from metamotivo.base import BaseConfig
from metamotivo.base_model import BaseModel, BaseModelConfig

from ...base_model import load_model, save_model
from ...nn_models import (
    AugmentatorArchiConfig,
    BackwardArchiConfig,
    DrQEncoderArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from ...normalizers import AVAILABLE_NORMALIZERS, IdentityNormalizerConfig


class SFModelArchiConfig(BaseConfig):
    L_dim: int = 100
    z_dim: int = 100
    norm_z: bool = True
    successor_features: ForwardArchiConfig = ForwardArchiConfig()
    features: BackwardArchiConfig = BackwardArchiConfig()
    actor: SimpleActorArchiConfig = pydantic.Field(SimpleActorArchiConfig(), discriminator="name")
    left_encoder: BackwardArchiConfig | IdentityNNConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")
    # a shared image encoder config that is used for all networks
    rgb_encoder: IdentityNNConfig | DrQEncoderArchiConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")
    augmentator: IdentityNNConfig | AugmentatorArchiConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")


class SFModelConfig(BaseModelConfig):
    name: tp.Literal["SFModel"] = "SFModel"

    archi: SFModelArchiConfig = SFModelArchiConfig()
    obs_normalizer: AVAILABLE_NORMALIZERS = pydantic.Field(IdentityNormalizerConfig(), discriminator="name")
    actor_std: float = 0.2
    # if True, the learned features are first centered by subtracting their running mean before being passed to SFs
    center_features: bool = False
    actor_encode_obs: bool = True

    def build(self, obs_space, action_dim) -> "SFModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return SFModel


class SFModel(BaseModel):
    config_class = SFModelConfig

    def __init__(self, obs_space, action_dim, cfg: SFModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: SFModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device

        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)
        self._rgb_encoder = arch.rgb_encoder.build(obs_space)
        self._augmentator = arch.augmentator.build(obs_space)
        self._sf_encoder = arch.rgb_encoder.build(obs_space)
        self._left_encoder = arch.left_encoder.build(self._sf_encoder.output_space, arch.L_dim)

        # NOTE: self._features must be accessed *only* by the feature learning algorithm
        # all other parts of the agent (eg SF learning, z mixing) must use self.features()
        # this is to make sure we apply feature centering correctly
        self._features = arch.features.build(self._rgb_encoder.output_space, arch.z_dim)
        self._sf = arch.successor_features.build(self._left_encoder.output_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(
            self._left_encoder.output_space if self.cfg.actor_encode_obs else self._sf_encoder.output_space, arch.z_dim, action_dim
        )
        self.register_buffer("features_mean", torch.zeros(arch.z_dim))
        self.register_buffer("_z_cov", torch.eye(arch.z_dim))

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        self._target_sf = copy.deepcopy(self._sf)
        self._target_features = copy.deepcopy(self._features)
        self._target_left_encoder = copy.deepcopy(self._left_encoder)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def features(self, obs: torch.Tensor, norm_obs: bool = True, encode_image: bool = True):
        if norm_obs:
            obs = self._normalize(obs)
        if encode_image:
            obs = self._rgb_encoder(obs)
        phi = self._features(obs)
        if self.cfg.center_features:
            return phi - self.features_mean
        return phi

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        obs = self._sf_encoder(self._normalize(obs))
        obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
        return self._actor(obs, z, std)

    @torch.no_grad()
    def _update_features_stats(self, obs: torch.Tensor):
        # Must be called only by the training algorithm
        # NOTE: if cfg.norm_obs=True, this expects obs to be already normalized
        # NOTE: when training from pixels, obs must also be already encoded
        phi = self._features(obs)
        self.features_mean = 0.995 * self.features_mean + 0.005 * phi.mean(dim=0)
        self._z_cov.mul_(0.995).add_(0.005 * torch.matmul(phi.T, phi) / phi.shape[0])

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    def reward_inference(self, next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        phi = self.features(next_obs)
        z = torch.linalg.lstsq(phi, reward).solution.T
        return self.project_z(z)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None) -> "SFModel":
        return load_model(path, device, strict=strict, config_class=cls.config_class, build_kwargs=build_kwargs)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
