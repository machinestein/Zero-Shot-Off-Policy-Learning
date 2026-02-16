# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import typing as tp

import gymnasium
import numpy as np
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


class TDJEPAModelArchiConfig(BaseConfig):
    phi_dim: int = 50
    psi_dim: int = 50
    norm_z: bool = True
    # convolutional part of the encoders
    rgb_encoder: IdentityNNConfig | DrQEncoderArchiConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")
    augmentator: IdentityNNConfig | AugmentatorArchiConfig = pydantic.Field(IdentityNNConfig(), discriminator="name")
    # Pred(phi, a, z) --> psi
    phi_predictor: ForwardArchiConfig = ForwardArchiConfig()
    # Pred(psi, a, z) --> phi
    psi_predictor: ForwardArchiConfig = ForwardArchiConfig()
    # the following define the MLP head of the phi and psi encoders
    # the full encoders are given by
    #    - phi_mlp_encoder(phi_rgb_encoder(obs))
    #    - psi_mlp_encoder(psi_rgb_encoder(obs))
    phi_mlp_encoder: BackwardArchiConfig = BackwardArchiConfig()
    psi_mlp_encoder: BackwardArchiConfig = BackwardArchiConfig()
    actor: SimpleActorArchiConfig = pydantic.Field(SimpleActorArchiConfig(), discriminator="name")


class TDJEPAModelConfig(BaseModelConfig):
    name: tp.Literal["TDJEPAModel"] = "TDJEPAModel"

    archi: TDJEPAModelArchiConfig = TDJEPAModelArchiConfig()
    obs_normalizer: AVAILABLE_NORMALIZERS = pydantic.Field(IdentityNormalizerConfig(), discriminator="name")
    actor_std: float = 0.2
    # if True, the actor takes as input the output of phi_mlp_encoder(phi_rgb_encoder(obs))
    # if False, the actor takes as input the output of phi_rgb_encoder(obs)
    actor_use_full_encoder: bool = True
    # if True, use a single encoder for phi and psi (in which case only the configs for phi are used)
    # if False, phi and psi use different mlp and rgb networks
    symmetric: bool = False

    def build(self, obs_space, action_dim) -> "TDJEPAModel":
        return TDJEPAModel(obs_space, action_dim, self)


class TDJEPAModel(BaseModel):
    def __init__(self, obs_space, action_dim, cfg: TDJEPAModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: TDJEPAModelConfig = cfg
        arch: TDJEPAModelArchiConfig = self.cfg.archi
        self.device = self.cfg.device

        # create networks
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)
        self._augmentator = arch.augmentator.build(obs_space)
        self.z_dim = arch.phi_dim if self.cfg.symmetric else arch.psi_dim

        # phi encoder and predictor
        phi_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(arch.phi_dim,), dtype=np.float32)
        self._phi_rgb_encoder = arch.rgb_encoder.build(obs_space)
        phi_obs_space = self._phi_rgb_encoder.output_space
        self._phi_mlp_encoder = arch.phi_mlp_encoder.build(phi_obs_space, arch.phi_dim)
        self._phi_predictor = arch.phi_predictor.build(phi_space, self.z_dim, action_dim, output_dim=self.z_dim)
        self._actor_input_space = phi_space if self.cfg.actor_use_full_encoder else phi_obs_space
        self._actor = arch.actor.build(self._actor_input_space, self.z_dim, action_dim)

        if not self.cfg.symmetric:
            # psi encoder and predictor
            psi_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(arch.psi_dim,), dtype=np.float32)
            self._psi_rgb_encoder = arch.rgb_encoder.build(obs_space)
            psi_obs_space = self._psi_rgb_encoder.output_space
            self._psi_mlp_encoder = arch.psi_mlp_encoder.build(psi_obs_space, arch.psi_dim)
            self._psi_predictor = arch.psi_predictor.build(psi_space, self.z_dim, action_dim, output_dim=arch.phi_dim)

        self.register_buffer("_z_cov", torch.eye(self.z_dim))

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        self._target_phi_mlp_encoder = copy.deepcopy(self._phi_mlp_encoder)
        self._target_phi_predictor = copy.deepcopy(self._phi_predictor)
        if not self.cfg.symmetric:
            self._target_psi_mlp_encoder = copy.deepcopy(self._psi_mlp_encoder)
            self._target_psi_predictor = copy.deepcopy(self._psi_predictor)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def _update_z_stats(self, z: torch.Tensor):
        self._z_cov.mul_(0.995).add_(0.005 * torch.matmul(z.T, z) / z.shape[0])

    @torch.no_grad()
    def phi(self, obs: torch.Tensor):
        return self._phi_mlp_encoder(self._phi_rgb_encoder(self._normalize(obs)))

    @torch.no_grad()
    def psi(self, obs: torch.Tensor):
        if self.cfg.symmetric:
            return self.phi(obs)
        return self._psi_mlp_encoder(self._psi_rgb_encoder(self._normalize(obs)))

    @torch.no_grad()
    def phi_predictor(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._phi_predictor(self.phi(obs), z, action)

    @torch.no_grad()
    def psi_predictor(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._psi_predictor(self.psi(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        actor_in = self.phi(obs) if self.cfg.actor_use_full_encoder else self._phi_rgb_encoder(self._normalize(obs))
        return self._actor(actor_in, z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
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
        next_obs = next_obs.to(self.device)
        reward = reward.to(self.device)
        z = torch.linalg.lstsq(self.psi(next_obs), reward).solution.T
        return self.project_z(z)

    @classmethod
    def load(
        cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None
    ) -> "TDJEPAModel":
        return load_model(path, device, strict=strict, config_class=TDJEPAModelConfig, build_kwargs=build_kwargs)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
