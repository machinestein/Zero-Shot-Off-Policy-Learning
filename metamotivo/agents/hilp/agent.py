# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch

from ...nn_models import _soft_update_params
from ..sf.agent import SFAgent, SFAgentConfig, SFAgentTrainConfig
from .model import HilpModelConfig


class HilpAgentTrainConfig(SFAgentTrainConfig):
    expectile: float = 0.5


class HilpAgentConfig(SFAgentConfig):
    name: Literal["HilpAgent"] = "HilpAgent"
    model: HilpModelConfig
    train: HilpAgentTrainConfig

    @property
    def object_class(self):
        return HilpAgent


class HilpAgent(SFAgent):
    config_class = HilpAgentConfig

    def __init__(self, obs_space, action_dim, cfg: HilpAgentConfig):
        super().__init__(obs_space, action_dim, cfg)

    def setup_training(self) -> None:
        super().setup_training()

        self._features2_paramlist = tuple(x for x in self._model._features2.parameters())
        self._target_features2_paramlist = tuple(x for x in self._model._target_features2.parameters())

        self.features_optimizer = torch.optim.Adam(
            list(self._model._features.parameters()) + list(self._model._features2.parameters()),
            lr=self.cfg.train.lr_features,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    def update_target_networks(self):
        super().update_target_networks()
        with torch.no_grad():
            _soft_update_params(self._features2_paramlist, self._target_features2_paramlist, self.cfg.train.features_target_tau)

    def value(self, obs: torch.Tensor, goals: torch.Tensor, is_target: bool = False):
        if is_target:
            phi1 = self._model._target_features
            phi2 = self._model._target_features2
        else:
            phi1 = self._model._features
            phi2 = self._model._features2

        phi1_s = phi1(obs)
        phi1_g = phi1(goals)

        phi2_s = phi2(obs)
        phi2_g = phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def enc_for_features(self, obs, next_obs, future_obs):
        with torch.no_grad():
            next_obs = self._model._rgb_encoder(next_obs)
        obs = self._model._rgb_encoder(obs)
        future_obs = self._model._rgb_encoder(future_obs)
        return obs, next_obs, future_obs

    def _reward(self, obs, goals):
        return (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float()

    def feature_loss(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor, discount: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        goals = future_obs
        rewards = self._reward(obs, goals)
        masks = 1.0 - rewards
        rewards = rewards - 1.0

        next_v1, next_v2 = self.value(next_obs, goals, is_target=True)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.cfg.train.discount * masks * next_v

        v1_t, v2_t = self.value(obs, goals, is_target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.cfg.train.discount * masks * next_v1
        q2 = rewards + self.cfg.train.discount * masks * next_v2
        v1, v2 = self.value(obs, goals, is_target=False)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.cfg.train.expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.cfg.train.expectile).mean()
        value_loss = value_loss1 + value_loss2

        with torch.no_grad():
            metrics = {
                "value_loss": value_loss,
                "v_mean": v.mean(),
                "v_max": v.max(),
                "v_min": v.min(),
                "abs_adv_mean": torch.abs(adv).mean(),
                "adv_mean": adv.mean(),
                "adv_max": adv.max(),
                "adv_min": adv.min(),
                "accept_prob": (adv >= 0).float().mean(),
            }

        return value_loss, metrics
