# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch

from ..sf.agent import SFAgent, SFAgentConfig, SFAgentTrainConfig
from ..sf.model import SFModelConfig


class LaplacianAgentTrainConfig(SFAgentTrainConfig):
    ortho_coef: float = 1.0


class LaplacianAgentConfig(SFAgentConfig):
    name: Literal["LaplacianAgent"] = "LaplacianAgent"
    model: SFModelConfig
    train: LaplacianAgentTrainConfig

    @property
    def object_class(self):
        return LaplacianAgent


class LaplacianAgent(SFAgent):
    config_class = LaplacianAgentConfig

    def __init__(self, obs_space, action_dim, cfg: LaplacianAgentConfig):
        super().__init__(obs_space, action_dim, cfg)

    def enc_for_features(self, obs, next_obs, future_obs):
        obs = self._model._rgb_encoder(obs)
        next_obs = self._model._rgb_encoder(next_obs)
        return obs, next_obs, None

    def feature_loss(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor, discount: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        phi = self._model._features(obs)
        next_phi = self._model._features(next_obs)
        lap_loss = (phi - next_phi).pow(2).mean()
        # the loss is not scaled by 0.5, compared to FB
        Cov = torch.matmul(phi, phi.T)
        Id = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~Id.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        phi_loss = lap_loss + self.cfg.train.ortho_coef * orth_loss

        with torch.no_grad():
            metrics = {"phi_loss": phi_loss, "laplacian_loss": lap_loss, "orth_loss": orth_loss}

        return phi_loss, metrics
