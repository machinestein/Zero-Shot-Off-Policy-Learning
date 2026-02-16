# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch

from ..sf.agent import SFAgent, SFAgentConfig, SFAgentTrainConfig
from .model import BYOLModelConfig


class BYOLAgentTrainConfig(SFAgentTrainConfig):
    lr_predictor: float = 1e-4
    ortho_coef: float = 1.0
    # if True, sample future obs from the successor distribution
    # if False, just use the next observation
    multi_step: bool = False


class BYOLAgentConfig(SFAgentConfig):
    name: Literal["BYOLAgent"] = "BYOLAgent"
    model: BYOLModelConfig
    train: BYOLAgentTrainConfig

    @property
    def object_class(self):
        return BYOLAgent


class BYOLAgent(SFAgent):
    config_class = BYOLAgentConfig

    @property
    def optimizer_dict(self):
        return {**super().optimizer_dict, "predictor_optimizer": self.predictor_optimizer.state_dict()}

    def setup_training(self) -> None:
        super().setup_training()

        self.predictor_optimizer = torch.optim.Adam(
            self._model._predictor.parameters(),
            lr=self.cfg.train.lr_predictor,
            capturable=False,
            weight_decay=self.cfg.train.weight_decay,
        )

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    def enc_for_features(self, obs, next_obs, future_obs):
        with torch.no_grad():
            next_obs = self._model._rgb_encoder(next_obs)
            future_obs = self._model._rgb_encoder(future_obs) if self.cfg.train.multi_step else None
        obs = self._model._rgb_encoder(obs)
        return obs, next_obs, future_obs

    def feature_loss(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor, discount: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            target_obs = future_obs if self.cfg.train.multi_step else next_obs
            next_phi = self._model._target_features(target_obs)  # batch x z_dim

        phi = self._model._features(obs)  # batch x z_dim
        preds = self._model._predictor(phi, action)  # num_parallel x batch x z_dim

        sp_loss = (preds - next_phi).pow(2).sum(-1).mean()

        # compute orthonormality loss for encoder
        Cov = torch.matmul(phi, phi.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag

        total_loss = sp_loss + self.cfg.train.ortho_coef * orth_loss

        with torch.no_grad():
            output_metrics = {
                "phi": phi.mean(),
                "phi_norm": torch.norm(phi, dim=-1).mean(),
                "sp_loss": sp_loss,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "total_loss": total_loss,
            }
        return total_loss, output_metrics

    def optimizers_zero_grad(self):
        self.predictor_optimizer.zero_grad(set_to_none=True)
        super().optimizers_zero_grad()

    def optimizers_step(self):
        self.predictor_optimizer.step()
        super().optimizers_step()
