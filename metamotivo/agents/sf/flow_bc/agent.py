# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch

from ..agent import SFAgent, SFAgentConfig, SFAgentTrainConfig
from .model import SFFlowBCModelConfig


class SFFlowBCAgentTrainConfig(SFAgentTrainConfig):
    flow_steps: int = 10
    lr_actor_vf: float = 3e-4


class SFFlowBCAgentConfig(SFAgentConfig):
    name: Literal["SFFlowBCAgent"] = "SFFlowBCAgent"
    model: SFFlowBCModelConfig
    train: SFFlowBCAgentTrainConfig

    @property
    def object_class(self):
        return SFFlowBCAgent


class SFFlowBCAgentMixin:
    @property
    def optimizer_dict(self):
        return {"actor_vf_optimizer": self.actor_vf_optimizer.state_dict()}

    def setup_training(self) -> None:
        self.actor_vf_optimizer = torch.optim.Adam(
            self._model._actor_vf.parameters(),
            lr=self.cfg.train.lr_actor_vf,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        noises = torch.randn((z.shape[0], self.action_dim), device=z.device, dtype=z.dtype)
        action = self._model._actor(obs, z, noises)
        return action

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x_1 = action
        x_0 = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
        t = torch.rand((x_1.shape[0], 1), device=action.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        # flow matching l2 loss
        pred = self._model._actor_vf(obs, x_t, t)
        bc_flow_loss = torch.pow(pred - vel, 2).mean()

        # Q loss.
        with torch.no_grad():
            left_enc = self._model._left_encoder(obs)
        actor_in = left_enc if self.cfg.model.actor_encode_obs else obs
        noises = torch.randn_like(x_1, device=action.device, dtype=action.dtype)
        actor_actions = self._model._actor(actor_in, z, noises)
        Fs = self._model._sf(left_enc, z, actor_actions)  # num_parallel x batch x z_dim
        Qs = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
        actor_loss = -Q.mean()

        # compute bc loss
        bc_loss = torch.tensor([0.0], device=action.device)
        if self.cfg.train.bc_coeff > 0:
            with torch.no_grad():
                target_flow_actions = self.compute_flow_actions(obs, noises)
            bc_error = torch.pow(actor_actions - target_flow_actions, 2).mean()
            bc_loss = self.cfg.train.bc_coeff * bc_error
            actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        actor_loss = actor_loss + bc_flow_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor_vf_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_vf_optimizer.step()

        metrics = {
            "actor_loss": actor_loss.mean().detach(),
            "bc_flow_loss": bc_flow_loss.detach(),
            "bc_error": bc_error.detach(),
            "q": Q.mean().detach(),
        }
        return metrics

    def compute_flow_actions(self, obs: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        actions = noises
        for i in range(self.cfg.train.flow_steps):
            t = torch.ones((noises.shape[0], 1), device=noises.device) * i / self.cfg.train.flow_steps
            vels = self._model._actor_vf(obs, actions, t)
            actions = actions + vels / self.cfg.train.flow_steps
        actions = torch.clamp(actions, min=-1, max=1)
        return actions


class SFFlowBCAgent(SFFlowBCAgentMixin, SFAgent):
    config_class = SFFlowBCAgentConfig

    @property
    def optimizer_dict(self):
        return super(SFAgent, self).optimizer_dict | super(SFFlowBCAgent, self).optimizer_dict

    def setup_training(self) -> None:
        SFAgent.setup_training(self)
        SFFlowBCAgentMixin.setup_training(self)
