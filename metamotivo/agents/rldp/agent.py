# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch.amp import autocast

from ...nn_models import _soft_update_params, eval_mode
from ..fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig
from .model import RLDPModelConfig


class RLDPAgentTrainConfig(FBAgentTrainConfig):
    pass


class RLDPAgentConfig(FBAgentConfig):
    name: Literal["RLDPAgent"] = "RLDPAgent"
    model: RLDPModelConfig
    train: FBAgentTrainConfig

    @property
    def object_class(self):
        return RLDPAgent


class RLDPAgent(FBAgent):
    config_class = RLDPAgentConfig

    def setup_training(self) -> None:
        super().setup_training()
        self.backward_optimizer = torch.optim.Adam(
            list(self._model._backward_map.parameters())
            + list(self._model._bw_encoder.parameters())
            + list(self._model._predictor.parameters()),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    @torch.no_grad()
    def aug(self, obs, next_obs, future_obs):
        """
        Augments observations when training from pixels, does nothing otherwise.
        """
        return (
            self._model._augmentator(obs),
            self._model._augmentator(next_obs),
            self._model._augmentator(future_obs) if future_obs is not None else future_obs,
        )

    def enc(self, obs, next_obs, future_obs):
        """
        Encodes observations when training from pixels, does nothing otherwise.
        """
        obs, obs_bw = self._model._fw_encoder(obs), self._model._bw_encoder(obs)
        with torch.no_grad():
            goal = self._model._bw_encoder(next_obs)
            next_obs = self._model._fw_encoder(next_obs)
            future_obs = self._model._bw_encoder(future_obs) if future_obs is not None else future_obs
        return obs, obs_bw, next_obs, goal, future_obs

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device),
            batch["action"].to(self.device),
            batch["next"]["observation"].to(self.device),
            batch["next"]["terminated"].to(self.device),
        )
        discount = self.cfg.train.discount * ~terminated

        future_obs, future_act = None, None
        if "traj_action" in batch:
            future_obs = batch["next"]["traj_observation"]
            future_obs = future_obs.reshape(-1, *future_obs.shape[2:])
            future_act = batch["next"]["traj_action"]

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)
            future_obs = self._model._obs_normalizer(future_obs) if future_obs is not None else future_obs

        torch.compiler.cudagraph_mark_step_begin()

        obs, next_obs, future_obs = self.aug(obs, next_obs, future_obs)
        obs, obs_bw, next_obs, goal, future_obs = self.enc(obs, next_obs, future_obs)

        z = self.sample_mixed_z(train_goal=goal).clone()

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics = self.update_fb(
            obs=obs,
            obs_bw=obs_bw,
            action=action,
            future_act=future_act,
            discount=discount,
            next_obs=next_obs,
            future_obs=future_obs,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(
            self.update_actor(
                obs=obs.detach(),
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.cfg.train.f_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.cfg.train.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.cfg.train.f_target_tau)

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor,
        obs_bw: torch.Tensor,
        action: torch.Tensor,
        future_act: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                next_left_enc = self._model._target_left_encoder(next_obs)  # batch x L_dim
                actor_in = next_left_enc if self.cfg.model.actor_encode_obs else next_obs
                next_action = self.sample_action_from_norm_obs(actor_in, z)
                target_Fs = self._model._target_forward_map(next_left_enc, z, next_action)  # num_parallel x batch x z_dim
                target_B = self._model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

            # compute FB loss
            left_enc = self._model._left_encoder(obs)  # batch x L_dim
            Fs = self._model._forward_map(left_enc, z, action)  # num_parallel x batch x z_dim
            # important! do not update B through the contrastive loss
            B = self._model._backward_map(goal).detach()  # batch x z_dim
            Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

            diff = Ms - discount * target_M  # num_parallel x batch x batch
            fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
            fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
            fb_loss = fb_offdiag + fb_diag

            # self-predictive loss
            with torch.no_grad():
                targets = [self._model._target_backward_map(next_obs)]
                actions = [action]
                if future_obs is not None:
                    future_phi = self._model._target_backward_map(future_obs)
                    future_phi = future_phi.reshape(*future_act.shape[:2], -1)  # batch_size x horizon x z_dim
                    targets += [future_phi[:, i] for i in range(future_phi.shape[1])]
                    actions += [future_act[:, i] for i in range(future_act.shape[1])]
            B = self._model._backward_map(obs_bw)
            curr = B
            sp_loss = 0
            for act, target in zip(actions, targets):
                curr = self._model._predictor(curr, act)  # num_parallel x batch x z_dim
                sp_loss += (curr - target.unsqueeze(0)).pow(2).sum(-1).mean()
            fb_loss += sp_loss / len(actions)

            # compute orthonormality loss for backward embedding
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -Cov.diag().mean()
            orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = orth_loss_offdiag + orth_loss_diag
            fb_loss += self.cfg.train.ortho_coef * orth_loss

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                    # we disable autocast here to make sure B and cov have the same dtype (otherwise torch.linalg.solve fails)
                    with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=False):
                        cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                    B_inv_conv = torch.linalg.solve(cov, B, left=False)
                    implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._left_encoder.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "sp_loss": sp_loss,
                "q_loss": q_loss,
            }
        return output_metrics
