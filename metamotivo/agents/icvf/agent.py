# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Literal

import torch
import torch.nn.functional as F

from ...nn_models import _soft_update_params, eval_mode
from ..sf.agent import SFAgent, SFAgentConfig, SFAgentTrainConfig
from .model import ICVFModelConfig


class ICVFAgentTrainConfig(SFAgentTrainConfig):
    expectile: float = 0.9
    prob_same_goal: float = 0.5


class ICVFAgentConfig(SFAgentConfig):
    name: Literal["ICVFAgent"] = "ICVFAgent"
    model: ICVFModelConfig
    train: ICVFAgentTrainConfig

    @property
    def object_class(self):
        return ICVFAgent


class ICVFAgent(SFAgent):
    config_class = ICVFAgentConfig

    @property
    def optimizer_dict(self):
        return {**super().optimizer_dict, "t_optimizer": self.sf_optimizer.state_dict()}

    def setup_training(self) -> None:
        super().setup_training()
        self._t_paramlist = tuple(x for x in self._model._t.parameters())
        self._target_t_paramlist = tuple(x for x in self._model._target_t.parameters())

        left_params = list(self._model._left_encoder.parameters()) + list(self._model._sf_encoder.parameters())
        self.sf_optimizer = torch.optim.Adam(
            list(self._model._sf.parameters()) + left_params,
            lr=self.cfg.train.lr_sf,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.t_optimizer = torch.optim.Adam(
            list(self._model._t.parameters()) + left_params,
            lr=self.cfg.train.lr_features,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    def update_target_networks(self):
        super().update_target_networks()
        with torch.no_grad():
            _soft_update_params(self._t_paramlist, self._target_t_paramlist, self.cfg.train.features_target_tau)

    def value(self, obs: torch.Tensor, goals: torch.Tensor, desired_goals: torch.Tensor, is_target: bool = False):
        if is_target:
            phi = self._model._target_left_encoder
            t = self._model._target_t
            psi = self._model._target_features
        else:
            phi = self._model._left_encoder
            t = self._model._t
            psi = self._model._features
        v = (t(phi(obs), psi(desired_goals)) * psi(goals)).sum(-1, keepdims=True)
        return v.detach() if is_target else v

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def enc_for_features(self, obs, next_obs, future_obs, desired_obs):
        with torch.no_grad():
            next_obs = self._model._rgb_encoder(next_obs)
            obs = self._model._rgb_encoder(obs)
        future_obs = self._model._rgb_encoder(future_obs)
        desired_obs = self._model._rgb_encoder(desired_obs)
        return obs, next_obs, future_obs, desired_obs

    def _reward(self, obs, goals):
        obs = obs.reshape(obs.shape[0], -1)
        goals = goals.reshape(goals.shape[0], -1)
        if isinstance(obs, dict):
            obs = torch.cat([v for v in obs.values()], dim=-1)
            goals = torch.cat([v for v in goals.values()], dim=-1)
        return (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float().unsqueeze(0).unsqueeze(-1)

    @torch.no_grad()
    def _same_goals(self, desired_obs, future_obs):
        # set future_obs to desired_obs
        mask_shape = tuple([self.cfg.train.batch_size] + [1] * (desired_obs.ndim - 1))
        mask = torch.rand(mask_shape, device=self.device) < self.cfg.train.prob_same_goal
        return torch.where(mask, desired_obs, future_obs)

    def same_goals(self, desired_obs, future_obs):
        return self._same_goals(desired_obs, future_obs)

    def aug(self, obs, next_obs, future_obs, desired_obs):
        obs, next_obs, future_obs = super().aug(obs, next_obs, future_obs)
        with torch.no_grad():
            desired_obs = self._model._augmentator(desired_obs)
        return obs, next_obs, future_obs, desired_obs

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device),
            batch["action"].to(self.device),
            batch["next"]["observation"].to(self.device),
            batch["next"]["terminated"].to(self.device),
        )
        future_obs = batch.get("future_observation", next_obs)
        future_obs = future_obs.to(self.device)
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs, future_obs = (
                self._model._obs_normalizer(obs),
                self._model._obs_normalizer(next_obs),
                self._model._obs_normalizer(future_obs),
            )

        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        desired_obs = future_obs[perm]

        torch.compiler.cudagraph_mark_step_begin()

        # NOTE this must be done before encoding the obs to ensure correct gradient computation
        if self.cfg.train.prob_random_goal > 0:
            future_obs = self.mix_goals(next_obs, future_obs)
            desired_obs = self.mix_goals(next_obs, desired_obs)

        if self.cfg.train.prob_same_goal > 0:
            future_obs = self.same_goals(desired_obs, future_obs)

        # compute rewards before augmentation
        rewards = self._reward(next_obs, future_obs) - 1.0
        desired_rewards = self._reward(next_obs, desired_obs) - 1.0

        obs, next_obs, future_obs, desired_obs = self.aug(obs, next_obs, future_obs, desired_obs)
        obs_phi, next_obs_phi, future_obs_phi, desired_obs_phi = self.enc_for_features(obs, next_obs, future_obs, desired_obs)
        obs_sf, next_obs_sf = self.enc_for_sf(obs, next_obs)
        # ICVF needs different features with respect to SF methods
        phi_loss, feature_metrics = self.feature_loss(
            obs=obs_sf,
            next_obs=next_obs_sf,
            goals=future_obs_phi,
            desired_goals=desired_obs_phi,
            discount=discount,
            rewards=rewards,
            desired_rewards=desired_rewards,
        )

        # keep track of features statistics (this is needed for centering when enabled)
        self._model._update_features_stats(obs_phi.detach())

        # when actor_encode_obs=True, the SFs and actor take as input the features' output, so we overwrite obs and next_obs
        # NOTE: self._model.features() returns detached tensors as required here
        next_phi = self._model.features(next_obs_phi, norm_obs=False, encode_image=False)

        z = self.sample_mixed_z(next_phi)

        sf_loss, sf_metrics = self.sf_loss(obs=obs_sf, action=action, discount=discount, next_obs=next_obs_sf, next_phi=next_phi, z=z)

        self.optimizers_zero_grad()
        phi_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.optimizers_step()

        # optimize actor
        actor_metrics = self.update_actor(obs_sf.detach(), action, z)

        self.update_target_networks()

        metrics = {}
        for m in [feature_metrics, sf_metrics, actor_metrics]:
            metrics.update(m)
        return metrics

    def feature_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        goals: torch.Tensor,
        desired_goals: torch.Tensor,
        discount: torch.Tensor,
        rewards: torch.Tensor,
        desired_rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        masks, desired_masks = -desired_rewards, -desired_rewards
        with torch.no_grad():
            next_v_gz = self.value(next_obs, goals, desired_goals, is_target=True)
            q_gz = rewards + discount * masks * next_v_gz

            next_v_zz = self.value(next_obs, desired_goals, desired_goals, is_target=True)
            next_v_zz = next_v_zz.min(0, keepdims=True)[0]
            q_zz = desired_rewards + discount * desired_masks * next_v_zz
            v_zz = self.value(obs, desired_goals, desired_goals).mean(0)
            adv = q_zz - v_zz

        v_gz = self.value(obs, goals, desired_goals)

        value_loss = self.expectile_loss(adv, q_gz - v_gz, self.cfg.train.expectile).mean()

        with torch.no_grad():
            metrics = {
                "value_loss": value_loss,
                "v_mean": v_gz.mean(),
                "v_max": v_gz.max(),
                "v_min": v_gz.min(),
                "abs_adv_mean": torch.abs(adv).mean(),
                "adv_mean": adv.mean(),
                "adv_max": adv.max(),
                "adv_min": adv.min(),
                "accept_prob": (adv >= 0).float().mean(),
            }

        return value_loss, metrics

    def sf_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_phi: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        left_enc = self._model._left_encoder(obs)  # batch x L_dim
        SFs = self._model._sf(left_enc, z, action)  # num_parallel x batch x z_dim

        with torch.no_grad():
            next_left_enc = self._model._target_left_encoder(next_obs)  # batch x L_dim
            actor_in = next_left_enc if self.cfg.model.actor_encode_obs else next_obs
            next_action = self.sample_action_from_norm_obs(actor_in, z)
            next_SFs = self._model._target_sf(next_left_enc, z, next_action)  # num_parallel x batch x z_dim

        if not self.cfg.train.q_loss:
            with torch.no_grad():
                _, _, next_SF = self.get_targets_uncertainty(next_SFs, self.cfg.train.sf_pessimism_penalty)  # batch
                target_SF = next_phi + discount * next_SF
                target_SF = target_SF.expand(SFs.shape[0], -1, -1)

            sf_loss = 0.5 * SFs.shape[0] * F.mse_loss(SFs, target_SF)
        else:
            with torch.no_grad():
                next_Qs = (next_SFs * z[None, ...]).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.sf_pessimism_penalty)  # batch
                implicit_reward = (next_phi * z).sum(dim=-1)  # batch
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                expanded_targets = target_Q.expand(SFs.shape[0], -1)

            Qs = (SFs * z).sum(dim=-1)  # num_parallel x batch
            sf_loss = 0.5 * SFs.shape[0] * F.mse_loss(Qs, expanded_targets)

        with torch.no_grad():
            output_metrics = {
                "SF1": SFs[0].mean(),
                "features_norm": torch.norm(next_phi, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "sf_loss": sf_loss,
            }
        return sf_loss, output_metrics

    def optimizers_zero_grad(self):
        """
        Calls zero grad on every optimizer.
        """
        super().optimizers_zero_grad()
        self.t_optimizer.zero_grad(set_to_none=True)

    def optimizers_step(self):
        """
        Performs a step on every optimizer.
        """
        super().optimizers_step()
        self.t_optimizer.step()
