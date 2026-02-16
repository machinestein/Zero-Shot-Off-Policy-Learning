# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
from pathlib import Path
from typing import Dict, Literal, Tuple

import safetensors
import torch
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast

from ...base import BaseConfig
from ...envs.utils.gym_spaces import json_to_space, space_to_json
from ...nn_models import _soft_update_params, eval_mode, weight_init
from .model import FBModel, FBModelConfig


class ResidualCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + action_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class ZOLConfig(BaseConfig):
    lr: float = 3e-3
    num_steps: int = 500
    n_mu: int = 512
    early_stop_patience: int = 500
    early_stop_tol: float = 1e-8
    chi2_coef: float = 0.01
    trust_l2_coef: float = 0.001
    weight_clip: float = 20.0
    center_rewards: bool = True


class FBAgentTrainConfig(BaseConfig):
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    f_target_tau: float = 0.01
    b_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.0
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float = 0.99
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000
    bc_coeff: float = 0.0
    discrete_temperature: float = 10.0
    zol: ZOLConfig = ZOLConfig()

class FBAgentConfig(BaseConfig):
    name: Literal["FBAgent"] = "FBAgent"
    model: FBModelConfig
    train: FBAgentTrainConfig
    cudagraphs: bool = False
    compile: bool = True
    discrete: bool = False

    # ENTRY POINT TO THE AGENT INSTANTIATION
    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBAgent


class FBAgent:
    config_class = FBAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self.discrete = cfg.discrete
        self._model: FBModel = self.cfg.model.build(obs_space, action_dim, discrete=self.discrete)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.device)

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            list(self._model._backward_map.parameters()) + list(self._model._bw_encoder.parameters()),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            list(self._model._forward_map.parameters())
            + list(self._model._left_encoder.parameters())
            + list(self._model._fw_encoder.parameters()),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())
        self._left_encoder_paramlist = tuple(x for x in self._model._left_encoder.parameters())
        self._target_left_encoder_paramlist = tuple(x for x in self._model._target_left_encoder.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            # feel free to re-enable compilation if https://github.com/pytorch/pytorch/issues/166604 is resolved
            # self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)
            self.aug = torch.compile(self.aug, mode=mode)
            self.enc = torch.compile(self.enc, mode=mode)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
            if train_goal is not None:
                perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
                train_goal = train_goal[perm]
                goals = self._model._backward_map(train_goal)
                goals = self._model.project_z(goals)
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
                z = torch.where(mask, goals, z)
        return z

    @torch.no_grad()
    def aug(self, obs, next_obs):
        """
        Augments observations when training from pixels, does nothing otherwise.
        """
        return self._model._augmentator(obs), self._model._augmentator(next_obs)

    def enc(self, obs, next_obs):
        """
        Encodes observations when training from pixels, does nothing otherwise.
        """
        obs = self._model._fw_encoder(obs)
        goal = self._model._bw_encoder(next_obs)
        with torch.no_grad():
            next_obs = self._model._fw_encoder(next_obs)
        return obs, next_obs, goal

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device),
            batch["action"].to(self.device),
            batch["next"]["observation"].to(self.device),
            batch["next"]["terminated"].to(self.device),
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()

        obs, next_obs = self.aug(obs, next_obs)
        obs, next_obs, goal = self.enc(obs, next_obs)

        z = self.sample_mixed_z(train_goal=goal).clone()

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        if not self.discrete:
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

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            if not self.discrete:
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
                B = self._model._backward_map(goal)  # batch x z_dim
                Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

                diff = Ms - discount * target_M  # num_parallel x batch x batch
                fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
                fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
                fb_loss = fb_offdiag + fb_diag

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
            else:
                with torch.no_grad():
                    next_left_enc = self._model._target_left_encoder(next_obs)
                    target_Fs = self._model._target_forward_map(next_left_enc, z)
                    next_Qs = torch.einsum("psza, sz -> psa", target_Fs, z)

                    # Pessimistic discrete policy alignment
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)
                    pi = F.softmax(next_Q / self.cfg.train.discrete_temperature, dim=-1)
                    greedy_action = pi.argmax(-1)  # (S,)

                    # Expand greedy_action to [P, S, Z, 1] for gather
                    greedy_action_idx = greedy_action.view(1, -1, 1, 1).expand(
                        target_Fs.shape[0], -1, z.shape[-1], 1
                    ).long()
                    target_Fs = target_Fs.gather(-1, greedy_action_idx).squeeze(-1)  # (P, S, Z)

                    target_B = self._model._target_backward_map(goal)  # (S, Z)
                    target_Ms = torch.matmul(target_Fs, target_B.T)  # (P, S, S)
                    _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # (S, S)

                # compute FB loss
                left_enc = self._model._left_encoder(obs)  # (S, L)
                Fs = self._model._forward_map(left_enc, z)  # (P, S, Z, A)

                # Expand action to [P, S, Z, 1] for gather
                action_idx = action.view(1, -1, 1, 1).expand(Fs.shape[0], -1, z.shape[-1], 1).long()
                Fs = Fs.gather(-1, action_idx).squeeze(-1)  # (P, S, Z)

                B = self._model._backward_map(goal)  # (S, Z)
                Ms = torch.matmul(Fs, B.T)  # (P, S, S)

                diff = Ms - discount * target_M  # (P, S, S)
                fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
                fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
                fb_loss = fb_offdiag + fb_diag

                # Q-loss for discrete branch is still disabled / commented out here.
                q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)

                # orthogonality loss on backward embeddings (sample-wise covariance)
                Cov = torch.matmul(B, B.T)
                orth_loss_diag = -Cov.diag().mean()
                orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
                orth_loss = orth_loss_offdiag + orth_loss_diag
                fb_loss += self.cfg.train.ortho_coef * orth_loss

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
                "q_loss": q_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, action=action, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                left_enc = self._model._left_encoder(obs)
            actor_in = left_enc if self.cfg.model.actor_encode_obs else obs
            dist = self._model._actor(actor_in, z, self._model.cfg.actor_std)
            actor_action = dist.sample(clip=self.cfg.train.stddev_clip)
            Fs = self._model._forward_map(left_enc, z, actor_action)  # num_parallel x batch x z_dim
            Qs = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
            actor_loss = -Q.mean()

            # compute bc loss
            bc_error = torch.tensor([0.0], device=action.device)
            if self.cfg.train.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                bc_loss = self.cfg.train.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "bc_error": bc_error.detach(), "q": Q.mean().detach()}

    @torch.no_grad()
    def _extract_obs(self, obs_data):
        if isinstance(obs_data, dict):
            if "observation" in obs_data:
                raw_obs = obs_data["observation"]
            elif "state" in obs_data:
                raw_obs = obs_data["state"]
            else:
                raw_obs = next(iter(obs_data.values()))
        else:
            raw_obs = getattr(obs_data, "observation", obs_data)
        return raw_obs

    # @torch.no_grad()
    # def collect_initial_obs(self, env, n_init, device):
    #     obs_list = []
    #     for _ in range(n_init):
    #         reset_res = env.reset()
    #         if isinstance(reset_res, (list, tuple)) and len(reset_res) == 2:
    #             obs_data, _ = reset_res
    #         else:
    #             obs_data = reset_res
    #         raw = self._extract_obs(obs_data)
    #         obs = torch.as_tensor(raw, dtype=torch.float32, device=device)
    #         if obs.ndim == 1:
    #             obs = obs.unsqueeze(0)
    #         obs_list.append(obs)
    #     return torch.cat(obs_list, dim=0)
    
    @torch.no_grad()
    def collect_initial_obs(self, env, n_init, device, expected_obs_dim=None):
        """
        Robustly collect initial observations from env.reset().

        - Supports Gymnasium-style (obs, info) returns.
        - Supports dm_env TimeStep returns via `.observation`.
        - Supports dict observations by concatenating all fields in sorted key order.
        - Flattens to 1D vectors and stacks into [n_init, obs_dim].

        Args:
            env: environment with reset()
            n_init: number of resets
            device: torch device
            expected_obs_dim: if provided, assert obs_dim matches (catches DMC dict bugs)

        Returns:
            obs_batch: torch.FloatTensor [n_init, obs_dim]
        """

        def _to_numpy_obs(obs_data):
            # Handle dm_env TimeStep or objects with `.observation`
            raw = obs_data
            if not isinstance(raw, dict):
                raw = getattr(raw, "observation", raw)

            # Unwrap nested {"observation": ...} if present
            if isinstance(raw, dict) and "observation" in raw:
                raw = raw["observation"]

            # If still a dict, concatenate all pieces deterministically
            if isinstance(raw, dict):
                parts = []
                for k in sorted(raw.keys()):
                    v = raw[k]
                    v = np.asarray(v, dtype=np.float32).ravel()
                    parts.append(v)
                raw = np.concatenate(parts, axis=0) if len(parts) > 0 else np.zeros((0,), dtype=np.float32)
            else:
                raw = np.asarray(raw, dtype=np.float32).ravel()

            return raw

        obs_list = []
        obs_dim_seen = None

        for i in range(n_init):
            reset_res = env.reset()

            # Gymnasium style: (obs, info)
            if isinstance(reset_res, (list, tuple)) and len(reset_res) == 2:
                obs_data, _info = reset_res
            else:
                obs_data = reset_res

            raw = _to_numpy_obs(obs_data)

            if expected_obs_dim is not None and raw.shape[0] != expected_obs_dim:
                raise ValueError(
                    f"collect_initial_obs: obs_dim mismatch at i={i}: got {raw.shape[0]}, expected {expected_obs_dim}. "
                    f"This usually means dict obs flattening doesn't match dataset construction."
                )

            if obs_dim_seen is None:
                obs_dim_seen = raw.shape[0]
            elif raw.shape[0] != obs_dim_seen:
                raise ValueError(
                    f"collect_initial_obs: inconsistent obs_dim at i={i}: got {raw.shape[0]}, previously {obs_dim_seen}."
                )

            obs = torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]
            obs_list.append(obs)

        return torch.cat(obs_list, dim=0)  # [n_init, obs_dim]


    def estimate_mu_pi_z_from_obs(self, obs_tensor, z):
        N = obs_tensor.shape[0]
        if z.ndim == 1:
            z = z.unsqueeze(0)
        
        z_proj = self._model.project_z(z)
        z_batch = z_proj.expand(N, -1)
        
        action = self._model.act_zol(obs_tensor, z_batch, mean=True)
        F = self._model.forward_map_zol(obs=obs_tensor, z=z_batch, action=action)
        return F.mean(dim=0)

    def zol_latent_search(
        self,
        env,
        batch_obs_tensor,
        rewards_tensor,
        initial_z,
        *,
        mu_source: str = "batch",         # "init" | "batch" | "mix"
        mu_mix_frac: float = 0.5,         # only used if mu_source="mix"
        mu_reward_top_frac: float = 0.0,  # if >0, add some top-reward batch states into mu set
        use_exp_weights: bool = True,     # else softplus
        weight_temp: float = 1.0,         # beta for exp weights
        self_normalized_obj: bool = True, # use sum(w r)/sum(w) instead of mean(w r)
    ):
        """
        Improved ZOL latent search:
        - Renormalize after clipping (important).
        - Optionally estimate mu on batch or mix distributions to reduce mismatch.
        - Optional inclusion of top-reward states in mu estimation set.
        - Optionally use exp(beta * logit) weights for sharper focus.
        - Optionally use self-normalized objective for reward-seeking behavior.
        """
        cfg = self.cfg.train.zol
        device = self.device
        gamma = float(self.cfg.train.discount)

        if batch_obs_tensor.device != device:
            batch_obs_tensor = batch_obs_tensor.to(device)
        if rewards_tensor.device != device:
            rewards_tensor = rewards_tensor.to(device)
        rewards_tensor = rewards_tensor.to(torch.float32)

        # Precompute B(s) ONCE (constant w.r.t z)
        with torch.no_grad():
            B_s = self._model.backward_map_zol(batch_obs_tensor)  # [N, d]

        # Collect init obs (only if needed)
        init_obs = None
        if mu_source in ("init", "mix"):
            init_obs = self.collect_initial_obs(env, cfg.n_mu, device)

        # Build mu estimation set
        def build_mu_obs():
            if mu_source == "init":
                mu_obs = init_obs
            elif mu_source == "batch":
                # Use a random subset of the batch for mu estimation (size cfg.n_mu)
                N = batch_obs_tensor.shape[0]
                idx = torch.randint(0, N, (cfg.n_mu,), device=device)
                mu_obs = batch_obs_tensor[idx]
            elif mu_source == "mix":
                # Mix init + batch
                n_init = int(cfg.n_mu * mu_mix_frac)
                n_batch = cfg.n_mu - n_init
                N = batch_obs_tensor.shape[0]
                idx = torch.randint(0, N, (n_batch,), device=device)
                mu_obs = torch.cat([init_obs[:n_init], batch_obs_tensor[idx]], dim=0)
            else:
                raise ValueError(f"Unknown mu_source={mu_source}")

            # Optionally append some top-reward states into mu set
            if mu_reward_top_frac and mu_reward_top_frac > 0.0:
                N = batch_obs_tensor.shape[0]
                k = max(1, int(mu_reward_top_frac * N))
                top_idx = torch.topk(rewards_tensor, k=k, largest=True).indices
                # sample up to cfg.n_mu//4 from top reward
                m = min(top_idx.shape[0], max(1, cfg.n_mu // 4))
                pick = top_idx[torch.randint(0, top_idx.shape[0], (m,), device=device)]
                mu_obs = torch.cat([mu_obs, batch_obs_tensor[pick]], dim=0)

            return mu_obs

        mu_obs = build_mu_obs()

        def compute_weights(z_eval):
            # Always evaluate weights using projected z for consistency
            z_eval = z_eval.detach() if not z_eval.requires_grad else z_eval
            z_proj = self._model.project_z(z_eval.unsqueeze(0)).squeeze(0)

            mu_pi_z = self.estimate_mu_pi_z_from_obs(mu_obs, z_proj)

            # force mu to shape [d, 1]
            if mu_pi_z.ndim == 1:
                mu_col = mu_pi_z.unsqueeze(1)
            elif mu_pi_z.shape[0] == 1 and mu_pi_z.shape[1] > 1:
                mu_col = mu_pi_z.T
            else:
                mu_col = mu_pi_z
                if mu_col.ndim == 2 and mu_col.shape[1] != 1:
                    mu_col = mu_col.mean(dim=0, keepdim=True).T  # fallback

            logit = (1.0 - gamma) * torch.matmul(B_s, mu_col).squeeze(1)  # [N]

            if use_exp_weights:
                # Stable exp weights with temperature
                x = weight_temp * logit
                x = x - x.max()
                w = torch.exp(x)
            else:
                w = F.softplus(logit)

            # Mean-normalize
            w = w / (w.mean() + 1e-8)

            # Clip THEN renormalize (important!)
            if cfg.weight_clip is not None:
                w = torch.clamp(w, max=float(cfg.weight_clip))
                w = w / (w.mean() + 1e-8)

            return w, z_proj

        initial_z_device = initial_z.clone().detach().to(device)
        z0_proj = self._model.project_z(initial_z_device.unsqueeze(0)).squeeze(0).detach()

        def get_loss(z_eval):
            w, z_proj = compute_weights(z_eval)

            local_rewards = rewards_tensor
            if cfg.center_rewards:
                local_rewards = local_rewards - local_rewards.mean()
            if local_rewards.shape != w.shape:
                local_rewards = local_rewards.view_as(w)

            if self_normalized_obj:
                # self-normalized importance weighting
                J = torch.sum(w * local_rewards) / (torch.sum(w) + 1e-8)
            else:
                J = torch.mean(w * local_rewards)

            chi2 = torch.mean((w - 1.0) ** 2)
            trust = torch.mean((z_proj - z0_proj) ** 2)

            obj = J - cfg.chi2_coef * chi2 - cfg.trust_l2_coef * trust
            return -obj

        with torch.no_grad():
            best_loss = get_loss(initial_z_device).item()
            best_z = initial_z_device.clone()

        z_opt = initial_z_device.clone().detach().to(device)
        z_opt.requires_grad_(True)
        optimizer = torch.optim.Adam([z_opt], lr=cfg.lr)

        patience_counter = 0
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            loss = get_loss(z_opt)
            loss.backward()
            optimizer.step()

            # Project back after step (still ok)
            with torch.no_grad():
                z_opt.copy_(self._model.project_z(z_opt.unsqueeze(0)).squeeze(0))

            current_loss = float(loss.item())
            if current_loss < best_loss - cfg.early_stop_tol:
                best_loss = current_loss
                best_z = z_opt.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.early_stop_patience:
                break

        return self._model.project_z(best_z.unsqueeze(0)).squeeze(0).detach()

    
    def zol_latent_search_old(self, env, batch_obs_tensor, rewards_tensor, initial_z):
        cfg = self.cfg.train.zol
        device = self.device
        gamma = self.cfg.train.discount

        if batch_obs_tensor.device != device:
            batch_obs_tensor = batch_obs_tensor.to(device)
        if rewards_tensor.device != device:
            rewards_tensor = rewards_tensor.to(device)
        rewards_tensor = rewards_tensor.to(torch.float32)

        # Collect initial observations once for efficient mu estimation
        init_obs = self.collect_initial_obs(env, cfg.n_mu, device)

        def get_loss(z_eval):
            mu_pi_z = self.estimate_mu_pi_z_from_obs(init_obs, z_eval)
            B_s = self._model.backward_map_zol(batch_obs_tensor)
            if mu_pi_z.ndim == 1:
                mu_pi_z = mu_pi_z.unsqueeze(1)
            elif mu_pi_z.shape[0] == 1 and mu_pi_z.shape[1] > 1:
                mu_pi_z = mu_pi_z.T
                
            w = torch.matmul(B_s, mu_pi_z).squeeze(1)
            w = (1.0 - gamma) * w
            
            # Weight normalization and clipping as in tutorial
            w = F.softplus(w)
            w = w / (w.mean() + 1e-8)
            if cfg.weight_clip is not None:
                w = torch.clamp(w, max=float(cfg.weight_clip))
            
            local_rewards = rewards_tensor
            if cfg.center_rewards:
                local_rewards = local_rewards - local_rewards.mean()

            if local_rewards.shape != w.shape:
                local_rewards = local_rewards.view_as(w)
                
            J = torch.mean(w * local_rewards)

            # Regularization terms
            chi2 = torch.mean((w - 1.0) ** 2)
            
            z_proj = self._model.project_z(z_eval.unsqueeze(0)).squeeze(0)
            z0_proj = self._model.project_z(initial_z_device.unsqueeze(0)).squeeze(0).detach()
            trust = torch.mean((z_proj - z0_proj) ** 2)

            obj = J - cfg.chi2_coef * chi2 - cfg.trust_l2_coef * trust
            return -obj

        initial_z_device = initial_z.clone().detach().to(device)
        with torch.no_grad():
            initial_loss = get_loss(initial_z_device).item()

        best_loss = initial_loss
        best_z = initial_z_device.clone()
        
        z_opt = initial_z.clone().detach().to(device)
        z_opt.requires_grad_(True)
        
        optimizer = torch.optim.Adam(
            [z_opt], 
            lr=cfg.lr,
        )
        
        patience_counter = 0
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            loss = get_loss(z_opt)
            loss.backward()
            optimizer.step()
            
            # Project Z back to manifold after step
            with torch.no_grad():
                z_opt.copy_(self.project_z(z_opt))
            
            current_loss = loss.item()
            if current_loss < best_loss - cfg.early_stop_tol:
                best_loss = current_loss
                best_z = z_opt.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= cfg.early_stop_patience:
                break
        
        return self.project_z(best_z.unsqueeze(0)).squeeze(0).detach()

    def forward_map_zol(self, obs, z, action):
        return self._model.forward_map_zol(obs, z, action)

    def backward_map_zol(self, obs):
        return self._model.backward_map_zol(obs)

    def project_z(self, z):
        return self._model.project_z(z)

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    @classmethod
    def load(cls, path: str, device: str | None = None, **kwargs):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device

        if (path / "init_kwargs.pkl").exists():
            # Load arguments from a pickle file
            with (path / "init_kwargs.pkl").open("rb") as f:
                args = pickle.load(f)
            obs_space = args["obs_space"]
            action_dim = args["action_dim"]
        else:
            # load argeuments from a json file
            with (path / "init_kwargs.json").open("r") as f:
                args = json.load(f)
            obs_space = json_to_space(args["obs_space"])
            action_dim = args["action_dim"]

        # Validate that the loaded agent is compatible with the current environment
        if "obs_space" in kwargs:
            expected_obs_space = kwargs["obs_space"]
            if obs_space != expected_obs_space:
                raise RuntimeError(
                    f"Observation space mismatch during agent loading: path={path}, expected={expected_obs_space}, loaded={obs_space}"
                )
        if "action_dim" in kwargs:
            expected_action_dim = kwargs["action_dim"]
            if action_dim != expected_action_dim:
                raise RuntimeError(
                    f"Action dimension mismatch during agent loading: path={path}, expected={expected_action_dim}, loaded={action_dim}"
                )

        config = cls.config_class(**loaded_config)
        agent = config.build(obs_space, action_dim)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True)
        for k, v in optimizers.items():
            getattr(agent, k).load_state_dict(v)
        safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device)
        agent._model.train()
        agent._model.requires_grad_(True)
        return agent

    def rela_fast_adaptation(self, env, zr, num_episodes=50, lr_z=1e-4, lr_q=1e-4, batch_size=1024, tau=0.005):
        """Residual Latent Adaptation (ReLA)."""
        device = self.device
        obs_dim = self.obs_space.shape[0]
        action_dim = self.action_dim
        zr = zr.detach().to(device)

        # Initialize residual critics and targets
        q1 = ResidualCritic(obs_dim, action_dim).to(device)
        q2 = ResidualCritic(obs_dim, action_dim).to(device)
        q1_target = ResidualCritic(obs_dim, action_dim).to(device)
        q2_target = ResidualCritic(obs_dim, action_dim).to(device)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
        q1_target.requires_grad_(False)
        q2_target.requires_grad_(False)

        optimizer_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr_q)
        
        z = zr.clone().detach().requires_grad_(True)
        optimizer_z = torch.optim.Adam([z], lr=lr_z)

        replay_buffer = []  # Simple online buffer: [(s, a, r, s', done)]
        gamma = self.cfg.train.discount

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                # Select action using current z
                with torch.no_grad():
                    obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                    action = self._model.act(obs_t, z).cpu().numpy()[0]
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append((obs, action, reward, next_obs, done))
                obs = next_obs

                if len(replay_buffer) >= batch_size:
                    # Sample minibatch
                    indices = np.random.choice(len(replay_buffer), batch_size)
                    batch = [replay_buffer[i] for i in indices]
                    s_b = torch.tensor(np.array([x[0] for x in batch]), device=device, dtype=torch.float32)
                    a_b = torch.tensor(np.array([x[1] for x in batch]), device=device, dtype=torch.float32)
                    r_b = torch.tensor(np.array([x[2] for x in batch]), device=device, dtype=torch.float32).unsqueeze(1)
                    sn_b = torch.tensor(np.array([x[3] for x in batch]), device=device, dtype=torch.float32)
                    d_b = torch.tensor(np.array([x[4] for x in batch]), device=device, dtype=torch.float32).unsqueeze(1)

                    # Critic Update
                    with torch.no_grad():
                        an_b = self._model.act(sn_b, z.expand(batch_size, -1))
                        psi_next = self._model.forward_map_zol(sn_b, z.expand(batch_size, -1), an_b)
                        v_base_next = (psi_next * zr).sum(dim=-1, keepdim=True)
                        v_res_next = torch.min(q1_target(sn_b, an_b), q2_target(sn_b, an_b))
                        target_q = r_b + gamma * (1 - d_b) * (v_base_next + v_res_next)

                    psi_curr = self._model.forward_map_zol(s_b, z.expand(batch_size, -1), a_b)
                    q_base_curr = (psi_curr * zr).sum(dim=-1, keepdim=True)
                    
                    q1_loss = F.mse_loss(q_base_curr + q1(s_b, a_b), target_q)
                    q2_loss = F.mse_loss(q_base_curr + q2(s_b, a_b), target_q)
                    
                    optimizer_q.zero_grad()
                    (q1_loss + q2_loss).backward()
                    optimizer_q.step()

                    # Actor Update (on latent z)
                    a_z = self._model.act(s_b, z.expand(batch_size, -1))
                    psi_z = self._model.forward_map_zol(s_b, z.expand(batch_size, -1), a_z)
                    obj = (psi_z * zr).sum(dim=-1, keepdim=True) + torch.min(q1(s_b, a_z), q2(s_b, a_z))
                    loss_z = -obj.mean()

                    optimizer_z.zero_grad()
                    loss_z.backward()
                    optimizer_z.step()
                    
                    with torch.no_grad():
                        z.copy_(self.project_z(z))

                    # Soft Update
                    with torch.no_grad():
                        _soft_update_params(list(q1.parameters()), list(q1_target.parameters()), tau)
                        _soft_update_params(list(q2.parameters()), list(q2_target.parameters()), tau)

        return z.detach()

    def lola_fast_adaptation(self, env, zr, num_episodes=50, lr_mu=0.05, n_lookahead=100, k_samples=10, sigma=0.1, state_buffer=None):
        """Lookahead Latent Adaptation (LoLA)."""
        device = self.device
        zr = zr.detach().to(device)
        mu = zr.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=lr_mu)
        gamma = self.cfg.train.discount

        for ep in range(num_episodes):
            # 1. Sample s0 from d0 or state_buffer
            if state_buffer is not None and len(state_buffer) > 0 and np.random.random() > 0.5:
                s0 = state_buffer[np.random.randint(len(state_buffer))]
            else:
                s0 = None # reset to d0

            # 2. Sample k latents from N(mu, sigma)
            z_samples = mu + torch.randn(k_samples, zr.shape[-1], device=device) * sigma
            z_samples = self.project_z(z_samples)
            
            returns = []
            log_probs = []

            for i in range(k_samples):
                zi = z_samples[i]
                
                # Reset to s0
                if s0 is not None:
                    # Generic reset for DMC/Gym if possible, or specialized for DMC
                    if hasattr(env, "physics"): # DMC
                        with env.physics.reset_context():
                            env.physics.set_state(s0)
                        obs = env._task.get_observation(env.physics) # approximate
                    else:
                        obs, _ = env.reset() # fallback
                else:
                    obs, _ = env.reset()
                
                ep_reward = 0
                for t in range(n_lookahead):
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                        action = self._model.act(obs_t, zi.unsqueeze(0)).cpu().numpy()[0]
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += (gamma ** t) * reward
                    obs = next_obs
                    if terminated or truncated:
                        break
                
                # Terminal value bootstrap using psi
                with torch.no_grad():
                    obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                    act_t = self._model.act(obs_t, zi.unsqueeze(0))
                    psi_term = self._model.forward_map_zol(obs_t, zi.unsqueeze(0), act_t)
                    bootstrap = (gamma ** (t + 1)) * (psi_term * zr).sum()
                
                returns.append(ep_reward + bootstrap)
                dist = torch.distributions.Normal(mu, sigma)
                log_probs.append(dist.log_prob(zi).sum())

            returns = torch.stack(returns)
            log_probs = torch.stack(log_probs)

            # Leave-one-out baseline
            if k_samples > 1:
                baselines = (returns.sum() - returns) / (k_samples - 1)
                advantages = returns - baselines
            else:
                advantages = returns

            loss = -(log_probs * advantages.detach()).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                mu.copy_(self.project_z(mu))

        return mu.detach()

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        json_dump = self.cfg.model_dump()
        with (output_folder / "config.json").open("w+") as f:
            json.dump(json_dump, f, indent=4)
        # save optimizer
        torch.save(
            self.optimizer_dict,
            output_folder / "optimizers.pth",
        )
        # save model
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))

        # Save the arguments required to create this agent (in addition to the config)
        init_kwargs = {
            "obs_space": space_to_json(self.obs_space),
            "action_dim": self.action_dim,
        }
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)
