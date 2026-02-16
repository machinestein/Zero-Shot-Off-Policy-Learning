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
import torch.nn.functional as F

from metamotivo.base import BaseConfig

from ...nn_models import _soft_update_params, eval_mode, weight_init
from .model import SFModel, SFModelConfig


class SFAgentTrainConfig(BaseConfig):
    lr_sf: float = 1e-4
    lr_features: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    sf_target_tau: float = 0.01
    features_target_tau: float = 0.005
    train_goal_ratio: float = 0.5
    sf_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.0
    stddev_clip: float = 0.3
    q_loss: bool = False
    batch_size: int = 1024
    discount: float = 0.99
    prob_random_goal: float = 0.0
    bc_coeff: float = 0.0


class SFAgentConfig(BaseConfig):
    name: Literal["SFAgent"] = "SFAgent"
    model: SFModelConfig
    train: SFAgentTrainConfig
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return SFAgent


class SFAgent:
    config_class = SFAgentConfig

    def __init__(self, obs_space, action_dim, cfg: SFAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self._model: SFModel = self.cfg.model.build(obs_space, action_dim)
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
            "features_optimizer": self.features_optimizer.state_dict(),
            "sf_optimizer": self.sf_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self._sf_paramlist = tuple(x for x in self._model._sf.parameters())
        self._target_sf_paramlist = tuple(x for x in self._model._target_sf.parameters())
        self._features_paramlist = tuple(x for x in self._model._features.parameters())
        self._target_features_paramlist = tuple(x for x in self._model._target_features.parameters())
        self._left_encoder_paramlist = tuple(x for x in self._model._left_encoder.parameters())
        self._target_left_encoder_paramlist = tuple(x for x in self._model._target_left_encoder.parameters())

        self.sf_optimizer = torch.optim.Adam(
            list(self._model._sf.parameters()) + list(self._model._left_encoder.parameters()) + list(self._model._sf_encoder.parameters()),
            lr=self.cfg.train.lr_sf,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.features_optimizer = torch.optim.Adam(
            list(self._model._features.parameters()) + list(self._model._rgb_encoder.parameters()),
            lr=self.cfg.train.lr_features,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            # feel free to re-enable compilation if https://github.com/pytorch/pytorch/issues/166604 is resolved
            # self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode)
            # self.mix_goals = torch.compile(self.mix_goals, mode=mode)
            self.sf_loss = torch.compile(self.sf_loss, mode=mode)
            self.update_actor = torch.compile(self.update_actor, mode=mode)
            self.feature_loss = torch.compile(self.feature_loss, mode=mode)
            self.aug = torch.compile(self.aug, mode=mode)
            self.enc_for_features = torch.compile(self.enc_for_features, mode=mode)
            self.enc_for_sf = torch.compile(self.enc_for_sf, mode=mode)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, next_phi: torch.Tensor):
        """
        Samples a batch from the z distribution used to update the networks

        NOTE: next_phi must contain the features' output at next_obs
        """
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        phi = next_phi[perm]
        inv_cov = torch.inverse(self._model._z_cov + 1e-6 * torch.eye(*self._model._z_cov.size(), device=z.device))
        new_z = torch.matmul(phi, inv_cov)
        new_z = self._model.project_z(new_z)
        mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
        z = torch.where(mask, new_z, z)
        return z

    def aug(self, obs, next_obs, future_obs):
        """
        Augments observations when training from pixels, does nothing otherwise.
        """
        with torch.no_grad():
            obs = self._model._augmentator(obs)
            next_obs = self._model._augmentator(next_obs)
            future_obs = self._model._augmentator(future_obs)
        return obs, next_obs, future_obs

    def enc_for_features(self, obs, next_obs, future_obs):
        """
        Encodes observations when training from pixels, does nothing otherwise.

        NOTE: by default here we compute gradients wrt all 3 observations.
        To speed up code, when extending this class for a particular feature learner:
        - push within torch.no_grad() the encoding of observations whose gradient is not needed
        - don't encode future_obs if not needed, just return None
        """
        obs = self._model._rgb_encoder(obs)
        next_obs = self._model._rgb_encoder(next_obs)
        future_obs = self._model._rgb_encoder(future_obs)
        return obs, next_obs, future_obs

    def enc_for_sf(self, obs, next_obs):
        """
        Encodes observations when training from pixels and using a separate image encoder for SFs, does nothing otherwise.
        """
        obs = self._model._sf_encoder(obs)
        with torch.no_grad():
            next_obs = self._model._sf_encoder(next_obs)
        return obs, next_obs

    @torch.no_grad()
    def _mix_goals(self, next_obs, future_obs):
        # mix future goals with random obs
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        random_obs = next_obs[perm]
        mask_shape = tuple([self.cfg.train.batch_size] + [1] * (next_obs.ndim - 1))
        mask = torch.rand(mask_shape, device=self.device) < self.cfg.train.prob_random_goal
        goals = torch.where(mask, random_obs, future_obs)
        return goals

    def mix_goals(self, next_obs, future_obs):
        return self._mix_goals(next_obs, future_obs)

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

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

        torch.compiler.cudagraph_mark_step_begin()

        # NOTE this must be done before encoding the obs to ensure correct gradient computation
        if self.cfg.train.prob_random_goal > 0:
            future_obs = self.mix_goals(next_obs, future_obs)

        obs, next_obs, future_obs = self.aug(obs, next_obs, future_obs)
        obs_phi, next_obs_phi, future_obs_phi = self.enc_for_features(obs, next_obs, future_obs)

        phi_loss, feature_metrics = self.feature_loss(
            obs=obs_phi, action=action, next_obs=next_obs_phi, future_obs=future_obs_phi, discount=discount
        )

        # keep track of features statistics (this is needed for centering when enabled)
        self._model._update_features_stats(obs_phi.detach())

        # when actor_encode_obs=True, the SFs and actor take as input the features' output, so we overwrite obs and next_obs
        # NOTE: self._model.features() returns detached tensors as required here
        next_phi = self._model.features(next_obs_phi, norm_obs=False, encode_image=False)
        obs_sf, next_obs_sf = self.enc_for_sf(obs, next_obs)

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

    def update_target_networks(self):
        """
        Update all target networks

        NOTE: when extending this class, override this function if your feature learner has extra target networks to update (remembder to call super)
        """
        with torch.no_grad():
            _soft_update_params(self._sf_paramlist, self._target_sf_paramlist, self.cfg.train.sf_target_tau)
            _soft_update_params(self._features_paramlist, self._target_features_paramlist, self.cfg.train.features_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.cfg.train.sf_target_tau)

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

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            left_enc = self._model._left_encoder(obs)
        actor_in = left_enc if self.cfg.model.actor_encode_obs else obs
        actor_action = self.sample_action_from_norm_obs(actor_in, z)
        Fs = self._model._sf(left_enc, z, actor_action)  # num_parallel x batch x z_dim
        Qs = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
        actor_loss = -Q.mean()

        # compute bc loss
        bc_error = torch.tensor([0.0], device=z.device)
        if self.cfg.train.bc_coeff > 0:
            bc_error = F.mse_loss(actor_action, action)
            bc_loss = self.cfg.train.bc_coeff * bc_error
            actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "bc_error": bc_error.detach()}

    def feature_loss(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor, discount: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        The feature learning loss be implemented by each method extending this class.
        By default this does nothing, which means running this agent would train SFs over random features.
        """
        return torch.zeros(1, device=obs.device, dtype=obs.dtype), {}  # return feature loss and metrics

    def optimizers_zero_grad(self):
        """
        Calls zero grad on every optimizer.

        NOTE: when extending this class, override this function if your feature learner has extra optimizers (remembder to call super)
        """
        self.sf_optimizer.zero_grad(set_to_none=True)
        self.features_optimizer.zero_grad(set_to_none=True)

    def optimizers_step(self):
        """
        Performs a step on every optimizer.

        NOTE: when extending this class, override this function if your feature learner has extra optimizers (remembder to call super)
        """
        self.features_optimizer.step()
        self.sf_optimizer.step()

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
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device
        config = cls.config_class(**loaded_config)

        # Load arguments from a pickle file
        with (path / "init_kwargs.pkl").open("rb") as f:
            args = pickle.load(f)
        obs_space = args["obs_space"]
        action_dim = args["action_dim"]

        agent = config.build(obs_space, action_dim)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True)
        for k, v in optimizers.items():
            getattr(agent, k).load_state_dict(v)
        safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device)
        agent._model.train()
        agent._model.requires_grad_(True)
        return agent

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
        with (output_folder / "init_kwargs.pkl").open("wb") as f:
            pickle.dump(
                {
                    "obs_space": self.obs_space,
                    "action_dim": self.action_dim,
                },
                f,
                protocol=5,
            )
