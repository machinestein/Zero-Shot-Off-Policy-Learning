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
from metamotivo.envs.utils.gym_spaces import json_to_space, space_to_json

from ...nn_models import _soft_update_params, eval_mode, weight_init
from .model import TDJEPAModel, TDJEPAModelConfig


class TDJEPAAgentTrainConfig(BaseConfig):
    lr_predictor: float = 1e-4
    lr_phi: float = 1e-4
    lr_psi: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    encoder_target_tau: float = 0.005
    predictor_target_tau: float = 0.005
    phi_ortho_coef: float = 1.0
    psi_ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    predictor_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.0
    stddev_clip: float = 0.3
    batch_size: int = 1024
    discount: float = 0.98
    bc_coeff: float = 0.0
    log_eigvals: bool = False
    scale_train_goals: bool = False


class TDJEPAAgentConfig(BaseConfig):
    name: Literal["TDJEPAAgent"] = "TDJEPAAgent"
    model: TDJEPAModelConfig
    train: TDJEPAAgentTrainConfig
    compile: bool = False

    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return TDJEPAAgent


class TDJEPAAgent:
    config_class = TDJEPAAgentConfig

    def __init__(self, obs_space, action_dim, cfg: TDJEPAAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self._model: TDJEPAModel = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.device)

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        optimizers = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "phi_encoder_optimizer": self.phi_encoder_optimizer.state_dict(),
            "phi_predictor_optimizer": self.phi_predictor_optimizer.state_dict(),
        }
        if not self.cfg.model.symmetric:
            optimizers["psi_encoder_optimizer"] = self.psi_encoder_optimizer.state_dict()
            optimizers["psi_predictor_optimizer"] = self.psi_predictor_optimizer.state_dict()
        return optimizers

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.phi_encoder_optimizer = torch.optim.Adam(
            list(self._model._phi_mlp_encoder.parameters()) + list(self._model._phi_rgb_encoder.parameters()),
            lr=self.cfg.train.lr_phi,
            capturable=False,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.phi_predictor_optimizer = torch.optim.Adam(
            self._model._phi_predictor.parameters(),
            lr=self.cfg.train.lr_predictor,
            capturable=False,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=False,
            weight_decay=self.cfg.train.weight_decay,
        )

        # parameter lists will be used for target update
        self._phi_predictor_paramlist = tuple(x for x in self._model._phi_predictor.parameters())
        self._target_phi_predictor_paramlist = tuple(x for x in self._model._target_phi_predictor.parameters())
        self._phi_mlp_encoder_paramlist = tuple(x for x in self._model._phi_mlp_encoder.parameters())
        self._target_phi_mlp_encoder_paramlist = tuple(x for x in self._model._target_phi_mlp_encoder.parameters())

        if not self.cfg.model.symmetric:
            self.psi_encoder_optimizer = torch.optim.Adam(
                list(self._model._psi_mlp_encoder.parameters()) + list(self._model._psi_rgb_encoder.parameters()),
                lr=self.cfg.train.lr_psi,
                capturable=False,
                weight_decay=self.cfg.train.weight_decay,
            )
            self.psi_predictor_optimizer = torch.optim.Adam(
                self._model._psi_predictor.parameters(),
                lr=self.cfg.train.lr_predictor,
                capturable=False,
                weight_decay=self.cfg.train.weight_decay,
            )

            # parameter lists will be used for target update
            self._psi_predictor_paramlist = tuple(x for x in self._model._psi_predictor.parameters())
            self._target_psi_predictor_paramlist = tuple(x for x in self._model._target_psi_predictor.parameters())
            self._psi_mlp_encoder_paramlist = tuple(x for x in self._model._psi_mlp_encoder.parameters())
            self._target_psi_mlp_encoder_paramlist = tuple(x for x in self._model._target_psi_mlp_encoder.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead"
            print(f"compiling with mode '{mode}'")
            self.update_tdjepa_asym = torch.compile(self.update_tdjepa_asym, mode=mode)
            self.update_tdjepa_sym = torch.compile(self.update_tdjepa_sym, mode=mode)
            self.update_actor = torch.compile(self.update_actor, mode=mode)
            # feel free to re-enable compilation if https://github.com/pytorch/pytorch/issues/166604 is resolved
            # self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)
            self.augment_image = torch.compile(self.augment_image, mode=mode)
            self.encode_image = torch.compile(self.encode_image, mode=mode)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        if train_goal is not None:
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            train_goal = train_goal[perm]
            # NOTE: this assumes that train_goal has already been passed through the psi_rgb_encoder and obs_normalizer
            goals = self._model._phi_mlp_encoder(train_goal) if self.cfg.model.symmetric else self._model._psi_mlp_encoder(train_goal)
            if self.cfg.train.scale_train_goals:
                inv_cov = torch.inverse(self._model._z_cov + 1e-6 * torch.eye(*self._model._z_cov.size(), device=z.device))
                goals = torch.matmul(goals, inv_cov)
            goals = self._model.project_z(goals)
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
            z = torch.where(mask, goals, z)
        return z

    @torch.no_grad()
    def augment_image(self, obs, next_obs):
        """
        Augments observations when training from pixels, does nothing otherwise.
        """
        return self._model._augmentator(obs), self._model._augmentator(next_obs)

    def encode_image(self, obs, next_obs):
        """
        Encodes observations when training from pixels, does nothing otherwise.
        """
        with torch.no_grad():
            phi_next_obs = self._model._phi_rgb_encoder(next_obs)
            psi_next_obs = phi_next_obs if self.cfg.model.symmetric else self._model._psi_rgb_encoder(next_obs)
        phi_obs = self._model._phi_rgb_encoder(obs)
        psi_obs = phi_obs if self.cfg.model.symmetric else self._model._psi_rgb_encoder(obs)
        return phi_obs, phi_next_obs, psi_obs, psi_next_obs

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

        obs, next_obs = self.augment_image(obs, next_obs)
        phi_obs, phi_next_obs, psi_obs, psi_next_obs = self.encode_image(obs, next_obs)

        z = self.sample_mixed_z(train_goal=psi_next_obs).clone()

        metrics = self.update_tdjepa(
            phi_obs=phi_obs,
            psi_obs=psi_obs,
            action=action,
            discount=discount,
            phi_next_obs=phi_next_obs,
            psi_next_obs=psi_next_obs,
            z=z,
        )

        if self.cfg.train.log_eigvals:
            with torch.no_grad():
                # eigvalsh cannot be compiled, so we compute it here
                eigvals = torch.linalg.eigvalsh(self._model._z_cov).sort()[0].flip(0)
                area_under_eigval_cumsum = eigvals.cumsum(0).mean() / eigvals.sum()
                area_under_unif_cumsum = torch.ones_like(eigvals).cumsum(0).mean() / len(eigvals)
                metrics.update(
                    {
                        "eigval_ratio": eigvals[-1] / eigvals[0],
                        "eigenval_early_enrichment": area_under_eigval_cumsum - area_under_unif_cumsum,
                    }
                )

        metrics.update(
            self.update_actor(
                phi_obs=phi_obs.detach(),
                action=action,
                z=z,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._phi_predictor_paramlist, self._target_phi_predictor_paramlist, self.cfg.train.predictor_target_tau)
            _soft_update_params(self._phi_mlp_encoder_paramlist, self._target_phi_mlp_encoder_paramlist, self.cfg.train.encoder_target_tau)
            if not self.cfg.model.symmetric:
                _soft_update_params(
                    self._psi_predictor_paramlist, self._target_psi_predictor_paramlist, self.cfg.train.predictor_target_tau
                )
                _soft_update_params(
                    self._psi_mlp_encoder_paramlist, self._target_psi_mlp_encoder_paramlist, self.cfg.train.encoder_target_tau
                )

        return metrics

    def sample_action_from_latent(self, latent: torch.Tensor, z: torch.Tensor, mean: bool = False) -> torch.Tensor:
        dist = self._model._actor(latent, z, self._model.cfg.actor_std)
        if mean:
            return dist.mean
        action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action

    def _orth_loss(self, enc: torch.Tensor):
        Cov = torch.matmul(enc, enc.T)
        diag = -Cov.diag().mean()
        offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        loss = offdiag + diag
        return loss, diag, offdiag

    def update_tdjepa(
        self,
        phi_obs: torch.Tensor,
        psi_obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        phi_next_obs: torch.Tensor,
        psi_next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.cfg.model.symmetric:
            return self.update_tdjepa_sym(phi_obs=phi_obs, action=action, discount=discount, phi_next_obs=phi_next_obs, z=z)
        return self.update_tdjepa_asym(
            phi_obs=phi_obs, psi_obs=psi_obs, action=action, discount=discount, phi_next_obs=phi_next_obs, psi_next_obs=psi_next_obs, z=z
        )

    def update_tdjepa_asym(
        self,
        phi_obs: torch.Tensor,
        psi_obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        phi_next_obs: torch.Tensor,
        psi_next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # encode next obs
            next_phi_enc = self._model._target_phi_mlp_encoder(phi_next_obs)  # batch x phi_dim
            next_psi_enc = self._model._target_psi_mlp_encoder(psi_next_obs)  # batch x psi_dim
            # compute next action
            actor_in = next_phi_enc if self.cfg.model.actor_use_full_encoder else phi_next_obs
            next_action = self.sample_action_from_latent(actor_in, z)
            # compute target predictor Pred_phi(phi(s'), a', z)
            target_phi_predictors = self._model._target_phi_predictor(next_phi_enc, z, next_action)  # num_parallel x batch x psi_dim
            _, _, target_phi_predictor = self.get_targets_uncertainty(
                target_phi_predictors, self.cfg.train.predictor_pessimism_penalty
            )  # batch x psi_dim
            td_target_phi = next_psi_enc + discount * target_phi_predictor  # batch x psi_dim
            # compute target predictor Pred_psi(psi(s'), a', z)
            target_psi_predictors = self._model._target_psi_predictor(next_psi_enc, z, next_action)  # num_parallel x batch x phi_dim
            _, _, target_psi_predictor = self.get_targets_uncertainty(
                target_psi_predictors, self.cfg.train.predictor_pessimism_penalty
            )  # batch x phi_dim
            td_target_psi = next_phi_enc + discount * target_psi_predictor  # batch x phi_dim

        # compute predictor Pred_phi(phi(s), a, z)
        phi_enc = self._model._phi_mlp_encoder(phi_obs)  # batch x phi_dim
        phi_preds = self._model._phi_predictor(phi_enc, z, action)  # num_parallel x batch x psi_dim
        # compute predictor Pred_psi(psi(s), a, z)
        psi_enc = self._model._psi_mlp_encoder(psi_obs)  # batch x psi_dim
        psi_preds = self._model._psi_predictor(psi_enc, z, action)  # num_parallel x batch x phi_dim

        # compute td-jepa losses
        phi_tdjepa_loss = (phi_preds - td_target_phi).pow(2).sum(-1).mean()
        psi_tdjepa_loss = (psi_preds - td_target_psi).pow(2).sum(-1).mean()
        tdjepa_loss = phi_tdjepa_loss + psi_tdjepa_loss

        # compute orthonormality losses
        phi_orth_loss, phi_orth_loss_diag, phi_orth_loss_offdiag = self._orth_loss(phi_enc)
        psi_orth_loss, psi_orth_loss_diag, psi_orth_loss_offdiag = self._orth_loss(psi_enc)

        total_loss = tdjepa_loss + self.cfg.train.phi_ortho_coef * phi_orth_loss + self.cfg.train.psi_ortho_coef * psi_orth_loss

        self.phi_predictor_optimizer.zero_grad(set_to_none=True)
        self.psi_predictor_optimizer.zero_grad(set_to_none=True)
        self.phi_encoder_optimizer.zero_grad(set_to_none=True)
        self.psi_encoder_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.phi_predictor_optimizer.step()
        self.psi_predictor_optimizer.step()
        self.phi_encoder_optimizer.step()
        self.psi_encoder_optimizer.step()

        self._model._update_z_stats(psi_enc)

        with torch.no_grad():
            output_metrics = {
                "phi_encoder": phi_enc.mean(),
                "psi_encoder": psi_enc.mean(),
                "phi_encoder_norm": torch.norm(phi_enc, dim=-1).mean(),
                "psi_encoder_norm": torch.norm(psi_enc, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "tdjepa_loss": tdjepa_loss,
                "phi_orth_loss": phi_orth_loss,
                "phi_orth_loss_diag": phi_orth_loss_diag,
                "phi_orth_loss_offdiag": phi_orth_loss_offdiag,
                "psi_orth_loss": psi_orth_loss,
                "psi_orth_loss_diag": psi_orth_loss_diag,
                "psi_orth_loss_offdiag": psi_orth_loss_offdiag,
                "total_loss": total_loss,
                "td_target_phi": td_target_phi.mean(),
                "phi_tdjepa_loss": phi_tdjepa_loss,
                "psi_tdjepa_loss": psi_tdjepa_loss,
            }
        return output_metrics

    def update_tdjepa_sym(
        self,
        phi_obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        phi_next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # encode next obs
            next_phi_enc = self._model._target_phi_mlp_encoder(phi_next_obs)  # batch x phi_dim
            # compute next action
            actor_in = next_phi_enc if self.cfg.model.actor_use_full_encoder else phi_next_obs
            next_action = self.sample_action_from_latent(actor_in, z)
            # compute target predictor Pred_phi(phi(s'), a', z)
            target_phi_predictors = self._model._target_phi_predictor(next_phi_enc, z, next_action)  # num_parallel x batch x phi_dim
            _, _, target_phi_predictor = self.get_targets_uncertainty(
                target_phi_predictors, self.cfg.train.predictor_pessimism_penalty
            )  # batch x psi_dim
            td_target_phi = next_phi_enc + discount * target_phi_predictor  # batch x phi_dim

        # compute predictor Pred_phi(phi(s), a, z)
        phi_enc = self._model._phi_mlp_encoder(phi_obs)  # batch x phi_dim
        phi_preds = self._model._phi_predictor(phi_enc, z, action)  # num_parallel x batch x phi_dim

        # compute td-jepa losses
        phi_tdjepa_loss = (phi_preds - td_target_phi).pow(2).sum(-1).mean()
        tdjepa_loss = phi_tdjepa_loss

        # compute orthonormality losses
        phi_orth_loss, phi_orth_loss_diag, phi_orth_loss_offdiag = self._orth_loss(phi_enc)

        total_loss = tdjepa_loss + self.cfg.train.phi_ortho_coef * phi_orth_loss

        self.phi_predictor_optimizer.zero_grad(set_to_none=True)
        self.phi_encoder_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.phi_predictor_optimizer.step()
        self.phi_encoder_optimizer.step()

        self._model._update_z_stats(phi_enc)

        with torch.no_grad():
            output_metrics = {
                "phi_encoder": phi_enc.mean(),
                "phi_encoder_norm": torch.norm(phi_enc, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "tdjepa_loss": tdjepa_loss,
                "phi_orth_loss": phi_orth_loss,
                "phi_orth_loss_diag": phi_orth_loss_diag,
                "phi_orth_loss_offdiag": phi_orth_loss_offdiag,
                "total_loss": total_loss,
                "td_target_phi": td_target_phi.mean(),
                "phi_tdjepa_loss": phi_tdjepa_loss,
            }
        return output_metrics

    def update_actor(
        self,
        phi_obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            phi_enc = self._model._phi_mlp_encoder(phi_obs)
        actor_in = phi_enc if self.cfg.model.actor_use_full_encoder else phi_obs
        dist = self._model._actor(actor_in, z, self._model.cfg.actor_std)
        actor_action = dist.sample(clip=self.cfg.train.stddev_clip)
        preds = self._model._phi_predictor(phi_enc, z, actor_action)  # num_parallel x batch x psi_dim
        Qs = (preds * z).sum(-1)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.actor_pessimism_penalty)  # batch
        actor_loss = -Q.mean()

        # compute bc loss
        bc_error = torch.tensor([0.0], device=z.device)
        if self.cfg.train.bc_coeff > 0:
            bc_error = F.mse_loss(actor_action, action)
            bc_loss = self.cfg.train.bc_coeff * bc_error
            actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "bc_error": bc_error.detach(), "q": Q.mean().detach()}

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
        init_kwargs = {
            "obs_space": space_to_json(self.obs_space),
            "action_dim": self.action_dim,
        }
        with (output_folder / "init_kwargs.json").open("w") as f:
            json.dump(init_kwargs, f, indent=4)
