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
from .model import TD3Model, TD3ModelConfig


class TD3AgentTrainConfig(BaseConfig):
    lr: float = 1e-4
    critic_target_tau: float = 0.005
    stddev_clip: float = 0.3
    pessimism_penalty: float = 0.0
    batch_size: int = 1024
    discount: float = 0.98
    bc_coeff: float = 0.0


class TD3AgentConfig(BaseConfig):
    name: Literal["TD3Agent"] = "TD3Agent"
    model: TD3ModelConfig = TD3ModelConfig()
    train: TD3AgentTrainConfig = TD3AgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return TD3Agent


class TD3Agent:
    config_class = TD3AgentConfig

    def __init__(self, obs_space, action_dim, cfg: TD3AgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self._model: TD3Model = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.cfg.model.device)

    @property
    def device(self):
        return self._model.device

    @property
    def optimizer_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.critic_optimizer = torch.optim.Adam(
            list(self._model._critic.parameters()) + list(self._model._encoder.parameters()),
            lr=self.cfg.train.lr,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
        )

        # prepare parameter list
        self._critic_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_paramlist = tuple(x for x in self._model._target_critic.parameters())

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_actor = torch.compile(self.update_actor, mode=mode)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor, z: None = None, mean: bool = True) -> torch.Tensor:
        # TD3 just ignores the context z
        return self._model.act(obs, z=z, mean=mean)

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
        with torch.no_grad():
            next_obs = self._model._encoder(next_obs)
        return self._model._encoder(obs), next_obs

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated, reward = (
            batch["observation"].to(self.device),
            batch["action"].to(self.device),
            batch["next"]["observation"].to(self.device),
            batch["next"]["terminated"].to(self.device),
            batch["reward"].to(self.device),
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = (
                self._model._obs_normalizer(obs),
                self._model._obs_normalizer(next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()

        obs, next_obs = self.aug(obs, next_obs)
        obs, next_obs = self.enc(obs, next_obs)

        metrics = self.update_critic(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs)
        metrics.update(self.update_actor(obs.detach(), action))

        with torch.no_grad():
            _soft_update_params(
                self._critic_paramlist,
                self._target_critic_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # compute target critic
        with torch.no_grad():
            next_action = self.sample_action_from_norm_obs(next_obs)
            next_Qs = self._model._target_critic(next_obs, next_action)  # num_parallel x batch x 1
            _, _, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.pessimism_penalty)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(next_Qs.shape[0], -1, -1).float()

        # compute critic loss
        Qs = self._model._critic(obs, action)  # num_parallel x batch x 1
        critic_loss = 0.5 * Qs.shape[0] * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            metrics = {
                "target_Q": target_Q.detach().mean(),
                "critic_loss": critic_loss.detach(),
            }

        return metrics

    def update_actor(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        # compute actor loss
        dist = self._model._actor(obs, self._model.cfg.actor_std)
        actor_action = dist.sample()
        Qs = self._model._critic(obs, actor_action)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(Qs, self.cfg.train.pessimism_penalty)  # batch
        actor_loss = -Q.mean()

        # compute bc loss
        bc_loss = torch.tensor([0.0], device=action.device)
        if self.cfg.train.bc_coeff > 0:
            bc_error = F.mse_loss(actor_action, action)
            bc_loss = self.cfg.train.bc_coeff * Q.abs().mean().detach() * bc_error
            actor_loss = actor_loss + bc_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            metrics = {"actor_loss": actor_loss.detach(), "bc_loss": bc_loss.detach()}

        return metrics

    def sample_action_from_norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._model._actor(obs, self._model.cfg.actor_std)
        next_action = dist.sample(clip=self.cfg.train.stddev_clip)
        return next_action

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
