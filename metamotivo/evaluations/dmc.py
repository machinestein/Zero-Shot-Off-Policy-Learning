# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import mujoco
import numpy as np
import pydantic
import torch
from tqdm.auto import tqdm
import wandb

from metamotivo.base import BaseConfig
from metamotivo.envs.dmc import DMCEnvConfig
from metamotivo.envs.dmc_tasks import dmc
from metamotivo.envs.utils.rollout import rollout
from metamotivo.evaluations.base import extract_model
from metamotivo.nn_models import eval_mode


class DMCRewardEvalConfig(BaseConfig):
    name: tp.Literal["dmc_reward_eval"] = "dmc_reward_eval"
    name_in_logs: str = "reward"
    env: DMCEnvConfig

    tasks: list[str] = pydantic.Field(default_factory=lambda: [])
    num_episodes: int = 100

    num_inference_samples: int = 50_000
    disable_tqdm: bool = False
    record_video: bool = False

    def build(self):
        return DMCRewardEvaluation(self)


class DMCRewardEvaluation:
    def __init__(self, config: DMCRewardEvalConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, replay_buffer, logger, **kwargs):
        wandb_dict = {}
        eval_metrics = {}
        model = extract_model(agent_or_model)

        pbar = tqdm(self.cfg.tasks, leave=False, disable=self.cfg.disable_tqdm,
                    dynamic_ncols=True, smoothing=0.1, colour='green', position=1)
        for task in pbar:
            # pbar.set_description(f"task {task}")
            eval_env, _ = self.cfg.env.model_copy(update={"task": task}).build()
            # pbar.set_description(f"task {task} (inference)")
            ctx, relabel_metrics = self._reward_inference(agent_or_model, task, replay_buffer)
            # pbar.set_description(f"task {task} (rollout)")
            # Removed redundant ctx = [None] wrap
            with torch.no_grad(), eval_mode(model):
                st, _, frames = rollout(
                    eval_env,
                    agent=model,
                    num_episodes=self.cfg.num_episodes,
                    ctx=ctx,
                    render=self.cfg.record_video,
                    render_num_episodes=1,
                )  # return statistics and episodes

                if self.cfg.record_video and frames:
                    video = np.stack(frames).transpose(0, 3, 1, 2)  # T, C, H, W
                    wandb_dict[f"videos/{task}"] = wandb.Video(video, fps=30, format="mp4")

            eval_metrics[task] = st
            mean_reward = np.mean(st["reward"])
            pbar.set_postfix({"reward": mean_reward, "task": task})
            
            wandb_dict[f"{task}/reward"] = mean_reward
            wandb_dict[f"{task}/reward#std"] = np.std(st["reward"])
            for k, v in relabel_metrics.items():
                wandb_dict[f"{task}/{k}"] = v
            eval_env.close()

        rewards = np.concatenate([el["reward"] for el in eval_metrics.values()])
        wandb_dict["eval/reward"] = np.mean(rewards)
        wandb_dict["eval/reward#std"] = np.std(rewards)

        # log reward results
        if logger is not None:
            for k, v in eval_metrics.items():
                # task and timestamp needs to be repeated length of metrics times so the logger accepts it
                random_key = list(v.keys())[0]
                n = len(v[random_key])
                v["task"] = [k] * n
                v["timestep"] = [timestep] * n
                logger.log(v)

        return eval_metrics, wandb_dict

    def _reward_inference(self, agent_or_model, task, replay_buffer) -> torch.Tensor:
        model = extract_model(agent_or_model)
        if not hasattr(model, "reward_inference"):
            return torch.zeros((1, 1)), {}
        env = dmc.make(f"{self.cfg.env.domain}_{task}")
        num_samples = self.cfg.num_inference_samples
        batch = replay_buffer["train"].sample(num_samples)
        rewards = []
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(batch["next"]["physics"][i].cpu().numpy())
                env._physics.set_control(batch["action"][i].cpu().detach().numpy())
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            rewards.append(env._task.get_reward(env._physics))
        rewards = np.array(rewards).reshape(-1, 1)
        relabel_metrics = {
            "relabel_reward#mean": np.mean(rewards),
            "relabel_reward#nonzero": np.count_nonzero(rewards.ravel()),
            "relabel_reward#num_samples": rewards.size,
        }
        z = agent_or_model._model.reward_inference(
            next_obs=batch["next"]["observation"].to(agent_or_model.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=agent_or_model.device),
        )
        return z.reshape(1, -1), relabel_metrics
