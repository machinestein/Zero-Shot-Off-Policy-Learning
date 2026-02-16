# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pydantic
import torch
import tqdm
import wandb

from metamotivo.base import BaseConfig
from metamotivo.envs.ogbench import OGBenchEnvConfig
from metamotivo.envs.utils.rollout import rollout
from metamotivo.evaluations.base import extract_model
from metamotivo.nn_models import eval_mode


class OGBenchRewardEvalConfig(BaseConfig):
    name: tp.Literal["ogbench_reward_eval"] = "ogbench_reward_eval"
    name_in_logs: str = "reward"
    env: OGBenchEnvConfig

    tasks: list[str] = pydantic.Field(default_factory=lambda: [])
    num_episodes: int = 100

    num_inference_samples: int = 50_000
    shift_reward: float = 0.0
    disable_tqdm: bool = True
    record_video: bool = False

    def build(self):
        return OGBenchRewardEvaluation(self)


class OGBenchRewardEvaluation:
    def __init__(self, config: OGBenchRewardEvalConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, replay_buffer, logger, **kwargs):
        wandb_dict = {}
        eval_metrics = {}
        model = extract_model(agent_or_model)

        pbar = tqdm.tqdm(self.cfg.tasks, leave=False, disable=self.cfg.disable_tqdm)
        for task in pbar:
            pbar.set_description(f"task {task}")
            eval_env, _ = self.cfg.env.model_copy(update={"task": task}).build()
            pbar.set_description(f"task {task} (inference)")
            ctx, relabel_metrics = self._reward_inference(agent_or_model, task, replay_buffer)
            pbar.set_description(f"task {task} (rollout)")
            # Removed redundant ctx = [None] wrap
            with torch.no_grad(), eval_mode(model):
                # We now do a single rollout for all episodes.
                # If record_video is True, it will capture frames for the first episode.
                st, infos, frames = rollout(
                    eval_env,
                    agent=model,
                    num_episodes=self.cfg.num_episodes,
                    ctx=ctx,
                    render=self.cfg.record_video,
                    render_num_episodes=1,  # only record video for the first episode
                )

                if self.cfg.record_video and frames:
                    video = np.stack(frames).transpose(0, 3, 1, 2)  # T, C, H, W
                    wandb_dict[f"videos/{task}"] = wandb.Video(video, fps=30, format="mp4")

            st["success"] = [any([step.get("success", False) for step in info]) for info in infos]
            eval_metrics[task] = st
            wandb_dict[f"{task}/reward"] = np.mean(st["reward"])
            wandb_dict[f"{task}/success"] = np.mean(st["success"])
            wandb_dict[f"{task}/reward#std"] = np.std(st["reward"])
            for k, v in relabel_metrics.items():
                wandb_dict[f"{task}/{k}"] = v
            eval_env.close()
            del eval_env

        rewards = np.concatenate([el["reward"] for el in eval_metrics.values()])
        wandb_dict["eval/reward"] = np.mean(rewards)
        wandb_dict["eval/reward#std"] = np.std(rewards)
        successes = np.concatenate([el["success"] for el in eval_metrics.values()])
        wandb_dict["eval/success"] = np.mean(successes)

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
        num_samples = self.cfg.num_inference_samples
        batch = replay_buffer["train"].sample(num_samples)
        rewards = self.cfg.env.get_relabel_fn(task)(batch["next"]["physics"].cpu().numpy(), batch["action"].cpu().numpy())
        rewards += self.cfg.shift_reward

        non_zero_rewards = np.count_nonzero(rewards.ravel())
        relabel_metrics = {
            "relabel_reward#mean": np.mean(rewards),
            "relabel_reward#nonzero": non_zero_rewards,
            "relabel_reward#zero": rewards.size - non_zero_rewards,
            "relabel_reward#num_samples": rewards.size,
        }
        z = agent_or_model._model.reward_inference(
            next_obs=batch["next"]["observation"].to(agent_or_model.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=agent_or_model.device),
        )
        return z.reshape(1, -1), relabel_metrics
