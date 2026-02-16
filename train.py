# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"  # for headless rendering

import torch

torch.set_float32_matmul_precision("high")

torch._inductor.config.autotune_local_cache = False

import json
import time
import typing as tp
from pathlib import Path
from typing import Dict, List

import gymnasium
import numpy as np
import pydantic
import torch
import tyro
from tqdm.auto import tqdm
import wandb

from metamotivo.agents import Agent
from metamotivo.base import BaseConfig
from metamotivo.data_loading.dmc import DMCDataConfig
from metamotivo.data_loading.ogbench import OGBenchDataConfig
from metamotivo.envs.dmc import DMCEnvConfig
from metamotivo.envs.ogbench import OGBenchEnvConfig
from metamotivo.evaluations.dmc import DMCRewardEvalConfig
from metamotivo.evaluations.ogbench import OGBenchRewardEvalConfig
from metamotivo.misc.loggers import CSVLogger
from metamotivo.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"

CHECKPOINT_DIR_NAME = "checkpoint"


Env = DMCEnvConfig | OGBenchEnvConfig
DataLoading = DMCDataConfig | OGBenchDataConfig

# Stackoverflow #70914419
Evaluation = tp.Annotated[
    tp.Union[DMCRewardEvalConfig, OGBenchRewardEvalConfig],
    pydantic.Field(discriminator="name"),
]


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")

    env: Env = pydantic.Field(discriminator="name")
    data: DataLoading = pydantic.Field(discriminator="name")
    relabel_dataset: bool = False

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("train_dmc"))

    seed: int = 0
    log_every_updates: int = 5_000
    num_train_steps: int = 1_000_000
    checkpoint_every_steps: int = 250_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    buffer_device: str | None = None  # if None, use the agent's device

    # eval
    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])

    eval_every_steps: int = 100_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    def model_post_init(self, context):
        if self.relabel_dataset:
            if not isinstance(self.env, (DMCEnvConfig, OGBenchEnvConfig)):
                raise ValueError("Relabeling is only supported for DMC and OGBench environments")

    def build(self):
        return Workspace(self)


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        # read train status
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device, **agent_build_kwargs)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time


def init_wandb(cfg: TrainConfig):
    wandb_config = cfg.model_dump()
    wandb.init(project=cfg.wandb_pname, group=cfg.wandb_gname, name=cfg.wandb_ename, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        sample_env, _ = cfg.env.build()
        self.obs_space = sample_env.observation_space
        assert isinstance(self.obs_space, gymnasium.spaces.Box), "Only Box observation spaces are supported"

        self.action_space = sample_env.action_space
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported"
        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir,
            self.cfg,
            agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim),
        )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        elif isinstance(self.cfg.evaluations, dict):
            self.evaluations = {name: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0
        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name, eval_cfg in self.evaluations.items()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        buffer_device = self.agent.device if self.cfg.buffer_device is None else self.cfg.buffer_device
        relabel_fn = self.cfg.env.get_relabel_fn(self.cfg.env.task) if self.cfg.relabel_dataset else None
        replay_buffer = self.cfg.data.build(buffer_device, self.cfg.agent.train.batch_size, self.cfg.env.frame_stack, relabel_fn)
        # print(replay_buffer["train"])

        total_metrics = None
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)
        pbar = tqdm(range(self._checkpoint_time, int(self.cfg.num_train_steps) + 1), leave=True,
                    dynamic_ncols=True, smoothing=0.1, colour='yellow', position=0)
        for t in pbar:
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if self.evaluate and eval_time_checker.check(t):
                eval_time_checker.update_last_step(t)
                self.eval(t, replay_buffer=replay_buffer)

            metrics = self.agent.update(replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if log_time_checker.check(t):
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                total_metrics = None
                fps_start_time = time.time()
        return

    def eval(self, t, replay_buffer):
        evaluation_results = {}

        self.agent._model.train(False)

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        all_wandb_dict = {}
        for evaluation_name in self.evaluations:
            evaluation = self.evaluations[evaluation_name]
            logger = self.eval_loggers[evaluation_name]

            evaluation_metrics, wandb_dict = evaluation.run(
                timestep=t,
                agent_or_model=self.agent,
                replay_buffer=replay_buffer,
                logger=logger,
            )
            # Collect wandb dict
            if wandb_dict is not None:
                for k, v in wandb_dict.items():
                    all_wandb_dict[f"eval/{evaluation_name}/{k}"] = v

            evaluation_results[evaluation_name] = evaluation_metrics

        # For wandb dict, put it on wandb
        if self.cfg.use_wandb and all_wandb_dict:
            wandb.log(all_wandb_dict, step=t)

        # ---------------------------------------------------------------
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    workspace = tyro.cli(Workspace)
    try:
        workspace.train()
        wandb.finish()
    except KeyboardInterrupt:
        print("Finishing wandb run")
        wandb.finish()
