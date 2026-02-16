# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from functools import partial

import gymnasium
import numpy as np
from ogbench.utils import make_env_and_datasets

from metamotivo.base import BaseConfig

from .utils.wrappers import PixelWrapper

CUBE_DOMAINS = ["cube-single-play-v0", "cube-double-play-v0"]
PUZZLE_DOMAINS = ["puzzle-3x3-play-v0"]
SCENE_DOMAINS = ["scene-play-v0"]
ANT_DOMAINS = []
for size in ["medium", "large", "giant"]:
    for data_type in ["navigate", "stitch", "explore"]:
        ANT_DOMAINS += [f"antmaze-{size}-{data_type}-v0"]
ALL_DOMAINS = CUBE_DOMAINS + PUZZLE_DOMAINS + SCENE_DOMAINS + ANT_DOMAINS
ALL_TASKS = {}
for d in ALL_DOMAINS:
    ALL_TASKS.update({d: [d[:-2] + f"singletask-task{i + 1}-" + d[-2:] for i in range(5)]})


def cube_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    num_cubes = target_position.shape[0]
    # find all cubes
    cube_positions = [qpos[..., 14:17], qpos[..., 21:24], qpos[..., 28:31], qpos[..., 35:38]][:num_cubes]
    # find distance to target for all cubes
    distances = [np.linalg.norm(cpos - tpos, axis=-1) for cpos, tpos in zip(cube_positions, target_position)]
    # check which cubes are close enough
    successes = sum([(d < threshold).astype(float) for d in distances])
    # the reward is the negative number of misplaced cubes
    return (successes - num_cubes).reshape(-1, 1)


def puzzle_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray) -> np.ndarray:
    return (qpos[:, -len(target_position) :] == target_position).sum(-1, keepdims=True).astype(float) - len(target_position)


def scene_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    cube = (np.linalg.norm(qpos[..., 14:17] - target_position[:3], axis=-1) <= threshold).astype(float)
    button1 = (qpos[..., -2] == target_position[3]).astype(float)
    button2 = (qpos[..., -1] == target_position[4]).astype(float)
    drawer = (np.abs(qpos[..., -4] - target_position[5]) <= threshold).astype(float)
    window = (np.abs(qpos[..., -3] - target_position[6]) <= threshold).astype(float)
    return (cube + button1 + button2 + drawer + window - 5).reshape(-1, 1)


def ant_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.linalg.norm(qpos[..., :2] - target_position, axis=-1).reshape(-1, 1) <= threshold) - 1.0


def create_ogbench_env(
    task,
    wrappers=None,
    seed=None,
    render_height: int = 256,
    render_width: int = 256,
    obs_type: str = "state",
):
    match obs_type:
        case "state":
            task = task
        case "pixels":
            task = "visual-" + task
        case _:
            raise ValueError(f"Unsupported observation type {obs_type}")
    env_kwargs = {"height": render_height, "width": render_width, "render_mode": "rgb_array"}
    env = make_env_and_datasets(task, env_only=True, **env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    env.reset(seed=seed)  # this is used to pass the seed to the environment
    return env, {}


class OGBenchEnvConfig(BaseConfig):
    name: tp.Literal["ogbench"] = "ogbench"

    domain: tp.Literal[tuple(ALL_DOMAINS)]
    task: str

    seed: int = 0

    # observation type
    # state_pixels is not supported atm
    obs_type: tp.Literal["state", "pixels"] = "state"

    # vision based parameter
    camera_id: int | None = None
    render_height: int = 256
    render_width: int = 256
    frame_stack: int = 1

    def build(self) -> tp.Tuple[gymnasium.Env, tp.Any]:
        wrappers = []
        if self.obs_type == "pixels":
            wrappers.append(lambda env: PixelWrapper(env, self.frame_stack))
        return create_ogbench_env(
            task=self.task,
            seed=self.seed,
            wrappers=wrappers,
            render_height=self.render_height,
            render_width=self.render_width,
            obs_type=self.obs_type,
        )

    def get_relabel_fn(self, task):
        env = make_env_and_datasets(task, env_only=True)
        env.reset()  # necessary for antmaze
        if self.domain in CUBE_DOMAINS:
            return partial(cube_reward_fn, target_position=env.unwrapped.cur_task_info["goal_xyzs"])
        if self.domain in PUZZLE_DOMAINS:
            return partial(puzzle_reward_fn, target_position=env.unwrapped.cur_task_info["goal_button_states"])
        if self.domain in SCENE_DOMAINS:
            return partial(
                scene_reward_fn,
                target_position=np.concatenate([np.array([v]).ravel() for v in env.unwrapped.cur_task_info["goal"].values()]),
            )
        if self.domain in ANT_DOMAINS:
            return partial(ant_reward_fn, target_position=env.unwrapped.get_oracle_rep())
        raise NotImplementedError("Unknown relabeling function for domain: ", self.domain)
