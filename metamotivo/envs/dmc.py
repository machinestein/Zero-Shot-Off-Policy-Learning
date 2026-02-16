# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import time
import typing as tp

import gymnasium
import mujoco
import numpy as np

from metamotivo.base import BaseConfig
from metamotivo.envs.dmc_tasks import dmc
from metamotivo.envs.dmc_tasks.dmc2gym import DmcGymWrapper

from .utils.wrappers import PixelWrapper

CAMERA_IDS = {
    "walker": 0,
    "cheetah": 0,
    "quadruped": 2,
    "pointmass": 0,
}


def create_dmc_env(
    wrappers=None,
    **env_kwargs,
):
    import random

    seed = env_kwargs.pop("seed", random.randint(0, 9999))
    env = dmc.make(f"{env_kwargs.get('domain')}_{env_kwargs.get('task')}", seed=env_kwargs.get("seed", 1))
    env = DmcGymWrapper(
        env,
        height=env_kwargs.get("render_height", 256),
        width=env_kwargs.get("render_width", 256),
        camera_id=env_kwargs.get("camera_id", 0),
        obs_type=env_kwargs.get("obs_type", "state"),
    )
    if wrappers is not None:
        for wrapper in wrappers:
            env = wrapper(env)
    env.reset(seed=seed)  # this is used to pass the seed to the environment
    return env, {}


class DMCEnvConfig(BaseConfig):
    name: tp.Literal["dmc"] = "dmc"

    domain: tp.Literal["walker", "cheetah", "quadruped", "pointmass"]
    task: str

    seed: int = 0

    # observation type
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
        return create_dmc_env(
            domain=self.domain,
            task=self.task,
            seed=self.seed,
            wrappers=wrappers,
            obs_type=self.obs_type,
            camera_id=self.camera_id or CAMERA_IDS[self.domain],
            render_height=self.render_height,
            render_width=self.render_width,
        )

    def get_relabel_fn(self, task):
        def fn(next_physics, actions):
            print("Relabeling train buffer.")
            env = dmc.make(f"{self.domain}_{task}")
            start = time.time()
            reward = np.zeros((len(actions), 1))
            for i in range(len(actions)):
                with env._physics.reset_context():
                    env._physics.set_state(next_physics[i])
                    env._physics.set_control(actions[i])
                mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                reward[i] = env._task.get_reward(env._physics)
            print(f"Relabeling time: {time.time() - start}s")
            return reward

        return fn
