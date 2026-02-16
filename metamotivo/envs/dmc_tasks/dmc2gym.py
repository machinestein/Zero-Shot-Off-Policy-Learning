# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from collections import OrderedDict
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, Literal

import gymnasium
import mujoco
import numpy as np
from dm_control.rl.control import flatten_observation
from dm_env import _environment as dm_env_environment
from dm_env import specs
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec


def extract_min_max(s):
    assert s.dtype == np.float64 or s.dtype == np.float32
    dim = int(np.prod(s.shape))
    if isinstance(s, specs.BoundedArray):
        zeros = np.zeros(dim, dtype=np.float32)
        return s.minimum + zeros, s.maximum + zeros
    elif isinstance(s, specs.Array):
        bound = np.inf * np.ones(dim, dtype=np.float32)
        return -bound, bound


def _spec_to_gym(spec, dtype=None):
    if isinstance(spec, Iterable):
        output = {}
        for s in spec:
            mn, mx = extract_min_max(s)
            _dtype = dtype or mn.dtype
            output[s.name] = spaces.Box(mn, mx, dtype=_dtype)
        output = spaces.Dict(output)
    else:
        mn, mx = extract_min_max(spec)
        _dtype = dtype or mn.dtype
        output = spaces.Box(mn, mx, dtype=_dtype)

    return output


class ObsType(Enum):
    state = 0
    pixels = 1
    state_pixels = 2


class DmcGymWrapper(gymnasium.Env):
    def __init__(
        self,
        env: dm_env_environment.Environment,
        height: int = 300,
        width: int = 300,
        camera_id: int = 0,
        seed: int = 0,
        obs_type: Literal["state", "pixels"] = "state",
    ):
        self.obs_type = ObsType[obs_type]
        self.render_mode = "rgb_array"
        self._render_height = height
        self._render_width = width
        self._render_camera_id = camera_id
        self._env = env
        self.spec = EnvSpec("dmc_gym_env", max_episode_steps=int(self._env._step_limit))

        self._action_space = _spec_to_gym(self._env.action_spec())
        match self.obs_type:
            case ObsType.state:
                observation_space = _spec_to_gym(self._env.observation_spec())
            case ObsType.pixels:
                observation_space = spaces.Box(low=0, high=255, shape=(self._render_height, self._render_width, 3), dtype=np.uint8)
            case _:
                raise ValueError(f"Unknown observation type: {self.obs_type}")
        self._observation_space = observation_space
        self.reset(seed=seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        reward = timestep.reward
        termination = False
        truncation = timestep.last()
        return self._get_obs(timestep), reward, termination, truncation, self._get_info(timestep)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            assert isinstance(seed, numbers.Integral), "Seed must be an integer."
            self._env.task._random = np.random.RandomState(seed=seed)
        if options is None:
            timestep = self._env.reset()
        else:
            physics = np.concat([options["qpos"], options["qvel"]])
            with self._env.physics.reset_context():
                self._env.physics.set_state(physics)
                if "action" in options:
                    self._env.physics.set_control(options["action"])
                else:
                    self._env.physics.set_control(np.zeros_like(self._env.physics.data.ctrl))

            mujoco.mj_forward(self._env.physics.model.ptr, self._env.physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(self._env.physics.model.ptr, self._env.physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(self._env.physics.model.ptr, self._env.physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(self._env.physics.model.ptr, self._env.physics.data.ptr)  # pylint: disable=no-member

            observation = self._env._task.get_observation(self._env.physics)
            observation = flatten_observation(observation)
            timestep = dm_env_environment.TimeStep(
                step_type=dm_env_environment.StepType.FIRST, reward=None, discount=None, observation=observation["observations"]
            )

        return self._get_obs(timestep), self._get_info(timestep)

    def _get_info(self, timestep) -> Dict[str, np.ndarray]:
        return {"discount": timestep.discount, "physics": self._env.physics.get_state()}

    def _get_obs(self, timestep) -> Dict[str, np.ndarray]:
        obs = OrderedDict()
        match self.obs_type:
            case ObsType.state:
                obs = timestep.observation
            case ObsType.pixels:
                obs = self.render()
            case _:
                raise ValueError(f"Unknown observation type: {self.obs_type}")
        return obs

    def render(self, height=None, width=None, camera_id=None):
        height = height or self._render_height
        width = width or self._render_width
        camera_id = camera_id or self._render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
