# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import gymnasium
import numpy as np


class PixelWrapper(gymnasium.wrappers.FrameStackObservation):
    """
    Wrapper designed for environment involving RGB observations.
    Deals with frame stacking, and enforces a channel-first format (CHW).
    """

    def __init__(self, env: gymnasium.Env, num_stack: int = 1):
        observation_space = self._concat_space(env.observation_space, num_stack)
        super().__init__(env, num_stack)
        self.observation_space = observation_space

    def step(self, action: np.ndarray):
        return self._reshape(*super().step(action))

    def reset(self, *, seed: int | None = None, options: tp.Dict[str, tp.Any] | None = None):
        return self._reshape(*super().reset(seed=seed, options=options))

    def _concat_space(self, space: gymnasium.spaces.Space, num_stack=1) -> gymnasium.spaces.Space:
        """Moves the last dimension of the space to the first dimension and repeats it num_stack times."""
        if isinstance(space, gymnasium.spaces.Dict):
            assert "pixels" in space.spaces, "The environment must have a 'pixels' observation space."
            return gymnasium.spaces.Dict({k: self._concat_space(v, num_stack) for k, v in space.items()})
        elif isinstance(space, gymnasium.spaces.Box):
            assert len(space.shape) in [1, 3], "Box space must be 1D or 3D (e.g., (C,) or (H, W, C))"
            return gymnasium.spaces.Box(
                low=np.moveaxis(np.tile(space.low, num_stack), -1, 0),
                high=np.moveaxis(np.tile(space.high, num_stack), -1, 0),
                shape=(space.shape[-1] * num_stack, *space.shape[:-1]),
                dtype=space.dtype,
            )
        raise ValueError(f"Unsupported space type: {type(space)}")

    def _reshape(self, obs, *args):
        """Reshapes the observation to have the channel dimension first, and concatenates frames instead of stacking."""
        if isinstance(obs, dict):
            return {k: self._reshape(v)[0] for k, v in obs.items()}, *args
        return np.moveaxis(obs, -1, 1).reshape(obs.shape[0] * obs.shape[-1], *obs.shape[1:-1]), *args
