# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import gymnasium
import numpy as np

from ...base_model import load_model
from ...nn_models import VForwardArchiConfig
from ..fb.model import FBModel, FBModelArchiConfig, FBModelConfig


class RLDPModelArchiConfig(FBModelArchiConfig):
    predictor: VForwardArchiConfig = VForwardArchiConfig()


class RLDPModelConfig(FBModelConfig):
    name: tp.Literal["RLDPModel"] = "RLDPModel"
    archi: RLDPModelArchiConfig = RLDPModelArchiConfig()

    @property
    def object_class(self):
        return RLDPModel


class RLDPModel(FBModel):
    def __init__(self, obs_space, action_dim, cfg: RLDPModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        z_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.archi.z_dim,), dtype=np.float32)
        self._predictor = cfg.archi.predictor.build(z_space, action_dim, output_dim=self.cfg.archi.z_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None) -> "RLDPModel":
        return load_model(path, device, strict=strict, config_class=RLDPModelConfig, build_kwargs=build_kwargs)
