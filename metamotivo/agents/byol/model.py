# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import gymnasium
import numpy as np

from ...base_model import load_model, save_model
from ...nn_models import VForwardArchiConfig
from ..sf.model import SFModel, SFModelArchiConfig, SFModelConfig


class BYOLModelArchiConfig(SFModelArchiConfig):
    predictor: VForwardArchiConfig = VForwardArchiConfig()


class BYOLModelConfig(SFModelConfig):
    name: tp.Literal["BYOLModel"] = "BYOLModel"
    archi: BYOLModelArchiConfig = BYOLModelArchiConfig()

    @property
    def object_class(self):
        return BYOLModel


class BYOLModel(SFModel):
    config_class = BYOLModelConfig

    def __init__(self, obs_space, action_dim, cfg: BYOLModelConfig):
        super().__init__(obs_space, action_dim, cfg)

        arch: BYOLModelArchiConfig = self.cfg.archi
        z_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(arch.z_dim,), dtype=np.float32)
        self._predictor = arch.predictor.build(z_space, action_dim, output_dim=arch.z_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None) -> "BYOLModel":
        return load_model(path, device, strict=strict, config_class=cls.config_class, build_kwargs=build_kwargs)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
