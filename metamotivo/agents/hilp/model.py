# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp

from ...base_model import load_model, save_model
from ..sf.model import SFModel, SFModelConfig


class HilpModelConfig(SFModelConfig):
    name: tp.Literal["HilpModel"] = "HilpModel"

    def build(self, obs_space, action_dim) -> "HilpModel":
        return HilpModel(obs_space, action_dim, self)


class HilpModel(SFModel):
    config_class = HilpModelConfig

    def __init__(self, obs_space, action_dim, cfg: HilpModelConfig):
        super().__init__(obs_space, action_dim, cfg)

        arch = self.cfg.archi
        feat_obs_space = self._rgb_encoder.output_space
        self._features2 = arch.features.build(feat_obs_space, arch.z_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_features2 = copy.deepcopy(self._features2)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None) -> "HilpModel":
        return load_model(path, device, strict=strict, config_class=cls.config_class, build_kwargs=build_kwargs)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
