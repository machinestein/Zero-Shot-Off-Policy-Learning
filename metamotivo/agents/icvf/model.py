# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp

from ...base_model import load_model
from ..sf.model import SFModel, SFModelArchiConfig, SFModelConfig
from .nn_models import ICVFForwardArchiConfig


class ICVFModelArchiConfig(SFModelArchiConfig):
    t: ICVFForwardArchiConfig = ICVFForwardArchiConfig()


class ICVFModelConfig(SFModelConfig):
    name: tp.Literal["ICVFModel"] = "ICVFModel"

    archi: ICVFModelArchiConfig = ICVFModelArchiConfig()

    @property
    def object_class(self):
        return ICVFModel


class ICVFModel(SFModel):
    def __init__(self, obs_space, action_dim, cfg: ICVFModelConfig):
        super().__init__(obs_space, action_dim, cfg)

        assert cfg.archi.z_dim == cfg.archi.L_dim, "Latent dimensions need to match for ICVF"
        self._t = self.cfg.archi.t.build(self._left_encoder.output_space, self.cfg.archi.z_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_t = copy.deepcopy(self._t)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True, build_kwargs: dict[str, tp.Any] | None = None) -> "ICVFModel":
        return load_model(path, device, strict=strict, config_class=ICVFModelConfig, build_kwargs=build_kwargs)
