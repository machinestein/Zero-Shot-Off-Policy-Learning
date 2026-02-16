# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ...sf.flow_bc.model import SFFlowBCModelArchiConfig, SFFlowBCModelConfig, SFFlowBCModelMixin
from ..model import ICVFModel, ICVFModelArchiConfig, ICVFModelConfig


class ICVFFlowBCModelArchiConfig(SFFlowBCModelArchiConfig, ICVFModelArchiConfig):
    pass


class ICVFFlowBCModelConfig(SFFlowBCModelConfig, ICVFModelConfig):
    name: tp.Literal["ICVFFlowBCModel"] = "ICVFFlowBCModel"
    archi: ICVFFlowBCModelArchiConfig = ICVFFlowBCModelArchiConfig()

    def build(self, obs_space, action_dim) -> "ICVFFlowBCModel":
        return ICVFFlowBCModel(obs_space, action_dim, self)


class ICVFFlowBCModel(SFFlowBCModelMixin, ICVFModel):
    config_class = ICVFFlowBCModelConfig

    def __init__(self, obs_space, action_dim, cfg: ICVFModelConfig):
        ICVFModel.__init__(self, obs_space, action_dim, cfg)
        SFFlowBCModelMixin.__init__(self, obs_space, action_dim, cfg)
