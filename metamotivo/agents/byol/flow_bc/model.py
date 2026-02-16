# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ...sf.flow_bc.model import SFFlowBCModelArchiConfig, SFFlowBCModelConfig, SFFlowBCModelMixin
from ..model import BYOLModel, BYOLModelArchiConfig, BYOLModelConfig


class BYOLFlowBCModelArchiConfig(SFFlowBCModelArchiConfig, BYOLModelArchiConfig):
    pass


class BYOLFlowBCModelConfig(SFFlowBCModelConfig, BYOLModelConfig):
    name: tp.Literal["BYOLFlowBCModel"] = "BYOLFlowBCModel"
    archi: BYOLFlowBCModelArchiConfig = BYOLFlowBCModelArchiConfig()

    @property
    def object_class(self):
        return BYOLFlowBCModel


class BYOLFlowBCModel(SFFlowBCModelMixin, BYOLModel):
    config_class = BYOLFlowBCModelConfig

    def __init__(self, obs_space, action_dim, cfg: BYOLModelConfig):
        BYOLModel.__init__(self, obs_space, action_dim, cfg)
        SFFlowBCModelMixin.__init__(self, obs_space, action_dim, cfg)
