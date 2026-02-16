# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ...sf.flow_bc.model import SFFlowBCModelArchiConfig, SFFlowBCModelConfig, SFFlowBCModelMixin
from ..model import HilpModel, HilpModelConfig


class HilpFlowBCModelConfig(SFFlowBCModelConfig, HilpModelConfig):
    name: tp.Literal["HilpFlowBCModel"] = "HilpFlowBCModel"
    archi: SFFlowBCModelArchiConfig = SFFlowBCModelArchiConfig()

    def build(self, obs_space, action_dim) -> "HilpFlowBCModel":
        return HilpFlowBCModel(obs_space, action_dim, self)


class HilpFlowBCModel(SFFlowBCModelMixin, HilpModel):
    config_class = HilpFlowBCModelConfig

    def __init__(self, obs_space, action_dim, cfg: HilpModelConfig):
        HilpModel.__init__(self, obs_space, action_dim, cfg)
        SFFlowBCModelMixin.__init__(self, obs_space, action_dim, cfg)
