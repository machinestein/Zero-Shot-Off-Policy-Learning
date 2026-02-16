# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ...sf.flow_bc.agent import SFFlowBCAgentConfig, SFFlowBCAgentMixin, SFFlowBCAgentTrainConfig
from ..agent import HilpAgent, HilpAgentConfig, HilpAgentTrainConfig
from .model import HilpFlowBCModelConfig


class HilpFlowBCAgentTrainConfig(SFFlowBCAgentTrainConfig, HilpAgentTrainConfig):
    pass


class HilpFlowBCAgentConfig(SFFlowBCAgentConfig, HilpAgentConfig):
    name: tp.Literal["HilpFlowBCAgent"] = "HilpFlowBCAgent"
    model: HilpFlowBCModelConfig
    train: HilpFlowBCAgentTrainConfig

    @property
    def object_class(self):
        return HilpFlowBCAgent


class HilpFlowBCAgent(SFFlowBCAgentMixin, HilpAgent):
    config_class = HilpFlowBCAgentConfig

    @property
    def optimizer_dict(self):
        return super(HilpAgent, self).optimizer_dict | super(HilpFlowBCAgent, self).optimizer_dict

    def setup_training(self) -> None:
        HilpAgent.setup_training(self)
        SFFlowBCAgentMixin.setup_training(self)
