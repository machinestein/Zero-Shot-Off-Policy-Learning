# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ...sf.flow_bc.agent import SFFlowBCAgentConfig, SFFlowBCAgentMixin, SFFlowBCAgentTrainConfig
from ..agent import BYOLAgent, BYOLAgentConfig, BYOLAgentTrainConfig
from .model import BYOLFlowBCModelConfig


class BYOLFlowBCAgentTrainConfig(SFFlowBCAgentTrainConfig, BYOLAgentTrainConfig):
    pass


class BYOLFlowBCAgentConfig(SFFlowBCAgentConfig, BYOLAgentConfig):
    name: tp.Literal["BYOLFlowBCAgent"] = "BYOLFlowBCAgent"
    model: BYOLFlowBCModelConfig
    train: BYOLFlowBCAgentTrainConfig

    @property
    def object_class(self):
        return BYOLFlowBCAgent


class BYOLFlowBCAgent(SFFlowBCAgentMixin, BYOLAgent):
    config_class = BYOLFlowBCAgentConfig

    @property
    def optimizer_dict(self):
        return super(BYOLAgent, self).optimizer_dict | super(BYOLFlowBCAgent, self).optimizer_dict

    def setup_training(self) -> None:
        BYOLAgent.setup_training(self)
        SFFlowBCAgentMixin.setup_training(self)
