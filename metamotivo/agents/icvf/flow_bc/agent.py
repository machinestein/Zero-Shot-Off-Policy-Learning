# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ...sf.flow_bc.agent import SFFlowBCAgentConfig, SFFlowBCAgentMixin, SFFlowBCAgentTrainConfig
from ..agent import ICVFAgent, ICVFAgentConfig, ICVFAgentTrainConfig
from .model import ICVFFlowBCModelConfig


class ICVFFlowBCAgentTrainConfig(SFFlowBCAgentTrainConfig, ICVFAgentTrainConfig):
    pass


class ICVFFlowBCAgentConfig(SFFlowBCAgentConfig, ICVFAgentConfig):
    name: tp.Literal["ICVFFlowBCAgent"] = "ICVFFlowBCAgent"
    model: ICVFFlowBCModelConfig
    train: ICVFFlowBCAgentTrainConfig

    @property
    def object_class(self):
        return ICVFFlowBCAgent


class ICVFFlowBCAgent(SFFlowBCAgentMixin, ICVFAgent):
    config_class = ICVFFlowBCAgentConfig

    @property
    def optimizer_dict(self):
        return super(ICVFAgent, self).optimizer_dict | super(ICVFFlowBCAgent, self).optimizer_dict

    def setup_training(self) -> None:
        ICVFAgent.setup_training(self)
        SFFlowBCAgentMixin.setup_training(self)
