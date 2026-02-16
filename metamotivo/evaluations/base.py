# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import metamotivo


def extract_model(agent_or_model):
    if isinstance(agent_or_model, metamotivo.base_model.BaseModel):  # If this is a raw model
        return agent_or_model
    return agent_or_model._model  # If this is an agent
