# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic
import torch
from gymnasium import spaces
from torch import nn

from metamotivo.base import BaseConfig


class BatchNormNormalizerConfig(BaseConfig):
    momentum: float = 0.01

    def build(self, obs_space) -> "BatchNormNormalizer":
        return BatchNormNormalizer(obs_space, self)


class BatchNormNormalizer(nn.Module):
    def __init__(self, obs_space: spaces.Space, cfg: BatchNormNormalizerConfig):
        super().__init__()
        assert len(obs_space.shape) == 1, "BatchNormNormalizer only supports 1D observation spaces"
        self._normalizer = nn.BatchNorm1d(num_features=obs_space.shape[0], affine=False, momentum=cfg.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalizer(x)


class IdentityNormalizerConfig(BaseConfig):
    def build(self, obs_space) -> nn.Identity:
        return nn.Identity()


class RGBNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize RGB images to [-0.5, 0.5] range."""
        # Assuming x is in [0, 255] range
        return (x / 255.0) - 0.5


class RGBNormalizerConfig(BaseConfig):
    def build(self, obs_space) -> RGBNorm:
        return RGBNorm()


AVAILABLE_NORMALIZERS = tp.Annotated[
    tp.Union[
        BatchNormNormalizerConfig,
        IdentityNormalizerConfig,
        RGBNormalizerConfig,
    ],
    pydantic.Field(discriminator="name"),
]
