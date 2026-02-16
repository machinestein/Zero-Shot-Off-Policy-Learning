# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

from metamotivo.data_loading.base import BaseDataConfig
from metamotivo.envs.ogbench import ALL_DOMAINS


class OGBenchDataConfig(BaseDataConfig):
    name: tp.Literal["ogbench"] = "ogbench"
    domain: tp.Literal[tuple(ALL_DOMAINS)]
    load_n_episodes: int = 1_000

    def build(self, buffer_device, batch_size, frame_stack, relabel_fn=None) -> tp.Dict:
        domain = ("visual-" if self.obs_type == "pixels" else "") + self.domain
        path = Path(self.dataset_root) / f"{domain}/buffer"
        return super().build_from_path(path, buffer_device, batch_size, frame_stack, relabel_fn)
