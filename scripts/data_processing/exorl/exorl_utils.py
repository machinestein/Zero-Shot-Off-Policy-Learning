# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import numpy as np


def set_seed_everywhere(seed):
    import random

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def load_episode(fn):
    with fn.open("rb") as f:
        try:
            episode = pickle.load(f, allow_pickle=True)
        except Exception:
            episode = np.load(f, allow_pickle=True)
            episode = {k: episode[k] for k in episode.keys()}
        return episode
