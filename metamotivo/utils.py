# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import random
import string
from pathlib import Path

import numpy as np
import torch

N_RANDOM_CHARACTERS = 4


class EveryNStepsChecker:
    def __init__(self, current_step: int, every_n_steps: int, step_zero_should_trigger: bool = True):
        # if step_zero_should_trigger is True, `check` will return True for step=0
        # this is to be consistent with the original modulo logic (i.e. step % N == 0)
        self.step_zero_should_trigger = step_zero_should_trigger
        self.last_step = current_step
        self.every_n_steps = every_n_steps

    def check(self, step: int) -> bool:
        return (step - self.last_step) >= self.every_n_steps or (self.step_zero_should_trigger and step == 0)

    def update_last_step(self, step: int):
        self.last_step = step


def get_unique_name() -> str:
    # Timestamp + unique letters
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = f"{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}:{now.second}"
    random_letters = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N_RANDOM_CHARACTERS))
    return f"{timestamp}-{random_letters}"


def get_local_workdir(name: str = "") -> str:
    return str(Path.cwd() / "workdir" / name / get_unique_name())


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
