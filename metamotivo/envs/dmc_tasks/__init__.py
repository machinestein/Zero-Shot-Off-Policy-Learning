# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

ALL_TASKS = {
    "walker": ["stand", "walk", "run", "spin"],
    "cheetah": ["walk", "walk_backward", "run", "run_backward"],
    "quadruped": ["stand", "walk", "run", "jump"],
    "pointmass": [
        "reach_top_left",
        "reach_top_right",
        "reach_bottom_left",
        "reach_bottom_right",
        # "reach_bottom_left_long",
        # "loop",
        # "square",
        # "fast_slow",
    ],
}
