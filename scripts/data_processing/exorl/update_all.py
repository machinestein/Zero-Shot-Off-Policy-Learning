# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import tyro

from scripts.data_processing.exorl.update_data import main as update_data_main


def main(
    input_folder: str,
    output_folder: str,
    num_workers: int = 0,
    save_rgb: bool = False,
):
    for domain in ["walker", "cheetah", "quadruped", "pointmass"]:
        update_data_main(
            num_workers=num_workers,
            env_name=domain,
            expl_agent="rnd",
            datasets_dir=input_folder,
            new_dataset_dir=output_folder,
            save_rgb=save_rgb,
        )


if __name__ == "__main__":
    tyro.cli(main)
