# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import tyro
from tqdm import tqdm


def extract(
    env_name: str,
    output_folder: str,
    dataset_path: str | None = None,
) -> None:
    if dataset_path is not None:
        dataset_path = Path(dataset_path) / (env_name + ".npz")
    else:
        # download and load from default location
        from ogbench.utils import DEFAULT_DATASET_DIR, make_env_and_datasets

        make_env_and_datasets(env_name)
        dataset_path = Path(DEFAULT_DATASET_DIR).expanduser() / (env_name + ".npz")

    print(f"Dataset: {dataset_path}")
    output_folder = Path(output_folder) / (env_name + "/buffer")
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}")

    data = np.load(dataset_path)
    data = {k: data[k] for k in data.keys()}

    nz = data["terminals"].ravel().nonzero()[0]
    ends = np.arange(data["terminals"].shape[0])[nz]
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts
    ep_idx = 0
    for start, length, end in tqdm(zip(starts, lengths, ends)):
        episode = {}
        episode["observation"] = data["observations"][start : end + 1]
        if "visual" in env_name:
            episode["observation"] = np.zeros((length + 1, 1), dtype=np.float32)
            episode["pixels"] = np.moveaxis(data["observations"][start : end + 1], -1, 1)
        else:
            episode["observation"] = data["observations"][start : end + 1]
            episode["pixels"] = np.zeros((length + 1, 1), dtype=np.float32)
        episode["action"] = data["actions"][start : end + 1]
        episode["action"][1:] = episode["action"][:-1]
        episode["action"][0] = 0.0
        episode["physics"] = data["qpos"][start : end + 1]
        if "button_states" in data.keys():
            episode["physics"] = np.concatenate([episode["physics"], data["button_states"][start : end + 1]], axis=-1)
        episode["reward"] = np.zeros((length + 1, 1), dtype=np.float32)
        episode["discount"] = np.ones((length + 1, 1), dtype=np.float32)

        filename = f"episode_{format(ep_idx, '06')}_{length}.npz"
        np.savez_compressed(output_folder / filename, **episode)
        ep_idx += 1


def main(
    output_folder: str,
    dataset_path: str | None = None,
):
    for env_name in [
        "antmaze-medium-navigate-v0",
        "visual-antmaze-medium-navigate-v0",
        "antmaze-large-navigate-v0",
        "visual-antmaze-large-navigate-v0",
        "antmaze-medium-stitch-v0",
        "visual-antmaze-medium-stitch-v0",
        "antmaze-large-stitch-v0",
        "visual-antmaze-large-stitch-v0",
        "antmaze-medium-explore-v0",
        "visual-antmaze-medium-explore-v0",
        "antmaze-large-explore-v0",
        "visual-antmaze-large-explore-v0",
        "puzzle-3x3-play-v0",
        "visual-puzzle-3x3-play-v0",
        "scene-play-v0",
        "visual-scene-play-v0",
        "cube-single-play-v0",
        "visual-cube-single-play-v0",
        "cube-double-play-v0",
        "visual-cube-double-play-v0",
    ]:
        extract(
            env_name=env_name,
            output_folder=output_folder,
            dataset_path=dataset_path,
        )


if __name__ == "__main__":
    tyro.cli(main)
