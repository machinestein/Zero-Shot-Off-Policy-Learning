# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import tyro


# This script was not used for TD-JEPA, see extract_all.py instead
def main(
    dataset_path="/fsx-unicorns/shared/datasets/ogbench_original/visual-cube-single-play-v0.npz",
    output_folder: str = "/fsx-unicorns/shared/datasets/ogbench_original/splitted/",
) -> None:
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  1, ...]

    dataset_path = Path(dataset_path)
    task = str(dataset_path.name)
    task = task.replace(".npz", "")
    print(f"Task: {task}")
    breakpoint()

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    file = np.load(dataset_path)
    data = {}
    for k in file.files:
        assert file[k].shape[0] == file["terminals"].shape[0]
        data[k] = file[k]
    del file
    nz = data["terminals"].ravel().nonzero()[0]
    ends = np.arange(data["terminals"].shape[0])[nz]
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts
    ep_idx = 0
    for start, length, end in zip(starts, lengths, ends):
        # print(f"Trajectory length: {length}")
        # print(f"Start index: {start}")
        # print(f"End index: {start + length - 1}")
        assert start + length == end
        npz_vars = {}
        for k in data.keys():
            # plus one is to include the last index
            npz_vars[k] = data[k][start : end + 1]
            # TODO: do we need the same representation as in EXORL?
            # # shift the actions by one, the first is not used
            # if k == "actions":
            #     npz_vars[k] = np.concatenate((np.zeros(1, npz_vars[k].shape[1]), npz_vars[k][:-1]), axis=0)
            # shoudld we add discount?

        filename = f"ogbench_trajectory_{ep_idx}_{length}.npz"
        np.savez_compressed(filename, **npz_vars)
        ep_idx += 1


if __name__ == "__main__":
    tyro.cli(main)
