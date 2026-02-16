# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from metamotivo.base import BaseConfig
from metamotivo.buffers.parallel import ParallelBuffer
from metamotivo.buffers.transition import DictBuffer


def tree_check_batch_size(x, batch_size, prefix=""):
    """Manual recursive check the batch size (first dim) of pytree of tensors"""
    if isinstance(x, dict):
        for key, item in x.items():
            tree_check_batch_size(item, batch_size, prefix=f"{prefix}.{key}")
    elif isinstance(x, torch.Tensor):
        if x.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch at {prefix}: expected {batch_size}, got {x.shape[0]}")


def load_transitions(
    episode_fns: tp.List[str],
    obs_type: tp.Literal["state", "pixels"] = "state",
):
    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {"observation": [], "terminated": [], "physics": []},
    }
    for f in episode_fns:
        data = np.load(str(f))
        match obs_type:
            case "state":
                obs_key = "observation"
            case "pixels":
                obs_key = "pixels"
            case _:
                raise ValueError(f"Unknown observation type {obs_type}")
        storage["observation"].append(data[obs_key][:-1].astype(np.float32))
        storage["next"]["observation"].append(data[obs_key][1:].astype(np.float32))
        storage["action"].append(data["action"][1:].astype(np.float32))
        storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=bool))
        storage["physics"].append(data["physics"][:-1])
        storage["next"]["physics"].append(data["physics"][1:])

    # Concatenate all the individual tensors into single tensors
    # `is_leaf` determines on which items we shoud apply the `np.concat` function, which in this case is the list of arrays to concat
    storage = tree_map(lambda x: np.concatenate(x), storage, is_leaf=lambda x: isinstance(x, list))

    expected_n_items = storage["action"].shape[0]
    tree_check_batch_size(storage, expected_n_items)

    return storage


def load_trajectories(
    episode_fns: tp.List[str],
    obs_type: tp.Literal["state", "pixels"] = "state",
):
    data = np.load(str(episode_fns[0]))
    # EXORL data only has fixed-length trajectories
    # in order to handle trajectories of varying length we would need to read all files here and compute the total number of transitions
    traj_len = data["observation"].shape[0]
    n = traj_len * len(episode_fns)
    match obs_type:
        case "state":
            obs_storage = np.zeros((n, data["observation"].shape[1]), dtype=np.float32)
        case "pixels":
            obs_storage = np.zeros((n, *data["pixels"].shape[1:]), dtype=data["pixels"].dtype)
        case _:
            raise ValueError(f"Unknown observation type {obs_type}")
    storage = {
        "observation": obs_storage,
        "action": np.zeros((n, data["action"].shape[1]), dtype=np.float32),
        "physics": np.zeros((n, data["physics"].shape[1]), dtype=data["physics"].dtype),
        "truncated": np.zeros((n, 1), dtype=bool),
        "terminated": np.zeros((n, 1), dtype=bool),
    }

    idx = 0
    for f in episode_fns:
        data = np.load(str(f))
        n = data["observation"].shape[0]
        assert n == traj_len, f"All trajectories must have the same lengths. Found {traj_len} and {n}"
        match obs_type:
            case "state":
                storage["observation"][idx : idx + n] = data["observation"].astype(np.float32)
            case "pixels":
                storage["observation"][idx : idx + n] = data["pixels"]
            case _:
                raise ValueError(f"Unknown observation type {obs_type}")
        act = np.concatenate([data["action"][1:].astype(np.float32), np.zeros((1, data["action"].shape[1]), dtype=np.float32)], axis=0)
        storage["action"][idx : idx + n] = act
        terminated = np.concatenate([np.zeros((1, 1), dtype=bool), np.array(1 - data["discount"][1:], dtype=bool)], axis=0)
        storage["terminated"][idx : idx + n] = terminated
        truncated = np.zeros_like(terminated, dtype=bool)
        truncated[-1] = 1
        storage["truncated"][idx : idx + n] = truncated
        storage["physics"][idx : idx + n] = data["physics"]
        idx += n

    expected_n_items = storage["action"].shape[0]
    tree_check_batch_size(storage, expected_n_items)
    return storage


class BaseDataConfig(BaseConfig):
    name: str
    domain: str
    dataset_root: str
    load_n_episodes: int = 5_000
    obs_type: tp.Literal["state", "pixels", "state_pixels"] = "state"
    buffer_type: tp.Literal["dict", "parallel"] = "dict"
    future: float = 0.99
    num_workers: int = 8  # for parallel buffer
    horizon: int = 1  # for RLDP only

    def build(self, buffer_device, batch_size, frame_stack, relabel_fn=None) -> tp.Dict:
        raise NotImplementedError

    def build_from_path(self, path, buffer_device, batch_size, frame_stack, relabel_fn=None) -> tp.Dict:
        print(f"Loading data from: {path}")
        files = list(Path(path).glob("*.npz"))
        if len(files) == 0:
            raise FileNotFoundError(f"No .npz files found at: {path}")
        num_episodes = min(self.load_n_episodes, len(files))

        buffer_type = self.buffer_type
        if self.obs_type == "pixels":
            print("Enforcing parallel buffer when learning from pixels.")
            buffer_type = "parallel"

        match buffer_type:
            case "dict":
                data = load_transitions(
                    files[:num_episodes],
                    self.obs_type,
                )
                replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=buffer_device)}
                replay_buffer["train"].extend(data)
                del data
                if relabel_fn is not None:
                    rewards = relabel_fn(
                        replay_buffer["train"].storage["next"]["physics"].cpu().numpy(),
                        replay_buffer["train"].storage["action"].cpu().numpy(),
                    )
                    replay_buffer["train"].storage["reward"] = torch.tensor(
                        rewards, dtype=torch.float32, device=replay_buffer["train"].device
                    )
            case "parallel":
                replay_buffer = {
                    "train": ParallelBuffer(
                        files[:num_episodes],
                        load_trajectories,
                        batch_size=batch_size,
                        frame_stack=frame_stack,
                        obs_type=self.obs_type,
                        relabel_fn=relabel_fn,
                        future=self.future,
                        num_workers=self.num_workers,
                        horizon=self.horizon,
                    )
                }
            case _:
                raise ValueError(f"Unsupported buffer type {self.buffer_type}")
        return replay_buffer
