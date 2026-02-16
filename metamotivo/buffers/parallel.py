# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclasses.dataclass(kw_only=True)
class Buffer(IterableDataset):
    episode_fns: Tuple  # list of filenames that need to be loaded across workers
    load_fn: callable  # function that loads the data from the filenames
    relabel_fn: Optional[callable]  # relabeling function
    num_workers: int = 0
    batch_size: int = 1
    frame_stack: int = 1
    frame_stack_key: str = "observation"
    obs_type: str = "pixels"
    output_key: Tuple[str] = ("observation", "physics", "action", "reward")
    output_key_next: Tuple[str] = ("observation", "physics", "terminated")
    end_key: str = "truncated"
    # if future < 1, additionally sample an observation in the future of each transition
    # this is done by geometric distribution with p=1-future
    future: float = 0.99
    H: int = 1  # length of trajectories to load. only RLDP uses H\=1

    def __post_init__(self) -> None:
        self.storage = None
        self.sampleable_idxs = None
        self.timesteps = None
        self.ready = False

    def _fetch(self):
        # loads the assigned part of the dataset
        try:  # find the worker id
            worker_id = torch.utils.data.get_worker_info().id
        except:  # noqa: E722
            worker_id = 0
        # only load the correct share of all episodes
        episode_fns = [fn for i, fn in enumerate(self.episode_fns) if i % max(1, self.num_workers) == worker_id]
        self.storage = self.load_fn(episode_fns, self.obs_type)  # trajectory storage
        self.sampleable_idxs = ~self.storage[self.end_key]  # mark states from which a transition (or sequence) can be sampled
        for _ in range(self.H - 1):  # if working with sequences, propagate done signal backward to avoid sampling across episode ends
            for i in range(1, len(self.sampleable_idxs)):
                self.sampleable_idxs[i - 1] = self.sampleable_idxs[i - 1] if self.sampleable_idxs[i] else self.sampleable_idxs[i]
        self.sampleable_idxs = np.where(self.sampleable_idxs)[0]  # idxs that contain the first state in a valid transition
        self.timesteps = np.zeros(len(self.storage[self.end_key]), dtype=np.int32)  # temporal distance from first state in the trajectory
        self.lengths = np.zeros(len(self.storage[self.end_key]), dtype=np.int32)  # length of each trajectory
        prev = 0  # now compute lengths of all episodes
        for i in range(1, len(self.timesteps)):
            if not self.storage[self.end_key][i - 1]:
                self.timesteps[i] = self.timesteps[i - 1] + 1
            else:
                self.timesteps[i] = 0
                self.lengths[prev:i] = self.timesteps[i - 1]
                prev = i
        self.lengths[prev:] = self.timesteps[-1]
        if self.relabel_fn is not None:
            self.storage["reward"] = np.zeros((len(self.storage["action"]), 1))
            self.storage["reward"][:-1] = self.relabel_fn(self.storage["physics"][1:], self.storage["action"][:-1])
            self.storage["reward"][self.storage[self.end_key]] = 0.0  # set rewards to zero for truncated episodes
        assert not isinstance(self.storage[self.frame_stack_key], dict), (
            "Only flat observation dictionaries are allowed for the parallel buffer"
        )
        self.ready = True  # ensures this function will not be called any longer

    def _sample(self):
        if not self.ready:
            self._fetch()
        idxs = np.random.choice(self.sampleable_idxs, size=self.batch_size)
        timesteps = self.timesteps[idxs]
        # these offsets are added to the timesteps to sample all observation we will need for one transition, considering frame_stacking and horizon
        offsets = [np.maximum(-i, -timesteps) for i in range(1, self.frame_stack)][::-1] + list(range(self.H + 1))
        # prepare sequence of observations
        obs = [self.storage[self.frame_stack_key][idxs + offset] for offset in offsets]
        batch = {
            # take first frame_stack observations for s
            self.frame_stack_key: (np.concatenate(obs[: self.frame_stack], 1) if self.frame_stack > 1 else obs[0]),
            "next": {
                # shift by one for s'
                self.frame_stack_key: (np.concatenate(obs[1 : 1 + self.frame_stack], 1) if self.frame_stack > 1 else obs[1]),
            },
        }
        # add other keys to batch
        for k in self.output_key:
            if k != self.frame_stack_key and k in self.storage:
                batch[k] = self.storage[k][idxs]
        for k in self.output_key_next:
            if k != self.frame_stack_key and k in self.storage:
                batch["next"][k] = self.storage[k][idxs + 1]

        if self.H > 1:
            # also add actions and states from the rest of the sequence to the batch
            batch["next"]["traj_" + self.frame_stack_key] = np.stack(
                [
                    np.concatenate(obs[1 + h : 1 + h + self.frame_stack], 1) if self.frame_stack > 1 else obs[1 + h]
                    for h in range(1, self.H)
                ],
                1,
            )
            if "action" in self.output_key and "action" in self.storage:
                batch["next"]["traj_action"] = np.stack([self.storage["action"][idxs + h] for h in range(1, self.H)], 1)

        if self.future < 1:
            # sample future states (for HILP of ICVF)
            future_timesteps = timesteps + np.random.geometric(p=(1 - self.future), size=self.batch_size) - 1
            future_timesteps = np.clip(future_timesteps, a_min=None, a_max=self.lengths[idxs] - 1)
            future_idxs = idxs + future_timesteps - timesteps
            future_offsets = [np.maximum(-i, -future_timesteps) for i in range(1, self.frame_stack - 1)][::-1] + [0, 1]
            future_obs = [self.storage[self.frame_stack_key][future_idxs + offset] for offset in future_offsets]
            batch["future_observation"] = np.concatenate(future_obs, 1) if self.frame_stack > 1 else future_obs[-1]

        return batch

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    # see https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate(batch_list, batch_size):
    batch = {}
    for k in batch_list[0].keys():
        if isinstance(batch_list[0][k], dict):
            batch[k] = _collate([b[k] for b in batch_list], batch_size)
        else:
            batch[k] = torch.cat([b[k] for b in batch_list], 0)[:batch_size]
    return batch


class ParallelBuffer:
    # This buffer is designed for learning from pixels: in this setting, it should be faster than other buffers.
    # It only supports offline learning: to extend it to online learning, each worker could have a routine that
    # checks a shared folder for newly saved .npz episodes, and loads then when available. See Denis's DrQv2
    # codebase for an example.
    def __init__(
        self, episode_fns, load_fn, batch_size, frame_stack=1, obs_type="pixels", relabel_fn=None, num_workers=8, future=0.99, horizon=1
    ):
        self._batch_size = batch_size  # for compatibility with standard dataloaders, we use a fixed batch size, but we will manually collate if larger ones are requested
        iterable = Buffer(
            episode_fns=episode_fns,
            load_fn=load_fn,
            relabel_fn=relabel_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            frame_stack=frame_stack,
            obs_type=obs_type,
            future=future,
            H=horizon,
        )
        self.loader = torch.utils.data.DataLoader(
            iterable, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn
        )
        self._replay_iter = iter(self.loader)

    @torch.no_grad
    def sample(self, batch_size):
        if batch_size == self._batch_size:
            return next(self._replay_iter)
        # collate batches to the requested batch size
        # useful for regression of reward embeddings
        n_batches = 1 + batch_size // self._batch_size
        return _collate([next(self._replay_iter) for _ in range(n_batches)], batch_size)
