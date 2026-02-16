# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import functools
import numbers
import os
from multiprocessing import cpu_count, current_process, get_context
from pathlib import Path
from typing import Any, List, Literal

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import tqdm
import tyro

from metamotivo.envs.dmc_tasks import dmc as cdmc
from scripts.data_processing.exorl.exorl_utils import get_domain, load_episode, set_seed_everywhere


def _atleast2d(x):
    if x.ndim == 1:
        return np.expand_dims(x, axis=1)
    return x


def relabel_episode(env, episode, env_name, save_rgb=False):
    states = episode["physics"]
    state0 = states[0]
    og_actions = episode["action"]

    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(state0)
    obs = [np.hstack([o for o in env.task.get_observation(env.physics).values()])]
    physics = [env.physics.get_state()]
    new_actions = [og_actions[0]]
    if episode["reward"][0].size > 1:
        # Ad hoc for maze rnd data since reward are 4 dimensional vector
        rewards = [episode["reward"][0][0] if isinstance(episode["reward"][0][0], numbers.Number) else episode["reward"][0][0].item()]
    else:
        rewards = [episode["reward"][0].item()]
    discount = [episode["discount"][0].item()]

    if save_rgb:
        size = 64
        camera_id = ({"quadruped": 2}).get(env_name, 0)
        img = env.physics.render(camera_id=camera_id, height=size, width=size)
        img = np.moveaxis(img, 2, 0)
        image_list = [img]

    for action in og_actions[1:]:
        time_step = env.step(action)
        phy = env.physics.get_state()
        obs.append(time_step.observation["observations"])
        physics.append(phy)
        new_actions.append(action)
        rewards.append(time_step.reward)
        discount.append(time_step.discount)

        if save_rgb:
            img = env.physics.render(camera_id=camera_id, height=size, width=size)
            img = np.moveaxis(img, 2, 0)
            image_list.append(img)
    episode = {
        "observation": _atleast2d(np.array(obs, dtype=np.float32)),
        "physics": _atleast2d(np.array(physics, dtype=np.float64)),
        "action": _atleast2d(np.array(new_actions, dtype=np.float32)),
        "reward": _atleast2d(np.array(rewards, dtype=np.float32)),
        "discount": _atleast2d(np.array(discount, dtype=np.float32)),
    }
    if save_rgb:
        episode["pixels"] = np.stack(image_list)
    return episode


def main(
    seed: int = 0,
    num_workers: int = 0,
    env_name: Literal[
        "walker",
        "cartpole",
        "cheetah",
        "jaco",
        "pointmass",
        "quadruped",
    ] = "walker",
    expl_agent: Literal[
        "aps",
        "icm_apt",
        "diayn",
        "disagreement",
        "icm",
        "proto",
        "random",
        "rnd",
        "smm",
    ] = "proto",
    datasets_dir: str = "",
    new_dataset_dir: str = "",
    save_rgb: bool = False,
):
    if env_name == "walker":
        task = "walker_stand"
    elif env_name == "cartpole":
        task = "cartpole_balance"
    elif env_name == "cheetah":
        task = "cheetah_run"
    elif env_name == "jaco":
        task = "jaco_reach_top_left"
    elif env_name == "pointmass":
        task = "pointmass_reach_top_left"
    elif env_name == "quadruped":
        task = "quadruped_run"
    set_seed_everywhere(seed)

    # create data storage
    domain = get_domain(task)
    replay_dir = Path(datasets_dir).resolve() / domain / expl_agent / "buffer"
    replay_img_dir = Path(new_dataset_dir).resolve() / domain / expl_agent / "buffer"
    os.makedirs(replay_img_dir, exist_ok=True)
    print(f"Replay dir: {replay_dir}")

    eps_fns = sorted(replay_dir.glob("*.npz"))

    print(f"Using {num_workers} workers out of {cpu_count()}")
    if num_workers == 0:
        save_new_files(eps_fns, seed, task, replay_img_dir, env_name, save_rgb)
    else:
        ctx = get_context("spawn")
        list_eps = np.array_split(np.array(eps_fns), num_workers)
        assert len(list_eps) == num_workers
        with ctx.Pool(num_workers) as pool:
            f = functools.partial(
                save_new_files,
                seed=seed,
                task=task,
                replay_img_dir=replay_img_dir,
                env_name=env_name,
                save_rgb=save_rgb,
            )
            pool.map(f, list_eps)


def save_new_files(
    eps_fns: List[Any],
    seed: int,
    task: str,
    replay_img_dir: Path,
    env_name: str,
    save_rgb: bool = False,
):
    _task = task.replace(f"{env_name}_", "")
    env = cdmc._make_dmc(domain=env_name, task=_task, seed=seed)

    try:
        disable = current_process()._identity[0] != 1
    except Exception:
        disable = False

    for eps_fn in tqdm.tqdm(eps_fns, disable=disable):
        episode = load_episode(eps_fn)
        episode = relabel_episode(env, episode, env_name, save_rgb)
        file_name = eps_fn.name
        np.savez(replay_img_dir / file_name, **episode)


if __name__ == "__main__":
    tyro.cli(main)
