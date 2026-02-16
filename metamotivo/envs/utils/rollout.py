# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def rollout(
    env: Any,
    agent: Any,
    num_episodes: int,
    ctx: torch.Tensor | None = None,
    render: bool = False,
    max_steps: int | None = None,
    render_num_episodes: int = 1,
) -> Tuple[Dict[str, Any], List[List[Dict[str, Any]]], List[np.ndarray] | None]:
    observation, info = env.reset()
    returns, lengths, infos = [0.0], [0], [[info]]

    # We only record frames if render=True AND we haven't exceeded render_num_episodes
    frames = [] if render else None

    def should_render():
        return render and len(returns) <= render_num_episodes

    if should_render():
        frames.append(env.render())

    input_ctx = {} if ctx is None else {"z": ctx}
    step_count = 0
    while True:
        input_dict = {"obs": torch.tensor(observation, device=agent.device, dtype=torch.float32)[None], **input_ctx}
        action = agent.act(**input_dict).detach().cpu().numpy()[0]
        observation, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated
        returns[-1] += reward
        lengths[-1] += 1
        infos[-1] += [info]

        if should_render():
            frames.append(env.render())

        if done:
            if len(returns) >= num_episodes:
                break
            observation, info = env.reset()
            returns.append(0.0)
            lengths.append(0)
            infos.append([info])
            if should_render():
                frames.append(env.render())
        elif max_steps is not None and step_count >= max_steps:
            break

    return {"reward": returns, "length": lengths}, infos, frames
