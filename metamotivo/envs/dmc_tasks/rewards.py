# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid, device):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise."""
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError("`value_at_1` must be nonnegative and smaller than 1, got {}.".format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError("`value_at_1` must be strictly between 0 and 1, got {}.".format(value_at_1))

    x_tensor = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)
    value_at_1 = torch.tensor(value_at_1, dtype=torch.get_default_dtype(), device=device)

    if sigmoid == "gaussian":
        scale = torch.sqrt(-2 * torch.log(value_at_1))
        return torch.exp(-0.5 * (x_tensor * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = torch.acosh(1.0 / value_at_1)
        return 1 / torch.cosh(x_tensor * scale)

    elif sigmoid == "long_tail":
        scale = torch.sqrt(1 / value_at_1 - 1)
        return 1 / ((x_tensor * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (torch.abs(x_tensor) * scale + 1)

    elif sigmoid == "cosine":
        scale = torch.acos(2 * value_at_1 - 1) / torch.pi
        scaled_x = x_tensor * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="invalid value encountered in cos")
            cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
        return torch.where(torch.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, torch.tensor(0.0, dtype=x_tensor.dtype))

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x_tensor * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x, torch.tensor(0.0, dtype=x_tensor.dtype))

    elif sigmoid == "quadratic":
        scale = torch.sqrt(1 - value_at_1)
        scaled_x = x_tensor * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x**2, torch.tensor(0.0, dtype=x_tensor.dtype))

    elif sigmoid == "tanh_squared":
        scale = torch.arctanh(torch.sqrt(1 - value_at_1))
        return 1 - torch.tanh(x_tensor * scale) ** 2

    else:
        raise ValueError("Unknown sigmoid type {!r}.".format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=_DEFAULT_VALUE_AT_MARGIN, device="cpu"):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise."""
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    x_tensor = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)
    in_bounds = torch.logical_and(lower <= x_tensor, x_tensor <= upper)
    if margin == 0:
        value = torch.where(
            in_bounds, torch.tensor(1.0, dtype=x_tensor.dtype, device=device), torch.tensor(0.0, dtype=x_tensor.dtype, device=device)
        )
    else:
        d = torch.where(x_tensor < lower, lower - x_tensor, x_tensor - upper) / margin
        value = torch.where(
            in_bounds, torch.tensor(1.0, dtype=x_tensor.dtype, device=device), _sigmoids(d, value_at_margin, sigmoid, device=device)
        )

    # Return a float if input was scalar, else tensor
    return value


def pointmass_reward(obs, action, *, task="loop", device="cpu"):
    """Returns a reward to the agent."""

    GOALS = {
        "reach_top_left": torch.tensor([-0.15, 0.15], device=device),
        "reach_top_right": torch.tensor([0.15, 0.15], device=device),
        "reach_bottom_left": torch.tensor([-0.15, -0.15], device=device),
        "reach_bottom_right": torch.tensor([0.15, -0.15], device=device),
        "reach_bottom_left_long": torch.tensor([-0.15, -0.15], device=device),
    }
    obs = obs.to(device=device)
    action = action.to(device=device)
    x, y = obs[:2]
    vx, vy = obs[2:]
    zero = torch.tensor([0.0], dtype=torch.get_default_dtype(), device=device)
    match task:
        case "fast_slow":
            up = ((y > 0.2) & (y < 0.28)).to(torch.int)
            right = ((x > 0.2) & (x < 0.28)).to(torch.int)
            left = ((x < -0.2) & (x > -0.28)).to(torch.int)
            down = ((y < -0.2) & (y > -0.28)).to(torch.int)

            up_rew = tolerance(vx, bounds=(-0.05, -0.04), margin=0.01, value_at_margin=0, sigmoid="linear", device=device) * up
            right_rew = tolerance(vy, bounds=(0.09, 0.1), margin=0.01, value_at_margin=0, sigmoid="linear", device=device) * right
            left_rew = tolerance(vy, bounds=(-0.1, -0.09), margin=0.01, value_at_margin=0, sigmoid="linear", device=device) * left
            down_rew = tolerance(vx, bounds=(0.04, 0.05), margin=0.01, value_at_margin=0, sigmoid="linear", device=device) * down

            # reward = 0 if up + right + left + down > 1 else up_rew + right_rew + left_rew + down_rew
            reward = torch.where(up + right + left + down > 1, zero, up_rew + right_rew + left_rew + down_rew)

        case "square":  # square
            up = (y > 0.2).to(torch.int)
            right = (x > 0.2).to(torch.int)
            left = (x < -0.2).to(torch.int)
            down = (y < -0.2).to(torch.int)

            up_rew = torch.abs(torch.clamp(vx, 0, 0.1) * 10 * up)
            right_rew = torch.abs(torch.clamp(vy, -0.1, 0) * 10 * right)
            left_rew = torch.abs(torch.clamp(vy, 0, 0.1) * 10 * left)
            down_rew = torch.abs(torch.clamp(vx, -0.1, 0) * 10 * down)

            reward = torch.where(up + right + left + down > 1, zero, up_rew + right_rew + left_rew + down_rew)

        case "loop":
            # Compute all possible vx_rew and vy_rew values
            vx_rew_tl = tolerance(vx, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vy_rew_tl = tolerance(vy, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vx_rew_tr = tolerance(vx, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vy_rew_tr = tolerance(vy, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vx_rew_bl = tolerance(vx, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vy_rew_bl = tolerance(vy, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vx_rew_br = tolerance(vx, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            vy_rew_br = tolerance(vy, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid="linear", device=device)
            tl = (x <= 0) & (y >= 0)
            tr = (x > 0) & (y >= 0)
            bl = (x <= 0) & (y < 0)
            br = (x > 0) & (y < 0)
            # Convert conditions to tensors (0 or 1)
            tl = tl.to(torch.bool)
            tr = tr.to(torch.bool)
            bl = bl.to(torch.bool)
            br = br.to(torch.bool)

            # Select vx_rew and vy_rew based on conditions
            vx_rew = torch.where(
                tl,
                vx_rew_tl,
                torch.where(tr, vx_rew_tr, torch.where(bl, vx_rew_bl, torch.where(br, vx_rew_br, torch.tensor(float("nan"))))),
            )  # or some default
            vy_rew = torch.where(
                tl,
                vy_rew_tl,
                torch.where(tr, vy_rew_tr, torch.where(bl, vy_rew_bl, torch.where(br, vy_rew_br, torch.tensor(float("nan"))))),
            )  # or some default
            # For a, b, c, create tensors and select similarly
            a = torch.where(
                tl,
                torch.tensor([1.0], dtype=torch.get_default_dtype(), device=device),
                torch.where(
                    tr,
                    torch.tensor([-1.0], dtype=torch.get_default_dtype(), device=device),
                    torch.where(
                        bl,
                        torch.tensor([-1.0], dtype=torch.get_default_dtype(), device=device),
                        torch.where(br, torch.tensor([1.0], dtype=torch.get_default_dtype(), device=device), torch.tensor(float("nan"))),
                    ),
                ),
            )
            b = torch.where(
                tl,
                torch.tensor([-1.0], dtype=torch.get_default_dtype(), device=device),
                torch.where(
                    tr,
                    torch.tensor([-1.0], dtype=torch.get_default_dtype(), device=device),
                    torch.where(
                        bl,
                        torch.tensor([-1.0], dtype=torch.get_default_dtype(), device=device),
                        torch.where(br, torch.tensor([-1.0], dtype=torch.get_default_dtype(), device=device), torch.tensor(float("nan"))),
                    ),
                ),
            )
            c = torch.where(
                tl,
                torch.tensor([0.24], dtype=torch.get_default_dtype(), device=device),
                torch.where(
                    tr,
                    torch.tensor([0.24], dtype=torch.get_default_dtype(), device=device),
                    torch.where(
                        bl,
                        torch.tensor([-0.24], dtype=torch.get_default_dtype(), device=device),
                        torch.where(br, torch.tensor([-0.24], dtype=torch.get_default_dtype(), device=device), torch.tensor(float("nan"))),
                    ),
                ),
            )

            dist = torch.abs(a * x + b * y + c) / 1.4142  # torch.sqrt(torch.tensor(2.0))
            dist_rew = tolerance(dist, bounds=(0, 0.02), margin=0.02, value_at_margin=0, sigmoid="linear", device=device)
            reward = (dist_rew + vx_rew + vy_rew) / 3

        case "reach_bottom_left_long":
            target = GOALS["reach_bottom_left"]
            target_size = 0.015
            control_reward = tolerance(action, margin=1, value_at_margin=0, sigmoid="quadratic", device=device).mean()
            small_control = (control_reward + 4) / 5
            near_target = tolerance(torch.linalg.norm(target - obs[:2]), bounds=(0, target_size), margin=6 * target_size, device=device)
            reach_reward = near_target * small_control
            up = (y > 0.15).to(torch.int)
            right = (x > 0.15).to(torch.int)
            left = (x < -0.15).to(torch.int)
            down = (y < -0.15).to(torch.int)
            up_rew = torch.where(vx >= 0, torch.clamp(vx, -0.1, 0.1) * up * 5, torch.clamp(vx, -0.1, 0.1) * up * 100)
            right_rew = -torch.where(vy <= 0, torch.clamp(vy, -0.1, 0.1) * right * 5, torch.clamp(vy, -0.1, 0.1) * right * 100)
            left_rew = torch.where(vy >= 0, torch.clamp(vy, -0.1, 0.1) * left * 5, torch.clamp(vy, -0.1, 0.1) * left * 100)
            down_rew = -torch.where(vx >= 0, torch.clamp(vx, -0.1, 0.1) * down * 5, torch.clamp(vx, -0.1, 0.1) * down * 100)
            path_reward = torch.where(up + right + left + down > 1, zero, up_rew + right_rew + left_rew + down_rew)
            reward = torch.where(reach_reward > 0.01, reach_reward, path_reward)
        case _:
            target = GOALS[task]
            target_size = 0.015
            control_reward = tolerance(action, margin=1, value_at_margin=0, sigmoid="quadratic", device=device).mean()
            small_control = (control_reward + 4) / 5
            near_target = tolerance(torch.linalg.norm(target - obs[:2]), bounds=(0, target_size), margin=6 * target_size, device=device)

            reward = near_target * small_control

    return reward
