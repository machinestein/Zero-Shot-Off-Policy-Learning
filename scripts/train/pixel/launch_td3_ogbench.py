# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses

import tyro

from metamotivo.envs.ogbench import ALL_TASKS
from metamotivo.misc.launcher_utils import all_combinations_of_nested_dicts_for_sweep, flatten, launch_trials

BASE_CFG = {
    "relabel_dataset": True,
    "num_train_steps": 1_000_000,
    "data": {
        "name": "ogbench",
        "domain": "cube-single-play-v0",
        "obs_type": "pixels",
        "buffer_type": "parallel",
    },
    "env": {
        "name": "ogbench",
        "domain": "cube-single-play-v0",
        "task": "cube-single-play-singletask-task1-v0",
        "obs_type": "pixels",
        "frame_stack": 3,
    },
    "agent": {
        "name": "TD3Agent",
        "compile": True,
        "model": {
            "device": "cuda",
            "obs_normalizer": {
                "name": "RGBNormalizerConfig",
            },
            "archi": {
                "critic": {"hidden_dim": 512, "hidden_layers": 4, "layer_norm": True},
                "actor": {"hidden_dim": 512, "hidden_layers": 4},
                "rgb_encoder": {
                    "name": "drq",
                    "feature_dim": 256,
                },
                "augmentator": {
                    "name": "random_shifts",
                    "pad": 2,
                },
            },
        },
        "train": {
            "batch_size": 256,
            "discount": 0.99,
            "lr": 3.0e-4,
            "critic_target_tau": 0.005,
        },
    },
}


def sweep_cube_single():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["cube-single-play-v0"],
        "env.task": [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.3, 1.0, 3.0],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_cube_double():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["cube-double-play-v0"],
        "env.task": [f"cube-double-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.3, 1.0, 3.0],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_scene():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["scene-play-v0"],
        "env.task": [f"scene-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.3, 1.0, 3.0],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_puzzle_3x3():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["puzzle-3x3-play-v0"],
        "env.task": [f"puzzle-3x3-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.3, 1.0, 3.0],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_antmaze_medium_navigate():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-navigate-v0"],
        "env.task": [f"antmaze-medium-navigate-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.03, 0.1, 0.3],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_antmaze_large_navigate():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-navigate-v0"],
        "env.task": [f"antmaze-large-navigate-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.03, 0.1, 0.3],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_antmaze_medium_stitch():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-stitch-v0"],
        "env.task": [f"antmaze-medium-stitch-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.03, 0.1, 0.3],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_antmaze_large_stitch():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-stitch-v0"],
        "env.task": [f"antmaze-large-stitch-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.03, 0.1, 0.3],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_antmaze_medium_explore():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-explore-v0"],
        "env.task": [f"antmaze-medium-explore-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.03, 0.1, 0.3],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


def sweep_antmaze_large_explore():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-explore-v0"],
        "env.task": [f"antmaze-large-explore-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.03, 0.1, 0.3],
            "model.archi.actor_vf": {"hidden_layers": [4], "hidden_dim": [512]},
        },
    }
    return conf


@dataclasses.dataclass
class LaunchArgs:
    # dataset and working paths
    data_path: str = "/path/to/datasets"
    workdir_root: str = "/path/to/workdir"
    # wandb config
    use_wandb: bool = False
    wandb_gname: str | None = "td_jepa"
    wandb_ename: str | None = None
    wandb_pname: str | None = "td_jepa"
    # specify to run sweeps
    sweep_config: str | None = None
    # instead of launching all experiments, only run the first one
    first_only: bool = False
    # print out the configs instead of running the experiments
    dry: bool = False
    # launch with slurm
    slurm: bool = False
    # launch with exca
    exca: bool = False


def main(args: LaunchArgs):
    base_cfg = copy.deepcopy(BASE_CFG)
    base_cfg["work_dir"] = args.workdir_root
    base_cfg["data"]["dataset_root"] = args.data_path

    if args.sweep_config is None:
        sweep_params = {}
    else:
        if args.sweep_config in globals().keys():
            sweep_params = globals()[args.sweep_config]()
        else:
            raise RuntimeError("Unknown sweep configuration")

    trials = []
    for i, trial in enumerate(all_combinations_of_nested_dicts_for_sweep(sweep_params)):
        trial = flatten(trial)
        trial.update(
            flatten(
                {
                    "use_wandb": args.use_wandb,
                    "wandb_ename": args.wandb_ename,
                    "wandb_pname": args.wandb_pname,
                    "wandb_gname": args.wandb_gname,
                    "work_dir": f"{args.workdir_root}/{i}",
                    "data.domain": trial["env.domain"],
                    "env.task": ALL_TASKS[trial["env.domain"]][0],
                    "evaluations": [
                        {
                            "name": "ogbench_reward_eval",
                            "shift_reward": 1,
                            "env": {
                                "name": "ogbench",
                                "domain": trial["env.domain"],
                                "task": trial["env.task"],
                                "obs_type": "pixels",
                                "frame_stack": base_cfg["env"]["frame_stack"],
                            },
                            "tasks": [trial["env.task"]],
                            "num_episodes": 10,
                            "num_inference_samples": 10_000,
                        },
                    ],
                }
            )
        )
        trials.append(trial)

    launch_trials(base_cfg, trials, args.first_only, args.dry, args.slurm, args.exca)


if __name__ == "__main__":
    args = tyro.cli(LaunchArgs)
    main(args)
    # uv run -m scripts.train.pixel.launch_td3_ogbench --use_wandb --wandb_gname td3_antmaze_pixel --data_path datasets --workdir_root results --sweep_config sweep_antmaze_medium_navigate
