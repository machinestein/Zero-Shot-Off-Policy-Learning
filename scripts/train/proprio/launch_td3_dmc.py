# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses

import tyro

from metamotivo.envs.dmc_tasks import ALL_TASKS
from metamotivo.misc.launcher_utils import all_combinations_of_nested_dicts_for_sweep, flatten, launch_trials

BASE_CFG = {
    "relabel_dataset": True,
    "num_train_steps": 3_000_000,
    "data": {
        "name": "dmc",
        "domain": "walker",
        "load_n_episodes": 5_000,
        "obs_type": "state",
    },
    "env": {"name": "dmc", "domain": "walker", "task": "walk"},
    "agent": {
        "name": "TD3Agent",
        "compile": True,
        "model": {
            "device": "cuda",
            "obs_normalizer": {
                "name": "IdentityNormalizerConfig",
            },
            "archi": {
                "critic": {
                    "hidden_dim": 1024,
                    "hidden_layers": 2,
                },
                "actor": {
                    "hidden_dim": 1024,
                    "hidden_layers": 2,
                },
            },
        },
        "train": {
            "batch_size": 1024,
            "discount": 0.98,
            "lr": 1e-4,
            "critic_target_tau": 0.005,
        },
    },
}


def sweep_walker():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["walker"],
        "env.task": ALL_TASKS["walker"],
    }
    return conf


def sweep_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "env.task": ALL_TASKS["cheetah"],
    }
    return conf


def sweep_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "env.task": ALL_TASKS["quadruped"],
    }
    return conf


def sweep_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ALL_TASKS["pointmass"],
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
                            "name": "dmc_reward_eval",
                            "env": {
                                "name": "dmc",
                                "domain": trial["env.domain"],
                                "task": "walk",
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
    # uv run -m scripts.train.proprio.launch_td3_dmc --use_wandb --wandb_gname td3_walker_proprio --data_path datasets --workdir_root results --sweep_config sweep_walker
