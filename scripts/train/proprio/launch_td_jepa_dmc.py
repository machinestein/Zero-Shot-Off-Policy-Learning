# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses

import tyro

from metamotivo.envs.dmc_tasks import ALL_TASKS
from metamotivo.misc.launcher_utils import all_combinations_of_nested_dicts_for_sweep,\
    flatten, launch_trials, setup_wandb_config

BASE_CFG = {
    "num_train_steps": 1_000_000,
    "data": {
        "name": "dmc",
        "domain": "walker",
        "load_n_episodes": 5_000,
        "obs_type": "state",
    },
    "env": {"name": "dmc", "domain": "walker", "task": "walk"},
    "agent": {
        "name": "TDJEPAAgent",
        "compile": True,
        "model": {
            "device": "cuda",
            "obs_normalizer": {
                "name": "IdentityNormalizerConfig",
            },
            "archi": {
                "phi_predictor": {"name": "ForwardArchi", "hidden_dim": 1024, "hidden_layers": 1},
                "psi_predictor": {"name": "ForwardArchi", "hidden_dim": 1024, "hidden_layers": 1},
                "actor": {"hidden_dim": 1024, "hidden_layers": 1, "name": "simple"},
                "phi_mlp_encoder": {
                    "hidden_dim": 256,
                    "hidden_layers": 0,
                    "norm": True,
                },
                "psi_mlp_encoder": {
                    "hidden_dim": 256,
                    "hidden_layers": 2,
                    "norm": True,
                },
                "phi_dim": 256,
                "psi_dim": 50,
                "norm_z": True,
            },
        },
        "train": {
            "batch_size": 1024,
            "discount": 0.98,
            "phi_ortho_coef": 0.1,
            "psi_ortho_coef": 0.1,
            "encoder_target_tau": 0.001,
            "predictor_target_tau": 0.001,
        },
    },
}


def sweep_walker():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["walker"],
        "agent": {
            "train": {"lr_psi": [1e-4], "psi_ortho_coef": [1]},
        },
    }
    return conf


def sweep_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "agent": {
            "train": {"lr_psi": [1e-4, 1e-5], "psi_ortho_coef": [1]},
        },
    }
    return conf


def sweep_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "agent": {
            "train": {"lr_psi": [1e-4], "psi_ortho_coef": [1]},
        },
    }
    return conf


def sweep_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ["reach_top_left"],
        "agent": {
            "train": {"lr_psi": [1e-4], "psi_ortho_coef": [1], "discount": [0.99], "lr_actor": [1e-6]},
        },
    }
    return conf


@dataclasses.dataclass
class LaunchArgs:
    # dataset and working paths
    data_path: str = "/home/jovyan/.exorl/expl_datasets"
    workdir_root: str = "/path/to/workdir"
    # wandb config
    use_wandb: bool = False
    wandb_gname: str | None = None
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
                    **setup_wandb_config(
                        trial,
                        base_cfg,
                        args_use_wandb=args.use_wandb,
                        args_wandb_pname=args.wandb_pname,
                        args_wandb_gname=args.wandb_gname,
                        args_wandb_ename=args.wandb_ename,
                    ),
                    "work_dir": f"{args.workdir_root}/{trial['env.domain']}/{i}",
                    "data.domain": trial["env.domain"],
                    "env.task": ALL_TASKS[trial["env.domain"]][0],
                    "evaluations": [
                        {
                            "name": "dmc_reward_eval",
                            "env": {
                                "name": "dmc",
                                "domain": trial["env.domain"],
                                "task": ALL_TASKS[trial["env.domain"]][0],
                            },
                            "tasks": ALL_TASKS[trial["env.domain"]],
                            "num_episodes": 10,
                            "num_inference_samples": 10_000,
                        },
                    ],
                }
            )
        )
        trials.append(trial)

    launch_trials(base_cfg, trials, args.first_only, args.dry)


if __name__ == "__main__":
    args = tyro.cli(LaunchArgs)
    main(args)
    # uv run -m scripts.train.proprio.launch_td_jepa_dmc --use_wandb --wandb_gname td_jepa_walker_proprio --data_path datasets --workdir_root results --sweep_config sweep_walker
    # uv run -m scripts.train.proprio.launch_td_jepa_dmc --use_wandb --wandb_gname td_jepa_cheetah_proprio --data_path datasets --workdir_root results --sweep_config sweep_cheetah
    # uv run -m scripts.train.proprio.launch_td_jepa_dmc --use_wandb --wandb_gname td_jepa_quadruped_proprio --data_path datasets --workdir_root results --sweep_config sweep_quadruped
    # uv run -m scripts.train.proprio.launch_td_jepa_dmc --use_wandb --wandb_gname td_jepa_pointmass_proprio --data_path datasets --workdir_root results --sweep_config sweep_pointmass
