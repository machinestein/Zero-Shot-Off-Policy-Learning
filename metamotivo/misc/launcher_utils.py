# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import pathlib
from copy import deepcopy

from train import TrainConfig


def get_algo_name(agent_name: str) -> str:
    """
    Get a short algorithm name from the agent name.
    """
    if not agent_name:
        return "unknown"
    if "FBFlowBC" in agent_name:
        return "fb"
    if "TDJEPAFlowBC" in agent_name:
        return "tdjepa"
    if "RLDPFlowBC" in agent_name:
        return "rldp"
    if "SFFlowBC" in agent_name:
        return "sf"
    if "HilpFlowBC" in agent_name:
        return "hilp"
    if "LaplacianFlowBC" in agent_name:
        return "laplacian"
    if "BYOLFlowBC" in agent_name:
        return "byol"
    if "ICVFFlowBC" in agent_name:
        return "icvf"
    if "TD3FlowBC" in agent_name:
        return "td3-fb"
    if "TD3Agent" in agent_name:
        return "td3"
    return agent_name.replace("Agent", "").replace("FlowBC", "").lower()


def setup_wandb_config(
    trial: dict,
    base_cfg: dict,
    args_use_wandb: bool,
    args_wandb_pname: str | None,
    args_wandb_gname: str | None = None,
    args_wandb_ename: str | None = None,
) -> dict:
    """
    Setup WandB configuration for a trial.
    """
    # Get agent name from trial or base_cfg
    agent_name = trial.get("agent.name") or flatten(base_cfg).get("agent.name")
    algo = get_algo_name(agent_name)
    domain = trial.get("env.domain") or flatten(base_cfg).get("env.domain")
    seed = trial.get("seed") or flatten(base_cfg).get("seed")

    # Use provided names or fallback to automatic ones
    # Group: algo-domain
    wandb_gname = args_wandb_gname or f"{algo}-{domain}"
    # Name: seed
    wandb_ename = args_wandb_ename or f"seed-{seed}"

    return {
        "use_wandb": args_use_wandb,
        "wandb_pname": args_wandb_pname,
        "wandb_gname": wandb_gname,
        "wandb_ename": wandb_ename,
    }


def flatten(nested_dict: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested dictionary.
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(flat_dict: dict, sep: str = ".") -> dict:
    """
    Unflatten a dictionary.
    """
    unflat_dict = {}
    for k, v in flat_dict.items():
        keys = k.split(sep)
        d = unflat_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = v
    return unflat_dict


def all_combinations_of_nested_dicts_for_sweep(nested_dict):
    """
    Flatten the dict, get all combinations of the values and return a list of dicts with the combinations.
    """
    flat_dict = flatten(nested_dict)
    keys = list(flat_dict.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[flat_dict[k] for k in keys])]


def launch_locally(
    base_config: dict,
    trials: list[dict],
    first_only: bool = False,
    dry: bool = False,
):
    base_config = flatten(base_config)
    for trial in trials:
        config = deepcopy(base_config)
        config.update(flatten(trial))
        config = unflatten(config)
        if dry:
            print(config)
        else:
            TrainConfig(**config).build().train()
            if config.get("use_wandb"):
                import wandb

                wandb.finish()
        if first_only:
            break


def slurm_entry_point(config_path: str):
    """
    Entry point for slurm jobs.
    """
    with open(config_path) as f:
        cfg = TrainConfig.model_validate_json(f.read())
    cfg.build().train()


def launch_with_sbatch(
    base_config: dict,
    trials: list[dict],
):
    import shutil
    import stat
    from subprocess import PIPE, TimeoutExpired, run

    # TODO: edit these requirements as needed
    JOB_PREFIX = """#!/bin/bash
#SBATCH --time=1440
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=td_jepa
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1"""

    try:
        os.mkdir(base_config["work_dir"])
    except FileExistsError:
        # ask what to do if results_dir exists
        print(f"Directory {base_config['work_dir']} exists. Delete?")
        if input().lower() in ["y", "yes"]:
            print("Deleting result directory.")
            shutil.rmtree(base_config["work_dir"], ignore_errors=True)
            os.mkdir(base_config["work_dir"])
        else:
            print("Exiting.")
            exit(0)

    # optional: copy code for reproducibility
    # we recommend running a fresh clone instead
    # os.mkdir(base_config["work_dir"] / "code")
    # for path in ["metamotivo", "scripts", "uv.lock", "pyproject.toml"]:
    #     os.system(f"cp -r {path} {base_config['work_dir'] / 'code' / path}")

    base_config = flatten(base_config)
    for trial in trials:
        config = deepcopy(base_config)
        config.update(flatten(trial))
        config = TrainConfig(**unflatten(config))

        os.mkdir(trial["work_dir"])
        json_path = trial["work_dir"] + "/config.json"
        with open(json_path, "w") as f:
            f.write(config.model_dump_json())
        job_script = JOB_PREFIX + "\n" + f"#SBATCH --output={trial['work_dir'] + '/job.sh.out'}\n"
        job_script += f"#SBATCH --error={trial['work_dir'] + '/job.sh.err'}\n"
        # TODO: this is a didactic example, there are better ways to launch your jobs
        job_script += f"python -c \"from metamotivo.misc.launcher_utils import slurm_entry_point; slurm_entry_point('{json_path}')\""
        with open(trial["work_dir"] + "/job.sh", "w") as file:
            file.write(job_script)
        st = os.stat(trial["work_dir"] + "/job.sh")
        os.chmod(trial["work_dir"] + "/job.sh", st.st_mode | stat.S_IEXEC)

        print("Submitting...")
        try:
            result = run(
                ["sbatch " + trial["work_dir"] + "/job.sh"],
                cwd=str(os.getcwd()),
                shell=True,
                stdout=PIPE,
                timeout=20.0,
            )
        except TimeoutExpired:
            print("Submission hangs. Exiting. Check for orphan jobs.")
            exit()
        cluster_id = int(result.stdout.decode("utf-8").split(" ")[-1])
        print(f"Cluster ID: {cluster_id}")
    print("Submitted!")


def launch_with_exca(
    base_config: dict,
    trials: list[dict],
):
    import exca as xk
    from exca.confdict import ConfDict

    _PATHS_TO_COPY = ["metamotivo", "scripts", "uv.lock", "pyproject.toml", "train.py"]
    # TODO: edit these requirements as needed
    CLUSTER_CONFIG = {
        "timeout_min": 24 * 60,
        "gpus_per_node": 1,
        "slurm_constraint": "",
        "slurm_partition": "",
        "cpus_per_task": 16,  # we recommend at least 16 cores when training from pixels
        "mem_gb": 80,  # we recommend at least 80GB of memory when training from pixels
        "cluster": "slurm",
    }

    class InfraTrainConfig(TrainConfig):
        infra: xk.TaskInfra = xk.TaskInfra(version="1")

        @infra.apply
        def process(self):
            ws = self.build()
            ws.train()

    workdir_root = pathlib.Path(base_config["work_dir"])
    exca_infra_args = CLUSTER_CONFIG.copy()
    # exca needs its own folder to store code, logs, etc
    exca_infra_args["folder"] = str(workdir_root / "_exca")
    # tell exca which paths to copy
    exca_infra_args["workdir"] = {"copied": _PATHS_TO_COPY, "includes": tuple()}
    # by default, force new runs and do not use cache
    exca_infra_args["mode"] = "force"
    base_config["infra"] = exca_infra_args

    base_config = InfraTrainConfig(**base_config)
    print("Using current Python environment for experiments.")

    # instantiate the base config as a config_cls object, and now launch the runs
    with base_config.infra.job_array(max_workers=1000, allow_empty=True) as array:
        for trial in trials:
            # ConfDict implements nested dicts with convenience features (recursive update, flattening to "."-separated list)
            trial = ConfDict(trial)
            # NOTE: use of clone_obj important for exca tracking (how many configs are created etc)
            # this will create clone of base_config, with "trial" ConfDict applied on top
            new_config = base_config.infra.clone_obj(trial)
            array.append(new_config)


def launch_trials(
    base_config: dict,
    trials: list[dict],
    first_only: bool = False,
    dry: bool = False,
):
    """
    Launch trials, applying their changes on the base_config.

    Args:
        base_config: The base configuration dictionary.
        trials: A list of trial configuration dictionaries.
        first_only: If True, only launch the first trial.
        dry: If True, print the configurations instead of launching them.
    """
    return launch_locally(base_config, trials, first_only, dry)
