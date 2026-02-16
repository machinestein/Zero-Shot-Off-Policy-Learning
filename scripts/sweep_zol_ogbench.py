import sys
import os
import itertools
import json
import torch
import numpy as np
import scipy.stats as st
from pathlib import Path
from functools import partial

# Set MUJOCO_GL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

# Add current directory to path if needed for metamotivo imports
sys.path.append(os.getcwd())

from metamotivo.agents.fb.flow_bc.agent import FBFlowBCAgent
from metamotivo.envs.ogbench import OGBenchEnvConfig, ALL_DOMAINS, ALL_TASKS
from metamotivo.data_loading.ogbench import OGBenchDataConfig
from metamotivo.envs.utils.rollout import rollout

# Config
RESULTS_FB_DIR = "/home/jovyan/bobrin/td_jepa/results_fb_ogbench_proprio"
DATASET_ROOT = "/home/jovyan/bobrin/td_jepa/ogbench_data"
DEVICE = "cuda"
OUTPUT_FILE = "zol_sweep_ogbench_results.json"

# Exclusion List (Domains to skip)
EXCLUDE_DOMAINS = ["antmaze-medium-navigate-v0",
                "antmaze-large-navigate-v0",
                "antmaze-medium-stitch-v0",] 

# Sweep Space
sweep_space = {
    "lr": [1e-3],
    "num_steps": [200],
    "chi2_coef": [0.01],
    "trust_l2_coef": [0.001],
    "weight_clip": [50.0],
    "n_mu": [1, 512],
    "weight_temp": [2.0],
}

def calculate_success(infos):
    return [any([step.get("success", False) for step in info]) for info in infos]

def get_stats(data):
    if not data or len(data) == 0:
        return 0.0, 0.0
    mean = np.mean(data)
    sem = st.sem(data)
    ci = 1.96 * sem
    return float(mean), float(ci)

def calculate_sensitivity(flat_results):
    """
    Calculate average score per hyperparameter value across all tasks.
    """
    sensitivity = {}
    if not flat_results:
        return {}

    param_keys = flat_results[0]["params"].keys()
    for key in param_keys:
        sensitivity[key] = {}
        values = set(res["params"][key] for res in flat_results)
        for val in values:
            scores = [res["score"] for res in flat_results if res["params"][key] == val]
            if scores:
                sensitivity[key][str(val)] = {
                    "mean_score": float(np.mean(scores)),
                    "count": len(scores)
                }
    return sensitivity

def find_checkpoints(base_dir):
    checkpoints = []
    base_path = Path(base_dir)
    print(f"Searching for OGBench checkpoints in {base_dir}...")
    
    # OGBench structure: results/DOMAIN/0/checkpoint/config.json
    for ckpt_config in base_path.glob("**/checkpoint/config.json"):
        ckpt_dir = ckpt_config.parent
        # The domain should be the parent of the seed directory
        # path structure is .../DOMAIN/SEED/checkpoint
        try:
            domain = ckpt_dir.parent.parent.name
            if domain in ALL_DOMAINS:
                checkpoints.append({
                    "path": str(ckpt_dir),
                    "domain": domain
                })
                print(f"  Found: {ckpt_dir} (Domain: {domain})")
            else:
                # Fallback: check if any domain name is in the parts
                for part in ckpt_dir.parts:
                    if part in ALL_DOMAINS:
                        checkpoints.append({
                            "path": str(ckpt_dir),
                            "domain": part
                        })
                        print(f"  Found (fallback): {ckpt_dir} (Domain: {part})")
                        break
        except Exception:
            continue
            
    return checkpoints

def run_sweep():
    print(f"Starting OGBench ZOL sweep")
    print(f"Device: {DEVICE}")
    print(f"Results Directory: {RESULTS_FB_DIR}")
    print(f"Dataset Root: {DATASET_ROOT}")

    checkpoints = find_checkpoints(RESULTS_FB_DIR)
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    all_domain_results = {}
    master_summary = {}
    all_flat_results = []

    # Load existing results if they exist to allow resuming or adding
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing results from {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, "r") as f:
                existing_data = json.load(f)
                master_summary = existing_data.get("summary", {})
                all_domain_results = existing_data.get("detailed", {})
                for ckpt in all_domain_results:
                    for task in all_domain_results[ckpt]:
                        all_flat_results.extend(all_domain_results[ckpt][task])
        except Exception as e:
            print(f"Failed to load existing results: {e}. Starting fresh.")

    keys = sweep_space.keys()
    values = sweep_space.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for ckpt_info in checkpoints:
        ckpt_path = ckpt_info["path"]
        domain = ckpt_info["domain"]

        if domain in EXCLUDE_DOMAINS:
            print(f"\nSkipping Checkpoint: {ckpt_path} (Domain {domain} excluded)")
            continue

        # If we already have the summary for all tasks of this checkpoint, we might want to skip
        # but usually we want to re-run or verify. Here we skip if all tasks for this domain are in summary.
        tasks = ALL_TASKS.get(domain, [])
        if not tasks:
            print(f"  No tasks found for domain {domain}. Skipping.")
            continue
            
        print(f"\n======== Processing Checkpoint: {ckpt_path} (Domain: {domain}) ========")

        # 1. Load Agent
        # Use first task to build agent
        env_cfg = OGBenchEnvConfig(domain=domain, task=tasks[0])
        temp_env, _ = env_cfg.build()
        
        try:
            agent = FBFlowBCAgent.load(
                ckpt_path,
                device=DEVICE,
                obs_space=temp_env.observation_space,
                action_dim=temp_env.action_space.shape[0]
            )
            agent._model.train(False)
        except Exception as e:
            print(f"  Failed to load agent from {ckpt_path}: {e}")
            temp_env.close()
            continue
            
        temp_env.close()

        # 2. Sample data for reward inference
        print(f"  Sampling data for domain: {domain}...")
        data_cfg = OGBenchDataConfig(domain=domain, dataset_root=DATASET_ROOT)
        inf_bs = agent.cfg.model.inference_batch_size
        replay_buffer = data_cfg.build(buffer_device=DEVICE, batch_size=inf_bs, frame_stack=1)
        batch = replay_buffer["train"].sample(inf_bs)
        
        batch_obs = batch["next"]["observation"].to(DEVICE)
        next_physics = batch["next"]["physics"].cpu().numpy()
        actions = batch["action"].cpu().numpy()

        if ckpt_path not in all_domain_results:
            all_domain_results[ckpt_path] = {}

        for task in tasks:
            summary_key = f"{ckpt_path}_{task}"
            if summary_key in master_summary:
                print(f"  Task {task} already in summary. Skipping.")
                continue

            print(f"\n--- [Task: {task}] ---")
            all_domain_results[ckpt_path][task] = []
            
            # OGBench reward relabeling
            task_env_cfg = OGBenchEnvConfig(domain=domain, task=task)
            relabel_fn = task_env_cfg.get_relabel_fn(task)
            rewards_np = relabel_fn(next_physics, actions)
            # CRITICAL: Shift rewards to [0, 1] as in training evaluation
            rewards_np += 1.0 
            rewards = torch.tensor(rewards_np, dtype=torch.float32).to(DEVICE)
            
            # Evaluate Baseline
            task_env, _ = task_env_cfg.build()
            initial_z = agent._model.reward_inference(batch_obs, rewards.reshape(-1, 1))
            
            print(f"  Evaluating Baseline (100 episodes)...")
            base_stats, base_infos, _ = rollout(task_env, agent=agent._model, ctx=initial_z, num_episodes=100)
            base_successes = calculate_success(base_infos)
            base_sr_mean, base_sr_ci = get_stats(base_successes)
            base_rew_mean, base_rew_ci = get_stats(base_stats['reward'])
            print(f"  Baseline: {base_sr_mean*100:.1f}% success ({base_rew_mean:.2f} reward)")

            best_task_score = -float('inf')
            best_task_params = None

            # Split params between config and search kwargs
            config_keys = {"lr", "num_steps", "n_mu", "early_stop_patience", "early_stop_tol", 
                           "chi2_coef", "trust_l2_coef", "weight_clip", "center_rewards"}

            for idx, params in enumerate(combinations):
                print(f"  [{idx+1}/{len(combinations)}] Testing: {params}")
                
                cfg_updates = {k: v for k, v in params.items() if k in config_keys}
                search_params = {k: v for k, v in params.items() if k not in config_keys}
                
                # Default search kwargs from notebook
                search_kwargs = {
                    "mu_source": "batch",
                    "use_exp_weights": True,
                    "weight_temp": 2.0,
                    "mu_reward_top_frac": 0.05,
                    "self_normalized_obj": True,
                }
                search_kwargs.update(search_params)
                
                # Apply Config Updates
                agent.cfg = agent.cfg.model_copy(
                    update={
                        "train": agent.cfg.train.model_copy(
                            update={
                                "zol": agent.cfg.train.zol.model_copy(update=cfg_updates)
                            }
                        )
                    }
                )
                
                # Perform ZOL Latent Search
                z_zol = agent.zol_latent_search(task_env, batch_obs, rewards.flatten(), initial_z, **search_kwargs)
                
                # Evaluate ZOL Agent (100 episodes for final sweep result)
                stats, infos, _ = rollout(task_env, agent=agent._model, ctx=z_zol, num_episodes=100)
                successes = calculate_success(infos)
                sr_mean, sr_ci = get_stats(successes)
                rew_mean, rew_ci = get_stats(stats['reward'])
                
                print(f"    Result: {sr_mean*100:.1f}% success ({rew_mean:.2f} reward)")
                
                res_entry = {
                    "checkpoint": ckpt_path,
                    "domain": domain,
                    "task": task,
                    "params": params,
                    "score": sr_mean,
                    "ci": sr_ci,
                    "reward": rew_mean,
                }
                all_domain_results[ckpt_path][task].append(res_entry)
                all_flat_results.append(res_entry)
                
                if sr_mean > best_task_score:
                    best_task_score = sr_mean
                    best_task_params = params

            master_summary[summary_key] = {
                "checkpoint": ckpt_path,
                "domain": domain,
                "task": task,
                "base_sr": base_sr_mean,
                "base_sr_ci": base_sr_ci,
                "base_reward": base_rew_mean,
                "best_zol_sr": best_task_score,
                "best_zol_params": best_task_params,
                "improvement": best_task_score - base_sr_mean
            }
            
            task_env.close()
            # Save progressively
            sensitivity_res = calculate_sensitivity(all_flat_results)
            with open(OUTPUT_FILE, "w") as f:
                json.dump({
                    "summary": master_summary,
                    "sensitivity": sensitivity_res,
                    "detailed": all_domain_results
                }, f, indent=4)

    print(f"\nSweep complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        run_sweep()
    except Exception as e:
        print(f"Sweep failed with error: {e}")
        import traceback
        traceback.print_exc()
