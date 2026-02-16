import sys
import os
import itertools
import json
import torch
import numpy as np
import scipy.stats as st
from pathlib import Path

# Set MUJOCO_GL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

# Add current directory to path if needed for metamotivo imports
sys.path.append(os.getcwd())

from metamotivo.agents.fb.agent import FBAgent, ZOLConfig
from metamotivo.envs.dmc import DMCEnvConfig
from metamotivo.data_loading.dmc import DMCDataConfig
from metamotivo.envs.dmc_tasks import dmc, ALL_TASKS
from metamotivo.envs.utils.rollout import rollout

# Config
RESULTS_FB_DIR = "/home/jovyan/bobrin/td_jepa/results_fb_dmc_proprio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "zol_sweep_all_results.json"

# Exclusion List (Domains to skip)
EXCLUDE_DOMAINS = ["walker", "cheetah"] 

# Sweep Space
sweep_space = {
    "lr": [5e-4, 1e-4],
    "num_steps": [100, 200],
    "chi2_coef": [0.0, 0.001, 0.005],
    "trust_l2_coef": [0.0, 0.02, 0.05],
    "weight_clip": [50.0, 100.0],
    "n_mu": [256],
}

def get_stats(data):
    if not data:
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
            sensitivity[key][str(val)] = {
                "mean_score": float(np.mean(scores)),
                "count": len(scores)
            }
    return sensitivity

def find_checkpoints(base_dir):
    checkpoints = []
    base_path = Path(base_dir)
    print(f"Searching for checkpoints in {base_dir}...")
    # Walk the directory to find folders named 'checkpoint' that contain config.json
    for ckpt_config in base_path.glob("**/checkpoint/config.json"):
        ckpt_dir = ckpt_config.parent
        # Infer domain from path
        domain = None
        for d in ALL_TASKS.keys():
            # Check if domain name is a component in the path
            if d in ckpt_dir.parts:
                domain = d
                break
        
        if domain:
            checkpoints.append({
                "path": str(ckpt_dir),
                "domain": domain
            })
            print(f"  Found: {ckpt_dir} (Domain: {domain})")
        else:
            print(f"  Warning: Could not infer domain for {ckpt_dir}")
    return checkpoints

def run_sweep():
    print(f"Starting Multi-Domain ZOL sweep")
    print(f"Device: {DEVICE}")
    print(f"Results Directory: {RESULTS_FB_DIR}")

    checkpoints = find_checkpoints(RESULTS_FB_DIR)
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    all_domain_results = {}
    master_summary = {}
    all_flat_results = []

    # 0. Load existing results if they exist
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing results from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "r") as f:
            existing_data = json.load(f)
            master_summary = existing_data.get("summary", {})
            all_domain_results = existing_data.get("detailed", {})
            # We reconstruct all_flat_results from detailed for sensitivity calculation
            for ckpt in all_domain_results:
                for task in all_domain_results[ckpt]:
                    all_flat_results.extend(all_domain_results[ckpt][task])

    keys = sweep_space.keys()
    values = sweep_space.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for ckpt_info in checkpoints:
        ckpt_path = ckpt_info["path"]
        domain = ckpt_info["domain"]

        # Filter excluded domains
        if domain in EXCLUDE_DOMAINS:
            print(f"\nSkipping Checkpoint: {ckpt_path} (Domain {domain} is in EXCLUDE_DOMAINS)")
            continue

        print(f"\n======== Processing Checkpoint: {ckpt_path} (Domain: {domain}) ========")

        # 1. Load Agent and Environment
        # Use first task to build temp env for spaces
        tasks = ALL_TASKS.get(domain, [])
        if not tasks:
            print(f"  No tasks found for domain {domain}. Skipping.")
            continue
            
        temp_env_cfg = DMCEnvConfig(domain=domain, task=tasks[0])
        temp_env, _ = temp_env_cfg.build()
        
        try:
            agent = FBAgent.load(
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
        data_cfg = DMCDataConfig(domain=domain, dataset_root="/home/jovyan/.exorl/expl_datasets")
        inf_bs = agent.cfg.model.inference_batch_size
        replay_buffer = data_cfg.build(buffer_device=DEVICE, batch_size=inf_bs, frame_stack=1)
        batch = replay_buffer["train"].sample(inf_bs)
        
        batch_obs = batch["next"]["observation"].to(DEVICE)
        next_physics = batch["next"]["physics"].cpu().numpy()

        all_domain_results[ckpt_path] = {}

        for task in tasks:
            print(f"\n--- [Task: {domain}_{task}] ---")
            all_domain_results[ckpt_path][task] = []
            
            target_env = dmc.make(f"{domain}_{task}")
            rewards_list = []
            print(f"  Relabeling {inf_bs} transitions for {task}...")
            for i in range(inf_bs):
                with target_env.physics.reset_context():
                    target_env.physics.set_state(next_physics[i])
                rewards_list.append(target_env._task.get_reward(target_env.physics))
            rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(DEVICE)
            
            # Compute Initial Z (Base FB)
            initial_z = agent._model.reward_inference(batch_obs, rewards.unsqueeze(1))
            
            # Baseline Evaluation
            env_cfg = DMCEnvConfig(domain=domain, task=task)
            task_env, _ = env_cfg.build()
            print(f"  Evaluating Base FB baseline (100 episodes)...")
            base_stats, _, _ = rollout(task_env, agent=agent._model, ctx=initial_z, num_episodes=100)
            base_mean, base_ci = get_stats(base_stats['reward'])
            print(f"  Baseline: {base_mean:.2f} \u00b1 {base_ci:.2f}")

            best_task_score = -float('inf')
            best_task_params = None

            for idx, params in enumerate(combinations):
                print(f"  [{idx+1}/{len(combinations)}] Testing: {params}")
                
                # Apply Params to Agent
                agent.cfg = agent.cfg.model_copy(
                    update={
                        "train": agent.cfg.train.model_copy(
                            update={
                                "zol": agent.cfg.train.zol.model_copy(update=params)
                            }
                        )
                    }
                )
                
                # Perform ZOL Latent Search
                z_zol = agent.zol_latent_search(target_env, batch_obs, rewards, initial_z)
                
                # Evaluate ZOL Agent (10 episodes during sweep)
                stats, _, _ = rollout(task_env, agent=agent._model, ctx=z_zol, num_episodes=100)
                mean, ci = get_stats(stats['reward'])
                
                print(f"    Result: {mean:.2f} \u00b1 {ci:.2f}")
                
                res_entry = {
                    "checkpoint": ckpt_path,
                    "domain": domain,
                    "params": params,
                    "score": mean,
                    "ci": ci
                }
                all_domain_results[ckpt_path][task].append(res_entry)
                all_flat_results.append(res_entry)
                
                if mean > best_task_score:
                    best_task_score = mean
                    best_task_params = params

            master_summary[f"{ckpt_path}_{task}"] = {
                "checkpoint": ckpt_path,
                "domain": domain,
                "task": task,
                "base_score": base_mean,
                "base_ci": base_ci,
                "best_zol_score": best_task_score,
                "best_zol_params": best_task_params,
                "improvement": best_task_score - base_mean
            }
            
            print(f"  [{task}] Best Score: {best_task_score:.2f} with {best_task_params}")
            
            task_env.close()
            target_env.close()

    # Calculate sensitivity
    sensitivity_res = calculate_sensitivity(all_flat_results)

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "summary": master_summary,
            "sensitivity": sensitivity_res,
            "detailed": all_domain_results
        }, f, indent=4)
    
    print(f"\nSweep complete. Results saved to {OUTPUT_FILE}")
    
    print("\nHYPERPARAMETER SENSITIVITY (Mean score across all tasks/checkpoints):")
    for param, vals in sensitivity_res.items():
        print(f"  [{param}]")
        sorted_vals = sorted(vals.items(), key=lambda x: x[1]["mean_score"], reverse=True)
        for val_str, data in sorted_vals:
            print(f"    {val_str:<10} : {data['mean_score']:.2f}")

    print("\nFINAL SUMMARY:")
    print(f"{'Domain/Task':<30} | {'Base FB':<10} | {'Best ZOL':<10} | {'Improvement':<10}")
    print("-" * 75)
    for key, res in master_summary.items():
        label = f"{res['domain']}_{res['task']}"
        print(f"{label:<30} | {res['base_score']:>10.2f} | {res['best_zol_score']:>10.2f} | {res['improvement']:>10.2f}")

if __name__ == "__main__":
    try:
        run_sweep()
    except Exception as e:
        print(f"Sweep failed with error: {e}")
        import traceback
        traceback.print_exc()
