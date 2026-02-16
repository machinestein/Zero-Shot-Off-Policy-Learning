import os
import sys
import json
import torch
import numpy as np
import scipy.stats as st
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Metamotivo Imports
from metamotivo.agents.fb.flow_bc.agent import FBFlowBCAgent
from metamotivo.envs.ogbench import OGBenchEnvConfig, ALL_TASKS
from metamotivo.data_loading.ogbench import OGBenchDataConfig
from metamotivo.envs.utils.rollout import rollout

def calculate_success(infos):
    """Computes binary success (True/False) for each episode in the rollout."""
    return [any([step.get("success", False) for step in info]) for info in infos]

def get_stats(data):
    mean = np.mean(data)
    sem = st.sem(data) if len(data) > 1 else 0.0
    ci = 1.96 * sem
    return mean, ci

def get_ogbench_config(results_path, domain, task, results_root="/home/jovyan/bobrin/td_jepa/results_fb_ogbench_proprio"):
    """Robustly find best OGBench params from sweep file or fallback to disk."""
    ckpt_path = None
    best_params = None
    
    # 1. Try to load from sweep results
    if results_path and os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
                summary = data.get("summary", {})
                best_entry = None
            for entry_data in summary.values():
                if entry_data.get("domain") == domain and entry_data.get("task") == task:
                    # Handle both 'best_zol_sr' and the older 'best_zol_score'
                    score = entry_data.get("best_zol_sr", entry_data.get("best_zol_score", -1))
                    best_score = best_entry.get("best_zol_sr", best_entry.get("best_zol_score", -1)) if best_entry else -1
                    if best_entry is None or score > best_score:
                        best_entry = entry_data
            
            if best_entry: 
                ckpt_path = best_entry.get("checkpoint")
                best_params = best_entry.get("best_zol_params")
        except (json.JSONDecodeError, IOError):
            pass
            
    # 2. Fallback to searching on disk
    if not ckpt_path:
        root = Path(results_root)
        domain_dir = root / domain
        if domain_dir.exists():
            for seed_dir in domain_dir.glob("*"):
                potential_ckpt = seed_dir / "checkpoint"
                if potential_ckpt.exists():
                    ckpt_path = str(potential_ckpt)
                    print(f"  Fallback: Found checkpoint on disk for {domain}: {ckpt_path}")
                    break
                    
    return ckpt_path, best_params

def run_full_adaptation_evaluation(agent, env, env_cfg, domain, task, batch, num_episodes, eval_episodes):
    """
    Runs evaluation for Baseline, ZOL, ReLA, and LoLA on OGBench.
    Returns a dictionary of metrics.
    """
    device = agent.device
    
    # 1. Relabel rewards with mandatory +1.0 shift for OGBench
    relabel_fn = env_cfg.get_relabel_fn(task)
    next_physics = batch["next"]["physics"].detach().cpu().numpy()
    actions = batch["action"].detach().cpu().numpy()
    rewards_np = relabel_fn(next_physics, actions)
    rewards_np += 1.0  # Shift to [0, 1]
    
    rewards = torch.tensor(rewards_np, dtype=torch.float32).to(device)
    batch_obs = batch["next"]["observation"].to(device)
    
    # 2. Compute Baseline zr
    z_base = agent._model.reward_inference(batch_obs, rewards.reshape(-1, 1))
    
    # 3. Adaptation Methods
    methods = {"Baseline": z_base}
    
    # ZOL
    print(f"    Running ZOL Latent Search...")
    # Default ZOL params for OGBench navigation
    zol_search_params = {
        "mu_source": "init",      
        "use_exp_weights": True,
        "weight_temp": 1.0,       
        "mu_reward_top_frac": 0.05,
        "self_normalized_obj": True,
    }
    methods["ZOL"] = agent.zol_latent_search(env, batch_obs, rewards.flatten(), z_base, **zol_search_params)
    
    # ReLA
    print(f"    Running ReLA Adaptation...")
    methods["ReLA"] = agent.rela_fast_adaptation(
        env, z_base, 
        num_episodes=num_episodes, 
        lr_z=1e-4, lr_q=1e-4, 
        batch_size=1024
    )
    
    # LoLA
    print(f"    Running LoLA Adaptation...")
    methods["LoLA"] = agent.lola_fast_adaptation(
        env, z_base, 
        num_episodes=num_episodes, 
        lr_mu=0.05, 
        n_lookahead=100, 
        k_samples=10, 
        sigma=0.1
    )

    results = {}
    for name, z in methods.items():
        stats, infos, _ = rollout(env, agent=agent._model, ctx=z, num_episodes=eval_episodes)
        
        successes = calculate_success(infos)
        sr_m, sr_ci = get_stats(successes)
        rew_m, rew_ci = get_stats(stats['reward'])
        
        results[name] = {"sr": float(sr_m), "reward": float(rew_m), "ci": float(sr_ci)} # CI is for SR here
        print(f"    {name.ljust(10)} | Success: {sr_m*100:>5.1f}% | Return: {rew_m:>6.2f}")
        
    return results

def main():
    # --- Configuration ---
    INFERENCE_BATCH_SIZE = 10_000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ADAPT_EPISODES = 20
    EVAL_EPISODES = 100
    RESULTS_PATH = "../zol_sweep_ogbench_results.json"
    EVAL_RESULTS_OUTPUT = "all_ogbench_eval_results.json"

    all_domain_results = {}

    for domain, tasks in ALL_TASKS.items():
        print(f"\n{'#'*60}\nProcessing Domain: {domain}\n{'#'*60}")
        all_domain_results[domain] = {}
        
        try:
            data_cfg = OGBenchDataConfig(domain=domain, dataset_root="/home/jovyan/bobrin/td_jepa/ogbench_data")
            replay_buffer = data_cfg.build(buffer_device=DEVICE, batch_size=INFERENCE_BATCH_SIZE, frame_stack=1)
            batch = replay_buffer["train"].sample(INFERENCE_BATCH_SIZE)
        except Exception as e:
            print(f"Skipping domain {domain}: Failed to load data buffer. Error: {e}")
            continue

        agent = None 

        for task in tasks:
            print(f"\n{'='*50}\nTask: {domain}_{task}\n{'='*50}")
            
            ckpt_path, best_params = get_ogbench_config(RESULTS_PATH, domain, task)
            if ckpt_path is None:
                print(f"Warning: No checkpoint found for {domain}_{task}. Skipping.")
                continue
                
            env_cfg = OGBenchEnvConfig(domain=domain, task=task)
            env, _ = env_cfg.build()
            
            if agent is None:
                try:
                    agent = FBFlowBCAgent.load(ckpt_path, device=DEVICE, 
                                            obs_space=env.observation_space, 
                                            action_dim=batch["action"].shape[-1])
                    agent._model.train(False)
                except Exception as e:
                    print(f"Error loading agent for {domain}: {e}")
                    env.close()
                    break

            # Update agent config if we have best params from sweep
            if best_params:
                config_keys = {"lr", "num_steps", "n_mu", "early_stop_patience", "early_stop_tol", 
                            "chi2_coef", "trust_l2_coef", "weight_clip", "center_rewards"}
                cfg_updates = {k: v for k, v in best_params.items() if k in config_keys}
                
                if cfg_updates:
                    agent.cfg = agent.cfg.model_copy(update={
                        "train": agent.cfg.train.model_copy(
                            update={"zol": agent.cfg.train.zol.model_copy(update=cfg_updates)}
                        )
                    })

            try:
                task_metrics = run_full_adaptation_evaluation(
                    agent=agent, 
                    env=env, 
                    env_cfg=env_cfg,
                    domain=domain,
                    task=task,
                    batch=batch, 
                    num_episodes=ADAPT_EPISODES,
                    eval_episodes=EVAL_EPISODES
                )
                all_domain_results[domain][task] = task_metrics
            except Exception as e:
                print(f"Error evaluating {domain}_{task}: {e}")
            
            env.close()

    # Save Results
    with open(EVAL_RESULTS_OUTPUT, "w") as f:
        json.dump(all_domain_results, f, indent=4)

    print(f"\n\nResults saved to {EVAL_RESULTS_OUTPUT}")

if __name__ == "__main__":
    main()
