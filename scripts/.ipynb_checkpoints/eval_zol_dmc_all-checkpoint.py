import os
import sys
import json
import torch
import numpy as np
import scipy.stats as st
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables for MuJoCo
os.environ['MUJOCO_GL'] = 'egl'

# Metamotivo Imports
from metamotivo.agents.fb.agent import FBAgent
from metamotivo.envs.dmc import DMCEnvConfig
from metamotivo.data_loading.dmc import DMCDataConfig
from metamotivo.envs.dmc_tasks import ALL_TASKS
from metamotivo.envs.utils.rollout import rollout

def get_zol_config(results_path, domain, task):
    """
    Finds the best checkpoint and ZOL parameters for a specific domain and task.
    """
    if not os.path.exists(results_path):
        print(f"Warning: {results_path} not found.")
        return None, None
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    summary = results.get("summary", {})
    best_entry = None
    
    for entry_data in summary.values():
        if entry_data.get("domain") == domain and entry_data.get("task") == task:
            if best_entry is None or entry_data.get("best_zol_score", -1) > best_entry.get("best_zol_score", -1):
                best_entry = entry_data
                
    if best_entry:
        return best_entry.get("checkpoint"), best_entry.get("best_zol_params")
    
    return None, None

def run_full_adaptation_evaluation(agent, env, batch, num_episodes, eval_episodes):
    """
    Runs evaluation for Baseline (zero-shot), ZOL, ReLA, and LoLA.
    Returns a dictionary of metrics.
    """
    device = agent.device
    batch_obs = batch["next"]["observation"].to(device)
    next_physics = batch["next"]["physics"].cpu().numpy()
    
    # 1. Relabel rewards for ZOL
    rewards = []
    for i in range(batch_obs.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(next_physics[i])
        if hasattr(env._task, "after_step"):
            env._task.after_step(env.physics)
        rewards.append(float(env._task.get_reward(env.physics)))
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    # 2. Compute Baseline zr
    z_base = agent._model.reward_inference(batch_obs, rewards.unsqueeze(1))
    
    # 3. Fast Adaptation Methods
    methods = {"Baseline": z_base}
    
    print(f"    Running ZOL Latent Search...")
    methods["ZOL"] = agent.zol_latent_search(
        env, batch_obs, rewards, z_base,
        mu_source="batch",
        use_exp_weights=True,
        weight_temp=2.0,
        mu_reward_top_frac=0.05,
        self_normalized_obj=True,
    )
    
    # Using recommended paper defaults for ReLA and LoLA
    print(f"    Running ReLA Adaptation...")
    methods["ReLA"] = agent.rela_fast_adaptation(
        env, z_base, 
        num_episodes=num_episodes, 
        lr_z=1e-4, lr_q=1e-4, 
        batch_size=1024
    )
    
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
        
        successes = [any([s.get("success", False) for s in info]) for info in infos]
        sr = np.mean(successes) if successes else 0.0
        
        rew_mean = np.mean(stats['reward'])
        rew_ci = 1.96 * st.sem(stats['reward']) if len(stats['reward']) > 1 else 0.0
        
        results[name] = {"sr": sr, "reward": rew_mean, "ci": rew_ci}
        print(f"    {name.ljust(10)} | Success: {sr*100:>5.1f}% | Return: {rew_mean:>6.2f} ± {rew_ci:>5.2f}")
        
    return results

def main():
    # --- Configuration ---
    INFERENCE_BATCH_SIZE = 10_000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ADAPT_EPISODES = 20   # As per paper's "fast adaptation" recommendation
    EVAL_EPISODES = 100
    RESULTS_PATH = "/home/jovyan/bobrin/td_jepa/zol_sweep_dmc_results.json"
    EVAL_RESULTS_OUTPUT = "all_domains_eval_results.json"

    all_domain_results = {}

    for domain, tasks in ALL_TASKS.items():
        print(f"\n{'#'*60}\nProcessing Domain: {domain}\n{'#'*60}")
        all_domain_results[domain] = {}
        
        try:
            data_cfg = DMCDataConfig(domain=domain, dataset_root="/home/jovyan/.exorl/expl_datasets")
            replay_buffer = data_cfg.build(buffer_device=DEVICE, batch_size=INFERENCE_BATCH_SIZE, frame_stack=1)
            batch = replay_buffer["train"].sample(INFERENCE_BATCH_SIZE)
        except Exception as e:
            print(f"Skipping domain {domain}: Failed to load data buffer. Error: {e}")
            continue

        agent = None 

        for task in tasks:
            print(f"\n{'='*50}\nTask: {domain}_{task}\n{'='*50}")
            
            ckpt_path, best_task_params = get_zol_config(RESULTS_PATH, domain, task)
            if ckpt_path is None:
                print(f"Warning: No checkpoint found in results for {domain}_{task}. Skipping.")
                continue
            print(best_task_params)
            env, _ = DMCEnvConfig(domain=domain, task=task).build()
            
            if agent is None:
                try:
                    agent = FBAgent.load(ckpt_path, device=DEVICE, 
                                        obs_space=env.observation_space, 
                                        action_dim=env.action_space.shape[0])
                    agent._model.train(False)
                except Exception as e:
                    print(f"Error loading agent for {domain}: {e}")
                    env.close()
                    break

            if best_task_params:
                agent.cfg = agent.cfg.model_copy(update={
                    "train": agent.cfg.train.model_copy(update={
                        "zol": agent.cfg.train.zol.model_copy(update=best_task_params)
                    })
                })

            try:
                task_metrics = run_full_adaptation_evaluation(
                    agent=agent, 
                    env=env, 
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
