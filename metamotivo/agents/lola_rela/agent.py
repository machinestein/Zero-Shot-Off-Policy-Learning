# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Tuple, Any
import numpy as np

from ...base import BaseConfig
from ...nn_models import weight_init, eval_mode
from ..fb.agent import FBAgent, FBAgentConfig
from .model import ResidualCritic

class ReLAAgentTrainConfig(BaseConfig):
    lr_z: float = 1e-3
    lr_critic: float = 1e-3
    batch_size: int = 1024
    discount: float = 0.99
    critic_target_tau: float = 0.01
    num_critic_layers: int = 3
    critic_hidden_dim: int = 256

class ReLAAgentConfig(BaseConfig):
    name: Literal["ReLAAgent"] = "ReLAAgent"
    fb_agent_path: str
    train: ReLAAgentTrainConfig = ReLAAgentTrainConfig()
    device: str = "cuda"

    def build(self, obs_space, action_dim):
        return ReLAAgent(obs_space, action_dim, self)

class ReLAAgent:
    def __init__(self, obs_space, action_dim, cfg: ReLAAgentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Load pre-trained FB agent
        self.fb_agent = FBAgent.load(cfg.fb_agent_path, device=cfg.device)
        self.fb_agent._model.train(False)
        self.fb_agent._model.requires_grad_(False)
        
        # Latent variable z (to be optimized)
        # Initialized from zero-shot or randomly if needed. 
        # For now, we'll initialize it in the update loop or provide a method to set it.
        self.z = nn.Parameter(torch.randn(1, self.fb_agent._model.cfg.archi.z_dim, device=self.device))
        
        # Residual Critic
        obs_dim = obs_space.shape[0]
        self.critic = ResidualCritic(
            obs_dim, action_dim, 
            hidden_dim=cfg.train.critic_hidden_dim, 
            num_layers=cfg.train.num_critic_layers
        ).to(self.device)
        self.target_critic = ResidualCritic(
            obs_dim, action_dim, 
            hidden_dim=cfg.train.critic_hidden_dim, 
            num_layers=cfg.train.num_critic_layers
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.z_optimizer = torch.optim.Adam([self.z], lr=cfg.train.lr_z)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.train.lr_critic)
        
        self.critic.apply(weight_init)

    def set_z_from_reward(self, reward_fn_or_z):
        # Allow setting z from zero-shot reward weight
        if isinstance(reward_fn_or_z, torch.Tensor):
            self.z.data.copy_(reward_fn_or_z)
        else:
            # Handle reward function if needed
            pass

    def get_q_value(self, obs, action, z, use_target=False):
        # Q_total = Q_base + Q_residual
        # Q_base = F(s, a, z)^T * z
        with torch.no_grad():
            # Encere observations
            norm_obs = self.fb_agent._model._normalize(obs)
            fw_enc = self.fb_agent._model._fw_encoder(norm_obs)
            left_enc = self.fb_agent._model._left_encoder(fw_enc)
            
            Fs = self.fb_agent._model._forward_map(left_enc, z, action) # num_parallel x batch x z_dim
            Q_base = (Fs * z.unsqueeze(1)).sum(dim=-1).mean(dim=0) # batch
            
        critic = self.target_critic if use_target else self.critic
        Q_residual = critic(obs, action).squeeze(-1) # batch
        return Q_base + Q_residual

    def update(self, replay_buffer, step: int):
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        obs = batch["observation"].to(self.device)
        action = batch["action"].to(self.device)
        next_obs = batch["next"]["observation"].to(self.device)
        reward = batch["next"]["reward"].to(self.device)
        terminated = batch["next"]["terminated"].to(self.device)
        discount = self.cfg.train.discount * (~terminated).float()

        # Update Critic
        with torch.no_grad():
            # Sample next action from frozen actor with current z
            next_action = self.fb_agent.act(next_obs, self.z.expand(next_obs.shape[0], -1), mean=False)
            target_Q = reward + discount * self.get_q_value(next_obs, next_action, self.z, use_target=True)
        
        current_Q = self.get_q_value(obs, action, self.z, use_target=False)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update z (Latent Actor Update)
        # Maximize Q_total(s, pi_z(s))
        with torch.no_grad():
            # Encere observations for actor
            norm_obs = self.fb_agent._model._normalize(obs)
            fw_enc = self.fb_agent._model._fw_encoder(norm_obs)
            left_enc = self.fb_agent._model._left_encoder(fw_enc)
        
        # Sample action from frozen actor, but we need to backprop through z?
        # The actor π_z(s) depends on z.
        # actor(obs, z, std)
        dist = self.fb_agent._model._actor(left_enc, self.z.expand(obs.shape[0], -1), self.fb_agent._model.cfg.actor_std)
        actor_action = dist.rsample() # rsample if supported, or use means
        
        Q_total = self.get_q_value(obs, actor_action, self.z)
        z_loss = -Q_total.mean()
        
        self.z_optimizer.zero_grad()
        z_loss.backward()
        self.z_optimizer.step()
        
        # Soft update target critic
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.cfg.train.critic_target_tau) + 
                    param.data * self.cfg.train.critic_target_tau
                )
        
        return {
            "critic_loss": critic_loss.item(),
            "z_loss": z_loss.item(),
            "q_mean": Q_total.mean().item()
        }

class LoLAAgentTrainConfig(BaseConfig):
    lr_mu: float = 1e-3
    sigma: float = 0.1
    num_samples_k: int = 4
    lookahead_n: int = 10
    batch_size: int = 64
    discount: float = 0.99

class LoLAAgentConfig(BaseConfig):
    name: Literal["LoLAAgent"] = "LoLAAgent"
    fb_agent_path: str
    train: LoLAAgentTrainConfig = LoLAAgentTrainConfig()
    device: str = "cuda"

    def build(self, obs_space, action_dim):
        return LoLAAgent(obs_space, action_dim, self)

class LoLAAgent:
    def __init__(self, obs_space, action_dim, cfg: LoLAAgentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.fb_agent = FBAgent.load(cfg.fb_agent_path, device=cfg.device)
        self.fb_agent._model.train(False)
        self.fb_agent._model.requires_grad_(False)
        
        self.mu = nn.Parameter(torch.randn(1, self.fb_agent._model.cfg.archi.z_dim, device=self.device))
        self.mu_optimizer = torch.optim.Adam([self.mu], lr=cfg.train.lr_mu)
        self.sigma = cfg.train.sigma

    def update(self, env, replay_buffer, step: int):
        # LoLA requires rollouts. This 'env' must support reset to specific states or we use the replay buffer states.
        # Sample s0 from replay buffer or initial distribution
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        obs_s0 = batch["observation"].to(self.device).detach()
        
        # Sample k latents z_i ~ N(mu, sigma)
        k = self.cfg.train.num_samples_k
        eps = torch.randn(self.cfg.train.batch_size, k, self.mu.shape[-1], device=self.device)
        z_samples = self.mu.detach() + self.sigma * eps # batch x k x z_dim
        
        # We need to compute gradients of log π(z_i | mu)
        # log π(z_i | mu) = -0.5 * ||(z_i - mu)/sigma||^2 + const
        
        # For each sample z_i, we need the lookahead return R_i(s0, z_i)
        returns = []
        for i in range(k):
            z_i = z_samples[:, i]
            # rollout_return = self._run_rollout(env, obs_s0, z_i)
            # returns.append(rollout_return)
            # For now, placeholder returns
            returns.append(torch.randn(self.cfg.train.batch_size, device=self.device))
        
        returns = torch.stack(returns, dim=1) # batch x k
        
        # RLOO Baseline: b_i = 1/(k-1) * sum_{j!=i} R_j
        sum_returns = returns.sum(dim=1, keepdim=True)
        baselines = (sum_returns - returns) / (k - 1)
        
        # Advantages
        advantages = returns - baselines # batch x k
        
        # Policy gradient: grad_mu = E [ (R - b) * grad_mu log p(z | mu) ]
        # grad_mu log p(z | mu) = (z - mu) / sigma^2
        
        # z_samples: batch x k x z_dim
        # mu: 1 x z_dim
        log_prob_grad = (z_samples - self.mu) / (self.sigma**2) # batch x k x z_dim
        
        # Final gradient estimate
        # advantages: batch x k -> batch x k x 1
        mu_grad = -(advantages.unsqueeze(-1) * log_prob_grad).mean(dim=(0, 1))
        
        self.mu_optimizer.zero_grad()
        self.mu.grad = mu_grad.reshape(self.mu.shape)
        self.mu_optimizer.step()
        
        # Project z back to hypersphere if needed (paper mentions this)
        with torch.no_grad():
            self.mu.data = self.fb_agent._model.project_z(self.mu.data)
        
        return {
            "mu_loss": -returns.mean().item(),
            "return_mean": returns.mean().item(),
            "advantage_std": advantages.std().item()
        }
    
    def _run_rollout(self, env, s0, z, n):
        # Placeholder for rollout logic
        # For each state s_t in batch s0:
        #   total_reward = 0
        #   s = s_t
        #   for t in range(n):
        #     a = self.fb_agent.act(s, z)
        #     s_next, r, term, ... = env.step(a)
        #     total_reward += gamma^t * r
        #     s = s_next
        #     if term: break
        #   total_reward += gamma^n * Q_base(s_next, act(s_next, z), z)
        return torch.zeros(s0.shape[0], device=self.device)
