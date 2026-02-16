import os
import math
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from metamotivo.agents.fb.agent import FBAgent, FBAgentConfig, FBModelConfig, FBAgentTrainConfig
from metamotivo.agents.fb.model import FBModelArchiConfig
from metamotivo.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig
from metamotivo.buffers.transition import DictBuffer
from metamotivo.normalizers import IdentityNormalizerConfig

class DonutEnv(gym.Env):
    """
    A 2D PointMass environment restricted to a donut region:
    0.25 <= r <= 1.5
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Uniformly sample inside the donut
        while True:
            pos = self.np_random.uniform(-1.5, 1.5, size=(2,))
            r = np.linalg.norm(pos)
            if 0.25 <= r <= 1.5:
                self.state = pos.astype(np.float32)
                break
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = self.state + action
        r = np.linalg.norm(next_state)
        
        # Boundary handling: stay in bounds
        if 0.25 <= r <= 1.5:
            self.state = next_state.astype(np.float32)
        
        return self.state, 0.0, False, False, {}

import torch.nn as nn
from metamotivo.agents.fb.model import FBModel

class FourierEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, scale=2.0):
        super().__init__()
        # Gaussian Random Fourier Features
        self.register_buffer("B_linear", torch.randn(in_dim, out_dim // 2) * scale)

    def forward(self, x):
        # Coordinate mapping to higher dimensionality
        x_proj = x @ self.B_linear
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ToyBackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, hidden_dim=512, fourier_dim=128, enable_fourier=False, num_layers=4):
        super().__init__()
        self.enable_fourier = enable_fourier
        if enable_fourier:
            self.fourier = FourierEncoding(obs_space.shape[0], fourier_dim, scale=2.0)
            in_features = fourier_dim
        else:
            in_features = obs_space.shape[0]

        layers = [nn.Linear(in_features, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.enable_fourier:
            x = self.fourier(x)
        x = self.net(x)
        return np.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)

import typing as tp

class ToyFBModelConfig(FBModelConfig):
    name: tp.Literal["FBModel"] = "FBModel"
    enable_fourier: bool = True
    def build(self, obs_space, action_dim, discrete: bool = False) -> "ToyFBModel":
        return ToyFBModel(obs_space, action_dim, self, discrete=discrete, enable_fourier=self.enable_fourier)

class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)

class ForwardMapWrapper(nn.Module):
    def __init__(self, forward_map):
        super().__init__()
        self.forward_map = forward_map
        self.norm = Norm()
    def forward(self, obs, z, action=None):
        out = self.forward_map(obs, z, action)
        return self.norm(out)

class ToyFBModel(FBModel):
    config_class = ToyFBModelConfig
    def __init__(self, obs_space, action_dim, cfg, discrete=False, enable_fourier=True):
        super().__init__(obs_space, action_dim, cfg, discrete=discrete)
        # Override with sharp backward map
        self._backward_map = ToyBackwardMap(obs_space, cfg.archi.z_dim, enable_fourier=enable_fourier)
        self._backward_map.to(self.device)
        # CRITICAL FIX: Normalize ForwardMap output
        self._forward_map = ForwardMapWrapper(self._forward_map).to(self.device)

class ToyFBAgent(FBAgent):
    def __init__(self, obs_space, action_dim, cfg, enable_fourier=True):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        self.discrete = cfg.discrete
        self._model = ToyFBModel(obs_space, action_dim, self.cfg.model, discrete=self.discrete, enable_fourier=enable_fourier)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.device)

def collect_data(env, capacity=100_000):
    obs_t = torch.empty((capacity, 2), dtype=torch.float32)
    next_obs_t = torch.empty((capacity, 2), dtype=torch.float32)
    act_t = torch.empty((capacity, 2), dtype=torch.float32)
    done_t = torch.zeros((capacity, 1), dtype=torch.bool)

    obs, _ = env.reset()
    for step in range(capacity):
        action = env.action_space.sample()
        next_obs, _, _, _, _ = env.step(action)

        # store transition
        obs_t[step] = torch.from_numpy(obs)
        next_obs_t[step] = torch.from_numpy(next_obs)
        act_t[step] = torch.from_numpy(action)

        obs = next_obs

    buffer = DictBuffer(capacity=capacity)
    buffer.extend({
        "observation": obs_t,
        "action": act_t,
        "next": {"observation": next_obs_t, "terminated": done_t}
    })
    return buffer

from reward_functions import (
    get_donut_mask,
    two_circles_reward,
    square_reward,
    cross_reward,
    grid_reward,
)

def sample_donut_points(n: int, inner_r=0.25, outer_r=1.5, low=-1.5, high=1.5, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        cand = rng.uniform(low, high, size=(n, 2))
        r = np.linalg.norm(cand, axis=1)
        mask = (r >= inner_r) & (r <= outer_r)
        pts.extend(cand[mask].tolist())
    pts = np.array(pts[:n], dtype=np.float32)
    return pts

@torch.no_grad()
def infer_z_ridge(model, s: torch.Tensor, r: torch.Tensor, lam: float = 1e-3, center: bool = True) -> torch.Tensor:
    B = model.backward_map(s)
    r = r.to(torch.float32)
    if center:
        r = r - r.mean()
    BtB = B.T @ B
    z_dim = BtB.shape[0]
    BtB = BtB + lam * torch.eye(z_dim, device=BtB.device, dtype=BtB.dtype)
    Btr = B.T @ r
    z_col = torch.linalg.solve(BtB, Btr)
    z = z_col.T
    return model.project_z(z)

def masked_heatmap(ax, X, Y, Z, mask, title, cmap="viridis", vmin=None, vmax=None):
    Zm = Z.copy()
    Zm[~mask] = np.nan
    im = ax.imshow(
        Zm,
        origin="lower",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(title)
    return im

def visualize_progress(agent, step, outdir="eval_progress"):
    os.makedirs(outdir, exist_ok=True)
    device = agent.device
    model = agent._model
    model.eval()
    
    grid_size = 400
    dataset_n = 100_000 #5000
    x_dim = np.linspace(-1.6, 1.6, grid_size)
    y_dim = np.linspace(-1.6, 1.6, grid_size)
    X, Y = np.meshgrid(x_dim, y_dim)
    mask = get_donut_mask(X, Y).astype(bool)
    
    tasks = [
        ("Square", square_reward), 
        ("Cross", cross_reward),
        ("TwoCircles", two_circles_reward),
        ("Grid", grid_reward),
        ("DonutMask", get_donut_mask),
    ]
    
    for name, r_fn in tasks:
        pts = sample_donut_points(dataset_n, seed=0)
        pts_t = torch.from_numpy(pts).to(device)
        r_pts = r_fn(pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1)).reshape(-1).astype(np.float32)
        r_t = torch.from_numpy(r_pts).to(device).view(-1, 1)

        z_ridge = infer_z_ridge(model, pts_t, r_t, lam=1e-3, center=True)
        z_corr = model.reward_inference(pts_t, r_t)
        
        grid = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
        grid_t = torch.from_numpy(grid).to(device)
        B_grid = model.backward_map(grid_t).detach()
        
        r_hat_ridge = (B_grid @ z_ridge.squeeze(0)).detach().cpu().numpy().reshape(X.shape)
        r_hat_corr = (B_grid @ z_corr.squeeze(0)).detach().cpu().numpy().reshape(X.shape)
        r_true = r_fn(X, Y).astype(np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        masked_heatmap(axes[0], X, Y, r_true, mask, f"Target: {name}", vmin=-1, vmax=1)
        masked_heatmap(axes[1], X, Y, r_hat_ridge, mask, f"FB Ridge")
        masked_heatmap(axes[2], X, Y, r_hat_corr, mask, f"FB Corr")
        
        fig.suptitle(f"Step {step}: {name}")
        outpath = os.path.join(outdir, f"step_{step:06d}_{name.lower()}.png")
        plt.savefig(outpath, dpi=300)
        plt.close(fig)
    
    model.train()

def train_agent(agent, buffer, iters=20_000, eval_steps=2500):
    pbar = tqdm(range(iters), desc="Training FB")
    for i in pbar:
        metrics = agent.update({"train": buffer}, i)
        
        if i % eval_steps == 0 or i == iters - 1:
            visualize_progress(agent, i)
            
        if i % 1000 == 0:
            pbar.set_postfix({
                "fb_loss": f"{metrics.get('fb_loss', 0):.4f}",
                "orth": f"{metrics.get('orth_loss', 0):.4f}",
                "q": f"{metrics.get('q', 0):.4f}"
            })

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = DonutEnv()
    buffer = collect_data(env, capacity=500_000)

    archi_cfg = FBModelArchiConfig(
        L_dim=256, # not used if left_encoder is identity
        z_dim=512,
        norm_z=True,
        # left_encoder=BackwardArchiConfig(hidden_dim=256, hidden_layers=2, norm=True),
        f=ForwardArchiConfig(hidden_dim=256, hidden_layers=3, num_parallel=2),
        b=BackwardArchiConfig(hidden_dim=256, hidden_layers=3, norm=True),
        actor=ActorArchiConfig(hidden_dim=256, hidden_layers=3),
    )
    enable_fourier = False
    fb_cfg = FBAgentConfig(
        compile=True,
        model=ToyFBModelConfig(
            obs_normalizer=IdentityNormalizerConfig(),
            archi=archi_cfg,
            actor_encode_obs=False,
            device=device,
            enable_fourier=enable_fourier
        ),
        train=FBAgentTrainConfig(
            batch_size=4096, 
            discount=0.99, 
            ortho_coef=1.0,
            lr_f=1e-3, 
            lr_b=1e-3,
            lr_actor=1e-3
        )
    )

    # Use the ToyFBAgent which uses ToyFBModel (Fourier toggleable)
    agent = ToyFBAgent(obs_space=env.observation_space, action_dim=env.action_space.shape[0], cfg=fb_cfg, enable_fourier=enable_fourier)
    
    train_agent(agent, buffer, iters=50_000, eval_steps=10_000)

    # Save model
    agent._model.save("fb_donut_model")
    print("Training complete. Model saved to fb_donut_model (with Fourier BackwardMap).")
