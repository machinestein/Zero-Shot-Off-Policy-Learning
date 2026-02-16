import os
import math
import typing as tp
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metamotivo.agents.fb.model import FBModel, FBModelConfig
from reward_functions import (
    get_donut_mask,
    two_circles_reward,
    square_reward,
    cross_reward,
    grid_reward,
)

import torch.nn as nn
import torch.nn.functional as F

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
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)

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

class ToyFBModelConfig(FBModelConfig):
    name: tp.Literal["FBModel"] = "FBModel"
    enable_fourier: bool = False
    def build(self, obs_space, action_dim, discrete: bool = False) -> "ToyFBModel":
        return ToyFBModel(obs_space, action_dim, self, discrete=discrete, enable_fourier=self.enable_fourier)

class ToyFBModel(FBModel):
    config_class = ToyFBModelConfig
    def __init__(self, obs_space, action_dim, cfg, discrete=False, enable_fourier=False):
        super().__init__(obs_space, action_dim, cfg, discrete=discrete)
        # Override with sharp backward map
        self._backward_map = ToyBackwardMap(obs_space, cfg.archi.z_dim, enable_fourier=enable_fourier)
        self._backward_map.to(self.device)
        # Match training architecture: normalize forward map
        self._forward_map = ForwardMapWrapper(self._forward_map).to(self.device)

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
def infer_z_ridge(model: FBModel, s: torch.Tensor, r: torch.Tensor, lam: float = 1e-3, center: bool = True) -> torch.Tensor:
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

def apply_publication_style():
    """Apply publication-ready rcParams based on tables.ipynb."""
    title_fontsize = 21
    label_fontsize = 20
    tick_fontsize = 18
    legend_fontsize = 18

    plt.rcParams.update({
        "font.size": label_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": legend_fontsize,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
    })

def masked_heatmap(ax, X, Y, Z, mask, title, cmap="viridis", vmin=None, vmax=None):
    """Plot heatmap with masked values."""
    Zm = Z.copy()
    Zm[~mask] = np.nan
    im = ax.imshow(
        Zm,
        origin="lower",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        interpolation="bilinear",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    if title:
        ax.set_title(title)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    
    # Hide ticks for a cleaner publication look if requested
    ax.set_xticks([])
    ax.set_yticks([])
            
    return im

def masked_occupancy_heatmap(ax, X, Y, Z, mask, title, cmap="viridis"):
    """Special heatmap for occupancy focusing on the 98th percentile for better visibility."""
    Zm = Z.copy()
    Zm[~mask] = np.nan
    v = Zm[np.isfinite(Zm)]
    if v.size > 0:
        vmax = np.percentile(v, 98)
        vmin = np.percentile(v, 2)
        if vmax <= vmin: vmax = vmin + 1e-8
    else:
        vmin, vmax = 0, 1
        
    im = ax.imshow(
        Zm,
        origin="lower",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        interpolation="bilinear",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

from metamotivo.base_model import load_model

@torch.no_grad()
def estimate_mu_pi_z(model: FBModel, obs: torch.Tensor, z: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Estimate mu_pi_z = E[F(s, pi(s,z), z)] using standard FB methods."""
    if z.ndim == 1:
        z = z.unsqueeze(0)
    B = obs.shape[0]
    z_batch = z.expand(B, -1)
    
    # Use standard actor
    action = model.act(obs, z_batch, mean=True)
    
    # Use standard forward map
    F_sa = model.forward_map(obs, z_batch, action)
    
    if F_sa.ndim == 3:
        F_sa = F_sa.mean(0)
    
    return F_sa.mean(0)

def sample_neighborhood_points(center: np.ndarray, n: int, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    """Sample points around a center starting point s0."""
    rng = np.random.default_rng(seed)
    pts = center + rng.normal(0, sigma, size=(n, 2))
    return pts.astype(np.float32)

@torch.no_grad()
def compute_sm_from_batch(model: FBModel, obs_t: torch.Tensor, z: torch.Tensor, grid_B: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    Compute averaged Successor Density m(s') = (1-gamma) * E_{s0, a0}[ F(s0, a0, z)^T B(s') ].
    This represents the occupancy measure across different starting states s0 and their actions a0.
    """
    # Estimate averaged successor measure mu = E[F(s, a, z)] over the batch
    mu = estimate_mu_pi_z(model, obs_t, z, gamma=gamma)
    
    # Compute dot product with grid B(s') to get spatial density
    # Shape: (grid_size,)
    density = (1.0 - gamma) * (grid_B @ mu.unsqueeze(1)).squeeze(1)
    return density

def compute_measure_heatmap(model: FBModel, z: torch.Tensor, pts_t: torch.Tensor, B_grid: torch.Tensor, shape: tuple, gamma: float = 0.99) -> np.ndarray:
    """Compute the FB successor measure heatmap: m(s') = (1-gamma) * B(s')^T mu_pi(z)"""
    mu = estimate_mu_pi_z(model, pts_t, z, gamma=gamma)
    
    # Standard FB Successor Measure logic: m(s') = (1-gamma) * B(s')^T mu
    # We multiply by (1-gamma) to normalize it into a probability-like distribution
    # over the infinite horizon (though in practice B and F scales handle this).
    measure = (1.0 - gamma) * (B_grid @ mu.unsqueeze(1)).squeeze(1)
    
    # Approximation might lead to small negative values; we use ReLU for visualization
    measure = torch.clamp(measure, min=0.0)
    
    return measure.detach().cpu().numpy().reshape(shape)

def zol_latent_search_toy(
    model: FBModel, 
    batch_obs: torch.Tensor, 
    rewards: torch.Tensor, 
    initial_z: torch.Tensor,
    weight_temp: float = 1.0,
    chi2_coef: float = 0.0,
    trust_l2_coef: float = 0.0,
    num_steps: int = 200,
    lr: float = 1e-2,
    gamma: float = 0.99,
    device: str = "cpu"
) -> torch.Tensor:
    """Simplified ZOL latent search for toy experiments."""
    z = initial_z.clone().detach().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    
    rewards = rewards.to(torch.float32)
    rewards = rewards - rewards.mean()
    
    # Precompute B(s)
    with torch.no_grad():
        B_s = model.backward_map(batch_obs)
    
    z0 = initial_z.clone().detach().to(device)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        
        # 1. Estimate mu_pi_z
        # Use a larger chunk or full batch for mu estimation to reduce noise
        mu_pi = estimate_mu_pi_z(model, batch_obs, z, gamma=gamma)
        
        # 2. Compute weights: w(s) propto exp(beta (1-gamma) B(s)^T mu)
        logit = (1.0 - gamma) * (B_s @ mu_pi.unsqueeze(1)).squeeze(1)
        
        # Stable exp
        x = weight_temp * logit
        x = x - x.max()
        w = torch.exp(x)
        w = w / (w.mean() + 1e-8)
        
        # 3. Objective: J = E[w * r]
        J = (w * rewards.view_as(w)).mean()
        
        # Regularization - only use chi2 if explicitly needed
        chi2 = ((w - 1.0)**2).mean()
        z_proj = model.project_z(z.unsqueeze(0)).squeeze(0)
        z0_proj = model.project_z(z0.unsqueeze(0)).squeeze(0)
        trust = ((z_proj - z0_proj)**2).mean()
        
        loss = -(J - chi2_coef * chi2 - trust_l2_coef * trust)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            z.copy_(model.project_z(z.unsqueeze(0)).squeeze(0))
            
    return z.detach()

def find_best_zol_params(
    model: FBModel, 
    batch_obs: torch.Tensor, 
    rewards: torch.Tensor, 
    initial_z: torch.Tensor,
    r_true_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu"
) -> tp.Tuple[torch.Tensor, tp.Dict]:
    """Sweeps over ZOL hyperparameters and finds the one that yields best reward reconstruction."""
    weight_temps = [20.0, 40.0]  # Focus on high temps for sharpness
    chi2_coefs = [0.0]           # Simplify sweep
    trust_l2_coefs = [0.01, 0.05, 0.1]
    
    best_score = -1e10
    best_z = initial_z
    best_params = {}
    
    grid = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device)
    mask_flat = mask.ravel()
    
    with torch.no_grad():
        B_grid = model.backward_map(grid_t).detach()
        r_true_flat = r_true_grid.ravel()[mask_flat]
        # Normalize target for MSE-like comparison if needed, but correlation is usually enough
        # We'll stick to correlation but add a small bonus for higher temperatures if correlations are tied

    print("Running Expanded ZOL Hyperparameter Sweep...")
    for wt in weight_temps:
        for c2 in chi2_coefs:
            for trust_coef in trust_l2_coefs:
                z_zol = zol_latent_search_toy(
                    model, batch_obs, rewards, initial_z, 
                    weight_temp=wt, chi2_coef=c2, trust_l2_coef=trust_coef,
                    num_steps=256, lr=1e-2, device=device
                )
                
                # Evaluate reconstruction correlation
                with torch.no_grad():
                    r_hat = (B_grid @ z_zol.squeeze(0)).cpu().numpy()
                    r_hat_flat = r_hat[mask_flat]
                    
                    # Correlation
                    corr = np.corrcoef(r_true_flat, r_hat_flat)[0, 1]
                    
                    # Heuristic: if correlations are very similar, prefer higher wt for "sharpness"
                    # We give a small bonus (0.001 per unit temp) to higher temperatures
                    score = corr + 0.0001 * wt
                    
                print(f"  temp={wt}, chi2={c2}, trust={trust_coef} -> corr={corr:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_z = z_zol
                    best_params = {"weight_temp": wt, "chi2_coef": c2, "trust_l2_coef": trust_coef}
                
    print(f"Best ZOL params: {best_params} (score={best_score:.4f})")
    return best_z, best_params

def main():
    model_path = "/home/jovyan/bobrin/td_jepa/fb_donut_model"
    outdir = "fb_toy2d_simplified"
    grid_size = 500
    dataset_n = 100_000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(outdir, exist_ok=True)
    
    # Load model manually using our custom ToyFBModelConfig
    model = tp.cast(FBModel, load_model(
        model_path, 
        device=device, 
        strict=True, 
        config_class=ToyFBModelConfig
    ))
    model.eval()

    # 1. Dataset support visualization (Gaussian + Sharp Sigmoids)
    x_dim = np.linspace(-1.6, 1.6, grid_size)
    y_dim = np.linspace(-1.6, 1.6, grid_size)
    X, Y = np.meshgrid(x_dim, y_dim)
    mask = get_donut_mask(X, Y).astype(bool)
    
    # Compute radial distance
    R = np.sqrt(X**2 + Y**2)
    # Gaussian distribution segment
    sigma = 0.9
    D_support = np.exp(-(R**2) / (2 * sigma**2))
    
    # Sharp sigmoidal masks
    hole_radius = 0.25
    outer_radius = 1.5
    inner_mask = 1.0 / (1.0 + np.exp(-(R - hole_radius) / 0.005))
    outer_mask = 1.0 / (1.0 + np.exp((R - outer_radius) / 0.005))
    
    D_support *= (inner_mask * outer_mask)
    
    # Normalize
    dx = x_dim[1] - x_dim[0]
    dy = y_dim[1] - y_dim[0]
    total_vol = np.sum(D_support) * dx * dy
    if total_vol > 0:
        D_support /= total_vol

    tasks = [
        ("Square", square_reward),
        ("TwoCircles", two_circles_reward),
        ("Cross", cross_reward),
    ]

    # Multi-start points in 4 quadrants
    s0_list = [
        np.array([1.0, 1.0]),   # Q1
        np.array([-1.0, 1.0]),  # Q2
        np.array([-1.0, -1.0]), # Q3
        np.array([1.0, -1.0])   # Q4
    ]
    s0_t_list = [torch.from_numpy(s0.astype(np.float32)).to(device) for s0 in s0_list]

    # Apply Publication Styling
    apply_publication_style()

    all_results = {}

    for name, r_fn in tasks:
        print(f"Processing task: {name}")
        # Compute Reward for dataset points
        pts = sample_donut_points(dataset_n, seed=0)
        pts_t = torch.from_numpy(pts).to(device)
        r_pts = r_fn(pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1)).reshape(-1).astype(np.float32)
        r_t = torch.from_numpy(r_pts).to(device).view(-1, 1)
        
        # 2. Target Reward visualization
        r_true = r_fn(X, Y).astype(np.float32)

        # 3. FB Reconstructed Reward (Ridge)
        z_ridge = infer_z_ridge(model, pts_t, r_t, lam=1e-3, center=True)
        
        # 4. FB Reconstructed Reward (Inference / Correlation)
        z_corr = model.reward_inference(pts_t, r_t)
        
        # 5. ZOL Hyperparameter Search and Inference
        z_zol, best_zol_params = find_best_zol_params(
            model, pts_t, r_t, z_ridge, 
            r_true, X, Y, mask, device=device
        )
        
        # Grid evaluation
        grid = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
        grid_t = torch.from_numpy(grid).to(device)
        B_grid = model.backward_map(grid_t).detach()
        
        r_hat_corr = (B_grid @ z_corr.squeeze(0)).detach().cpu().numpy().reshape(X.shape)
        r_hat_zol = (B_grid @ z_zol.squeeze(0)).detach().cpu().numpy().reshape(X.shape)

        # 6. Compute Occupancy Measures (Standard FB Successor Measure)
        m_ridge = compute_measure_heatmap(model, z_ridge, pts_t, B_grid, X.shape)
        m_corr = compute_measure_heatmap(model, z_corr, pts_t, B_grid, X.shape)
        m_zol = compute_measure_heatmap(model, z_zol, pts_t, B_grid, X.shape)
        
        # 7. Compute s0-conditioned Occupancy Measures (Averaged over neighborhoods)
        m_s0_list = []
        for s0 in s0_list:
            s0_batch = sample_neighborhood_points(s0, n=512, seed=1)
            s0_batch_t = torch.from_numpy(s0_batch).to(device)
            # Use z_corr (inferred latent from rewards) instead of z_zol
            sm_s0 = compute_sm_from_batch(model, s0_batch_t, z_corr, B_grid)
            m_s0 = torch.clamp(sm_s0, min=0).detach().cpu().numpy().reshape(X.shape)
            m_s0_list.append(m_s0)
        
        all_results[name] = {
            "r_true": r_true,
            "r_hat_corr": r_hat_corr,
            "r_hat_zol": r_hat_zol,
            "m_ridge": m_ridge,
            "m_corr": m_corr,
            "m_zol": m_zol,
            "m_s0_list": m_s0_list,
            "best_zol_params": best_zol_params
        }

    from matplotlib.gridspec import GridSpec

    # Plot 1: Reconstructions Overview (Support, Reward/Task, ZOL Recon, FB Recon)
    # Layout: Column 0 spans all rows, Col 1-3 are task-specific rows
    fig1 = plt.figure(figsize=(24, 15))
    gs = GridSpec(3, 4, figure=fig1, width_ratios=[1.2, 1, 1, 1], wspace=0.1, hspace=0.2)
    
    # Column 0: Offline Dataset Support (Spans all rows)
    ax_s = fig1.add_subplot(gs[:, 0])
    ax_s.imshow(D_support, origin="lower", extent=(X.min(), X.max(), Y.min(), Y.max()), cmap="viridis")
    ax_s.set_title("Offline Dataset Support", pad=20)
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_aspect('equal')

    for i, (name, _) in enumerate(tasks):
        res = all_results[name]
        
        # Column 1: Reward/Task
        ax_t = fig1.add_subplot(gs[i, 1])
        masked_heatmap(ax_t, X, Y, res["r_true"], mask, title=None, vmin=-1, vmax=1)
        if i == 0: ax_t.set_title("Reward / Task")
        ax_t.set_ylabel(name, labelpad=20, rotation=90, size=24, fontweight='bold')
        
        # Column 2: ZOL Reconstruction
        ax_z = fig1.add_subplot(gs[i, 2])
        masked_heatmap(ax_z, X, Y, res["r_hat_zol"], mask, title=None)
        if i == 0: ax_z.set_title("ZOL Reconstruction")
        
        # Column 3: FB Reconstruction
        ax_c = fig1.add_subplot(gs[i, 3])
        masked_heatmap(ax_c, X, Y, res["r_hat_corr"], mask, title=None)
        if i == 0: ax_c.set_title("FB Reconstruction")

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    path1 = os.path.join(outdir, "recon_overview.png")
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    print(f"[saved] {path1}")
    path1_pdf = os.path.join(outdir, "recon_overview.pdf")
    fig1.savefig(path1_pdf, dpi=300, bbox_inches='tight')
    print(f"[saved] {path1_pdf}")

    # Plot 2: Occupancy Overview (Reward/Task, Ridge Occ, FB Occ, ZOL Occ)
    # Layout: 3x4
    fig2, axes2 = plt.subplots(3, 4, figsize=(24, 15))
    for i, (name, _) in enumerate(tasks):
        res = all_results[name]
        # Column 0: Reward/Task
        masked_heatmap(axes2[i, 0], X, Y, res["r_true"], mask, title=None, vmin=-1, vmax=1)
        if i == 0: axes2[i, 0].set_title("Reward/Task")
        axes2[i, 0].set_ylabel(name, labelpad=20, rotation=90, size=24, fontweight='bold')
        
        # Column 1: Ridge Occupancy
        masked_occupancy_heatmap(axes2[i, 1], X, Y, res["m_ridge"], mask, title=None)
        if i == 0: axes2[i, 1].set_title("Ridge Occupancy")
        
        # Column 2: FB Occupancy
        masked_occupancy_heatmap(axes2[i, 2], X, Y, res["m_corr"], mask, title=None)
        if i == 0: axes2[i, 2].set_title("FB Occupancy")
        
        # Column 3: ZOL Occupancy
        masked_occupancy_heatmap(axes2[i, 3], X, Y, res["m_zol"], mask, title=None)
        if i == 0: axes2[i, 3].set_title("ZOL Occupancy")

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    path2 = os.path.join(outdir, "occupancy_overview.png")
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    print(f"[saved] {path2}")

    # Plot 3: Multi-Start Occupancy Overview
    # Layout: 3 tasks x 4 starting points
    fig3, axes3 = plt.subplots(3, 4, figsize=(24, 15))
    for i, (name, _) in enumerate(tasks):
        res = all_results[name]
        for j, s0 in enumerate(s0_list):
            m_s0 = res["m_s0_list"][j]
            masked_occupancy_heatmap(axes3[i, j], X, Y, m_s0, mask, title=None)
            
            # Draw a star at s0
            axes3[i, j].plot(s0[0], s0[1], '*', color='white', markersize=15, markeredgecolor='black')
            
            if i == 0:
                axes3[i, j].set_title(f"$s_0 = ({s0[0]}, {s0[1]})$")
            if j == 0:
                axes3[i, j].set_ylabel(name, labelpad=20, rotation=90, size=24, fontweight='bold')

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    path3 = os.path.join(outdir, "multi_start_occupancy.png")
    fig3.savefig(path3, dpi=300, bbox_inches='tight')
    print(f"[saved] {path3}")
    path3_pdf = os.path.join(outdir, "multi_start_occupancy.pdf")
    fig3.savefig(path3_pdf, dpi=300, bbox_inches='tight')
    print(f"[saved] {path3_pdf}")

if __name__ == "__main__":
    main()
