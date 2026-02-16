# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import numbers
import typing as tp
from typing import Optional, Any, Tuple
import copy
import abc

import gymnasium
import numpy as np
import torch
from torch import distributions as pyd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal

from .base import BaseConfig

##########################
# Initialization utils
##########################


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Creation utils
##########################


class ForwardArchiConfig(BaseConfig):
    name: tp.Literal["ForwardArchi"] = "ForwardArchi"
    hidden_dim: int = 1024
    model: tp.Literal["simple", "attention"] = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: tp.Literal["batch"] = "batch"

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None, discrete=False) -> torch.nn.Module:
        """Note: Forward model is also used for critics"""

        if self.ensemble_mode == "batch":
            return _build_batch_forward(self, obs_space, z_dim, action_dim, output_dim, discrete)
        else:
            raise ValueError(f"Unsupported ensemble_mode {self.ensemble_mode}")


def _build_batch_forward(cfg, obs_space, z_dim, action_dim, output_dim=None, discrete=False):
    if cfg.model == "simple":
        forward_cls = ForwardMap
    elif cfg.model == "attention":
        forward_cls = AttentionForwardRepresentation
    else:
        raise ValueError(f"Unsupported forward_map model {cfg.model}")
    return forward_cls(obs_space, z_dim, action_dim, cfg, output_dim=output_dim, discrete=discrete)


class AttentionForwardArchiConfig(BaseConfig):
    name: tp.Literal["AttentionForwardArchi"] = "AttentionForwardArchi"
    preprocessor_hidden_dim: int = 256
    preprocessor_output_dim: int = 256
    preprocessor_hidden_layers: int = 2
    forward_hidden_dim: int = 1024

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None, discrete=False) -> torch.nn.Module:
        return AttentionForwardRepresentation(obs_space, z_dim, action_dim, self, output_dim, discrete)


class ActorArchiConfig(BaseConfig):
    name: tp.Literal["actor"] = "actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim, action_dim):
        if self.model == "simple":
            return Actor(obs_space, z_dim, action_dim, self)
        else:
            raise ValueError(f"Unsupported actor model {self.model}. Define 'model' or use other configs explicitely")


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardArchiConfig(BaseConfig):
    name: tp.Literal["BackwardArchi"] = "BackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True

    def build(self, obs_space, z_dim: int):
        return BackwardMap(obs_space, z_dim, self)


class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg

        self.output_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(z_dim,), dtype=np.float32)

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        seq = [nn.Linear(obs_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]
        if cfg.hidden_layers == 0:
            seq = [nn.Linear(obs_space.shape[0], z_dim)]
        if cfg.norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1, discrete=False):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    if not discrete:
        seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    else:
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        
    return nn.Sequential(*seq)


def simple_embedding_custom_out(input_dim, hidden_dim, output_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, output_dim, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)

class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: ForwardArchiConfig,
        output_dim=None,
        discrete=False,
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg = cfg
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim
        self.discrete = discrete
        self.action_dim = action_dim
        # s + z
        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel, discrete=discrete)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel, discrete=discrete)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        if not discrete:
            seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        else:
            seq += [linear(cfg.hidden_dim, z_dim * action_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor=None):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            if action is not None:
                action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        if not self.discrete:
            sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
            return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))
        else:
            Fs = self.Fs(z_embedding)
            return Fs.reshape(self.num_parallel, -1, self.z_dim, self.action_dim)


class SimpleActorArchiConfig(ActorArchiConfig):
    name: tp.Literal["simple"] = "simple"
    model: tp.Literal["simple"] = "simple"

    def build(self, obs_space, z_dim: int, action_dim: int) -> "Actor":
        return Actor(obs_space, z_dim, action_dim, self)


class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: SimpleActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: SimpleActorArchiConfig = cfg

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist

## IDQL Diffusion Actor

class IDQLDiffusionActorArchiConfig(BaseConfig):
    name: tp.Literal["idql_diffusion"] = "idql_diffusion"
    model: tp.Literal["idql_diffusion"] = "idql_diffusion"
    time_embeding: tp.Literal["fixed", "learned"] = "fixed"
    time_dim: int = 64
    num_blocks: int = 3
    hidden_dim: int = 256
    ac_fn: tp.Literal["relu", "mish", "gelu"] = "mish"
    schedule: tp.Literal["linear", "cosine", "sigmoid", "vp"] = "cosine"
    num_timesteps: int = 5
    temperature: float = 1.0
    
    def build(self, obs_space, z_dim: int, action_dim: int) -> "Actor":
        return DiffusionAgent(
            policy=IDQLDiffusionActor(obs_space, z_dim, action_dim, self),
            schedule=self.schedule,
            num_timesteps=self.num_timesteps,
            temperature=self.temperature
        )

class BaseAgent(nn.Module):
    def __init__(
        self, 
        policy: torch.nn.Module,
        v_model: torch.nn.Module, 
        gamma: float = 0.99,
        utd: int = 2,
        start_steps: int = int(25e3),
        ema: float = 1e-3,
    ):       
        super().__init__()

        self.policy = policy
        self.v_model = v_model
        self.policy_target = copy.deepcopy(policy)
        self.v_model_target = copy.deepcopy(v_model)
        
        self.start_steps = start_steps
        self.utd = utd
        self.gamma = gamma
        self.ema = ema

    @property
    def device(self):
        return self.policy.device
            
    def ema_update_policy(self):
        for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
            if param.data is not None and target_param.data is not None:
                target_param.data.copy_(
                    self.ema * param.data + (1 - self.ema) * target_param.data
                )
                
    def next_state(self, state):
        """
        get the action during evaluation
        """
        pass
    
    def load(self, ckpt_path):
        pass
    
    def policy_loss(self):
        pass
    
    def v_loss(self):
        pass

def extract(a, x_shape):
    '''
    align the dimention of alphas_cumprod_t to x_shape
    
    a: alphas_cumprod_t, B
    x_shape: B x F x F x F
    output: alphas_cumprod_t B x 1 x 1 x 1]
    '''
    b, *_ = a.shape
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
    """Discret VP noise schedule
    """
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1

    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas       


SCHEDULE = {
    'linear': linear_beta_schedule,
    'cosine': cosine_beta_schedule,
    'sigmoid': sigmoid_beta_schedule,
    'vp': vp_beta_schedule
}

class DiffusionAgent(BaseAgent):
    def __init__(
        self, 
        policy: torch.nn.Module, 
        schedule: str = 'cosine',
        num_timesteps: int = 5,
        ema: float = 1e-3,
        device: str = 'cpu',
        temperature: float = 1.0,
    ):
        super().__init__(policy, None, None, ema=ema)
        self.temperature = temperature
        
        if schedule not in SCHEDULE.keys():
            raise ValueError(
                f"Invalid schedule '{schedule}'. Expected one of: {list(SCHEDULE.keys())}"
            )
        self.schedule = SCHEDULE[schedule]
        
        self.num_timesteps = num_timesteps
        self.register_buffer('betas', self.schedule(self.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def forward(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        cond: Optional[torch.Tensor] = None, 
        z: Optional[torch.Tensor] = None, 
        from_target: bool = False
    ) -> torch.Tensor:
        """
        predict the noise
        """
        if from_target:
            return self.policy_target(xt, t, cond, z)
        return self.policy(xt, t, cond, z)
    
    def predict_noise(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        predict the noise
        """
        noise_pred = self.policy(xt, t, cond, z)
        return noise_pred
    
    def policy_loss(
        self, 
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        calculate ddpm loss
        '''
        batch_size = x0.shape[0]
        
        noise = torch.randn_like(x0, device=self.device)
        t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
        
        xt = self.q_sample(x0, t, noise)
        
        noise_pred = self.predict_noise(xt, t, cond)
        loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
        
        return loss
    
    def policy_loss_with_weight(
        self, 
        weight: torch.Tensor, 
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        calculate ddpm loss
        '''
        batch_size = x0.shape[0]
        
        noise = torch.randn_like(x0, device=self.device)
        t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
        
        xt = self.q_sample(x0, t, noise)
        
        noise_pred = self.predict_noise(xt, t, cond, z)
        
        return (((noise_pred - noise) ** 2).sum(axis = -1) * weight).mean()
            
    def q_sample(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        sample noisy xt from x0, q(xt|x0), forward process
        """
        alphas_cumprod_t = self.alphas_cumprod[t]
        xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) \
            + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
        return xt
    
    def p_sample(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        clip_sample: bool = False,
        ddpm_temperature: float = 1., 
        from_target: bool = False
    ) -> torch.Tensor:
        """
        sample xt-1 from xt, p(xt-1|xt)
        """
        noise_pred = self.forward(xt, t, cond, z, from_target=from_target)
        
        alpha1 = 1 / torch.sqrt(self.alphas[t])
        alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
        
        xtm1 = alpha1 * (xt - alpha2 * noise_pred)
        
        noise = torch.randn_like(xtm1, device=self.device) * ddpm_temperature
        xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
        
        if clip_sample:
            xtm1 = torch.clip(xtm1, -1., 1.)
        return xtm1
    
    def ddpm_sampler(
        self, 
        shape: Tuple, 
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        from_target: bool = False,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        sample x0 from xT, reverse process
        """
        x = torch.randn(shape, device=self.device)
        temp = temperature if temperature is not None else self.temperature

        if len(shape) == 3:
            cond = cond.unsqueeze(1).repeat_interleave(shape[1], dim=1)
            z = z.unsqueeze(1).repeat_interleave(shape[1], dim=1)
            
            for t in reversed(range(self.num_timesteps)):
                x = self.p_sample(
                    xt=x, 
                    t=torch.full((shape[0], shape[1], 1), t, device=self.device), 
                    cond=cond, 
                    z=z, 
                    from_target=from_target,
                    ddpm_temperature=temp
                )
        else:
            cond = cond.repeat(x.shape[0], 1)
            z = z.repeat(x.shape[0], 1)

            for t in reversed(range(self.num_timesteps)):
                x = self.p_sample(
                    xt=x, 
                    t=torch.full((shape[0], 1), t, device=self.device), 
                    cond=cond, 
                    z=z, 
                    from_target=from_target,
                    ddpm_temperature=temp
                )
        return x

    def get_action(
        self, 
        state: torch.Tensor, 
        z: torch.Tensor,
        num: int = 1, 
        batch_input: bool = False, 
        from_target: bool = True,
        temperature: float | None = None,
    ) -> torch.Tensor:
        temp = temperature if temperature is not None else self.temperature
        if batch_input:
            return self.ddpm_sampler(
                (state.shape[0], num, self.policy.output_dim), 
                cond=state, 
                z=z, 
                from_target=from_target,
                temperature=temp
            )
        
        return self.ddpm_sampler(
            (num, self.policy.output_dim), 
            cond=state, 
            z=z, 
            from_target=from_target,
            temperature=temp
        )

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
            
    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.output_size // 2
        f = math.log(10000) / (half_dim - 1)
        f = torch.exp(torch.arange(half_dim, device=device) * -f)
        f = x * f[None, :]
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f

# learned positional embeds
class LearnedPosEmb(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)
            
    def forward(self, x: torch.Tensor):
        f = 2 * torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f       

AC_FN ={'relu': F.relu, 'mish': F.mish, 'gelu': F.gelu}

class MLP(nn.Module):
    def __init__(
        self, 
        input_size : int, 
        hidden_sizes : list, 
        output_size : int, 
        ac_fn: str = 'relu', 
        use_layernorm: bool = False, 
        dropout_rate: float = 0.
    ):
        super().__init__()             
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout_rate
        
        # initialize layers
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList() if use_layernorm else None
        self.ac_fn = AC_FN[ac_fn]
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                
        if self.use_layernorm:
            self.layernorms.append(nn.LayerNorm(input_size))
                
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

            
    def forward(self, x: torch.Tensor):
        if self.use_layernorm:
            x = self.layernorms[-1](x)
        
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x = self.ac_fn(x)

        x = self.layers[-1](x)
        return x
    

class MLPResNetBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim : int, 
        ac_fn: str ='relu', 
        use_layernorm: bool = False, 
        dropout_rate: int = 0.1, 
        condition_dim: Optional[torch.Tensor] = None,
    ):

        super(MLPResNetBlock, self).__init__()

        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ac_fn = AC_FN[ac_fn]
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        self.condition_dim = condition_dim
        if condition_dim is not None:
            self.film_gamma = nn.Linear(condition_dim, hidden_dim * 4)
            self.film_beta = nn.Linear(condition_dim, hidden_dim * 4)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        identity = x
        
        out = self.dropout(x)
        if self.use_layernorm:
            out = self.norm1(out)
        out = self.dense1(out)
        out = self.ac_fn(out)
        
        if self.condition_dim is not None:
            assert condition is not None, "give condition"
            gamma = self.film_gamma(condition)
            beta = self.film_beta(condition)
            out = gamma * out + beta
            
        out = self.dense2(out)
        return identity + out
    
class MLPResNet(nn.Module):
    def __init__(
            self, 
            num_blocks : int, 
            input_dim : int, 
            hidden_dim : int, 
            output_size : int, 
            ac_fn: str = 'relu', 
            use_layernorm: bool = True, 
            dropout_rate: float = 0.1, 
            condition_dim: Optional[torch.Tensor] = None,
        ):

        super(MLPResNet, self).__init__()
        
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.ac_fn = AC_FN[ac_fn]
        self.dense2 = nn.Linear(hidden_dim, output_size)
        self.mlp_res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_res_blocks.append(
                MLPResNetBlock(hidden_dim, ac_fn, use_layernorm, dropout_rate, condition_dim)
            )
            
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        out = self.dense1(x)
        for mlp_res_block in self.mlp_res_blocks:
            out = mlp_res_block(out, condition=condition)
        out = self.ac_fn(out)
        return self.dense2(out)

TIMEEMBED = {"fixed": SinusoidalPosEmb, "learned": LearnedPosEmb}

class IDQLDiffusionActor(nn.Module):
    """
    Diffusion model implementation for IDQL (Implicit Diffusion Q-Learning).
    
    Reference: 
        IDQL: Implicit Diffusion Q-Learning - arXiv:2304.10573
    """
    def __init__(
        self, 
        obs_dim,
        z_dim,
        action_dim,
        cfg: IDQLDiffusionActorArchiConfig
    ):
        super().__init__()
        self.cfg: IDQLDiffusionActorArchiConfig = cfg
        
        # time embedding
        if cfg.time_embeding not in TIMEEMBED.keys():
            raise ValueError(
                f"Invalid time_embedding '{cfg.time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}"
            )
        
        self.time_process = TIMEEMBED[cfg.time_embeding](1, cfg.time_dim)
        self.time_encoder = MLP(cfg.time_dim, [128], 128, ac_fn=cfg.ac_fn)
        
        # decoder
        if hasattr(obs_dim, 'shape'):
            obs_dim = obs_dim.shape[0]
        self.obs_dim = obs_dim
        self.output_dim = action_dim
        
        decoder_input_dim = obs_dim + 128 + action_dim
        self.decoder = MLPResNet(
            num_blocks=cfg.num_blocks, 
            input_dim=decoder_input_dim, 
            hidden_dim=cfg.hidden_dim, 
            output_size=action_dim, 
            ac_fn=cfg.ac_fn, 
            use_layernorm=True, 
            dropout_rate=0.1, 
            condition_dim=z_dim
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        """
        time = time.to(self.device)
        if condition is not None:
            condition = condition.to(self.device)
        if z is not None:
            z = z.to(self.device)
        # Process time embedding
        if x_t.dim() == 3:
            time_embedding = self.time_process(time)
        else:
            time_embedding = self.time_process(time.view(-1, 1))
            
        time_embedding = self.time_encoder(time_embedding)
        
        # Concatenate conditioning if provided
        if condition is not None:
            x_t = torch.cat([x_t, condition], dim=-1)
        
        # Prepare input for decoder
        decoder_input = torch.cat([time_embedding, x_t], dim=-1)

        noise_pred = self.decoder(decoder_input, z)
        return noise_pred      

#############################
## Backward with Attention ##
#############################

class AttentionBackwardArchiConfig(BaseConfig):
    name: tp.Literal["AttentionBackwardArchi"] = "AttentionBackwardArchi"
    observation_length: int
    z_dimension: int
    backward_hidden_dimension: int
    backward_hidden_layers: int
    device: torch.device

    def build(self, obs_space, z_dim: int) -> torch.nn.Module:
        return AttentionBackwardRepresentation(obs_space, z_dim, self)

class TransformerFull(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, src, tgt, tgt_mask=None) -> torch.Tensor:
        return self.transformer(src, tgt, tgt_mask=tgt_mask)

class AbstractFullTransformer(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for full transformer networks."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        device: torch.device,
        dropout: float = 0.1,
        preprocessor: bool = False,
    ):
        super().__init__()
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._d_model = d_model
        self._preprocessor = preprocessor

        # Input projection
        self.input_proj = torch.nn.Linear(input_dimension, d_model)
        
        # Positional encodings for encoder and decoder
        self.pos_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.GELU()
        )
        self.pos_decoder = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.GELU()
        )
        
        # Full Transformer
        self.transformer = TransformerFull(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = torch.nn.Linear(d_model, output_dimension)
        self.to(device)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

class BackwardTransformer(AbstractFullTransformer):
    """Backwards model implemented with full Transformer architecture."""
    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        d_model: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=z_dimension,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=hidden_layers,
            num_decoder_layers=hidden_layers,
            dim_feedforward=hidden_dimension,
            device=device,
            dropout=dropout,
        )
        self._z_dimension = z_dimension
        self.dropout = torch.nn.Dropout(dropout)
        
        # Learnable query for the decoder
        self.query_embed = torch.nn.Parameter(torch.randn(1, d_model))
        self.to(device)

    def forward(self, observation: torch.Tensor, position_encoding: bool = False) -> torch.Tensor:
        """
        Takes observation and processes it through full transformer architecture.
        Args:
            observation: state tensor of shape [batch_dim, observation_length]
        Returns:
            z: embedded tensor of shape [batch_dim, z_dimension]
        """
        batch_size = observation.shape[0]
        
        if position_encoding:
            # Create position encodings for encoder
            src_positions = torch.arange(observation.shape[1], dtype=torch.float32)\
                .expand(batch_size, -1).unsqueeze(-1).to(self.device)
            src_pos_encoding = self.pos_encoder(src_positions)

            # Project and add positional encoding for encoder input
            memory = self.input_proj(observation)
            memory = memory.unsqueeze(1)
            memory = memory + src_pos_encoding
        else:
            x = observation.unsqueeze(1).to(self.device)
            memory = self.input_proj(x)
        
        # Create decoder query
        query = self.query_embed.expand(batch_size, -1, -1)
        
        # Create position encoding for decoder
        tgt_positions = torch.zeros(batch_size, 1, 1, dtype=torch.float32).to(self.device)
        tgt_pos_encoding = self.pos_decoder(tgt_positions)
        query = query + tgt_pos_encoding

        # Generate target mask for decoder
        tgt_mask = torch.zeros((1, 1), dtype=torch.float32).to(self.device)
        
        # Pass through transformer
        output = self.transformer.transformer(
            src=memory,
            tgt=query,
            tgt_mask=tgt_mask
        )
        
        # Project to output dimension
        z = self.output_proj(output.squeeze(1))

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.float32, device=self.device)
        ) * torch.nn.functional.normalize(z, dim=1)

        return z

class AttentionBackwardRepresentation(torch.nn.Module):
    """Backward representation network."""

    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        device: torch.device,
    ):
        super().__init__()

        self.B = BackwardTransformer(
            observation_length=observation_length,
            z_dimension=z_dimension,
            hidden_dimension=backward_hidden_dimension,
            hidden_layers=backward_hidden_layers,
            device=device,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Estimates routes to observation via backwards model."""

        return self.B(observation)

############################
## Forward with Attention ##
############################

class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim = -1) * self.gamma * self.scale
    

class SelfAttention(nn.Module):
    def __init__(self, z_dim: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(z_dim, z_dim)
        self.key = nn.Linear(z_dim, z_dim)
        self.value = nn.Linear(z_dim, z_dim)
        self.z_dim = z_dim
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)   
        V = self.value(x) 

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.bmm(attention_weights, V)
        return output
    
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):

        super().__init__()

        inner_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = RMSNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))

class AttentionForwardRepresentation(nn.Module):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        device: torch.device,
    ):
        super().__init__()
        
        self.z_dimension = z_dimension
        self.device = device
        
        # Initialize preprocessors
        self.obs_action_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=action_length,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            activation=preprocessor_activation,
            device=device,
        )

        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            activation=preprocessor_activation,
            device=device,
        )

        self.self_attention_1 = SelfAttention(preprocessor_feature_space_dimension)
        self.feedforward_1 = FeedForward(preprocessor_feature_space_dimension)
        self.norm_1 = nn.LayerNorm(preprocessor_feature_space_dimension)
        self.linear_11 = nn.Linear(
            preprocessor_feature_space_dimension * 2, 
            forward_hidden_dimension
        )
        self.linear_12 = nn.Linear(forward_hidden_dimension, z_dimension)
        
        self.self_attention_2 = SelfAttention(preprocessor_feature_space_dimension)
        self.feedforward_2 = FeedForward(preprocessor_feature_space_dimension)
        self.norm_2 = nn.LayerNorm(preprocessor_feature_space_dimension)
        self.linear_21 = nn.Linear(
            preprocessor_feature_space_dimension * 2,
            forward_hidden_dimension
        )
        self.linear_22 = nn.Linear(forward_hidden_dimension, z_dimension)
        
        # Regularization components
        self.dropout = nn.Dropout(p=0.1)
        self.final_norm_1 = nn.LayerNorm(preprocessor_feature_space_dimension)
        self.final_norm_2 = nn.LayerNorm(preprocessor_feature_space_dimension)

        self.to(device)

    def forward(
        self, 
        observation: torch.Tensor, 
        action: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            observation: Input observation tensor
            action: Optional action tensor
            z: Optional latent z tensor  
            
        Returns:
            Tuple of two output tensors from the dual processing path
        """

        # Process observation-action pairs
        obs_action_input = torch.cat([observation, action], dim=-1)
        obs_action_embedding = self.obs_action_preprocessor(
            obs_action_input
        ).unsqueeze(1)


        # Process observation-z pairs
        obs_z_input = torch.cat([observation, z], dim=-1)
        obs_z_embedding = self.obs_z_preprocessor(obs_z_input).unsqueeze(1)

        # Combine embeddings for processing
        combined_embeddings = torch.cat(
            [obs_z_embedding, obs_action_embedding], 
            dim=1
        )

        # First processing block
        attended_1 = self.self_attention_1(combined_embeddings)
        residual_1 = attended_1 + self.feedforward_1(self.norm_1(attended_1))
        normalized_1 = self.final_norm_1(self.dropout(residual_1))
        
        # Second processing block  
        attended_2 = self.self_attention_2(normalized_1)
        residual_2 = attended_2 + self.feedforward_2(self.norm_2(attended_2))
        normalized_2 = self.final_norm_2(self.dropout(residual_2))
        
        # Flatten and project to output space
        flattened_features = normalized_2.flatten(start_dim=1)
        
        # Dual output pathways
        F1 = self.linear_12(self.linear_11(flattened_features))
        F2 = self.linear_22(self.linear_21(flattened_features))
        
        return F1, F2

class VForwardArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2

    def build(self, obs_space, z_dim: int, output_dim=None) -> torch.nn.Module:
        return VForwardMap(obs_space, z_dim, output_dim, self)

class VForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        output_dim=None,
        cfg: VForwardArchiConfig = VForwardArchiConfig(),
    ) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([s_embedding, z_embedding], dim=-1))


##########################
# Visual modules
##########################


class DrQEncoderArchiConfig(BaseConfig):
    name: tp.Literal["drq"] = "drq"
    feature_dim: int | None = None  # if not None, linearly project the output to feature_dim

    def build(self, obs_space):
        return DrQEncoder(obs_space, self)


class DrQEncoder(nn.Module):
    """RGB encoder from the DrQ-v2 paper"""

    def __init__(self, obs_space, cfg: DrQEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

        # courtesy of https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *obs_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.LayerNorm(self.cfg.feature_dim), nn.Tanh())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()
            print(
                "WARNING: using a DrQ encoder with feature_dim=None. This yields very large feature vectors that are fed as input to other networks"
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.trunk(obs))

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class AugmentatorArchiConfig(BaseConfig):
    name: tp.Literal["random_shifts"] = "random_shifts"
    pad: int = 4

    def build(self, obs_space):
        return Augmentator(obs_space, self)


class Augmentator(nn.Module):
    """Image augmentations from DrQ-v2"""

    def __init__(self, obs_space, cfg: AugmentatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert len(obs_space.shape) == 3, "obs_space must have a 3D shape (image)"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n, _, h, w = obs.size()
        assert h == w, "Augmentator only supports square images"
        padding = tuple([self.cfg.pad] * 4)
        obs = F.pad(obs, padding, "replicate")
        eps = 1.0 / (h + 2 * self.cfg.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.cfg.pad, device=obs.device, dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.cfg.pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype)
        shift *= 2.0 / (h + 2 * self.cfg.pad)
        grid = base_grid + shift
        return F.grid_sample(obs, grid, padding_mode="zeros", align_corners=False)


##########################
# FlowQ modules
##########################


class NoiseConditionedActorArchiConfig(BaseConfig):
    name: tp.Literal["noise_conditioned_actor"] = "noise_conditioned_actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim: int, action_dim: int) -> "NoiseConditionedActor":
        return NoiseConditionedActor(obs_space, z_dim, action_dim, self)


class NoiseConditionedActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: NoiseConditionedActorArchiConfig = cfg
        self.embed_z = simple_embedding(obs_dim + z_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z_embedding = self.embed_z(torch.cat([obs, z, noise], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(torch.cat([obs, noise], dim=-1))  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        actions = torch.tanh(self.policy(embedding))
        return actions


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class IdentityNNConfig(BaseConfig):
    name: tp.Literal["Identity"] = "Identity"

    def build(self, obs_space, *args) -> nn.Module:
        return IdentityNN(obs_space)


class IdentityNN(nn.Identity):
    def __init__(self, obs_space):
        super().__init__()
        self.obs_space = obs_space

    @property
    def output_space(self):
        return self.obs_space


##########################
# BREEZE Transformer & Attention models
##########################

class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class SelfAttention(nn.Module):
    def __init__(self, z_dim: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(z_dim, z_dim)
        self.key = nn.Linear(z_dim, z_dim)
        self.value = nn.Linear(z_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.bmm(attention_weights, V)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))




class AttentionBackwardArchiConfig(BaseConfig):
    name: tp.Literal["AttentionBackwardArchi"] = "AttentionBackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    d_model: int = 256
    nhead: int = 8
    dropout: float = 0.1

    def build(self, obs_space, z_dim: int):
        return AttentionBackwardMap(obs_space, z_dim, self)


class AttentionBackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: AttentionBackwardArchiConfig):
        super().__init__()
        self.cfg = cfg
        self._z_dimension = z_dim
        input_dim = obs_space.shape[0]

        self.input_proj = nn.Linear(input_dim, cfg.d_model)

        self.pos_encoder = nn.Sequential(nn.Linear(1, cfg.d_model), nn.GELU())
        self.pos_decoder = nn.Sequential(nn.Linear(1, cfg.d_model), nn.GELU())

        self.transformer = TransformerFull(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.hidden_layers,
            num_decoder_layers=cfg.hidden_layers,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        self.output_proj = nn.Linear(cfg.d_model, z_dim)
        self.query_embed = nn.Parameter(torch.randn(1, cfg.d_model))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, observation: torch.Tensor, position_encoding: bool = False) -> torch.Tensor:
        batch_size = observation.shape[0]
        device = observation.device

        if position_encoding:
            # Note: This assumes observation has a sequence dimension if position_encoding is True
            # In standard BREEZE it seems to be 1D, but let's keep it flexible if needed.
            # However, standard BREEZE uses 1D obs.
            src_positions = torch.arange(observation.shape[1], dtype=torch.float32).expand(batch_size, -1).unsqueeze(-1).to(device)
            src_pos_encoding = self.pos_encoder(src_positions)
            memory = self.input_proj(observation)
            memory = memory.unsqueeze(1) if memory.dim() == 2 else memory
            memory = memory + src_pos_encoding
        else:
            x = observation.unsqueeze(1) if observation.dim() == 2 else observation
            memory = self.input_proj(x)

        query = self.query_embed.expand(batch_size, -1, -1)
        tgt_positions = torch.zeros(batch_size, 1, 1, dtype=torch.float32).to(device)
        tgt_pos_encoding = self.pos_decoder(tgt_positions)
        query = query + tgt_pos_encoding

        tgt_mask = torch.zeros((1, 1), dtype=torch.float32, device=device)

        output = self.transformer(memory, query, tgt_mask=tgt_mask)
        z = self.output_proj(output.squeeze(1))

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = math.sqrt(self._z_dimension) * F.normalize(z, dim=1)
        return z


class AttentionForwardRepresentation(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: tp.Union["ForwardArchiConfig", "AttentionForwardArchiConfig"],
        output_dim=None,
        discrete=False,
    ):
        super().__init__()
        assert not discrete, "AttentionForwardRepresentation does not support discrete actions yet"
        obs_dim = obs_space.shape[0]
        self.z_dimension = z_dim

        if isinstance(cfg, AttentionForwardArchiConfig):
            pre_h = cfg.preprocessor_hidden_dim
            pre_o = cfg.preprocessor_output_dim
            pre_l = cfg.preprocessor_hidden_layers
            f_h = cfg.forward_hidden_dim
        else:
            pre_h = cfg.hidden_dim
            pre_o = cfg.hidden_dim // 2
            pre_l = cfg.embedding_layers
            f_h = cfg.hidden_dim

        # Reuse simple_embedding logic via simple_embedding_custom_out if possible
        self.obs_action_preprocessor = simple_embedding_custom_out(obs_dim + action_dim, pre_h, pre_o, pre_l)
        self.obs_z_preprocessor = simple_embedding_custom_out(obs_dim + z_dim, pre_h, pre_o, pre_l)

        preprocessor_feature_dim = pre_o

        self.self_attention_1 = SelfAttention(preprocessor_feature_dim)
        self.feedforward_1 = FeedForward(preprocessor_feature_dim)
        self.norm_1 = nn.LayerNorm(preprocessor_feature_dim)
        self.linear_11 = nn.Linear(preprocessor_feature_dim * 2, f_h)
        self.linear_12 = nn.Linear(f_h, output_dim if output_dim else z_dim)

        self.self_attention_2 = SelfAttention(preprocessor_feature_dim)
        self.feedforward_2 = FeedForward(preprocessor_feature_dim)
        self.norm_2 = nn.LayerNorm(preprocessor_feature_dim)
        self.linear_21 = nn.Linear(preprocessor_feature_dim * 2, f_h)
        self.linear_22 = nn.Linear(f_h, output_dim if output_dim else z_dim)

        self.dropout = nn.Dropout(p=0.1)
        self.final_norm_1 = nn.LayerNorm(preprocessor_feature_dim)
        self.final_norm_2 = nn.LayerNorm(preprocessor_feature_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process observation-action pairs
        obs_action_input = torch.cat([observation, action], dim=-1)
        obs_action_embedding = self.obs_action_preprocessor(obs_action_input).unsqueeze(1)

        # Process observation-z pairs
        obs_z_input = torch.cat([observation, z], dim=-1)
        obs_z_embedding = self.obs_z_preprocessor(obs_z_input).unsqueeze(1)

        # Combine embeddings for processing
        combined_embeddings = torch.cat([obs_z_embedding, obs_action_embedding], dim=1)

        # First processing block
        attended_1 = self.self_attention_1(combined_embeddings)
        residual_1 = attended_1 + self.feedforward_1(self.norm_1(attended_1))
        normalized_1 = self.final_norm_1(self.dropout(residual_1))

        # Second processing block
        attended_2 = self.self_attention_2(normalized_1)
        residual_2 = attended_2 + self.feedforward_2(self.norm_2(attended_2))
        normalized_2 = self.final_norm_2(self.dropout(residual_2))

        # Flatten and project to output space
        flattened_features = normalized_2.flatten(start_dim=1)

        # Dual output pathways
        f1 = self.linear_12(self.linear_11(flattened_features))
        f2 = self.linear_22(self.linear_21(flattened_features))

        return f1, f2


class VForwardMapArchiConfig(BaseConfig):
    name: tp.Literal["VForwardMapArchi"] = "VForwardMapArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: tp.Literal["relu", "gelu", "tanh"] = "relu"

    def build(self, obs_space, z_dim: int) -> nn.Module:
        return VForwardMap(obs_space, z_dim, self)


class VForwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: VForwardMapArchiConfig):
        super().__init__()
        input_dim = obs_space.shape[0] + z_dim
        
        # Use simple_embedding style or direct MLP?
        # Breeze original V_net is a simple MLP.
        # Let's use simple_embedding with hidden_layers+2 to match its internal structure if needed, 
        # or just implement it directly as it's very simple.
        # But wait, user said "AbstractPreprocessor should not be implemented, since it is the same as simple_embedding".
        # V_net in base.py is just an AbstractMLP.
        
        seq = [nn.Linear(input_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim)]
        if cfg.activation == "relu":
            seq.append(nn.ReLU())
        elif cfg.activation == "gelu":
            seq.append(nn.GELU())
        elif cfg.activation == "tanh":
            seq.append(nn.Tanh())
        
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU() if cfg.activation == "relu" else nn.GELU()]
        
        seq.append(nn.Linear(cfg.hidden_dim, 1))
        self.trunk = nn.Sequential(*seq)

    def forward(self, observation: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([observation, z], dim=-1))


def expectile_regression_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff**2)).mean()
