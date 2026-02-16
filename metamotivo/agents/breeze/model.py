import copy
import math
import typing as tp

import gymnasium
import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast

from metamotivo.base_model import BaseModel
from metamotivo.agents.fb.model import FBModelArchiConfig, FBModelConfig
from metamotivo.nn_models import IDQLDiffusionActorArchiConfig, VForwardMapArchiConfig, AttentionBackwardArchiConfig, AttentionForwardArchiConfig, eval_mode

class BreezeModelArchiConfig(FBModelArchiConfig):
    # noise conditioned actor
    actor: IDQLDiffusionActorArchiConfig = pydantic.Field(IDQLDiffusionActorArchiConfig(), discriminator="name")
    vfunc: VForwardMapArchiConfig = pydantic.Field(VForwardMapArchiConfig(), discriminator="name")
    b: AttentionBackwardArchiConfig = pydantic.Field(AttentionBackwardArchiConfig(), discriminator="name")
    f: AttentionForwardArchiConfig = pydantic.Field(AttentionForwardArchiConfig(), discriminator="name")
    
class BreezeModelConfig(FBModelConfig):
    name: tp.Literal["BreezeModel"] = "BreezeModel"
    archi: BreezeModelArchiConfig = BreezeModelArchiConfig()

    def build(self, obs_space, action_dim, discrete=False):
        return self.object_class(obs_space, action_dim, self, discrete)

    @property
    def object_class(self):
        return BreezeModel

class BreezeModel(BaseModel):
    def __init__(self, obs_space, action_dim, cfg: BreezeModelConfig, discrete: bool = False):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: BreezeModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16
        self.discrete = discrete
        
        # create networks
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)
        self._bw_encoder = arch.rgb_encoder.build(obs_space)
        self._augmentator = arch.augmentator.build(obs_space)
        self._fw_encoder = arch.rgb_encoder.build(obs_space)
        self._left_encoder = arch.left_encoder.build(self._fw_encoder.output_space, arch.L_dim)

        self._backward_map = arch.b.build(self._bw_encoder.output_space, arch.z_dim)
        self._forward_map = arch.f.build(self._left_encoder.output_space, arch.z_dim, action_dim, discrete=discrete)
        
        actor_obs_space = self._left_encoder.output_space if self.cfg.actor_encode_obs else self._fw_encoder.output_space
        self._actor = arch.actor.build(actor_obs_space, arch.z_dim, action_dim)
        self._vfunc = arch.vfunc.build(actor_obs_space, arch.z_dim)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_left_encoder = copy.deepcopy(self._left_encoder)

    def _normalize(self, obs: torch.Tensor):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._backward_map(self._bw_encoder(self._normalize(obs)))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._forward_map(self._left_encoder(self._fw_encoder(self._normalize(obs))), z, action)

    @torch.no_grad()
    def vfunc(self, obs: torch.Tensor, z: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
            return self._vfunc(obs, z)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
            obs = self._fw_encoder(self._normalize(obs))
            obs = self._left_encoder(obs) if self.cfg.actor_encode_obs else obs
            # Sample one action from the actor
            return self._actor.get_action(obs, z, num=1, batch_input=True, from_target=True)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def predict_q(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        # Double Q-learning style
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        F1, F2 = self.forward_map(obs, z, action)
        Q1 = (F1 * z).sum(-1)
        Q2 = (F2 * z).sum(-1)
        return torch.min(Q1, Q2)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True, keval: int = 1) -> torch.Tensor:
        # Move z to device upfront
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        
        # Preprocess observation for the actor
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            # _normalize already ensures obs is on device
            obs_encoded = self._fw_encoder(self._normalize(obs))
            if self.cfg.actor_encode_obs:
                obs_encoded = self._left_encoder(obs_encoded)
        
        # Sample actions from the diffusion actor using PREPROCESSED obs
        if keval > 1:
            # Sample multiple actions
            actions = self._actor.get_action(obs_encoded, z, num=keval, batch_input=False, from_target=True) # [keval, action_dim]
            
            # Predict-Q handles raw inputs (normalizes internally), but we can also pass processed obs if we want efficiency.
            # However, predict_q calls forward_map which calls _normalize again. 
            # To be safe and reuse existing logic, we pass the original obs (which will be re-normalized on device).
            
            # Repat raw obs and z for predict_q
            obs_expanded = obs.repeat(keval, 1) if obs.dim() == 2 else obs.expand(keval, -1)
            z_expanded = z.repeat(keval, 1) if z.dim() == 2 else z.expand(keval, -1)
            Qs = self.predict_q(obs_expanded, z_expanded, actions)
            best_idx = torch.argmax(Qs)
            return actions[best_idx].unsqueeze(0)
        else:
            # Standard single action sample
            return self._actor.get_action(obs_encoded, z, num=1, batch_input=False, from_target=True)