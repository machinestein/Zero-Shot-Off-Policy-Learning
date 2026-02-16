from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch.amp import autocast

from metamotivo.agents.fb.agent import FBAgent, FBAgentConfig, FBAgentTrainConfig
from metamotivo.nn_models import _soft_update_params, eval_mode, expectile_regression_loss
from .model import BreezeModelConfig


class BreezeAgentTrainConfig(FBAgentTrainConfig):
    lr_v: float = 1e-4
    expectile: float = 0.7
    freg_coef: float = 0.01
    guide_weight: float = 3.0
    ktrain: int = 1
    keval: int = 1

class BreezeAgentConfig(FBAgentConfig):
    name: Literal["BreezeAgent"] = "BreezeAgent"
    model: BreezeModelConfig
    train: BreezeAgentTrainConfig

class BreezeAgent(FBAgent):
    def setup_training(self) -> None:
        super().setup_training()
        self.v_optimizer = torch.optim.Adam(
            self._model._vfunc.parameters(),
            lr=self.cfg.train.lr_v,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device),
            batch["action"].to(self.device),
            batch["next"]["observation"].to(self.device),
            batch["next"]["terminated"].to(self.device),
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()

        obs, next_obs = self.aug(obs, next_obs)
        obs, next_obs, goal = self.enc(obs, next_obs)

        z = self.sample_mixed_z(train_goal=goal).clone()

        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=goal,
            z=z,
            clip_grad_norm=clip_grad_norm,
        )
        
        metrics.update(
            self.update_v(
                obs=obs.detach(),
                action=action, # Pass action for IQL V-update
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        metrics.update(
            self.update_actor(
                obs=obs.detach(),
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.cfg.train.f_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.cfg.train.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.cfg.train.f_target_tau)

        return metrics

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        # Pass keval from training config to the model's act method
        return self._model.act(obs, z, mean, keval=self.cfg.train.keval)

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                next_left_enc = self._model._target_left_encoder(next_obs)
                # In BREEZE, we use the actor to sample next actions
                actor_in = next_left_enc if self.cfg.model.actor_encode_obs else next_obs
                next_action = self._model._actor.get_action(actor_in, z, num=self.cfg.train.ktrain, batch_input=True, from_target=True)
                # If ktrain > 1, we should pick the best action according to Q, but for training often we just use one or mean.
                # Original BREEZE uses re-sampling if ktrain > 1.
                if self.cfg.train.ktrain > 1:
                    # next_action: [batch, ktrain, action_dim]
                    # We need to repeat next_left_enc and z
                    next_left_enc_rep = next_left_enc.unsqueeze(1).repeat(1, self.cfg.train.ktrain, 1).view(-1, next_left_enc.shape[-1])
                    z_rep = z.unsqueeze(1).repeat(1, self.cfg.train.ktrain, 1).view(-1, z.shape[-1])
                    next_action_flat = next_action.view(-1, next_action.shape[-1])
                    
                    target_Fs1, target_Fs2 = self._model._target_forward_map(next_left_enc_rep, z_rep, next_action_flat)
                    target_Qs = torch.min((target_Fs1 * z_rep).sum(-1), (target_Fs2 * z_rep).sum(-1))
                    target_Qs = target_Qs.view(-1, self.cfg.train.ktrain)
                    best_indices = torch.argmax(target_Qs, dim=1)
                    next_action = next_action[torch.arange(next_action.shape[0]), best_indices]
                else:
                    next_action = next_action.squeeze(1)

                target_Fs1, target_Fs2 = self._model._target_forward_map(next_left_enc, z, next_action)
                target_B = self._model._target_backward_map(goal)
                
                target_M1 = torch.einsum("sd, td -> st", target_Fs1, target_B)
                target_M2 = torch.einsum("sd, td -> st", target_Fs2, target_B)
                target_M = torch.min(target_M1, target_M2)

                next_V = self._model.vfunc(next_obs, z).squeeze()

            # Compute FB loss
            left_enc = self._model._left_encoder(obs)
            Fs1, Fs2 = self._model._forward_map(left_enc, z, action)
            B = self._model._backward_map(goal)
            
            M1 = torch.einsum("sd, td -> st", Fs1, B)
            M2 = torch.einsum("sd, td -> st", Fs2, B)
            
            fb_loss = 0.0
            for M in [M1, M2]:
                diff = M - discount * target_M
                fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
                fb_diag = -torch.diagonal(diff, dim1=0, dim2=1).mean()
                fb_loss += fb_offdiag + fb_diag
            
            # Orthonormality loss
            Cov = torch.matmul(B, B.T)
            orth_loss_diag = -2 * Cov.diag().mean()
            orth_loss_offdiag = (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
            orth_loss = self.cfg.train.ortho_coef * (orth_loss_offdiag + orth_loss_diag)
            fb_loss += orth_loss

            # Regularization (Q-learning loss)
            # In BREEZE, this uses IQL targets for V
            with torch.no_grad():
                # Compute implicit reward
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)
                target_Q = implicit_reward + self.cfg.train.discount * next_V

            Q1, Q2 = (Fs1 * z).sum(-1), (Fs2 * z).sum(-1)
            q_loss = self.cfg.train.freg_coef * 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))
            fb_loss += q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        return {
            "fb_loss": fb_loss.detach(),
            "orth_loss": orth_loss.detach(),
            "q_loss": q_loss.detach(),
            "target_M": target_M.mean().detach(),
            "M1": M1.mean().detach(),
            "B_norm": torch.norm(B, dim=-1).mean().detach(),
        }

    def update_v(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            with torch.no_grad():
                # Get Q values for the current actions (sampled from buffer or actor? usually buffer for IQL V update)
                # But wait, IQL V update: V(s) should be the expectile of Q(s, a) where a is from BUFFER.
                # So we need the actions from the batch.
                Q = self._model.predict_q(obs, z, action)
            
            V = self._model.vfunc(obs, z).squeeze()
            v_loss = self.expectile_regression_loss(Q - V, self.cfg.train.expectile)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._vfunc.parameters(), clip_grad_norm)
        self.v_optimizer.step()

        return {"v_loss": v_loss.detach(), "V": V.mean().detach()}

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            # IQL Actor update: weighted regression
            with torch.no_grad():
                Q = self._model.predict_q(obs, z, action)
                V = self._model.vfunc(obs, z).squeeze()
                weight = torch.exp(self.cfg.train.guide_weight * (Q - V)).clamp(max=100)

            actor_loss = self._model._actor.policy_loss_with_weight(weight, action, obs, z)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "weight": weight.mean().detach()}