"""Multi-horizon Lagrangian regime classifier with expressive latent dynamics.

Architecture:
  - Encoder: flatten -> 2-layer MLP -> z0_head + z_dot0_head
  - Richer MassNet: Linear -> GELU -> Linear -> Softplus + eps
  - DeepPotentialNet: 3-layer MLP (128 hidden, GELU)
  - Vector damping: softplus(gamma_net(q)), shape (batch, latent_dim)
  - Optional coord transform: q = coord_net(z), near-identity init
  - Symplectic Euler integrator (n_steps)
  - Multi-horizon heads: head_5, head_10, head_20
  - forward(x) returns 5-day logits (primary task, backward compat)
  - forward_multi(x) returns dict of all three logits
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from src.models.lagrangian_regime_net import _build_encoder, LagrangianConfig


def _softplus_inverse(y: float) -> float:
    if y <= 0:
        raise ValueError(f"softplus_inverse requires y > 0, got {y}")
    return math.log(math.expm1(y))


@dataclass
class LagrangianMHConfig:
    input_dim: int = 37
    window_len: int = 40
    latent_dim: int = 16
    hidden_dim: int = 64
    potential_hidden_dim: int = 128
    mass_hidden_dim: int = 64
    n_steps: int = 8
    damping: float = 0.1
    dt: float = 1.0
    use_forcing: bool = False
    use_vector_damping: bool = True
    use_coord_transform: bool = True
    eps: float = 1e-4
    seed: int = 42
    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 150
    patience: int = 30
    device: str = "cpu"
    multi_horizon: bool = True
    horizons: list[int] = field(default_factory=lambda: [5, 10, 20])
    horizon_weights: dict[int, float] = field(default_factory=lambda: {5: 1.0, 10: 0.5, 20: 0.5})
    # Encoder selection (mirrors LagrangianConfig)
    encoder_type: str = "mlp"
    encoder_dim: int = 64
    conv_channels: int = 64
    conv_kernel_size: int = 3
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])


class RichMassNet(nn.Module):
    """Richer diagonal mass: Linear -> GELU -> Linear -> Softplus + eps."""
    def __init__(self, latent_dim: int, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Init near identity: zero weights, bias at softplus_inverse(1.0)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
        nn.init.constant_(self.net[-1].bias, _softplus_inverse(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(z)) + self.eps


class DeepPotentialNet(nn.Module):
    """Deep scalar potential: 3-layer MLP with GELU, near-zero init."""
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (batch,)


class LagrangianRegimeNetMH(nn.Module):
    """Multi-horizon Lagrangian latent dynamics regime classifier."""

    def __init__(self, cfg: LagrangianMHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # Build a temporary LagrangianConfig to reuse _build_encoder
        _enc_cfg = LagrangianConfig(
            input_dim=cfg.input_dim,
            window_len=cfg.window_len,
            hidden_dim=cfg.hidden_dim,
            encoder_type=cfg.encoder_type,
            encoder_dim=cfg.encoder_dim,
            conv_channels=cfg.conv_channels,
            conv_kernel_size=cfg.conv_kernel_size,
            tcn_channels=cfg.tcn_channels,
            tcn_kernel_size=cfg.tcn_kernel_size,
            tcn_dilations=list(cfg.tcn_dilations),
        )
        self.encoder = _build_encoder(_enc_cfg)
        self.z0_head = nn.Linear(cfg.encoder_dim, cfg.latent_dim)
        self.z_dot0_head = nn.Linear(cfg.encoder_dim, cfg.latent_dim)

        self.mass_net = RichMassNet(cfg.latent_dim, cfg.mass_hidden_dim, cfg.eps)
        self.potential_net = DeepPotentialNet(cfg.latent_dim, cfg.potential_hidden_dim)

        # Scalar fallback damping (always present for compat)
        self.raw_gamma = nn.Parameter(torch.tensor(_softplus_inverse(cfg.damping)))

        if cfg.use_coord_transform:
            self.coord_net = nn.Linear(cfg.latent_dim, cfg.latent_dim)
            nn.init.eye_(self.coord_net.weight)
            nn.init.zeros_(self.coord_net.bias)

        if cfg.use_vector_damping:
            self.gamma_net = nn.Linear(cfg.latent_dim, cfg.latent_dim)
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.constant_(self.gamma_net.bias, _softplus_inverse(cfg.damping))

        if cfg.use_forcing:
            self.forcing_proj = nn.Linear(cfg.input_dim, cfg.latent_dim)

        self.norm = nn.LayerNorm(cfg.latent_dim)

        # Multi-horizon heads
        self.head_5 = nn.Linear(cfg.latent_dim, 4)
        if cfg.multi_horizon:
            self.head_10 = nn.Linear(cfg.latent_dim, 4)
            self.head_20 = nn.Linear(cfg.latent_dim, 4)

        self.last_trajectory: list[torch.Tensor] = []

    def _integrate(self, z: torch.Tensor, z_dot: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run symplectic Euler integration for n_steps. Returns final live z."""
        dt = self.cfg.dt
        trajectory = []

        for _ in range(self.cfg.n_steps):
            z = z.requires_grad_(True)

            if self.cfg.use_coord_transform:
                q = self.coord_net(z)
            else:
                q = z

            V = self.potential_net(q)

            if torch.is_grad_enabled():
                dV_dq = autograd.grad(V.sum(), q, create_graph=True)[0]
            else:
                with torch.enable_grad():
                    if self.cfg.use_coord_transform:
                        z_tmp = z.detach().requires_grad_(True)
                        q_tmp = self.coord_net(z_tmp)
                    else:
                        z_tmp = z.detach().requires_grad_(True)
                        q_tmp = z_tmp
                    V_tmp = self.potential_net(q_tmp)
                    dV_dq = autograd.grad(V_tmp.sum(), q_tmp)[0].detach()

            M_diag = self.mass_net(q)

            if self.cfg.use_vector_damping:
                gamma_vec = F.softplus(self.gamma_net(q))
                z_ddot = -(dV_dq + gamma_vec * z_dot) / M_diag
            else:
                gamma = F.softplus(self.raw_gamma)
                z_ddot = -(dV_dq + gamma * z_dot) / M_diag

            if self.cfg.use_forcing:
                z_ddot = z_ddot + self.forcing_proj(x[:, -1, :])

            z_dot = z_dot + dt * z_ddot
            z = z + dt * z_dot
            trajectory.append(z.detach())

        self.last_trajectory = trajectory  # diagnostic only, overwritten each pass
        return z  # live z for gradient flow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Primary forward — returns 5-day logits (batch, 4)."""
        h = self.encoder(x)
        z = self.z0_head(h)
        z_dot = self.z_dot0_head(h)
        z_T = self._integrate(z, z_dot, x)
        return self.head_5(self.norm(z_T))

    def forward_multi(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Multi-horizon forward — returns dict with logits_5, logits_10, logits_20."""
        h = self.encoder(x)
        z = self.z0_head(h)
        z_dot = self.z_dot0_head(h)
        z_T = self._integrate(z, z_dot, x)
        z_norm = self.norm(z_T)
        out = {"logits_5": self.head_5(z_norm)}
        if self.cfg.multi_horizon:
            out["logits_10"] = self.head_10(z_norm)
            out["logits_20"] = self.head_20(z_norm)
        return out

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training via src.training.train_lagrangian_mh")

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(X).float().to(device)
        logits = self(x)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
