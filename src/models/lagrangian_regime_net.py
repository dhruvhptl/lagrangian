"""Lagrangian-inspired discrete latent dynamics regime classifier.

Supports four encoder variants via LagrangianConfig.encoder_type:
  "mlp"          - flatten + 2-layer MLP (original)
  "conv1d"       - causal 1D-conv encoder
  "tcn"          - dilated causal TCN encoder
  "hybrid_conv"  - wider/deeper causal conv encoder

All encoders output h of shape (batch, encoder_dim), then two linear heads
map h -> z_0 and h -> z_dot_0. Downstream Lagrangian integrator unchanged.
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


@dataclass
class LagrangianConfig:
    input_dim: int = 37
    window_len: int = 40
    latent_dim: int = 8
    hidden_dim: int = 64
    n_steps: int = 4
    damping: float = 0.1
    dt: float = 1.0
    use_forcing: bool = False
    use_vector_damping: bool = False
    use_coord_transform: bool = False
    eps: float = 1e-4
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 150
    patience: int = 15
    device: str = "cpu"
    # Encoder selection
    encoder_type: str = "mlp"        # "mlp" | "conv1d" | "tcn" | "hybrid_conv"
    encoder_dim: int = 64            # output dim of any encoder before z_0/z_dot_0 heads
    conv_channels: int = 64
    conv_kernel_size: int = 3
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])


def _softplus_inverse(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y. Requires y > 0."""
    if y <= 0:
        raise ValueError(f"softplus_inverse requires y > 0, got {y}")
    return math.log(math.expm1(y))


# ---------------------------------------------------------------------------
# Dynamics sub-modules (unchanged)
# ---------------------------------------------------------------------------

class MassNet(nn.Module):
    """Diagonal mass matrix: Linear(latent_dim -> latent_dim) -> Softplus + eps.

    Initialized near identity: bias = softplus_inverse(1.0) so M ≈ 1 at init.
    """

    def __init__(self, latent_dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, _softplus_inverse(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.linear(z)) + self.eps


class PotentialNet(nn.Module):
    """Scalar potential V(z): Linear -> Tanh -> Linear -> scalar.

    Initialized near zero so V ≈ 0 at the start of training.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (batch,)


class DeepPotentialNet(nn.Module):
    """Deeper scalar potential: 3-layer MLP with GELU, hidden_dim=128. Near-zero init."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Encoder modules
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """Flatten + 2-layer MLP encoder (original behavior)."""

    def __init__(self, input_dim: int, window_len: int, hidden_dim: int, encoder_dim: int) -> None:
        super().__init__()
        flat_dim = window_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        return self.net(x.reshape(x.shape[0], -1))


class CausalConv1d(nn.Module):
    """Single causal Conv1d layer with left-padding to preserve sequence length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # left-only pad
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class Conv1dEncoder(nn.Module):
    """Causal 1D-conv encoder: two causal conv layers, last-step readout."""

    def __init__(self, input_dim: int, conv_channels: int, kernel_size: int, encoder_dim: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(input_dim, conv_channels, kernel_size)
        self.conv2 = CausalConv1d(conv_channels, conv_channels, kernel_size)
        self.proj = nn.Linear(conv_channels, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F) -> (batch, F, T)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        h = x[:, :, -1]  # last timestep: (batch, conv_channels)
        return self.proj(h)


class TCNResidualBlock(nn.Module):
    """TCN residual block: two causal dilated convs + residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        return F.gelu(x + residual)


class TCNEncoder(nn.Module):
    """Dilated causal TCN encoder with dilation rates [1,2,4,8], last-step readout."""

    def __init__(
        self,
        input_dim: int,
        tcn_channels: int,
        kernel_size: int,
        dilations: List[int],
        encoder_dim: int,
    ) -> None:
        super().__init__()
        # Input projection to tcn_channels
        self.input_proj = CausalConv1d(input_dim, tcn_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNResidualBlock(tcn_channels, kernel_size, d) for d in dilations
        ])
        self.proj = nn.Linear(tcn_channels, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F) -> (batch, F, T)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        h = x[:, :, -1]  # last timestep: (batch, tcn_channels)
        return self.proj(h)


class HybridConvEncoder(nn.Module):
    """Wider/deeper causal conv encoder: 3 layers (F->64->128->64), last-step readout."""

    def __init__(self, input_dim: int, kernel_size: int, encoder_dim: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(input_dim, 64, kernel_size)
        self.conv2 = CausalConv1d(64, 128, kernel_size)
        self.conv3 = CausalConv1d(128, 64, kernel_size)
        self.proj = nn.Linear(64, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F) -> (batch, F, T)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        h = x[:, :, -1]  # last timestep: (batch, 64)
        return self.proj(h)


def _build_encoder(cfg: LagrangianConfig) -> nn.Module:
    enc = cfg.encoder_type
    if enc == "mlp":
        return MLPEncoder(cfg.input_dim, cfg.window_len, cfg.hidden_dim, cfg.encoder_dim)
    elif enc == "conv1d":
        return Conv1dEncoder(cfg.input_dim, cfg.conv_channels, cfg.conv_kernel_size, cfg.encoder_dim)
    elif enc == "tcn":
        return TCNEncoder(
            cfg.input_dim, cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_dilations, cfg.encoder_dim
        )
    elif enc == "hybrid_conv":
        return HybridConvEncoder(cfg.input_dim, cfg.conv_kernel_size, cfg.encoder_dim)
    else:
        raise ValueError(f"Unknown encoder_type '{enc}'. Choose from: mlp, conv1d, tcn, hybrid_conv")


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LagrangianRegimeNet(nn.Module):
    """Discrete Lagrangian-inspired latent dynamics classifier.

    Encoder (modular, selected by cfg.encoder_type) produces h of shape
    (batch, encoder_dim). Two linear heads map h -> z_0 and h -> z_dot_0.
    Symplectic Euler integrator evolves latent state for n_steps.
    Classifier head: LayerNorm -> Linear(latent_dim, 4).
    """

    def __init__(self, cfg: LagrangianConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.encoder = _build_encoder(cfg)
        self.z0_head = nn.Linear(cfg.encoder_dim, cfg.latent_dim)
        self.z_dot0_head = nn.Linear(cfg.encoder_dim, cfg.latent_dim)

        self.mass_net = MassNet(cfg.latent_dim, cfg.eps)
        self.potential_net = PotentialNet(cfg.latent_dim, cfg.hidden_dim)

        if cfg.use_vector_damping:
            self.potential_net = DeepPotentialNet(cfg.latent_dim)

        if cfg.use_coord_transform:
            self.coord_net = nn.Linear(cfg.latent_dim, cfg.latent_dim)
            nn.init.eye_(self.coord_net.weight)
            nn.init.zeros_(self.coord_net.bias)

        if cfg.use_vector_damping:
            self.gamma_net = nn.Linear(cfg.latent_dim, cfg.latent_dim)
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.constant_(self.gamma_net.bias, _softplus_inverse(cfg.damping))

        self.raw_gamma = nn.Parameter(
            torch.tensor(_softplus_inverse(cfg.damping))
        )

        if cfg.use_forcing:
            self.forcing_proj = nn.Linear(cfg.input_dim, cfg.latent_dim)

        self.norm = nn.LayerNorm(cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, 4)

        self.last_trajectory: list[torch.Tensor] = []

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder -> (batch, encoder_dim)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, input_dim)
        h = self._encode(x)
        z = self.z0_head(h)           # (batch, latent_dim)
        z_dot = self.z_dot0_head(h)   # (batch, latent_dim)

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

        self.last_trajectory = trajectory

        return self.head(self.norm(z))  # (batch, 4)

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Training is handled by src.training.train_lagrangian — "
            "instantiate the model there, not via fit()."
        )

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
