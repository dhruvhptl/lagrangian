"""Lagrangian-inspired discrete latent dynamics regime classifier."""
from __future__ import annotations

import math
from dataclasses import dataclass

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


def _softplus_inverse(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y. Requires y > 0."""
    if y <= 0:
        raise ValueError(f"softplus_inverse requires y > 0, got {y}")
    return math.log(math.expm1(y))


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
    """Deeper, wider scalar potential: 3-layer MLP with GELU, hidden_dim=128. Near-zero init."""

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


class LagrangianRegimeNet(nn.Module):
    """Discrete Lagrangian-inspired latent dynamics classifier.

    Encoder flattens the input window -> MLP -> (z_0, z_dot_0).
    Symplectic Euler integrator evolves latent state for n_steps.
    Classifier head: LayerNorm -> Linear(latent_dim, 4).
    """

    def __init__(self, cfg: LagrangianConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        flat_dim = cfg.window_len * cfg.input_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.z0_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.z_dot0_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        self.mass_net = MassNet(cfg.latent_dim, cfg.eps)
        self.potential_net = PotentialNet(cfg.latent_dim, cfg.hidden_dim)

        # v5: replace shallow potential with deeper one when vector damping is enabled
        if cfg.use_vector_damping:
            self.potential_net = DeepPotentialNet(cfg.latent_dim)

        # v5: optional coordinate transform q = W z, near-identity init
        if cfg.use_coord_transform:
            self.coord_net = nn.Linear(cfg.latent_dim, cfg.latent_dim)
            nn.init.eye_(self.coord_net.weight)
            nn.init.zeros_(self.coord_net.bias)

        # v5: vector damping gamma(q) -> latent_dim positive values
        if cfg.use_vector_damping:
            self.gamma_net = nn.Linear(cfg.latent_dim, cfg.latent_dim)
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.constant_(self.gamma_net.bias, _softplus_inverse(cfg.damping))

        # Damping: always positive via softplus. Init so softplus(raw_gamma) ≈ cfg.damping
        self.raw_gamma = nn.Parameter(
            torch.tensor(_softplus_inverse(cfg.damping))
        )

        if cfg.use_forcing:
            self.forcing_proj = nn.Linear(cfg.input_dim, cfg.latent_dim)

        self.norm = nn.LayerNorm(cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, 4)

        self.last_trajectory: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, input_dim)
        batch = x.shape[0]
        h = self.encoder(x.reshape(batch, -1))
        z = self.z0_head(h)           # (batch, latent_dim)
        z_dot = self.z_dot0_head(h)   # (batch, latent_dim)

        dt = self.cfg.dt
        trajectory = []

        for _ in range(self.cfg.n_steps):
            # Enable grad on z for autograd.grad through potential
            z = z.requires_grad_(True)

            # Optional coordinate transform q = W z
            if self.cfg.use_coord_transform:
                q = self.coord_net(z)  # (batch, latent_dim)
            else:
                q = z

            V = self.potential_net(q)                          # (batch,)

            if torch.is_grad_enabled():
                # Differentiate V w.r.t. q; chain rule back to z handled by autograd
                dV_dq = autograd.grad(
                    V.sum(), q, create_graph=True
                )[0]                                           # (batch, latent_dim)
            else:
                # During inference (no_grad context), compute gradient manually
                # by temporarily enabling grad just for this computation
                with torch.enable_grad():
                    if self.cfg.use_coord_transform:
                        z_tmp = z.detach().requires_grad_(True)
                        q_tmp = self.coord_net(z_tmp)
                    else:
                        z_tmp = z.detach().requires_grad_(True)
                        q_tmp = z_tmp
                    V_tmp = self.potential_net(q_tmp)
                    dV_dq = autograd.grad(V_tmp.sum(), q_tmp)[0].detach()

            M_diag = self.mass_net(q)                          # (batch, latent_dim), positive

            if self.cfg.use_vector_damping:
                gamma_vec = F.softplus(self.gamma_net(q))  # (batch, latent_dim)
                z_ddot = -(dV_dq + gamma_vec * z_dot) / M_diag
            else:
                gamma = F.softplus(self.raw_gamma)
                z_ddot = -(dV_dq + gamma * z_dot) / M_diag    # (batch, latent_dim)

            if self.cfg.use_forcing:
                z_ddot = z_ddot + self.forcing_proj(x[:, -1, :])

            # Symplectic Euler: update velocity first, then position
            z_dot = z_dot + dt * z_ddot
            z = z + dt * z_dot

            # Store detached tensor — diagnostic only, must not hold training graph
            trajectory.append(z.detach())

        # Overwritten each forward pass — diagnostic state only
        self.last_trajectory = trajectory

        # Use live z (not trajectory[-1]) so gradients flow during training
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
