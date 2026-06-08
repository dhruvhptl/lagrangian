"""Neural ODE regime classifier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


@dataclass
class NODEConfig:
    input_dim: int = 37
    hidden_dim: int = 64
    ode_hidden_dim: int = 64
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    device: str = "cpu"
    solver: str = "dopri5"


class ODEFunc(nn.Module):
    """2-layer MLP modelling dh/dt."""

    def __init__(self, hidden_dim: int, ode_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(ode_hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class RegimeNODE(nn.Module):
    """Neural ODE classifier: input projection → ODE → LayerNorm → Linear(4)."""

    def __init__(self, cfg: NODEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.odefunc = ODEFunc(cfg.hidden_dim, cfg.ode_hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, 4)
        self.register_buffer('_t', torch.tensor([0.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, input_dim)
        h0 = torch.relu(self.input_proj(x[:, -1, :]))  # (batch, hidden_dim)
        h_traj = odeint(self.odefunc, h0, self._t, method=self.cfg.solver)  # (2, batch, hidden_dim)
        h1 = h_traj[-1]  # (batch, hidden_dim)
        return self.head(self.norm(h1))  # (batch, 4)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is handled by src.training.train_node — "
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
