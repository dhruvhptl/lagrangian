"""LSTM and GRU regime classifiers sharing a common RNNConfig."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RNNConfig:
    input_dim: int = 37
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    device: str = "cpu"


class _RegimeRNN(nn.Module):
    """Shared base: RNN encoder → LayerNorm → Linear(4)."""

    def __init__(self, cfg: RNNConfig, rnn_cell: type) -> None:
        super().__init__()
        self.cfg = cfg
        dropout = cfg.dropout if cfg.num_layers > 1 else 0.0
        self.rnn = rnn_cell(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(self.norm(last))

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


class RegimeLSTM(_RegimeRNN):
    def __init__(self, cfg: RNNConfig) -> None:
        super().__init__(cfg, nn.LSTM)


class RegimeGRU(_RegimeRNN):
    def __init__(self, cfg: RNNConfig) -> None:
        super().__init__(cfg, nn.GRU)
