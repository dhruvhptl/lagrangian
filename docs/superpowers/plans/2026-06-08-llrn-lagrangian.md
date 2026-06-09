# Lagrangian Regime Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `LagrangianRegimeNet` — a discrete Lagrangian-inspired latent dynamics model — with its Hydra training entrypoint, config, and 11 shape/behavior tests.

**Architecture:** MLP encoder flattens the input window and produces `(z_0, z_dot_0)` as initial position and velocity in a `latent_dim`-dimensional space. A discrete symplectic Euler integrator evolves the state for `n_steps` using a learned diagonal mass matrix (MassNet), scalar potential (PotentialNet), and softplus-constrained damping. The final latent position is classified by `LayerNorm → Linear(4)`. Training uses Adam + CrossEntropyLoss + early stopping + gradient clipping, mirroring `train_node.py`.

**Tech Stack:** Python 3.x, PyTorch (autograd.grad for potential gradient), Hydra/OmegaConf, pytest

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/models/lagrangian_regime_net.py` | **Create** | `LagrangianConfig`, `MassNet`, `PotentialNet`, `LagrangianRegimeNet` |
| `src/training/train_lagrangian.py` | **Create** | Hydra walk-forward training entrypoint |
| `configs/model/lagrangian.yaml` | **Create** | Hydra model config |
| `tests/test_shapes.py` | **Modify** | Append 11 Lagrangian tests |

---

## Codebase Context

**Existing patterns to follow:**
- `src/models/baseline_node.py` — model structure: `Config` dataclass, `nn.Module` with `forward`, `predict_proba`, `predict`, `fit` stub
- `src/training/train_node.py` — training entrypoint: Hydra `@hydra.main`, `_get_labels`, `_train_fold`, `main`
- `tests/test_shapes.py` — existing tests: all imports at top, `toy_seq_data` fixture produces `(200, 40, 37)` train / `(50, 40, 37)` val arrays, `node_cfg` fixture pattern

**`Fold` dataclass** (from `src/utils/dataset_builder.py`):
- `fold.train_X` — `(N, window_len, n_features)` np.float32, `flat=False`
- `fold.train_y` — `(N,)` np.int64, labels in {0,1,2,3}
- `fold.test_dates` — `pd.DatetimeIndex` of length `test_size`
- Date alignment: `test_dates = fold.test_dates[len(fold.test_dates) - n:]` where `n = len(y_pred)`

**Hydra patterns:**
- `project_root = Path(hydra.utils.get_original_cwd())` — always use for data/figure paths
- `cfg.model.device` — NOT `.get("device", "cpu")` (OmegaConf has no `.get()`)
- `model.to(device)` BEFORE `torch.optim.Adam(model.parameters(), ...)`

**softplus_inverse:** `math.log(math.exp(x) - 1)` — used to initialize `raw_gamma` so `softplus(raw_gamma) ≈ damping`.

---

## Task 20: `LagrangianRegimeNet` model class + 11 shape tests

**Files:**
- Create: `src/models/lagrangian_regime_net.py`
- Modify: `tests/test_shapes.py` (append)

- [ ] **Step 1: Append the failing tests to `tests/test_shapes.py`**

Open `tests/test_shapes.py`. The last line is currently `assert cfg.solver == "dopri5"`. Append the following **after** that line:

```python
from src.models.lagrangian_regime_net import LagrangianRegimeNet, LagrangianConfig


@pytest.fixture
def lag_cfg():
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=8,
        hidden_dim=32,
        n_steps=3,
        seed=42,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
def test_lagrangian_forward_output_shape(lag_cfg, batch_size):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4), f"Expected ({batch_size}, 4), got {out.shape}"


def test_lagrangian_predict_shape(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_lagrangian_predict_proba_shape(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_lagrangian_proba_sums_to_one(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_predict_in_range(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_lagrangian_predict_proba_switches_to_eval(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    model.train()
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


def test_lagrangian_trajectory_length(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == lag_cfg.n_steps


def test_lagrangian_trajectory_shape(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    for z in model.last_trajectory:
        assert z.shape == (4, lag_cfg.latent_dim)


def test_lagrangian_mass_positive(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    z = torch.randn(4, lag_cfg.latent_dim)
    m = model.mass_net(z)
    assert (m > 0).all(), "Mass diagonal must be strictly positive"


def test_lagrangian_damping_positive(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    gamma = torch.nn.functional.softplus(model.raw_gamma)
    assert gamma.item() > 0, "Damping must be positive"


def test_lagrangian_forward_finite(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    assert torch.isfinite(logits).all(), "Forward pass produced non-finite logits"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd c:\Users\dhruv\projects\lagrange
python -m pytest tests/test_shapes.py::test_lagrangian_forward_output_shape -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'src.models.lagrangian_regime_net'`

- [ ] **Step 3: Create `src/models/lagrangian_regime_net.py`**

```python
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
    eps: float = 1e-4
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 150
    patience: int = 15
    device: str = "cpu"


def _softplus_inverse(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y."""
    return math.log(math.exp(y) - 1.0)


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
        z = self.z0_head(h)       # (batch, latent_dim)
        z_dot = self.z_dot0_head(h)  # (batch, latent_dim)

        gamma = F.softplus(self.raw_gamma)
        dt = self.cfg.dt
        trajectory = []

        for _ in range(self.cfg.n_steps):
            # Enable grad on z for autograd.grad through potential
            z = z.requires_grad_(True)
            V = self.potential_net(z)             # (batch,)
            dV_dz = autograd.grad(
                V.sum(), z, create_graph=True
            )[0]                                  # (batch, latent_dim)

            M_diag = self.mass_net(z)             # (batch, latent_dim), positive
            z_ddot = -(dV_dz + gamma * z_dot) / M_diag  # (batch, latent_dim)

            if self.cfg.use_forcing:
                z_ddot = z_ddot + self.forcing_proj(x[:, -1, :])

            # Symplectic Euler: update velocity first, then position
            z_dot = z_dot + dt * z_ddot
            z = z + dt * z_dot

            trajectory.append(z.detach())

        # Diagnostic-only state: overwritten each forward, not part of training interface
        self.last_trajectory = trajectory

        z_T = trajectory[-1]
        return self.head(self.norm(z_T))  # (batch, 4)

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
```

- [ ] **Step 4: Run the Lagrangian tests**

```bash
python -m pytest tests/test_shapes.py -k "lagrangian" -v
```

Expected: 13 tests pass (forward×2 via parametrize + 11 others). If `test_lagrangian_forward_finite` fails with NaN, the most likely cause is an exploding potential gradient on random input — verify `PotentialNet` initialization is near-zero and `MassNet` bias is set to `_softplus_inverse(1.0) ≈ 0.541`.

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 61 existing tests + 13 new = 74 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/lagrangian_regime_net.py tests/test_shapes.py
git commit -m "feat: LagrangianRegimeNet model — discrete Lagrangian latent dynamics with 13 shape tests"
```

---

## Task 21: `configs/model/lagrangian.yaml` and `train_lagrangian.py`

**Files:**
- Create: `configs/model/lagrangian.yaml`
- Create: `src/training/train_lagrangian.py`

- [ ] **Step 1: Create `configs/model/lagrangian.yaml`**

```yaml
# @package model
name: lagrangian
input_dim: 37
window_len: 40
latent_dim: 8
hidden_dim: 64
n_steps: 4
damping: 0.1
dt: 1.0
use_forcing: false
eps: 0.0001
batch_size: 64
lr: 0.001
max_epochs: 150
patience: 15
device: cpu
```

- [ ] **Step 2: Verify config loads**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/model/lagrangian.yaml')
print(OmegaConf.to_yaml(cfg))
"
```

Expected: all 16 keys printed, no errors.

- [ ] **Step 3: Create `src/training/train_lagrangian.py`**

```python
"""Lagrangian Regime Network walk-forward training entrypoint.

Run with:
    python -m src.training.train_lagrangian model=lagrangian
    python -m src.training.train_lagrangian model=lagrangian model.latent_dim=16
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.data.download import fetch_all
from src.data.manager import DataManager
from src.evaluation.metrics import evaluate
from src.features.engineer import FeaturesConfig, build_features
from src.labels.quantile_labeler import LabelConfig, QuantileLabeler
from src.models.lagrangian_regime_net import LagrangianConfig, LagrangianRegimeNet
from src.utils.dataset_builder import SplitConfig, build_folds
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_summary,
    plot_regime_timeline,
)

log = logging.getLogger(__name__)

MODEL_NAME = "lagrangian"


def _get_labels(
    spy_prices: pd.DataFrame,
    label_cfg: LabelConfig,
    feature_index: pd.DatetimeIndex,
) -> pd.Series:
    labeler = QuantileLabeler(label_cfg)
    label_df = labeler.fit_transform(spy_prices)
    return label_df["label"].reindex(feature_index)


def _train_fold(
    model: nn.Module,
    fold,
    cfg: DictConfig,
    device: torch.device,
) -> nn.Module:
    """Train one fold: Adam + CrossEntropyLoss + early stopping + gradient clipping."""
    X_tr = torch.from_numpy(fold.train_X).float()
    y_tr = torch.from_numpy(fold.train_y).long()
    X_va = torch.from_numpy(fold.val_X).float().to(device)
    y_va = torch.from_numpy(fold.val_y).long().to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg.model.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(cfg.model.max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.model.patience:
                log.debug(f"  Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra.utils.get_original_cwd())

    set_global_seed(cfg.seed)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.model.device)

    dm = DataManager(
        raw_dir=project_root / cfg.data.raw_dir,
        processed_dir=project_root / cfg.data.processed_dir,
        tickers=list(cfg.data.tickers),
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    prices = fetch_all(dm)

    feat_cfg = FeaturesConfig(
        roll_windows=list(cfg.features.roll_windows),
        momentum_windows=list(cfg.features.momentum_windows),
        corr_windows=list(cfg.features.corr_windows),
        cross_assets=list(cfg.features.cross_assets),
        primary_asset=cfg.features.primary_asset,
    )
    features = build_features(prices, feat_cfg)
    n_features = features.shape[1]
    log.info(f"Features shape: {features.shape}")

    label_cfg = LabelConfig(
        horizon=cfg.labels.horizon,
        vol_window=cfg.labels.vol_window,
        return_quantile=cfg.labels.return_quantile,
        vol_quantile=cfg.labels.vol_quantile,
        smoothing=cfg.labels.smoothing,
        smoothing_min_periods=cfg.labels.smoothing_min_periods,
    )
    spy_prices = prices[cfg.labels.label_asset]
    labels = _get_labels(spy_prices, label_cfg, features.index)

    split_cfg = SplitConfig(
        train_start=cfg.splits.train_start,
        val_size=cfg.splits.val_size,
        test_size=cfg.splits.test_size,
        step_size=cfg.splits.step_size,
        min_train_size=cfg.splits.min_train_size,
    )

    lag_cfg = LagrangianConfig(
        input_dim=n_features,
        window_len=cfg.data.window_len,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_steps=cfg.model.n_steps,
        damping=cfg.model.damping,
        dt=cfg.model.dt,
        use_forcing=cfg.model.use_forcing,
        eps=cfg.model.eps,
        seed=cfg.seed,
        batch_size=cfg.model.batch_size,
        lr=cfg.model.lr,
        max_epochs=cfg.model.max_epochs,
        patience=cfg.model.patience,
        device=str(device),
    )

    output_dir = Path(".")
    figures_dir = project_root / cfg.figures_dir / MODEL_NAME

    all_metrics: list[dict] = []
    all_fold_ids: list[int] = []

    for fold in build_folds(
        features,
        labels,
        split_cfg,
        window_len=cfg.data.window_len,
        flat=False,
    ):
        log.info(
            f"Fold {fold.fold_id}: "
            f"train={len(fold.train_y)} val={len(fold.val_y)} test={len(fold.test_y)}"
        )

        torch.manual_seed(cfg.seed)
        model = LagrangianRegimeNet(lag_cfg)
        model = _train_fold(model, fold, cfg, device)

        y_pred = model.predict(fold.test_X)
        y_prob = model.predict_proba(fold.test_X)
        result = evaluate(fold.test_y, y_pred, y_prob)

        metrics_dict = {
            "fold_id": fold.fold_id,
            "model": MODEL_NAME,
            "macro_f1": result.macro_f1,
            "balanced_accuracy": result.balanced_accuracy,
            "brier_score": result.brier_score,
            "ece": result.ece,
            "switch_frequency": result.switch_frequency,
            "mean_entropy": result.mean_entropy,
            "val_start": str(fold.val_dates.min().date()),
            "test_start": str(fold.test_dates.min().date()),
            "test_end": str(fold.test_dates.max().date()),
        }
        all_metrics.append(metrics_dict)
        all_fold_ids.append(fold.fold_id)

        fold_dir = output_dir / f"fold_{fold.fold_id:02d}"
        fold_dir.mkdir(exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        # test_X has (test_size - window_len + 1) samples; align dates to the last n
        n = len(y_pred)
        test_dates = fold.test_dates[len(fold.test_dates) - n:]
        plot_regime_timeline(
            test_dates, fold.test_y[:n], y_pred,
            figures_dir / f"fold_{fold.fold_id:02d}_timeline.png",
        )
        plot_confusion_matrix(
            result.confusion_matrix, f"Fold {fold.fold_id} (lagrangian)",
            figures_dir / f"fold_{fold.fold_id:02d}_cm.png",
        )

        log.info(
            f"  Macro F1={result.macro_f1:.4f}  "
            f"Brier={result.brier_score:.4f}  "
            f"ECE={result.ece:.4f}"
        )

    if not all_metrics:
        log.warning("No folds produced — check split config and data date range.")
        return

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(output_dir / "walk_forward_summary.csv", index=False)

    numeric_cols = ["macro_f1", "balanced_accuracy", "brier_score", "ece"]
    summary_stats = summary_df[numeric_cols].agg(["mean", "std"])
    log.info(f"\nWalk-Forward Summary (lagrangian):\n{summary_stats.to_string()}")

    plot_fold_summary(
        all_fold_ids, summary_df["macro_f1"].tolist(),
        save_path=figures_dir / "fold_summary.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify import chain**

```bash
python -c "from src.training.train_lagrangian import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 74 tests pass.

- [ ] **Step 6: Commit**

```bash
git add configs/model/lagrangian.yaml src/training/train_lagrangian.py
git commit -m "feat: Lagrangian training entrypoint and Hydra config"
```

---

## Task 22: Smoke test — Lagrangian walk-forward (2 folds)

**Files:** None — runs `train_lagrangian.py` with reduced settings.

- [ ] **Step 1: Run smoke test**

```bash
cd c:\Users\dhruv\projects\lagrange
python -m src.training.train_lagrangian model=lagrangian "splits.min_train_size=200" "splits.val_size=50" "splits.test_size=50" "model.max_epochs=5" "model.patience=3" "model.n_steps=2"
```

Expected:
- At least 2 fold entries in log (`Fold 0:`, `Fold 1:`)
- Each fold logs `Macro F1=...`
- No Python exceptions (especially no NaN/inf from the integrator)
- `walk_forward_summary.csv` created

- [ ] **Step 2: Inspect CSV**

```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('outputs/**/**/walk_forward_summary.csv', recursive=True))
df = pd.read_csv(files[-1])
print(df[['fold_id','model','macro_f1','brier_score']].to_string())
"
```

Expected: rows with `model=lagrangian`, `macro_f1` between 0.0 and 1.0, no NaN.

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 74 tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: Lagrangian smoke test passes — walk-forward pipeline verified"
```

---

## Debugging Notes

**NaN in logits / loss:** Most likely cause is exploding `dV_dz` when `PotentialNet` weights are not near-zero. Verify `nn.init.normal_(layer.weight, std=0.01)` in `PotentialNet.__init__`. Also check gradient clipping is applied (`clip_grad_norm_` in `_train_fold`).

**`autograd.grad` error "One of the differentiated Tensors... does not require grad":** `z` must have `requires_grad_(True)` called at the top of each integration step. In the implementation above this is done with `z = z.requires_grad_(True)` inside the loop — verify this line is present.

**`RuntimeError: Trying to backward through the graph a second time`:** The trajectory stores `z.detach()` not `z` — if you accidentally store the live tensor, backprop will try to reuse freed buffers. Verify `trajectory.append(z.detach())` in the integration loop. Note: after `z = z + dt * z_dot`, the new `z` is used both as the live variable for the next step AND stored detached — this is correct.

**`create_graph=True` memory:** Each fold is short (< 2000 samples), so memory is not a concern at `batch_size=64`. If OOM occurs, reduce `batch_size` via CLI override.
