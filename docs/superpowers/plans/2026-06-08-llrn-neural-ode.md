# Neural ODE Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Neural ODE classifier baseline — `RegimeNODE` — that replaces the RNN encoder with a continuous-time ODE function, and a shared Hydra training entrypoint `train_node.py` that mirrors `train_rnn.py`.

**Architecture:** A `ODEFunc` MLP encodes the hidden state dynamics; `torchdiffeq.odeint` integrates from `t=0` to `t=1` over the last hidden state produced by a linear input projection. The final hidden state is passed through `LayerNorm → Linear(4)` identical to the RNN models. Training uses Adam + CrossEntropyLoss + early stopping on val loss, exactly as `train_rnn.py`.

**Tech Stack:** Python 3.x, PyTorch, torchdiffeq (CPU adjoint solver), Hydra/OmegaConf, pytest

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/models/baseline_node.py` | **Create** | `NODEConfig`, `ODEFunc`, `RegimeNODE` model class |
| `src/training/train_node.py` | **Create** | Hydra walk-forward training entrypoint for NODE |
| `configs/model/node.yaml` | **Create** | Hydra model config for NODE |
| `tests/test_shapes.py` | **Modify** | Append 5 NODE shape tests (forward, predict, predict_proba, proba_sums_to_one, predict_in_range) |

---

## Codebase Context

**Pattern to follow:** `src/models/baseline_lstm.py` and `src/training/train_rnn.py`. The NODE model must expose the same interface: `forward(x)`, `predict(X)`, `predict_proba(X)`, `fit()` (stub raising `NotImplementedError`).

**`Fold` dataclass** (from `src/utils/dataset_builder.py`):
- `fold.train_X` — `(N, window_len, n_features)` np.float32 (flat=False)
- `fold.train_y` — `(N,)` np.int64, labels in {0,1,2,3}
- `fold.val_X`, `fold.val_y`, `fold.test_X`, `fold.test_y` — same shapes
- `fold.test_dates` — `pd.DatetimeIndex` of length `test_size` (252)

**Date alignment:** `test_X` has `test_size - window_len + 1` samples (not `test_size`). Use `test_dates = fold.test_dates[len(fold.test_dates) - n:]` where `n = len(y_pred)`.

**Hydra patterns:**
- `project_root = Path(hydra.utils.get_original_cwd())` — always use this for data/figure paths
- `cfg.model.device` — NOT `cfg.model.get("device", "cpu")` (OmegaConf DictConfig has no `.get()`)
- Call `model.to(device)` **before** `torch.optim.Adam(model.parameters(), ...)` for correctness

**Config entrypoint:** `configs/config.yaml` sets `model: xgb` as default. Override with `model=node` on CLI.

**figures_dir pattern:** `project_root / cfg.figures_dir / "node"` — namespaced per model, mirrors `train_rnn.py`.

**Dependency:** `torchdiffeq` must be installed. Check with `pip show torchdiffeq`. Install with:
```bash
pip install torchdiffeq
```

---

### Task 17: `RegimeNODE` model class

**Files:**
- Create: `src/models/baseline_node.py`
- Modify: `tests/test_shapes.py` (append)

#### Background: Neural ODE for sequence classification

The input `x` is `(batch, window_len, n_features)`. We:
1. Project each timestep: `h0 = relu(linear_in(x[:, -1, :]))` — use only the **last** timestep as initial hidden state (avoids sequence-scan complexity while keeping the ODE as the dynamics model).
2. Define `ODEFunc`: a 2-layer MLP `h → tanh(W2·tanh(W1·h + b1) + b2)` that models `dh/dt`.
3. Integrate: `h1 = odeint(odefunc, h0, t=[0.0, 1.0], method='dopri5')[-1]` — shape `(batch, hidden_dim)`.
4. Classify: `head(norm(h1))` — shape `(batch, 4)`.

`torchdiffeq.odeint` signature:
```python
from torchdiffeq import odeint
# h0: (batch, hidden_dim)
# t: 1-D tensor [t0, t1]
# returns: (len(t), batch, hidden_dim) — take [-1] for final state
h_traj = odeint(func, h0, t, method='dopri5')  # (2, batch, hidden_dim)
h1 = h_traj[-1]  # (batch, hidden_dim)
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_shapes.py`:

```python
from src.models.baseline_node import RegimeNODE, NODEConfig


@pytest.fixture
def node_cfg():
    return NODEConfig(input_dim=37, hidden_dim=32, seed=42)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_node_forward_output_shape(node_cfg, batch_size):
    model = RegimeNODE(node_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4), f"Expected ({batch_size}, 4), got {out.shape}"


def test_node_predict_shape(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_node_predict_proba_shape(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_node_proba_sums_to_one(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_node_predict_in_range(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_node_predict_proba_switches_to_eval(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    model.train()
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd c:\Users\dhruv\projects\lagrange
python -m pytest tests/test_shapes.py::test_node_forward_output_shape -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'src.models.baseline_node'`

- [ ] **Step 3: Check torchdiffeq is installed**

```bash
pip show torchdiffeq
```

If not found:
```bash
pip install torchdiffeq
```

Expected after install: `Name: torchdiffeq` in output.

- [ ] **Step 4: Create `src/models/baseline_node.py`**

```python
"""Neural ODE regime classifier."""
from __future__ import annotations

from dataclasses import dataclass, field

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
        self._t = torch.tensor([0.0, 1.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_len, input_dim)
        h0 = torch.relu(self.input_proj(x[:, -1, :]))  # (batch, hidden_dim)
        t = self._t.to(x.device)
        h_traj = odeint(self.odefunc, h0, t, method=self.cfg.solver)  # (2, batch, hidden_dim)
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_shapes.py -k "node" -v
```

Expected: 7 tests pass (parametrize on batch_size=1 and batch_size=8 for forward test → 2 tests, plus 5 others).

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v
```

Expected: All 53 existing tests + 7 new NODE tests = 60 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/models/baseline_node.py tests/test_shapes.py
git commit -m "feat: RegimeNODE model — Neural ODE classifier with ODEFunc and 7 shape tests"
```

---

### Task 18: `configs/model/node.yaml` and `train_node.py`

**Files:**
- Create: `configs/model/node.yaml`
- Create: `src/training/train_node.py`

#### How `train_node.py` differs from `train_rnn.py`

`train_node.py` is nearly identical to `train_rnn.py` with these differences:
1. Imports `RegimeNODE, NODEConfig` instead of `RegimeLSTM, RegimeGRU, RNNConfig`
2. No `_MODEL_REGISTRY` dict — only one model class
3. `NODEConfig` has no `num_layers`, `dropout` fields — omit those from config instantiation
4. `figures_dir = project_root / cfg.figures_dir / "node"` (hardcoded, not from `cfg.model.name`)
5. `model_name = "node"` constant

The training loop `_train_fold` is **identical** to `train_rnn.py` — copy it verbatim.

- [ ] **Step 1: Create `configs/model/node.yaml`**

```yaml
# @package model
name: node
input_dim: 37
hidden_dim: 64
ode_hidden_dim: 64
batch_size: 64
lr: 0.001
max_epochs: 100
patience: 10
device: cpu
solver: dopri5
```

- [ ] **Step 2: Verify config is loadable**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/model/node.yaml')
print(OmegaConf.to_yaml(cfg))
"
```

Expected: YAML printed with all 9 keys.

- [ ] **Step 3: Write the failing smoke test**

Add to `tests/test_shapes.py`:

```python
def test_node_config_fields():
    cfg = NODEConfig()
    assert hasattr(cfg, "hidden_dim")
    assert hasattr(cfg, "ode_hidden_dim")
    assert hasattr(cfg, "solver")
    assert cfg.solver == "dopri5"
```

Run:
```bash
python -m pytest tests/test_shapes.py::test_node_config_fields -v
```

Expected: PASS (config dataclass already defined in Task 17).

- [ ] **Step 4: Create `src/training/train_node.py`**

```python
"""Neural ODE walk-forward training entrypoint.

Run with:
    python -m src.training.train_node model=node
    python -m src.training.train_node model=node model.hidden_dim=128
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
from src.models.baseline_node import RegimeNODE, NODEConfig
from src.utils.dataset_builder import SplitConfig, build_folds
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_summary,
    plot_regime_timeline,
)

log = logging.getLogger(__name__)

MODEL_NAME = "node"


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
    """Train model for one fold with Adam + cross-entropy + early stopping on val loss."""
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

    node_cfg = NODEConfig(
        input_dim=n_features,
        hidden_dim=cfg.model.hidden_dim,
        ode_hidden_dim=cfg.model.ode_hidden_dim,
        seed=cfg.seed,
        batch_size=cfg.model.batch_size,
        lr=cfg.model.lr,
        max_epochs=cfg.model.max_epochs,
        patience=cfg.model.patience,
        device=str(device),
        solver=cfg.model.solver,
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
        model = RegimeNODE(node_cfg)
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
            result.confusion_matrix, f"Fold {fold.fold_id} (node)",
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
    log.info(f"\nWalk-Forward Summary (node):\n{summary_stats.to_string()}")

    plot_fold_summary(
        all_fold_ids, summary_df["macro_f1"].tolist(),
        save_path=figures_dir / "fold_summary.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify import chain is clean**

```bash
python -c "from src.training.train_node import main; print('OK')"
```

Expected: `OK` with no import errors.

- [ ] **Step 6: Commit**

```bash
git add configs/model/node.yaml src/training/train_node.py tests/test_shapes.py
git commit -m "feat: Neural ODE training entrypoint and Hydra config"
```

---

### Task 19: Smoke test — Neural ODE walk-forward (2 folds)

**Files:**
- No new files — runs `train_node.py` with reduced config

This task verifies the full pipeline runs end-to-end on 2 folds with fast settings. It does NOT run all 21 folds (too slow for a smoke test). The smoke test uses reduced data size to confirm ODE integration works during training.

- [ ] **Step 1: Run smoke test (2 folds)**

```bash
cd c:\Users\dhruv\projects\lagrange
python -m src.training.train_node model=node "splits.min_train_size=200" "splits.val_size=50" "splits.test_size=50" "model.max_epochs=5" "model.patience=3"
```

Expected:
- Hydra creates `outputs/YYYY-MM-DD/HH-MM-SS/` directory
- Log shows `Fold 0:` and `Fold 1:` entries
- Log shows `Macro F1=...` for each fold
- No Python exceptions
- `outputs/.../walk_forward_summary.csv` created with 2 rows

- [ ] **Step 2: Inspect summary CSV**

```bash
python -c "
import pandas as pd, glob, os
files = sorted(glob.glob('outputs/**/**/walk_forward_summary.csv', recursive=True))
df = pd.read_csv(files[-1])
print(df[['fold_id','model','macro_f1','brier_score']].to_string())
"
```

Expected: 2 rows, `model` column = `node`, `macro_f1` between 0.0 and 1.0.

- [ ] **Step 3: Run full test suite one more time**

```bash
python -m pytest tests/ -v
```

Expected: 61 tests pass (60 from Task 17 + 1 config field test from Task 18).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: Neural ODE smoke test passes — 2 folds, end-to-end pipeline verified"
```

---

## Run Full Walk-Forward (after smoke test passes)

To run all 21 folds with default settings:

```bash
python -m src.training.train_node model=node
```

Expected: ~21 fold entries in log, `walk_forward_summary.csv` with 21 rows.

To compare with LSTM:

```bash
python -m src.training.train_rnn model=lstm
python -m src.training.train_node model=node
```
