# LLRN Phase 6–7: LSTM + GRU Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LSTM and GRU neural baselines to the LLRN project. Both models share a single Hydra training entrypoint (`train_rnn.py`), selected via `model=lstm` or `model=gru` in the config. They consume the same 3-D walk-forward `Fold` data (sequence format, NOT flat) produced by the existing `build_folds()` generator.

**Architecture:** Each model is an encoder stack (1–2 layers of LSTM or GRU) feeding into a `LayerNorm` → `Linear(hidden_dim, 4)` classification head with softmax output. Training uses Adam + cross-entropy + early stopping on val loss (patience configurable). The Hydra entrypoint mirrors `train_baseline.py` exactly in structure: data → features → labels → folds → train → eval → plot → CSV summary.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, existing project stack (hydra-core, src.* modules already built)

---

## File Map

| File | Responsibility |
|---|---|
| `src/models/baseline_lstm.py` | `RNNConfig` dataclass + `RegimeLSTM` and `RegimeGRU` model classes |
| `src/training/train_rnn.py` | Hydra entrypoint for LSTM/GRU walk-forward training (mirrors `train_baseline.py`) |
| `configs/model/lstm.yaml` | LSTM hyperparameters (`# @package model`) |
| `configs/model/gru.yaml` | GRU hyperparameters (`# @package model`) |
| `tests/test_shapes.py` | Extend with LSTM/GRU shape, forward-pass, and proba-sum tests |

---

## Existing Code to Know

**`src/utils/dataset_builder.py`** — `build_folds(features, labels, cfg, window_len, flat)`. When `flat=False`, yields `Fold` with:
- `train_X: np.ndarray shape (N, window_len, n_features)` dtype float32
- `train_y: np.ndarray shape (N,)` dtype int64
- Same for `val_X/val_y` and `test_X/test_y`
- `fold.train_dates`, `fold.val_dates`, `fold.test_dates`: `pd.DatetimeIndex`

**`src/evaluation/metrics.py`** — `evaluate(y_true, y_pred, y_prob) -> EvalResult` where `y_prob` is `(N, 4)` float32 numpy. `EvalResult.macro_f1`, `.brier_score`, `.ece`, `.switch_frequency`, `.mean_entropy`, `.balanced_accuracy`, `.confusion_matrix`.

**`src/training/train_baseline.py`** — Use as structural template. Key pattern:
```python
project_root = Path(hydra.utils.get_original_cwd())
dm = DataManager(raw_dir=project_root / cfg.data.raw_dir, ...)
figures_dir = project_root / cfg.figures_dir
```
Hydra changes CWD to output dir, so ALL data/figure paths must be resolved via `project_root`.

**`configs/config.yaml`** — defaults list references `model: xgb`. The RNN entrypoint will use `configs/model/lstm.yaml` or `configs/model/gru.yaml` directly (not overriding the global config default — user passes `python -m src.training.train_rnn model=lstm`).

**`configs/model/xgb.yaml`** — starts with `# @package model`. The LSTM/GRU yamls must do the same.

**`tests/test_shapes.py`** — currently has XGBoost tests only. LSTM/GRU tests will be appended to the same file.

---

## Task 14: LSTM and GRU Model Classes

**Files:**
- Create: `src/models/baseline_lstm.py`
- Create: `configs/model/lstm.yaml`
- Create: `configs/model/gru.yaml`

### Design

`RNNConfig` dataclass holds all hyperparameters shared by both architectures. `RegimeLSTM` and `RegimeGRU` are thin `nn.Module` wrappers with identical external interface:
- `forward(x: Tensor) -> Tensor` where `x` is `(batch, seq_len, n_features)` → returns `(batch, 4)` raw logits
- `predict(X: np.ndarray) -> np.ndarray` → class indices `(N,)`
- `predict_proba(X: np.ndarray) -> np.ndarray` → softmax probabilities `(N, 4)`, sums to 1

Both models: RNN encoder → `LayerNorm(hidden_dim)` → `Linear(hidden_dim, 4)`. The final hidden state of the last layer is the sequence representation (take the last timestep output).

- [ ] **Step 1: Write failing tests in `tests/test_shapes.py`**

Append the following to `tests/test_shapes.py` (after the existing XGBoost tests):

```python
import torch
from src.models.baseline_lstm import RegimeLSTM, RegimeGRU, RNNConfig


@pytest.fixture
def rnn_cfg():
    return RNNConfig(
        input_dim=37,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        seed=42,
    )


@pytest.fixture
def toy_seq_data():
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    seq_len, n_feat = 40, 37
    X_train = rng.standard_normal((n_train, seq_len, n_feat)).astype(np.float32)
    y_train = rng.integers(0, 4, n_train)
    X_val = rng.standard_normal((n_val, seq_len, n_feat)).astype(np.float32)
    y_val = rng.integers(0, 4, n_val)
    return X_train, y_train, X_val, y_val


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_forward_output_shape(rnn_cfg, ModelClass):
    model = ModelClass(rnn_cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_shape(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_proba_shape(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_proba_sums_to_one(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_in_range(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd c:\Users\dhruv\projects\lagrange
pytest tests/test_shapes.py -k "rnn" -v
```

Expected: `ImportError: cannot import name 'RegimeLSTM' from 'src.models.baseline_lstm'`

- [ ] **Step 3: Implement `src/models/baseline_lstm.py`**

```python
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
        last = out[:, -1, :]          # (batch, hidden_dim)
        return self.head(self.norm(last))   # (batch, 4) — raw logits

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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_shapes.py -k "rnn" -v
```

Expected: 10 tests pass (5 parametrized × 2 model classes).

- [ ] **Step 5: Create `configs/model/lstm.yaml`**

```yaml
# @package model
name: lstm
hidden_dim: 64
num_layers: 2
dropout: 0.1
batch_size: 64
lr: 0.001
max_epochs: 100
patience: 10
device: cpu
```

- [ ] **Step 6: Create `configs/model/gru.yaml`**

```yaml
# @package model
name: gru
hidden_dim: 64
num_layers: 2
dropout: 0.1
batch_size: 64
lr: 0.001
max_epochs: 100
patience: 10
device: cpu
```

- [ ] **Step 7: Commit**

```bash
git add src/models/baseline_lstm.py configs/model/lstm.yaml configs/model/gru.yaml tests/test_shapes.py
git commit -m "feat: RegimeLSTM and RegimeGRU models with shared RNNConfig"
```

---

## Task 15: Shared RNN Training Entrypoint

**Files:**
- Create: `src/training/train_rnn.py`

### Design

`train_rnn.py` is a Hydra entrypoint. It:
1. Loads data/features/labels exactly like `train_baseline.py`
2. Calls `build_folds(..., flat=False)` to get 3-D sequence data
3. For each fold: instantiates the right model class (LSTM or GRU) from `cfg.model.name`
4. Trains with Adam + `nn.CrossEntropyLoss()` using a mini-batch loop with early stopping on val loss
5. Evaluates on test split using `evaluate()` 
6. Writes per-fold metrics JSON + plots exactly like `train_baseline.py`
7. Writes `walk_forward_summary.csv` to Hydra output dir

The training loop uses `torch.utils.data.DataLoader` for mini-batching. Early stopping tracks val loss with `patience` from config.

**Run with:**
```bash
python -m src.training.train_rnn model=lstm
python -m src.training.train_rnn model=gru
```

Note: `configs/config.yaml` still defaults to `model: xgb`. The RNN entrypoint accepts `model=lstm` or `model=gru` as an override on the CLI.

- [ ] **Step 1: Implement `src/training/train_rnn.py`**

```python
"""LSTM/GRU walk-forward training entrypoint.

Run with:
    python -m src.training.train_rnn model=lstm
    python -m src.training.train_rnn model=gru
    python -m src.training.train_rnn model=lstm splits.val_size=126 model.hidden_dim=128
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
from src.models.baseline_lstm import RegimeGRU, RegimeLSTM, RNNConfig
from src.utils.dataset_builder import SplitConfig, build_folds
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_summary,
    plot_regime_timeline,
)

log = logging.getLogger(__name__)

_MODEL_REGISTRY = {"lstm": RegimeLSTM, "gru": RegimeGRU}


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
    """Train model for one fold. Returns trained model."""
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(cfg.model.max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_loss = criterion(val_logits, y_va).item()

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

    model_name = cfg.model.name
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(_MODEL_REGISTRY)}")
    ModelClass = _MODEL_REGISTRY[model_name]

    device = torch.device(cfg.model.device if hasattr(cfg.model, "device") else "cpu")

    # --- Data ---
    dm = DataManager(
        raw_dir=project_root / cfg.data.raw_dir,
        processed_dir=project_root / cfg.data.processed_dir,
        tickers=list(cfg.data.tickers),
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    prices = fetch_all(dm)

    # --- Features ---
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

    # --- Labels ---
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

    # --- Split config ---
    split_cfg = SplitConfig(
        train_start=cfg.splits.train_start,
        val_size=cfg.splits.val_size,
        test_size=cfg.splits.test_size,
        step_size=cfg.splits.step_size,
        min_train_size=cfg.splits.min_train_size,
    )

    # --- RNN config ---
    rnn_cfg = RNNConfig(
        input_dim=n_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        seed=cfg.seed,
        batch_size=cfg.model.batch_size,
        lr=cfg.model.lr,
        max_epochs=cfg.model.max_epochs,
        patience=cfg.model.patience,
        device=str(device),
    )

    output_dir = Path(".")
    figures_dir = project_root / cfg.figures_dir / model_name

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
        model = ModelClass(rnn_cfg)
        model = _train_fold(model, fold, cfg, device)

        y_pred = model.predict(fold.test_X)
        y_prob = model.predict_proba(fold.test_X)
        result = evaluate(fold.test_y, y_pred, y_prob)

        metrics_dict = {
            "fold_id": fold.fold_id,
            "model": model_name,
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

        n = len(y_pred)
        test_dates = fold.test_dates[:n]
        plot_regime_timeline(
            test_dates, fold.test_y[:n], y_pred,
            figures_dir / f"fold_{fold.fold_id:02d}_timeline.png",
        )
        plot_confusion_matrix(
            result.confusion_matrix, f"Fold {fold.fold_id} ({model_name})",
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
    log.info(f"\nWalk-Forward Summary ({model_name}):\n{summary_stats.to_string()}")

    plot_fold_summary(
        all_fold_ids, summary_df["macro_f1"].tolist(),
        save_path=figures_dir / "fold_summary.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the entrypoint at least imports cleanly**

```bash
cd c:\Users\dhruv\projects\lagrange
python -c "import src.training.train_rnn; print('OK')"
```

Expected: `OK` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add src/training/train_rnn.py
git commit -m "feat: shared LSTM/GRU walk-forward training entrypoint with early stopping"
```

---

## Task 16: Full Test Suite Pass + Smoke Test

**Files:**
- Modify: `tests/test_shapes.py` — already done in Task 14

- [ ] **Step 1: Run full test suite**

```bash
cd c:\Users\dhruv\projects\lagrange
pytest tests/ -v --tb=short
```

Expected: All prior tests + 10 new RNN tests pass. 51+ tests total.

- [ ] **Step 2: Smoke-test LSTM training on synthetic data**

The synthetic data in `data/raw/` from the MVP smoke test should already exist. If not, regenerate it:

```bash
python -c "
from pathlib import Path
from tests.conftest import _make_ohlcv
from src.data.manager import DataManager
tickers = ['SPY', 'QQQ', 'TLT', 'GLD', '^VIX']
dm = DataManager(raw_dir=Path('data/raw'), processed_dir=Path('data/processed'),
    tickers=tickers, start_date='2010-01-04', end_date='2013-12-31')
for i, t in enumerate(tickers):
    dm.save_raw(t, _make_ohlcv(800, seed=i))
print('Synthetic data written.')
"
```

Then run LSTM training:

```bash
python -m src.training.train_rnn model=lstm \
  "data.start_date=2010-01-04" \
  "data.end_date=2013-12-31" \
  "splits.min_train_size=200" \
  "splits.val_size=50" \
  "splits.test_size=50" \
  "splits.step_size=25" \
  "model.max_epochs=3" \
  "model.hidden_dim=16" \
  "model.num_layers=1" \
  "model.batch_size=32" \
  "model.patience=2" \
  "seed=42"
```

Expected: training completes, `walk_forward_summary.csv` written, no exceptions.

- [ ] **Step 3: Smoke-test GRU training on same data**

```bash
python -m src.training.train_rnn model=gru \
  "data.start_date=2010-01-04" \
  "data.end_date=2013-12-31" \
  "splits.min_train_size=200" \
  "splits.val_size=50" \
  "splits.test_size=50" \
  "splits.step_size=25" \
  "model.max_epochs=3" \
  "model.hidden_dim=16" \
  "model.num_layers=1" \
  "model.batch_size=32" \
  "model.patience=2" \
  "seed=42"
```

Expected: same as LSTM smoke test — completes without exception.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: LSTM/GRU baselines complete — models, configs, shared trainer, 10 new shape tests"
```

---

## Self-Review Against Spec

**Spec section coverage:**

| Spec Requirement | Covered by |
|---|---|
| LSTM baseline model | Task 14: `RegimeLSTM` in `baseline_lstm.py` |
| GRU baseline model | Task 14: `RegimeGRU` in `baseline_lstm.py` |
| Shared trainer | Task 15: `train_rnn.py` |
| Hydra config integration | Task 14: `lstm.yaml` + `gru.yaml` |
| Walk-forward evaluation | Task 15: calls `build_folds(..., flat=False)` |
| Early stopping | Task 15: val-loss patience loop in `_train_fold()` |
| Same eval metrics as XGBoost | Task 15: calls `evaluate()` from `src.evaluation.metrics` |
| Same plots as XGBoost | Task 15: `plot_regime_timeline`, `plot_confusion_matrix`, `plot_fold_summary` |
| Shape tests | Task 14: 10 parametrized tests in `test_shapes.py` |

**Type consistency:**
- `RNNConfig.input_dim` set from `features.shape[1]` in `train_rnn.py` ✓
- `build_folds(..., flat=False)` produces `(N, window_len, n_features)` — matches `RegimeLSTM/GRU.forward(x)` expecting `(batch, seq, feat)` ✓
- `model.predict_proba(X)` returns `(N, 4)` float32 numpy — matches `evaluate()` signature ✓
- Both models implement `.predict()` and `.predict_proba()` — same interface as `RegimeXGB` ✓

**Potential issues resolved:**
- `dropout` in `nn.LSTM/GRU` raises a warning when `num_layers=1` — handled by setting `dropout=0.0` when `num_layers <= 1` ✓
- `cfg.model.device` may not exist if user uses default xgb config — guarded with `hasattr` ✓
- Figures go to `reports/figures/lstm/` or `reports/figures/gru/` to avoid clobbering XGBoost figures ✓
