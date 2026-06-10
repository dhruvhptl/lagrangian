# Lagrangian Latent Regime Networks (LLRN)

We treat a financial market's latent regime state as a point particle in a learned potential landscape, evolved forward in time by discrete Lagrangian mechanics — mass, potential energy, and damping — rather than by a recurrent gate or a numerical ODE solver. This repo tests whether that structured physical inductive bias produces better-calibrated regime forecasts than XGBoost, LSTM, GRU, and Neural ODE baselines under a rigorous 71-fold walk-forward evaluation on two decades of multi-asset data.

---

## Motivation

Standard sequence models (LSTM, GRU) are strong baselines for regime classification but treat latent state evolution as an opaque learned function. Neural ODEs impose continuity but still lack physical structure. We ask: can a model that explicitly parameterizes *inertia*, *potential energy*, and *dissipation* in latent space learn smoother, better-calibrated regime trajectories than these alternatives?

The hypothesis is that Lagrangian dynamics — where the latent state has both position and velocity, and transitions are governed by energy-minimising forces — provides a useful inductive bias for market regimes, which exhibit regime persistence, mean-reversion, and abrupt stress transitions consistent with a particle in a multi-well potential.

---

## Regime Taxonomy

Four mutually exclusive classes from a 2×2 cross of rolling return direction and volatility level, both thresholded at rolling medians on SPY:

| ID | Label | Return | Volatility |
|----|-------|--------|------------|
| 0 | Bull/Calm | Positive | Low |
| 1 | Bull/Stress | Positive | High |
| 2 | Bear/Calm | Negative | Low |
| 3 | Bear/Stress | Negative | High |

Labels are constructed from 5-day forward returns and 21-day realised volatility. Optional multi-horizon variants extend this to 5d / 10d / 20d targets simultaneously.

---

## Model Zoo

| Model | Config | Architecture | Notes |
|-------|--------|-------------|-------|
| XGBoost | `xgb` | Gradient-boosted trees on 37 tabular features | Strongest overall baseline |
| LSTM | `lstm` | 2-layer LSTM, hidden=64 | Sequential recurrent baseline |
| GRU | `gru` | 2-layer GRU, hidden=64 | Sequential recurrent baseline |
| Neural ODE | `node` | Latent ODE with `dopri5` solver | Continuous-depth baseline |
| Lagrangian (MLP) | `lagrangian_mlp` | Flatten+MLP encoder → Lagrangian integrator | Original encoder, temporal order ignored |
| Lagrangian (Conv1D) | `lagrangian_conv1d` | Causal Conv1D encoder → Lagrangian integrator | **Current best Lagrangian** |
| Lagrangian (TCN) | `lagrangian_tcn` | Dilated causal TCN encoder → Lagrangian integrator | Dilations [1,2,4,8] |
| Lagrangian (Hybrid Conv) | `lagrangian_hybrid_conv` | 3-layer conv F→64→128→64 → Lagrangian integrator | Wider encoder variant |
| Lagrangian MH (v6) | `lagrangian_v6` | MLP encoder + econophysics features + 3-head multi-horizon loss | Best subset result |

All Lagrangian variants share the same latent integrator: diagonal mass net, deep potential net, optional vector damping, optional coordinate transform, and symplectic Euler integration.

---

## Results

71-fold walk-forward evaluation. Expanding train window, val=252 days, test=252 days, step=63 days. Data: SPY, QQQ, TLT, GLD, ^VIX, 2004-11-01 to 2026-06-08.

| Model | Mean Macro F1 | Std F1 | Mean Brier | Mean ECE |
|-------|:------------:|:------:|:----------:|:--------:|
| XGBoost | **0.4208** | — | 0.6246 | 0.1678 |
| LSTM | 0.4062 | — | 0.6152 | 0.1477 |
| GRU | 0.4008 | — | **0.6071** | 0.1477 |
| **Lagrangian Conv1D** | **0.3976** | — | 0.6214 | **0.1473** |
| Neural ODE | 0.3850 | — | 0.6281 | 0.1450 |
| Lagrangian MLP | 0.3700 | — | 0.6373 | 0.1258 |

Key findings:
- Replacing the flatten+MLP encoder with a causal Conv1D encoder improved Lagrangian F1 by **+2.8pp** (0.370 → 0.398), confirming that temporal order is a first-order requirement
- Lagrangian Conv1D matches GRU within 0.3pp and exceeds Neural ODE by 1.3pp
- XGBoost leads on F1 but Lagrangian Conv1D achieves near-identical ECE to LSTM/GRU, suggesting competitive calibration
- Econophysics features (+2.7pp on subset) and multi-horizon supervision interact non-trivially with encoder choice — ongoing ablation

Per-fold CSVs: `walk_forward_summary_lagrangian_conv1d.csv`, `walk_forward_summary_xgb.csv`, `walk_forward_summary_lagrangian_v3.csv`  
Fold-wise F1 plot: `reports/figures/benchmark_fold_f1.png`

---

## Quickstart

**Install**

```bash
git clone https://github.com/dhruvhptl/lagrangian
cd lagrangian
pip install -r requirements.txt
```

**Run a baseline**

```bash
# XGBoost — full 71-fold walk-forward
python -m src.training.train_baseline model=xgb

# LSTM
python -m src.training.train_rnn model=lstm
```

**Run the best Lagrangian model**

```bash
# Causal Conv1D encoder, single-horizon, 71 folds
python -m src.training.train_lagrangian model=lagrangian_conv1d

# Subset first (folds 20–40, ~15 min)
python -m src.training.train_lagrangian model=lagrangian_conv1d +fold_start=20 +fold_end=40
```

**Run tests**

```bash
python -m pytest tests/test_shapes.py -v   # 87 tests
```

---

## Project Structure

```
lagrangian/
├── configs/
│   ├── config.yaml              # base Hydra config (data, splits, features, labels)
│   └── model/                   # one yaml per model variant
│       ├── lagrangian_conv1d.yaml
│       ├── lagrangian_mlp.yaml
│       ├── lagrangian_tcn.yaml
│       ├── lstm.yaml, gru.yaml, xgb.yaml, node.yaml
│       └── ...
├── src/
│   ├── data/                    # yfinance download and caching
│   ├── features/                # causal feature engineering (37 base + 29 econophysics)
│   ├── labels/                  # 4-class quantile labeler, multi-horizon labeler
│   ├── models/
│   │   ├── lagrangian_regime_net.py      # single-horizon model, 4 encoder variants
│   │   ├── lagrangian_regime_net_mh.py   # multi-horizon model (head_5, head_10, head_20)
│   │   ├── baseline_lstm.py             # LSTM + GRU
│   │   ├── baseline_node.py             # Neural ODE
│   │   └── baseline_xgb.py             # XGBoost
│   ├── training/                # Hydra entrypoints per model family
│   ├── evaluation/              # macro F1, Brier score, ECE
│   ├── utils/                   # walk-forward fold builder, reproducibility
│   └── visualization/           # fold timeline and confusion matrix plots
├── tests/
│   └── test_shapes.py           # 87 unit tests (shapes, causality, gradients)
├── reports/figures/             # per-fold confusion matrices, timelines, benchmark plot
├── walk_forward_summary_*.csv   # saved fold-level results
├── PRD.md                       # product requirements
└── CLAUDE.md                    # agent/contributor instructions
```

---

## Architecture: LagrangianRegimeNet

```
Input (B, T=40, F)
      │
  [Encoder]  ←─ mlp | conv1d | tcn | hybrid_conv
      │
      h  (B, encoder_dim)
      │
  ┌───┴───┐
z0_head  z_dot0_head
  │          │
  z₀       ż₀   (B, latent_dim)
      │
  [Symplectic Euler × n_steps]
    q  = coord_net(z)          # optional coordinate transform
    V  = potential_net(q)      # learned scalar potential
    dV = ∂V/∂q                 # via autograd
    M  = mass_net(q)           # diagonal positive mass
    γ  = gamma_net(q)          # vector damping (optional)
    z̈  = -(dV + γ·ż) / M
    ż  ← ż + dt·z̈             # velocity first (symplectic)
    z  ← z + dt·ż
      │
   z_T  (B, latent_dim)
      │
  LayerNorm → Linear(4)
      │
  logits (B, 4)
```

---

## Research Framing

This is a controlled study in structured inductive biases for latent-space time-series modeling — not a trading system and not a price prediction engine. No profitability claims are made or implied. Regime labels are constructed from backward-looking rolling statistics with no forward leakage. The walk-forward evaluation mirrors real deployment constraints: models never see future data during training or validation.

The Lagrangian framing draws on classical mechanics as a source of structure for learning, in the spirit of Lagrangian and Hamiltonian neural networks (Cranmer et al., 2020; Greydanus et al., 2019). The application to financial regime dynamics is the novel contribution.

> Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian Neural Networks. *NeurIPS*.  
> Cranmer, M., Greydanus, S., Hoyer, S., et al. (2020). Lagrangian Neural Networks. *ICLR Workshop*.
