# Lagrangian Latent Regime Networks (LLRN)

Research project testing whether latent states evolved via learned Lagrangian
dynamics produce better-calibrated market regime forecasts than standard baselines.

## Regime Taxonomy

4 mutually exclusive classes (2×2 cross of return direction × volatility level):

| ID | Name | Return | Volatility |
|----|------|--------|-----------|
| 0 | Bull/Calm | Positive | Low |
| 1 | Bull/Stress | Positive | High |
| 2 | Bear/Calm | Negative | Low |
| 3 | Bear/Stress | Negative | High |

## Quickstart

```bash
pip install -r requirements.txt

# Download data and run XGBoost baseline walk-forward evaluation
python -m src.training.train_baseline

# Override config from CLI
python -m src.training.train_baseline labels.horizon=10 model.max_depth=4
```

## Project Structure

```
src/data/          — DataManager, yfinance download (pure pandas)
src/features/      — Causal feature engineering (pure pandas/numpy)
src/labels/        — Regime labeling (QuantileLabeler)
src/utils/         — Dataset builder, walk-forward splits, reproducibility
src/models/        — XGBoost, LSTM, GRU, NODE, Lagrangian models
src/training/      — Training entrypoints (Hydra)
src/evaluation/    — Metrics, walk-forward aggregation
src/visualization/ — Plots
```

## Research Framing

This is regime forecasting for risk-aware temporal representation learning.
Not a trading bot. Not a price prediction system. No profitability claims.
