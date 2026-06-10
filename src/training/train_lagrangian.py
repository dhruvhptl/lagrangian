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
        use_vector_damping=getattr(cfg.model, 'use_vector_damping', False),
        use_coord_transform=getattr(cfg.model, 'use_coord_transform', False),
        eps=cfg.model.eps,
        seed=cfg.seed,
        batch_size=cfg.model.batch_size,
        lr=cfg.model.lr,
        max_epochs=cfg.model.max_epochs,
        patience=cfg.model.patience,
        device=str(device),
        encoder_type=getattr(cfg.model, 'encoder_type', 'mlp'),
        encoder_dim=getattr(cfg.model, 'encoder_dim', 64),
        conv_channels=getattr(cfg.model, 'conv_channels', 64),
        conv_kernel_size=getattr(cfg.model, 'conv_kernel_size', 3),
        tcn_channels=getattr(cfg.model, 'tcn_channels', 64),
        tcn_kernel_size=getattr(cfg.model, 'tcn_kernel_size', 3),
        tcn_dilations=list(getattr(cfg.model, 'tcn_dilations', [1, 2, 4, 8])),
    )

    output_dir = Path(".")
    figures_dir = project_root / cfg.figures_dir / MODEL_NAME

    all_metrics: list[dict] = []
    all_fold_ids: list[int] = []

    fold_start = getattr(cfg, 'fold_start', None)
    fold_end = getattr(cfg, 'fold_end', None)

    for fold in build_folds(
        features,
        labels,
        split_cfg,
        window_len=cfg.data.window_len,
        flat=False,
    ):
        if fold_start is not None and fold.fold_id < fold_start:
            continue
        if fold_end is not None and fold.fold_id > fold_end:
            break

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
