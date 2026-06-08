"""XGBoost walk-forward training entrypoint.

Run with:
    python -m src.training.train_baseline
    python -m src.training.train_baseline model.n_estimators=100 labels.horizon=10
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.download import fetch_all
from src.data.manager import DataManager
from src.evaluation.metrics import evaluate
from src.features.engineer import FeaturesConfig, build_features
from src.labels.quantile_labeler import LabelConfig, QuantileLabeler
from src.models.baseline_xgb import RegimeXGB, XGBConfig
from src.utils.dataset_builder import SplitConfig, build_folds
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_fold_summary,
    plot_regime_timeline,
)

log = logging.getLogger(__name__)


def _get_labels(
    spy_prices: pd.DataFrame,
    label_cfg: LabelConfig,
    feature_index: pd.DatetimeIndex,
) -> pd.Series:
    """Fit QuantileLabeler on full data and return label series aligned to feature_index."""
    labeler = QuantileLabeler(label_cfg)
    label_df = labeler.fit_transform(spy_prices)
    return label_df["label"].reindex(feature_index)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Hydra changes CWD — resolve project root from original CWD
    project_root = Path(hydra.utils.get_original_cwd())

    set_global_seed(cfg.seed)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Data ---
    dm = DataManager(
        raw_dir=project_root / cfg.data.raw_dir,
        processed_dir=project_root / cfg.data.processed_dir,
        tickers=list(cfg.data.tickers),
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    prices = fetch_all(dm)
    log.info(f"Data hashes: {dm.log_hashes()}")

    # --- Features ---
    feat_cfg = FeaturesConfig(
        roll_windows=list(cfg.features.roll_windows),
        momentum_windows=list(cfg.features.momentum_windows),
        corr_windows=list(cfg.features.corr_windows),
        cross_assets=list(cfg.features.cross_assets),
        primary_asset=cfg.features.primary_asset,
    )
    features = build_features(prices, feat_cfg)
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

    # --- XGBoost config ---
    xgb_cfg = XGBConfig(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        eval_metric=cfg.model.eval_metric,
        seed=cfg.seed,
        n_jobs=cfg.model.n_jobs,
    )

    output_dir = Path(".")  # Hydra sets CWD to output dir
    figures_dir = project_root / cfg.figures_dir

    all_metrics: list[dict] = []
    all_fold_ids: list[int] = []
    all_importances: list[np.ndarray] = []
    feature_names = list(features.columns)

    for fold in build_folds(
        features,
        labels,
        split_cfg,
        window_len=cfg.data.window_len,
        flat=True,
    ):
        log.info(
            f"Fold {fold.fold_id}: "
            f"train={len(fold.train_y)} val={len(fold.val_y)} test={len(fold.test_y)}"
        )

        model = RegimeXGB(xgb_cfg)
        model.fit(fold.train_X, fold.train_y, fold.val_X, fold.val_y)

        y_pred = model.predict(fold.test_X)
        y_prob = model.predict_proba(fold.test_X)
        result = evaluate(fold.test_y, y_pred, y_prob)

        metrics_dict = {
            "fold_id": fold.fold_id,
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
        all_importances.append(model.feature_importances())

        fold_dir = output_dir / f"fold_{fold.fold_id:02d}"
        fold_dir.mkdir(exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        n = len(y_pred)
        test_dates = fold.test_dates[:n]
        plot_regime_timeline(
            test_dates, fold.test_y[:n], y_pred,
            figures_dir / f"fold_{fold.fold_id:02d}_timeline.png",
        )
        fig_cm = plot_confusion_matrix(
            result.confusion_matrix, f"Fold {fold.fold_id}",
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

    # --- Aggregate ---
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(output_dir / "walk_forward_summary.csv", index=False)

    numeric_cols = ["macro_f1", "balanced_accuracy", "brier_score", "ece"]
    summary_stats = summary_df[numeric_cols].agg(["mean", "std"])
    log.info(f"\nWalk-Forward Summary:\n{summary_stats.to_string()}")

    mean_importance = np.mean(all_importances, axis=0)
    plot_feature_importance(
        mean_importance, feature_names,
        save_path=figures_dir / "feature_importance.png",
    )
    plot_fold_summary(
        all_fold_ids, summary_df["macro_f1"].tolist(),
        save_path=figures_dir / "fold_summary.png",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
