"""Walk-forward builder that yields multi-horizon label arrays alongside features.

This is a drop-in companion to dataset_builder.py for the multi-horizon track.
It yields MultiHorizonFold objects that carry label arrays for each horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.dataset_builder import SplitConfig, _make_windows


@dataclass
class MultiHorizonFold:
    fold_id: int
    train_X: np.ndarray           # (N, window_len, n_features)
    train_y: dict[int, np.ndarray]  # {5: (N,), 10: (N,), 20: (N,)}
    val_X: np.ndarray
    val_y: dict[int, np.ndarray]
    test_X: np.ndarray
    test_y: dict[int, np.ndarray]
    scaler: StandardScaler
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def _make_windows_multi(
    features: np.ndarray,
    labels_dict: dict[int, np.ndarray],
    window_len: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Slide a window over features + multiple label arrays.

    A sample is included only if ALL horizons have a valid (non-NaN) label
    AND the feature window has no NaNs.
    """
    horizons = list(labels_dict.keys())
    X_list: list[np.ndarray] = []
    y_lists: dict[int, list[int]] = {h: [] for h in horizons}

    n = len(features)
    for i in range(window_len - 1, n):
        # Check all labels valid
        all_valid = all(not np.isnan(labels_dict[h][i]) for h in horizons)
        if not all_valid:
            continue
        window = features[i - window_len + 1: i + 1]
        if np.isnan(window).any():
            continue
        X_list.append(window)
        for h in horizons:
            y_lists[h].append(int(labels_dict[h][i]))

    if not X_list:
        shape = (0, window_len, features.shape[1])
        return np.empty(shape, dtype=np.float32), {h: np.empty((0,), dtype=np.int64) for h in horizons}

    X = np.array(X_list, dtype=np.float32)
    y = {h: np.array(y_lists[h], dtype=np.int64) for h in horizons}
    return X, y


def build_folds_multi(
    features: pd.DataFrame,
    labels_df: pd.DataFrame,   # columns: label_5, label_10, label_20
    cfg: SplitConfig,
    horizons: list[int],
    window_len: int = 40,
) -> Generator[MultiHorizonFold, None, None]:
    """Yield MultiHorizonFold with expanding train, fixed val/test."""
    all_dates = features.index
    train_start = pd.Timestamp(cfg.train_start)
    date_array = all_dates[all_dates >= train_start]

    fold_id = 0
    n = len(date_array)
    val_start_pos = cfg.min_train_size

    while val_start_pos + cfg.val_size + cfg.test_size <= n:
        train_end_pos = val_start_pos
        val_end_pos = val_start_pos + cfg.val_size
        test_end_pos = val_end_pos + cfg.test_size

        train_dates = date_array[:train_end_pos]
        val_dates = date_array[train_end_pos:val_end_pos]
        test_dates = date_array[val_end_pos:test_end_pos]

        feat_train = features.loc[train_dates].values.astype(np.float32)
        feat_val = features.loc[val_dates].values.astype(np.float32)
        feat_test = features.loc[test_dates].values.astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat_train = scaler.transform(feat_train)
        feat_val = scaler.transform(feat_val)
        feat_test = scaler.transform(feat_test)

        def _labels_dict(dates: pd.DatetimeIndex) -> dict[int, np.ndarray]:
            return {
                h: labels_df[f"label_{h}"].reindex(dates).values.astype(np.float32)
                for h in horizons
            }

        train_X, train_y = _make_windows_multi(feat_train, _labels_dict(train_dates), window_len)
        val_X, val_y = _make_windows_multi(feat_val, _labels_dict(val_dates), window_len)
        test_X, test_y = _make_windows_multi(feat_test, _labels_dict(test_dates), window_len)

        if any(len(train_y[h]) == 0 for h in horizons):
            val_start_pos += cfg.step_size
            continue
        if any(len(val_y[h]) == 0 for h in horizons):
            val_start_pos += cfg.step_size
            continue
        if any(len(test_y[h]) == 0 for h in horizons):
            val_start_pos += cfg.step_size
            continue

        yield MultiHorizonFold(
            fold_id=fold_id,
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            scaler=scaler,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
        )

        fold_id += 1
        val_start_pos += cfg.step_size
