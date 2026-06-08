"""Walk-forward dataset builder with expanding train window."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitConfig:
    train_start: str = "2000-01-01"
    val_size: int = 252
    test_size: int = 252
    step_size: int = 63
    min_train_size: int = 504


@dataclass
class Fold:
    fold_id: int
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    label_meta: pd.DataFrame


def _make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_len: int,
    flat: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over features/labels. Drop NaN-label and NaN-feature rows."""
    X_list, y_list = [], []
    n = len(features)
    for i in range(window_len - 1, n):
        y = labels[i]
        if np.isnan(y):
            continue
        window = features[i - window_len + 1 : i + 1]  # (window_len, n_feats)
        if np.isnan(window).any():
            continue
        X_list.append(window.flatten() if flat else window)
        y_list.append(int(y))
    if not X_list:
        shape = (0, features.shape[1] * window_len) if flat else (0, window_len, features.shape[1])
        return np.empty(shape, dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def build_folds(
    features: pd.DataFrame,
    labels: pd.Series,
    cfg: SplitConfig,
    window_len: int = 40,
    flat: bool = False,
) -> Generator[Fold, None, None]:
    """Yield Fold dataclasses with expanding train, fixed val/test windows.

    Scaler is fit on train features only and applied to all splits.
    """
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

        lbl_train = labels.reindex(train_dates).values.astype(np.float32)
        lbl_val = labels.reindex(val_dates).values.astype(np.float32)
        lbl_test = labels.reindex(test_dates).values.astype(np.float32)

        # Fit scaler on train only
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat_train = scaler.transform(feat_train)
        feat_val = scaler.transform(feat_val)
        feat_test = scaler.transform(feat_test)

        train_X, train_y = _make_windows(feat_train, lbl_train, window_len, flat)
        val_X, val_y = _make_windows(feat_val, lbl_val, window_len, flat)
        test_X, test_y = _make_windows(feat_test, lbl_test, window_len, flat)

        if len(train_y) == 0 or len(val_y) == 0 or len(test_y) == 0:
            val_start_pos += cfg.step_size
            continue

        label_meta = pd.DataFrame(
            {"label": pd.concat([
                labels.reindex(train_dates),
                labels.reindex(val_dates),
                labels.reindex(test_dates),
            ])},
        )

        yield Fold(
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
            label_meta=label_meta,
        )

        fold_id += 1
        val_start_pos += cfg.step_size
