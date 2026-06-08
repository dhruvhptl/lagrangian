# tests/test_splits.py
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.features.engineer import FeaturesConfig, build_features
from src.labels.quantile_labeler import LabelConfig, QuantileLabeler
from src.utils.dataset_builder import SplitConfig, build_folds, Fold


@pytest.fixture
def features_and_labels(synthetic_prices):
    fcfg = FeaturesConfig(roll_windows=[5, 10], momentum_windows=[5], corr_windows=[10],
                          cross_assets=["QQQ"], primary_asset="SPY")
    lcfg = LabelConfig(horizon=5, vol_window=10, smoothing=False)
    feats = build_features(synthetic_prices, fcfg)
    labeler = QuantileLabeler(lcfg)
    label_df = labeler.fit_transform(synthetic_prices["SPY"])
    return feats, label_df["label"]


@pytest.fixture
def split_cfg():
    return SplitConfig(
        train_start="2010-01-04",
        val_size=50,
        test_size=50,
        step_size=25,
        min_train_size=100,
    )


def test_fold_is_dataclass(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    folds = list(build_folds(feats, labels, split_cfg, window_len=10, flat=True))
    assert len(folds) > 0
    assert isinstance(folds[0], Fold)


def test_no_date_overlap_train_val(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    for fold in build_folds(feats, labels, split_cfg, window_len=10, flat=True):
        assert len(set(fold.train_dates) & set(fold.val_dates)) == 0


def test_no_date_overlap_val_test(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    for fold in build_folds(feats, labels, split_cfg, window_len=10, flat=True):
        assert len(set(fold.val_dates) & set(fold.test_dates)) == 0


def test_no_date_overlap_train_test(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    for fold in build_folds(feats, labels, split_cfg, window_len=10, flat=True):
        assert len(set(fold.train_dates) & set(fold.test_dates)) == 0


def test_folds_ordered_in_time(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    folds = list(build_folds(feats, labels, split_cfg, window_len=10, flat=True))
    for i in range(1, len(folds)):
        assert folds[i].train_dates.min() <= folds[i].train_dates.max()
        assert folds[i - 1].val_dates.max() < folds[i].val_dates.max()


def test_scaler_fit_on_train_only(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    fold = next(build_folds(feats, labels, split_cfg, window_len=10, flat=True))
    # Verify scaler was fit on training data, not val data.
    # scaler.n_samples_seen_ tracks samples per feature (may be array for NaN-aware scalers).
    # The max should equal len(train_dates) and must not equal val_size.
    n_seen = int(np.max(fold.scaler.n_samples_seen_))
    assert n_seen == len(fold.train_dates), (
        f"Scaler n_samples_seen_ max ({n_seen}) != "
        f"len(train_dates) ({len(fold.train_dates)}); scaler was not fit on train only"
    )
    assert n_seen != split_cfg.val_size, \
        "Scaler appears to have been fit on val data"


def test_window_construction_correct_shape(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    window_len = 10
    fold = next(build_folds(feats, labels, split_cfg, window_len=window_len, flat=False))
    assert fold.train_X.ndim == 3
    assert fold.train_X.shape[1] == window_len
    assert fold.train_X.shape[2] == feats.shape[1]


def test_flat_format_shape(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    window_len = 10
    fold = next(build_folds(feats, labels, split_cfg, window_len=window_len, flat=True))
    assert fold.train_X.ndim == 2
    assert fold.train_X.shape[1] == window_len * feats.shape[1]


def test_labels_are_integers(features_and_labels, split_cfg):
    feats, labels = features_and_labels
    fold = next(build_folds(feats, labels, split_cfg, window_len=10, flat=True))
    assert fold.train_y.dtype in (np.int32, np.int64, int)
    assert set(fold.train_y).issubset({0, 1, 2, 3})
