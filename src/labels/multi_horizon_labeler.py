"""Multi-horizon regime labels using the same 2x2 quadrant taxonomy.

Produces labels for horizons 5, 10, and 20 days simultaneously.
Uses the same QuantileLabeler logic per horizon — no leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.labels.quantile_labeler import LabelConfig, QuantileLabeler


@dataclass
class MultiHorizonLabelConfig:
    horizons: list[int] = field(default_factory=lambda: [5, 10, 20])
    vol_window: int = 21
    return_quantile: float = 0.5
    vol_quantile: float = 0.5
    smoothing: bool = True
    smoothing_min_periods: int = 3


class MultiHorizonLabeler:
    """Fits a QuantileLabeler per horizon and applies them together.

    Usage (mirrors QuantileLabeler API):
        labeler = MultiHorizonLabeler(cfg)
        labeler.fit(train_prices)
        labels_df = labeler.transform(prices)
        # labels_df has columns: label_5, label_10, label_20

    No look-ahead: thresholds are fit on training data only.
    The forward return for horizon h uses close.shift(-h), so the last h
    rows of each horizon will be NaN — this is intentional and correct.
    """

    def __init__(self, cfg: MultiHorizonLabelConfig) -> None:
        self.cfg = cfg
        self._labelers: dict[int, QuantileLabeler] = {}
        for h in cfg.horizons:
            label_cfg = LabelConfig(
                horizon=h,
                vol_window=cfg.vol_window,
                return_quantile=cfg.return_quantile,
                vol_quantile=cfg.vol_quantile,
                smoothing=cfg.smoothing,
                smoothing_min_periods=cfg.smoothing_min_periods,
            )
            self._labelers[h] = QuantileLabeler(label_cfg)

    def fit(self, data: pd.DataFrame) -> "MultiHorizonLabeler":
        """Fit thresholds on training data. data must have 'close' column."""
        for labeler in self._labelers.values():
            labeler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns label_5, label_10, label_20 (NaN where invalid)."""
        result = pd.DataFrame(index=data.index)
        for h, labeler in self._labelers.items():
            transformed = labeler.transform(data)
            result[f"label_{h}"] = transformed["label"]
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)
