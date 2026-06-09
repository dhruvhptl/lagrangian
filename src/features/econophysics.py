"""Econophysics-inspired rolling features for regime forecasting.

All features are computed causally — no look-ahead.
Returns a DataFrame with the same index as input prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_econophysics_features(
    prices: dict[str, pd.DataFrame],
    primary_asset: str = "SPY",
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute econophysics features from price data.

    Args:
        prices: dict mapping ticker -> OHLCV DataFrame (must have 'close' column)
        primary_asset: ticker to use as the primary series
        roll_windows: list of rolling window lengths (default: [21, 63])

    Returns:
        DataFrame with same index as primary asset prices, features as columns.
        NaNs at head where windows haven't filled.
    """
    if roll_windows is None:
        roll_windows = [21, 63]

    spy = prices[primary_asset]
    close = spy["close"].copy()
    log_ret = np.log(close / close.shift(1))

    parts: list[pd.Series] = []

    for w in roll_windows:
        min_p = w

        # 1. Realized volatility (annualized) — already in engineer.py but needed here for derived features
        rv = log_ret.rolling(w, min_periods=min_p).std() * np.sqrt(252)

        # 2. Volatility of volatility
        parts.append(rv.rolling(w, min_periods=min_p).std().rename(f"vol_of_vol_{w}"))

        # 3. Rolling kurtosis (excess) — fat tails proxy
        parts.append(log_ret.rolling(w, min_periods=min_p).kurt().rename(f"roll_kurtosis_{w}"))

        # 4. Rolling skewness
        parts.append(log_ret.rolling(w, min_periods=min_p).skew().rename(f"roll_skew_{w}"))

        # 5. Tail ratio: 95th pct / abs(5th pct) of rolling returns — asymmetry proxy
        q95 = log_ret.rolling(w, min_periods=min_p).quantile(0.95)
        q05 = log_ret.rolling(w, min_periods=min_p).quantile(0.05).abs()
        q05_safe = q05.where(q05 > 1e-8, np.nan)
        parts.append((q95 / q05_safe).rename(f"tail_ratio_{w}"))

        # 6. Rolling sign autocorrelation (lag-1) — return persistence / mean-reversion
        sign_ret = np.sign(log_ret)
        # autocorr = E[s_t * s_{t-1}]
        sign_autocorr = (sign_ret * sign_ret.shift(1)).rolling(w, min_periods=min_p).mean()
        parts.append(sign_autocorr.rename(f"sign_autocorr_{w}"))

        # 7. Squared return mean (proxy for volatility clustering strength)
        parts.append((log_ret ** 2).rolling(w, min_periods=min_p).mean().rename(f"sq_return_mean_{w}"))

        # 8. Absolute return mean
        parts.append(log_ret.abs().rolling(w, min_periods=min_p).mean().rename(f"abs_return_mean_{w}"))

    # 9. Rolling pairwise correlation proxy (average of available cross-asset corrs)
    # Only computed if multiple assets available
    spy_ret = log_ret.copy()
    cross_assets = [k for k in prices if k != primary_asset and k != "^VIX"]
    if len(cross_assets) >= 2:
        corr_series = []
        for asset in cross_assets:
            if "close" not in prices[asset].columns:
                continue
            asset_ret = np.log(prices[asset]["close"] / prices[asset]["close"].shift(1))
            asset_ret = asset_ret.reindex(spy_ret.index)
            for w in roll_windows:
                corr_series.append(spy_ret.rolling(w, min_periods=w).corr(asset_ret))
        if corr_series:
            avg_corr = pd.concat(corr_series, axis=1).mean(axis=1)
            parts.append(avg_corr.rename("avg_cross_corr"))

    result = pd.concat(parts, axis=1)
    result.index.name = "date"
    return result
