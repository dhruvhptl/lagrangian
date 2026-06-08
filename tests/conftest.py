"""Synthetic OHLCV fixtures for unit tests. No network calls."""
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int, seed: int, mu: float = 0.0003, sigma: float = 0.012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # GARCH-like vol clustering: variance follows AR(1)
    var = np.zeros(n)
    var[0] = sigma ** 2
    eps = rng.standard_normal(n)
    for t in range(1, n):
        var[t] = 0.05 * sigma ** 2 + 0.9 * var[t - 1] + 0.05 * (eps[t - 1] * np.sqrt(var[t - 1])) ** 2
    returns = mu + np.sqrt(var) * eps

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = close * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)

    dates = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="session")
def synthetic_prices() -> dict[str, pd.DataFrame]:
    """Dict of ticker -> OHLCV DataFrame with 500 business days, fixed seed."""
    tickers = ["SPY", "QQQ", "TLT", "GLD", "^VIX"]
    seeds = [42, 43, 44, 45, 46]
    return {t: _make_ohlcv(500, s) for t, s in zip(tickers, seeds)}


@pytest.fixture(scope="session")
def spy_prices(synthetic_prices) -> pd.DataFrame:
    return synthetic_prices["SPY"]
