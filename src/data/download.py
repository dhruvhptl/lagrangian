"""Download adjusted OHLCV data via yfinance. Cache to parquet."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from src.data.manager import DataManager

logger = logging.getLogger(__name__)


def download_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted OHLCV for one ticker. Raises on empty result."""
    logger.info(f"Downloading {ticker} from {start_date} to {end_date}")
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {ticker}")
    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "date"
    # VIX has no volume — fill with zeros so schema is consistent
    if "volume" not in raw.columns:
        raw["volume"] = 0.0
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    return raw[["open", "high", "low", "close", "volume"]]


def fetch_all(dm: "DataManager", force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch all tickers in dm.tickers. Uses cache unless force_download=True."""
    prices: dict[str, pd.DataFrame] = {}
    for ticker in dm.tickers:
        if not force_download and dm.raw_exists(ticker):
            logger.info(f"Loading {ticker} from cache")
            prices[ticker] = dm.load_raw(ticker)
        else:
            df = download_ticker(ticker, dm.start_date, dm.end_date)
            dm.save_raw(ticker, df)
            prices[ticker] = df
    return prices
