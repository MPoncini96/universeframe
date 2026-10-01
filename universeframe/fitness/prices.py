"""Multi-ticker adjusted-close price downloads via yfinance.

Monstra's production fitness analyzers pull from an internal Alpaca/Postgres
price pipeline (``AlgorithmContext`` / ``market_data_provider``). This module
is the standalone substitute, following the same per-ticker ``yfinance``
pattern ``vectorframe.pipeline.fetch_price_volume`` uses, assembled into one
wide DataFrame (date index, one column per ticker) - the shape every
function in ``force.py`` / ``draco.py`` expects.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_adjusted_close(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily adjusted closes for `tickers` between two ISO dates.

    Tickers that fail to download (bad symbol, no history in range, etc.)
    are silently dropped from the result rather than raising - callers
    decide whether the ticker they actually care about ended up present.
    """
    columns: dict[str, pd.Series] = {}
    for ticker in dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()):
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
        except Exception as exc:
            logger.warning("Could not download %s: %s", ticker, exc)
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        columns[ticker] = close

    if not columns:
        return pd.DataFrame()

    prices = pd.concat(columns, axis=1)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")
    return prices
