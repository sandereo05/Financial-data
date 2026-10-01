"""Data access: the ticker list and market data from yfinance."""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS_PATH = Path(__file__).resolve().parent.parent / "data" / "tickers.csv"
SUFFIX = ".OL"


def to_symbol(ticker: str) -> str:
    """Map an Oslo Børs ticker (EQNR) to its Yahoo symbol (EQNR.OL)."""
    return f"{ticker.strip().upper()}{SUFFIX}"


def clean(value) -> float | None:
    """Convert numpy/pandas scalars to JSON-safe floats; NaN, inf and junk become None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@lru_cache
def load_tickers() -> pd.DataFrame:
    df = pd.read_csv(TICKERS_PATH, comment="#", dtype=str)
    df = df.apply(lambda col: col.str.strip())
    df["ticker"] = df["ticker"].str.upper()
    return df


def find_ticker(ticker: str) -> dict | None:
    rows = load_tickers()
    match = rows[rows["ticker"] == ticker.strip().upper()]
    return None if match.empty else match.iloc[0].to_dict()


def _closes(prices: pd.DataFrame, symbol: str) -> pd.Series:
    """Extract the close series for one symbol from a yf.download result."""
    if prices.empty:
        return pd.Series(dtype=float)
    if isinstance(prices.columns, pd.MultiIndex):
        if symbol not in prices.columns.get_level_values(0):
            return pd.Series(dtype=float)
        frame = prices[symbol]
    else:
        frame = prices
    if "Close" not in frame:
        return pd.Series(dtype=float)
    return frame["Close"].dropna()


def _ticker_info(symbol: str) -> dict:
    """Market cap and currency from yfinance's lightweight fast_info."""
    try:
        info = yf.Ticker(symbol).fast_info
        return {"market_cap": clean(info.get("marketCap")), "currency": info.get("currency")}
    except Exception as exc:
        logger.warning("fast_info failed for %s: %s", symbol, exc)
        return {"market_cap": None, "currency": None}


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """Return last price, daily change in percent, market cap and currency per ticker."""
    symbols = [to_symbol(t) for t in tickers]
    try:
        prices = yf.download(
            symbols,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        logger.exception("Price download failed for %d symbols", len(symbols))
        prices = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = dict(zip(symbols, pool.map(_ticker_info, symbols), strict=True))

    quotes = {}
    for ticker, symbol in zip(tickers, symbols, strict=True):
        closes = _closes(prices, symbol)
        price = clean(closes.iloc[-1]) if len(closes) else None
        change_pct = None
        if len(closes) >= 2 and closes.iloc[-2]:
            change_pct = clean((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)
        if price is None:
            logger.warning("No price data for %s", symbol)
        quotes[ticker] = {"price": price, "change_pct": change_pct, **infos[symbol]}
    return quotes
