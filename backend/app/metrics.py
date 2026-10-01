"""Key figures computed from price history and company fundamentals.

Price-based functions expect a pandas Series/DataFrame indexed by tz-naive
dates in ascending order. All percentages are returned in percent (5.2 = 5.2 %).
"""

import math

import pandas as pd

from app.data import clean

TRADING_DAYS = 252


def moving_average(closes: pd.Series, window: int) -> pd.Series:
    return closes.rolling(window, min_periods=window).mean()


def period_return(closes: pd.Series, offset: pd.DateOffset) -> float | None:
    """Return in percent from the last close at or before (last date - offset)."""
    closes = closes.dropna()
    if closes.empty:
        return None
    start_date = closes.index[-1] - offset
    if closes.index[0] > start_date:
        return None
    base = closes.asof(start_date)
    return clean((closes.iloc[-1] / base - 1) * 100) if base else None


def ytd_return(closes: pd.Series) -> float | None:
    """Return in percent since the last close of the previous calendar year."""
    closes = closes.dropna()
    if closes.empty:
        return None
    previous_year = closes[closes.index.year < closes.index[-1].year]
    if previous_year.empty:
        return None
    base = previous_year.iloc[-1]
    return clean((closes.iloc[-1] / base - 1) * 100) if base else None


def annual_volatility(closes: pd.Series, days: int = TRADING_DAYS) -> float | None:
    """Annualised standard deviation of daily returns over the last `days`, in percent."""
    daily = closes.dropna().pct_change().dropna().tail(days)
    if len(daily) < 20:
        return None
    return clean(daily.std() * math.sqrt(TRADING_DAYS) * 100)


def high_low_52w(prices: pd.DataFrame) -> tuple[float | None, float | None]:
    if prices.empty:
        return None, None
    last_year = prices[prices.index > prices.index[-1] - pd.DateOffset(weeks=52)]
    return clean(last_year["High"].max()), clean(last_year["Low"].min())


def price_metrics(prices: pd.DataFrame) -> dict:
    """Metrics from daily OHLC history with both 'Close' and 'Adj Close' columns.

    Returns and volatility use adjusted closes (dividends reinvested); price
    levels and moving averages use the quoted close.
    """
    if prices.empty:
        return {
            key: None
            for key in (
                "price",
                "change_pct",
                "high_52w",
                "low_52w",
                "return_1m_pct",
                "return_1y_pct",
                "return_ytd_pct",
                "volatility_pct",
                "sma_50",
                "sma_200",
                "as_of",
            )
        }

    closes = prices["Close"].dropna()
    adjusted = prices["Adj Close"].dropna()
    high, low = high_low_52w(prices)
    change_pct = None
    if len(closes) >= 2 and closes.iloc[-2]:
        change_pct = clean((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)

    return {
        "price": clean(closes.iloc[-1]),
        "change_pct": change_pct,
        "high_52w": high,
        "low_52w": low,
        "return_1m_pct": period_return(adjusted, pd.DateOffset(months=1)),
        "return_1y_pct": period_return(adjusted, pd.DateOffset(years=1)),
        "return_ytd_pct": ytd_return(adjusted),
        "volatility_pct": annual_volatility(adjusted),
        "sma_50": clean(moving_average(closes, 50).iloc[-1]),
        "sma_200": clean(moving_average(closes, 200).iloc[-1]),
        "as_of": closes.index[-1].date().isoformat(),
    }


def _ratio(numerator, denominator) -> float | None:
    numerator, denominator = clean(numerator), clean(denominator)
    if numerator is None or not denominator:
        return None
    return clean(numerator / denominator)


def _percent(fraction) -> float | None:
    value = clean(fraction)
    return None if value is None else value * 100


def fundamental_metrics(info: dict, fx_rate: float | None) -> dict:
    """Valuation and balance sheet figures from a yfinance `info` dict.

    `fx_rate` converts the reporting currency into the trading currency. Yahoo
    reports enterprise value in the trading currency but EBITDA in the
    reporting currency, so EV/EBITDA is computed here instead of trusting
    `enterpriseToEbitda`. EPS, P/E and P/B are already in the trading currency.
    `dividendYield` is given in percent and `debtToEquity` as a percentage.
    """
    ebitda = clean(info.get("ebitda"))
    ev_ebitda = None
    if ebitda is not None and ebitda > 0 and fx_rate:
        ev_ebitda = _ratio(info.get("enterpriseValue"), ebitda * fx_rate)

    debt_to_equity = clean(info.get("debtToEquity"))
    return {
        "market_cap": clean(info.get("marketCap")),
        "pe": clean(info.get("trailingPE")),
        "pb": clean(info.get("priceToBook")),
        "ev_ebitda": ev_ebitda,
        "dividend_yield_pct": clean(info.get("dividendYield")),
        "eps": clean(info.get("trailingEps")),
        "roe_pct": _percent(info.get("returnOnEquity")),
        "debt_to_equity": None if debt_to_equity is None else debt_to_equity / 100,
    }
