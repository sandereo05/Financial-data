import math

import numpy as np
import pandas as pd
import pytest

from app import metrics


def series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def ohlc(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": closes, "Adj Close": closes, "High": closes + 1, "Low": closes - 1}
    )


def test_moving_average():
    result = metrics.moving_average(series([1, 2, 3, 4, 5]), 3)
    assert result.isna().sum() == 2
    assert result.iloc[-1] == 4


def test_period_return():
    closes = pd.Series(
        [100.0, 120.0, 150.0],
        index=pd.to_datetime(["2025-01-02", "2025-02-03", "2025-02-28"]),
    )
    # One month before 2025-02-28 is 2025-01-28; last close at or before that is 100.
    assert metrics.period_return(closes, pd.DateOffset(months=1)) == pytest.approx(50.0)


def test_period_return_needs_enough_history():
    assert metrics.period_return(series([1, 2, 3]), pd.DateOffset(years=1)) is None
    assert metrics.period_return(pd.Series(dtype=float), pd.DateOffset(months=1)) is None


def test_ytd_return():
    closes = pd.Series(
        [80.0, 100.0, 110.0],
        index=pd.to_datetime(["2024-12-30", "2024-12-31", "2025-03-01"]),
    )
    assert metrics.ytd_return(closes) == pytest.approx(10.0)
    assert metrics.ytd_return(closes.iloc[2:]) is None


def test_annual_volatility_constant_growth_is_zero():
    closes = series([100 * 1.01**i for i in range(60)])
    assert metrics.annual_volatility(closes) == pytest.approx(0.0, abs=1e-9)


def test_annual_volatility_matches_formula():
    rng = np.random.default_rng(0)
    closes = series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    expected = closes.pct_change().dropna().tail(252).std() * math.sqrt(252) * 100
    assert metrics.annual_volatility(closes) == pytest.approx(expected)


def test_annual_volatility_needs_data():
    assert metrics.annual_volatility(series([1, 2, 3])) is None


def test_price_metrics():
    closes = series(range(1, 301))
    result = metrics.price_metrics(ohlc(closes))
    assert result["price"] == 300
    assert result["change_pct"] == pytest.approx(300 / 299 * 100 - 100)
    assert result["sma_50"] == pytest.approx(275.5)
    assert result["sma_200"] == pytest.approx(200.5)
    assert result["high_52w"] == 301
    assert result["as_of"] == closes.index[-1].date().isoformat()


def test_price_metrics_short_history_gives_nulls():
    result = metrics.price_metrics(ohlc(series([10, 11])))
    assert result["sma_50"] is None
    assert result["return_1y_pct"] is None


def test_price_metrics_empty():
    result = metrics.price_metrics(pd.DataFrame())
    assert all(value is None for value in result.values())


def test_fundamental_metrics_converts_units():
    info = {
        "marketCap": 1e12,
        "trailingPE": 11.4,
        "priceToBook": float("nan"),
        "enterpriseValue": 980e9,
        "ebitda": 42e9,
        "dividendYield": 3.65,
        "trailingEps": 35.2,
        "returnOnEquity": 0.2127,
        "debtToEquity": 75.0,
    }
    result = metrics.fundamental_metrics(info, fx_rate=10.0)
    assert result["ev_ebitda"] == pytest.approx(980 / 420)
    assert result["pb"] is None
    assert result["dividend_yield_pct"] == 3.65
    assert result["roe_pct"] == pytest.approx(21.27)
    assert result["debt_to_equity"] == pytest.approx(0.75)


def test_fundamental_metrics_handles_missing_values():
    result = metrics.fundamental_metrics({"ebitda": -5e9, "enterpriseValue": 1e9}, fx_rate=1.0)
    assert all(value is None for value in result.values())
    assert (
        metrics.fundamental_metrics({"ebitda": 1, "enterpriseValue": 2}, None)["ev_ebitda"] is None
    )


def test_history_points_slices_after_computing_moving_averages():
    closes = series(range(1, 401))
    points = metrics.history_points(ohlc(closes), "1m")
    assert 19 <= len(points) <= 24
    assert points[-1]["close"] == 400
    assert points[0]["sma_200"] is not None
    assert points[-1]["date"] == closes.index[-1].date().isoformat()


def test_history_points_weekly_for_long_periods():
    closes = series(range(1, 401))
    closes = closes.iloc[:-1]  # end mid-week, not on a Friday
    points = metrics.history_points(ohlc(closes), "max")
    assert len(points) in (79, 80, 81)
    assert points[-1]["date"] == closes.index[-1].date().isoformat()
    assert points[0]["sma_50"] is None
    assert points[-1]["close"] == 399


def test_history_points_empty():
    assert metrics.history_points(pd.DataFrame(), "1y") == []
