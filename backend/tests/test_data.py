import math

import numpy as np
import pandas as pd
import pytest

from app import data

EMPTY_QUOTE = {"price": None, "change_pct": None, "market_cap": None, "currency": None}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, 1.5),
        (np.float64(2.0), 2.0),
        (np.int64(3), 3.0),
        (float("nan"), None),
        (np.nan, None),
        (math.inf, None),
        (None, None),
        ("abc", None),
        (True, None),
    ],
)
def test_clean(value, expected):
    assert data.clean(value) == expected


def test_to_symbol():
    assert data.to_symbol(" eqnr ") == "EQNR.OL"


def test_tickers_csv_is_valid():
    df = data.load_tickers()
    assert list(df.columns) == ["ticker", "name", "sector"]
    assert len(df) >= 20
    assert df["ticker"].is_unique
    assert not df.isna().any().any()


def test_find_ticker():
    assert data.find_ticker("eqnr")["name"] == "Equinor"
    assert data.find_ticker("NOPE") is None


def _fake_download(symbols, **kwargs):
    index = pd.date_range("2026-01-01", periods=3)
    frames = {
        "AAA.OL": pd.DataFrame({"Close": [100.0, 110.0, 99.0]}, index=index),
        "BBB.OL": pd.DataFrame({"Close": [np.nan, np.nan, np.nan]}, index=index),
    }
    return pd.concat(frames, axis=1)


class _FakeTicker:
    def __init__(self, symbol):
        self.fast_info = (
            {"marketCap": 1e9, "currency": "NOK"} if symbol == "AAA.OL" else {"marketCap": np.nan}
        )


def test_fetch_quotes(monkeypatch):
    monkeypatch.setattr(data.yf, "download", _fake_download)
    monkeypatch.setattr(data.yf, "Ticker", _FakeTicker)

    quotes = data.fetch_quotes(["AAA", "BBB", "CCC"])

    assert quotes["AAA"] == {
        "price": 99.0,
        "change_pct": pytest.approx(-10.0),
        "market_cap": 1e9,
        "currency": "NOK",
    }
    assert quotes["BBB"] == EMPTY_QUOTE
    assert quotes["CCC"]["price"] is None


def test_fetch_quotes_survives_download_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(data.yf, "download", boom)
    monkeypatch.setattr(data.yf, "Ticker", boom)

    quotes = data.fetch_quotes(["AAA"])
    assert quotes["AAA"] == EMPTY_QUOTE
