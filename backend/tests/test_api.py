import pandas as pd
import pytest

from app import data


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_stocks(client, monkeypatch):
    def fake_quotes(tickers):
        return {
            t: {"price": 10.0, "change_pct": None, "market_cap": None, "currency": "NOK"}
            for t in tickers
        }

    monkeypatch.setattr(data, "fetch_quotes", fake_quotes)

    response = client.get("/api/stocks")
    assert response.status_code == 200
    stocks = response.json()
    assert len(stocks) == len(data.load_tickers())
    equinor = next(s for s in stocks if s["ticker"] == "EQNR")
    assert equinor == {
        "ticker": "EQNR",
        "name": "Equinor",
        "sector": "Energi",
        "price": 10.0,
        "change_pct": None,
        "market_cap": None,
        "currency": "NOK",
    }


def _fake_history(ticker):
    index = pd.bdate_range("2024-01-01", periods=300)
    closes = pd.Series(range(100, 400), index=index, dtype=float)
    return pd.DataFrame({"Close": closes, "Adj Close": closes, "High": closes, "Low": closes})


def _fake_info(ticker):
    return {
        "currency": "NOK",
        "financialCurrency": "USD",
        "marketCap": 1e12,
        "trailingPE": float("nan"),
        "enterpriseValue": 100.0,
        "ebitda": 5.0,
    }


@pytest.fixture
def fake_market(monkeypatch):
    monkeypatch.setattr(data, "get_price_history", _fake_history)
    monkeypatch.setattr(data, "get_info", _fake_info)
    monkeypatch.setattr(data, "get_fx_rate", lambda source, target: 10.0)


def test_stock_details(client, fake_market):
    response = client.get("/api/stocks/eqnr")
    assert response.status_code == 200
    stock = response.json()
    assert stock["ticker"] == "EQNR"
    assert stock["name"] == "Equinor"
    assert stock["price"] == 399
    assert stock["pe"] is None
    assert stock["ev_ebitda"] == pytest.approx(2.0)
    assert stock["financial_currency"] == "USD"
    for key in ("pb", "eps", "roe_pct", "sma_200", "volatility_pct", "return_ytd_pct"):
        assert key in stock


def test_stock_details_unknown_ticker(client, fake_market):
    assert client.get("/api/stocks/NOPE").status_code == 404


def test_stock_history(client, fake_market):
    response = client.get("/api/stocks/EQNR/history?period=6m")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "EQNR"
    assert body["period"] == "6m"
    assert body["points"][-1] == {
        "date": "2025-02-21",
        "close": 399.0,
        "sma_50": pytest.approx(374.5),
        "sma_200": pytest.approx(299.5),
    }


def test_stock_history_defaults_to_one_year(client, fake_market):
    assert client.get("/api/stocks/EQNR/history").json()["period"] == "1y"


def test_stock_history_rejects_unknown_period(client, fake_market):
    assert client.get("/api/stocks/EQNR/history?period=3w").status_code == 422
