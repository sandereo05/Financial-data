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
