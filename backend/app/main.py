"""FastAPI application for the Oslo Børs explorer."""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import data, metrics

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Oslo Børs API", version="0.1.0")

allowed_origins = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stocks")
def list_stocks() -> list[dict]:
    tickers = data.load_tickers()
    quotes = data.get_quotes(tuple(tickers["ticker"]))
    return [
        {
            "ticker": row.ticker,
            "name": row.name,
            "sector": row.sector,
            **quotes.get(row.ticker, {}),
        }
        for row in tickers.itertuples(index=False)
    ]


def _lookup(ticker: str) -> dict:
    company = data.find_ticker(ticker)
    if company is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    return company


@app.get("/api/stocks/{ticker}")
def stock_details(ticker: str) -> dict:
    company = _lookup(ticker)
    info = data.get_info(company["ticker"])
    currency = info.get("currency")
    fx_rate = data.get_fx_rate(info.get("financialCurrency"), currency)
    return {
        **company,
        "currency": currency,
        "financial_currency": info.get("financialCurrency"),
        **metrics.fundamental_metrics(info, fx_rate),
        **metrics.price_metrics(data.get_price_history(company["ticker"])),
    }


@app.get("/api/stocks/{ticker}/history")
def stock_history(ticker: str, period: Literal["1m", "6m", "1y", "5y", "max"] = "1y") -> dict:
    company = _lookup(ticker)
    prices = data.get_price_history(company["ticker"])
    return {
        "ticker": company["ticker"],
        "period": period,
        "points": metrics.history_points(prices, period),
    }
