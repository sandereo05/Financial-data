"""FastAPI application for the Oslo Børs explorer."""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import data

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
