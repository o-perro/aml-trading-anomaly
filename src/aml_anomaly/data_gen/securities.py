"""Generate synthetic securities (stocks/tickers) data for AML simulation."""

import random
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

# The 11 standard GICS (Global Industry Classification Standard) sectors
GICS_SECTORS: list[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]

EXCHANGES: list[str] = ["NYSE", "NASDAQ", "OTC"]

# Per-tier parameters for log-normal volume/price distributions and realistic
# bid-ask spread, volatility, and exchange mix. Micro-cap stocks are deliberately
# given much lower volume, wider spreads, and higher volatility — matching real
# market microstructure. OTC weighting increases as cap tier decreases.
_TIER_PARAMS: dict[str, dict[str, Any]] = {
    "Large": {
        "volume_lognorm": (np.log(5_000_000), 0.8),
        "price_lognorm": (np.log(100), 0.8),
        "spread_range": (0.0005, 0.002),
        "vol_30d_range": (0.05, 0.20),
        "exchange_weights": [0.50, 0.40, 0.10],
    },
    "Mid": {
        "volume_lognorm": (np.log(500_000), 0.8),
        "price_lognorm": (np.log(30), 0.8),
        "spread_range": (0.002, 0.010),
        "vol_30d_range": (0.10, 0.30),
        "exchange_weights": [0.35, 0.40, 0.25],
    },
    "Small": {
        "volume_lognorm": (np.log(50_000), 0.8),
        "price_lognorm": (np.log(10), 0.7),
        "spread_range": (0.005, 0.030),
        "vol_30d_range": (0.15, 0.40),
        "exchange_weights": [0.20, 0.25, 0.55],
    },
    "Micro": {
        "volume_lognorm": (np.log(5_000), 0.7),
        "price_lognorm": (np.log(4), 0.7),
        "spread_range": (0.015, 0.070),
        "vol_30d_range": (0.25, 0.65),
        "exchange_weights": [0.05, 0.10, 0.85],
    },
}


def _unique_tickers(n: int, rng: np.random.Generator) -> list[str]:
    tickers: set[str] = set()
    while len(tickers) < n:
        length = int(rng.choice([3, 4], p=[0.4, 0.6]))
        tickers.add("".join(random.choices(string.ascii_uppercase, k=length)))
    return list(tickers)


def generate_securities(n: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Return a DataFrame of n synthetic securities with realistic market microstructure."""
    rng = np.random.default_rng(random_state)
    random.seed(random_state)
    fake = Faker()
    Faker.seed(random_state)

    tiers: np.ndarray = rng.choice(
        ["Large", "Mid", "Small", "Micro"],
        size=n,
        p=[0.30, 0.30, 0.25, 0.15],
    )
    tickers = _unique_tickers(n, rng)

    records = []
    for ticker, tier in zip(tickers, tiers):
        p = _TIER_PARAMS[str(tier)]

        avg_daily_volume = int(np.clip(rng.lognormal(*p["volume_lognorm"]), 100, 200_000_000))
        avg_price_usd = round(float(np.clip(rng.lognormal(*p["price_lognorm"]), 1.0, 500.0)), 2)
        spread_lo, spread_hi = p["spread_range"]
        vol_lo, vol_hi = p["vol_30d_range"]

        records.append(
            {
                "ticker": ticker,
                "company_name": fake.company(),
                "sector": str(rng.choice(GICS_SECTORS)),
                "market_cap_tier": str(tier),
                "avg_daily_volume": avg_daily_volume,
                "avg_price_usd": avg_price_usd,
                "bid_ask_spread_pct": round(float(rng.uniform(spread_lo, spread_hi)), 5),
                "price_volatility_30d": round(float(rng.uniform(vol_lo, vol_hi)), 4),
                "exchange": str(rng.choice(EXCHANGES, p=p["exchange_weights"])),
            }
        )

    df = pd.DataFrame(records)
    volume_threshold = df["avg_daily_volume"].quantile(0.20)
    df["is_illiquid"] = df["avg_daily_volume"] <= volume_threshold

    return df


if __name__ == "__main__":
    output_path = Path("data/raw/securities.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_securities()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} securities to {output_path}")
    print(df["market_cap_tier"].value_counts().to_string())
    print(f"Illiquid: {df['is_illiquid'].sum()} ({df['is_illiquid'].mean():.0%})")
