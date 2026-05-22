"""Generate synthetic customer account data for AML simulation.

All accounts are US-based retail brokerage accounts. Account holders are US
residents; state reflects their address. is_pep covers domestic politically
exposed persons (elected officials, senior government roles, etc.).
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

US_STATES: list[str] = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

# State populations (2020 census, millions) — used to weight address distribution
# so CA/TX/FL/NY are more common than WY/VT, matching real brokerage demographics.
US_STATE_WEIGHTS: list[float] = [
    4.9,
    0.7,
    7.2,
    3.0,
    39.5,
    5.8,
    3.6,
    1.0,
    21.5,
    10.7,
    1.4,
    1.9,
    12.8,
    6.8,
    3.2,
    2.9,
    4.5,
    4.7,
    1.3,
    6.2,
    7.0,
    10.1,
    5.7,
    3.0,
    6.2,
    1.1,
    2.0,
    3.1,
    1.4,
    9.3,
    2.1,
    20.2,
    10.4,
    0.8,
    11.8,
    4.0,
    4.2,
    13.0,
    1.1,
    5.1,
    0.9,
    7.0,
    29.1,
    3.3,
    0.6,
    8.6,
    7.7,
    1.8,
    5.9,
    0.6,
]

_TODAY = date.today()


def generate_accounts(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Return a DataFrame of n synthetic US brokerage customer accounts."""
    rng = np.random.default_rng(random_state)
    fake = Faker()
    Faker.seed(random_state)

    account_ids = [f"ACC-{i:05d}" for i in range(1, n + 1)]

    account_types: list[str] = list(
        rng.choice(
            ["Retail", "Institutional", "Broker-Dealer"],
            size=n,
            p=[0.70, 0.20, 0.10],
        )
    )

    # Account open dates: 1–15 years ago, uniform
    days_open = rng.integers(365, 365 * 15, size=n)
    open_dates = [_TODAY - timedelta(days=int(d)) for d in days_open]
    age_days = [int(d) for d in days_open]

    # All accounts are US-based; state weighted by population
    state_weights = np.array(US_STATE_WEIGHTS)
    state_weights = state_weights / state_weights.sum()
    states: list[str] = list(rng.choice(US_STATES, size=n, p=state_weights))

    # Risk tier: right-skewed — most accounts are low risk (tier 1–2)
    risk_tiers: list[int] = list(
        rng.choice([1, 2, 3, 4, 5], size=n, p=[0.40, 0.30, 0.15, 0.10, 0.05])
    )

    # Politically Exposed Persons: ~2% — domestic officials, executives, etc.
    is_pep = rng.random(n) < 0.02

    # Income: log-normal, clipped to $30K–$5M
    log_income = rng.normal(loc=np.log(80_000), scale=1.0, size=n)
    annual_income = np.clip(np.exp(log_income), 30_000, 5_000_000)

    # Net worth correlated with income: income × log-normal multiplier (median ~3×)
    nw_multiplier = np.clip(np.exp(rng.normal(loc=np.log(3), scale=0.8, size=n)), 0.5, 50)
    net_worth = np.clip(annual_income * nw_multiplier, 10_000, 50_000_000)

    return pd.DataFrame(
        {
            "account_id": account_ids,
            "customer_name": [fake.name() for _ in range(n)],
            "account_type": account_types,
            "account_open_date": open_dates,
            "account_age_days": age_days,
            "state": states,
            "risk_tier": risk_tiers,
            "is_pep": is_pep,
            "annual_income_usd": np.round(annual_income, 2),
            "net_worth_usd": np.round(net_worth, 2),
        }
    )


if __name__ == "__main__":
    output_path = Path("data/raw/accounts.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_accounts()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} accounts to {output_path}")
    print(df["account_type"].value_counts().to_string())
    print(f"PEP rate: {df['is_pep'].mean():.1%}")
    print(df["state"].value_counts().head(10).to_string())
