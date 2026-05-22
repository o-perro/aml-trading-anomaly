"""Generate synthetic trade transaction data, including injected anomaly patterns."""

from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Simulation window: 2 years of trading history
_SIM_START = date.today() - timedelta(days=730)
_SIM_END = date.today() - timedelta(days=1)

# Market hours: 9:30 AM – 4:00 PM ET (stored as seconds since midnight)
_MARKET_OPEN_SEC = 9 * 3600 + 30 * 60  # 34200
_MARKET_CLOSE_SEC = 16 * 3600  # 57600

# Anomaly pattern names — also used as column values in injected_anomalies.csv
ANOMALY_PATTERNS: list[str] = [
    "wash_trading",
    "velocity_spike",
    "smurfing",
    "illiquid_concentration",
    "off_hours_clustering",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _business_days(start: date, end: date) -> list[date]:
    """Return all Mon–Fri dates between start and end inclusive."""
    result = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _market_time(rng: np.random.Generator, off_hours: bool = False) -> time:
    """Return a trade time — market hours by default, outside hours if off_hours=True."""
    if off_hours:
        # Pick either pre-market (6:00–9:30) or after-hours (16:00–20:00)
        if rng.random() < 0.5:
            sec = int(rng.uniform(6 * 3600, _MARKET_OPEN_SEC))
        else:
            sec = int(rng.uniform(_MARKET_CLOSE_SEC, 20 * 3600))
    else:
        # Weight trade times toward mid-day (10am–2pm) as real volume clusters there
        sec = int(rng.triangular(_MARKET_OPEN_SEC, 12 * 3600, _MARKET_CLOSE_SEC))
    return time(sec // 3600, (sec % 3600) // 60, sec % 60)


def _is_round_value(value: float) -> bool:
    """True if the trade value is suspiciously close to a round number (multiple of $500)."""
    nearest_500 = round(value / 500) * 500
    return abs(value - nearest_500) < 25  # within $25 of a $500 multiple


def _trade_price(avg_price: float, rng: np.random.Generator) -> float:
    """Return a realistic execution price near the security's average price."""
    noise = rng.normal(0, avg_price * 0.005)  # ±0.5% noise
    return round(max(0.01, avg_price + noise), 2)


def _execution_ms(rng: np.random.Generator, algorithmic: bool = False) -> int:
    """Return milliseconds from order to fill — very fast if algorithmic."""
    if algorithmic:
        return int(np.clip(rng.lognormal(3, 1), 1, 500))  # 1–500 ms
    return int(np.clip(rng.lognormal(8, 1.5), 100, 60_000))  # 0.1s–60s


# ---------------------------------------------------------------------------
# Normal trade generation
# ---------------------------------------------------------------------------


def _assign_activity_levels(
    accounts: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Assign each account a baseline trades-per-month rate drawn from a log-normal.

    Institutional accounts trade more frequently than retail; broker-dealers most of all.
    This variance is what makes the self-baseline behavioral features meaningful.
    """
    base_rates = {"Retail": 2.0, "Institutional": 20.0, "Broker-Dealer": 50.0}
    activity: dict[str, float] = {}
    for _, row in accounts.iterrows():
        base = base_rates[row["account_type"]]
        activity[row["account_id"]] = float(np.clip(rng.lognormal(np.log(base), 0.7), 0.5, 500))
    return activity


def _generate_normal_trades(
    account_id: str,
    monthly_rate: float,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
    business_days: list[date],
    rng: np.random.Generator,
    trade_id_start: int,
) -> list[dict[str, Any]]:
    """Generate a realistic sequence of normal trades for one account."""
    # Convert monthly rate to daily probability of trading
    daily_prob = monthly_rate / 21  # ~21 trading days per month

    all_tickers = securities["ticker"].tolist()

    # Each account builds a "preferred" ticker list over time — more realistic
    # than purely random ticker selection
    n_preferred = max(1, int(rng.integers(2, min(15, len(all_tickers)))))
    preferred_tickers = list(rng.choice(all_tickers, size=n_preferred, replace=False))

    trades: list[dict[str, Any]] = []
    trade_id = trade_id_start

    for d in business_days:
        if rng.random() > daily_prob:
            continue

        # Number of trades on active days: usually 1–3, occasionally more
        n_trades = int(np.clip(rng.lognormal(0, 0.8), 1, 20))

        for _ in range(n_trades):
            # 80% chance of trading a preferred ticker, 20% new/random
            ticker_pool = preferred_tickers if rng.random() < 0.80 else all_tickers
            ticker = str(rng.choice(ticker_pool))

            sec_row = securities[securities["ticker"] == ticker].iloc[0]
            price = _trade_price(sec_row["avg_price_usd"], rng)

            # Quantity: log-normal, capped at 5% of the stock's average daily volume
            max_qty = max(1, int(sec_row["avg_daily_volume"] * 0.05))
            quantity = int(np.clip(rng.lognormal(np.log(max(1, max_qty // 20)), 1.0), 1, max_qty))
            trade_value = round(price * quantity, 2)

            # 5% chance of off-hours trade for normal accounts
            is_off = rng.random() < 0.05
            t = _market_time(rng, off_hours=is_off)

            trades.append(
                {
                    "trade_id": f"TRD-{trade_id:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": None,
                    "ticker": ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": str(rng.choice(["BUY", "SELL"])),
                    "quantity": quantity,
                    "price_usd": price,
                    "trade_value_usd": trade_value,
                    "order_type": str(
                        rng.choice(["MARKET", "LIMIT", "STOP"], p=[0.60, 0.30, 0.10])
                    ),
                    "time_to_execution_ms": _execution_ms(rng),
                    "is_off_hours": is_off,
                    "is_round_value": _is_round_value(trade_value),
                }
            )
            trade_id += 1

    return trades


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------


def _inject_wash_trading(
    account_id: str,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
    business_days: list[date],
    rng: np.random.Generator,
    trade_id_start: int,
) -> list[dict[str, Any]]:
    """Inject wash trading: rapid buy/sell cycles on the same ticker with one counterparty.

    AML signal: buy_sell_ratio_30d near 1.0, same_day_reversal_count elevated,
    top_counterparty_concentration_pct near 100%, avg_holding_period_minutes very short.
    """
    # Pick a counterparty from other accounts
    others = accounts[accounts["account_id"] != account_id]["account_id"].tolist()
    counterparty = str(rng.choice(others))

    # Focus on a small set of tickers — real wash trading uses specific stocks
    ticker_pool = securities["ticker"].tolist()
    focus_tickers = list(rng.choice(ticker_pool, size=3, replace=False))

    trades: list[dict[str, Any]] = []
    tid = trade_id_start

    # Generate wash trading activity over the last 60 days
    recent_days = [d for d in business_days if d >= date.today() - timedelta(days=60)]
    idx = rng.choice(len(recent_days), size=min(20, len(recent_days)), replace=False)
    active_days = [recent_days[int(i)] for i in idx]

    for d in sorted(active_days):
        ticker = str(rng.choice(focus_tickers))
        sec_row = securities[securities["ticker"] == ticker].iloc[0]
        price = _trade_price(sec_row["avg_price_usd"], rng)

        # Buy and sell the same quantity — equal sides create buy/sell ratio near 1.0
        quantity = int(np.clip(rng.lognormal(np.log(500), 0.5), 10, 5000))
        trade_value = round(price * quantity, 2)

        # Buy in the morning, sell later the same day (same-day reversal)
        buy_sec = int(rng.uniform(_MARKET_OPEN_SEC, 11 * 3600))
        sell_sec = int(rng.uniform(buy_sec + 600, min(buy_sec + 7200, _MARKET_CLOSE_SEC)))
        buy_time = time(buy_sec // 3600, (buy_sec % 3600) // 60, buy_sec % 60)
        sell_time = time(sell_sec // 3600, (sell_sec % 3600) // 60, sell_sec % 60)

        for direction, t in [("BUY", buy_time), ("SELL", sell_time)]:
            trades.append(
                {
                    "trade_id": f"TRD-{tid:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": counterparty,
                    "ticker": ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": direction,
                    "quantity": quantity,
                    "price_usd": price,
                    "trade_value_usd": trade_value,
                    "order_type": "LIMIT",
                    "time_to_execution_ms": _execution_ms(rng, algorithmic=True),
                    "is_off_hours": False,
                    "is_round_value": _is_round_value(trade_value),
                }
            )
            tid += 1

    return trades


def _inject_velocity_spike(
    account_id: str,
    securities: pd.DataFrame,
    business_days: list[date],
    rng: np.random.Generator,
    trade_id_start: int,
) -> list[dict[str, Any]]:
    """Inject a velocity spike: normal history then 10–20× activity in the last 7 days.

    AML signal: velocity_ratio_7d_vs_30d spiking, value_zscore_vs_self elevated.
    """
    trades: list[dict[str, Any]] = []
    tid = trade_id_start
    all_tickers = securities["ticker"].tolist()

    # Normal history: ~1–2 trades per active day over most of the window
    normal_days = [d for d in business_days if d < date.today() - timedelta(days=7)]
    for d in normal_days:
        if rng.random() > 0.15:  # trades ~3 days per month
            continue
        ticker = str(rng.choice(all_tickers))
        sec_row = securities[securities["ticker"] == ticker].iloc[0]
        price = _trade_price(sec_row["avg_price_usd"], rng)
        qty = int(np.clip(rng.lognormal(np.log(100), 0.8), 1, 1000))
        tv = round(price * qty, 2)
        t = _market_time(rng)
        trades.append(
            {
                "trade_id": f"TRD-{tid:07d}",
                "account_id": account_id,
                "counterparty_account_id": None,
                "ticker": ticker,
                "trade_date": d.isoformat(),
                "trade_time": t.strftime("%H:%M:%S"),
                "trade_direction": str(rng.choice(["BUY", "SELL"])),
                "quantity": qty,
                "price_usd": price,
                "trade_value_usd": tv,
                "order_type": "MARKET",
                "time_to_execution_ms": _execution_ms(rng),
                "is_off_hours": False,
                "is_round_value": _is_round_value(tv),
            }
        )
        tid += 1

    # Spike: 15–25 trades per day in the final 7 days
    spike_days = [d for d in business_days if d >= date.today() - timedelta(days=7)]
    for d in spike_days:
        n_burst = int(rng.integers(15, 26))
        for _ in range(n_burst):
            ticker = str(rng.choice(all_tickers))
            sec_row = securities[securities["ticker"] == ticker].iloc[0]
            price = _trade_price(sec_row["avg_price_usd"], rng)
            qty = int(np.clip(rng.lognormal(np.log(500), 0.8), 10, 10_000))
            tv = round(price * qty, 2)
            t = _market_time(rng)
            trades.append(
                {
                    "trade_id": f"TRD-{tid:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": None,
                    "ticker": ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": str(rng.choice(["BUY", "SELL"])),
                    "quantity": qty,
                    "price_usd": price,
                    "trade_value_usd": tv,
                    "order_type": "MARKET",
                    "time_to_execution_ms": _execution_ms(rng, algorithmic=True),
                    "is_off_hours": False,
                    "is_round_value": _is_round_value(tv),
                }
            )
            tid += 1

    return trades


def _inject_smurfing(
    account_id: str,
    securities: pd.DataFrame,
    business_days: list[date],
    rng: np.random.Generator,
    trade_id_start: int,
) -> list[dict[str, Any]]:
    """Inject smurfing: many trades just below $10,000 to avoid CTR filing thresholds.

    AML signal: round_value_trade_pct elevated, burst_event_count high,
    trade values clustering in $8,500–$9,999 band.
    """
    trades: list[dict[str, Any]] = []
    tid = trade_id_start
    all_tickers = securities["ticker"].tolist()

    # Active smurfing over the last 90 days: 8–15 structured trades per active day
    smurf_days = [d for d in business_days if d >= date.today() - timedelta(days=90)]
    idx = rng.choice(len(smurf_days), size=min(40, len(smurf_days)), replace=False)
    active_days = [smurf_days[int(i)] for i in idx]

    for d in sorted(active_days):
        n_trades = int(rng.integers(8, 16))
        for _ in range(n_trades):
            ticker = str(rng.choice(all_tickers))
            sec_row = securities[securities["ticker"] == ticker].iloc[0]

            # Target value: $8,500–$9,999 — deliberately below $10,000
            target_value = rng.uniform(8_500, 9_999)
            qty = max(1, int(target_value / sec_row["avg_price_usd"]))
            price = round(target_value / qty, 2)
            tv = round(price * qty, 2)

            t = _market_time(rng)
            trades.append(
                {
                    "trade_id": f"TRD-{tid:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": None,
                    "ticker": ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": "BUY",
                    "quantity": qty,
                    "price_usd": price,
                    "trade_value_usd": tv,
                    "order_type": str(rng.choice(["MARKET", "LIMIT"])),
                    "time_to_execution_ms": _execution_ms(rng),
                    "is_off_hours": False,
                    "is_round_value": True,  # structuring to round amounts is the signal
                }
            )
            tid += 1

    return trades


def _inject_illiquid_concentration(
    account_id: str,
    securities: pd.DataFrame,
    business_days: list[date],
    rng: np.random.Generator,
    trade_id_start: int,
) -> list[dict[str, Any]]:
    """Inject illiquid concentration: >60% of trade volume into one micro-cap stock.

    AML signal: illiquid_stock_trade_pct elevated, top_ticker_concentration_pct extreme,
    trade_size_vs_adv_max high (large trades relative to thin daily volume).
    """
    trades: list[dict[str, Any]] = []
    tid = trade_id_start

    # Pick the most illiquid micro-cap ticker available
    illiquid = securities[securities["is_illiquid"] & (securities["market_cap_tier"] == "Micro")]
    if len(illiquid) == 0:
        illiquid = securities[securities["is_illiquid"]]
    focus_ticker = str(illiquid.iloc[int(rng.integers(0, len(illiquid)))]["ticker"])
    focus_sec = securities[securities["ticker"] == focus_ticker].iloc[0]
    other_tickers = [t for t in securities["ticker"].tolist() if t != focus_ticker]

    idx = rng.choice(len(business_days), size=min(80, len(business_days)), replace=False)
    active_days = [business_days[int(i)] for i in idx]

    for d in sorted(active_days):
        n_focus = int(rng.integers(3, 8))  # heavy focus ticker activity
        n_other = int(rng.integers(0, 3))  # minimal diversification

        for _ in range(n_focus):
            price = _trade_price(focus_sec["avg_price_usd"], rng)
            # Trades that are large relative to avg_daily_volume
            max_qty = max(1, int(focus_sec["avg_daily_volume"] * 0.30))
            qty = int(np.clip(rng.lognormal(np.log(max(1, max_qty // 3)), 0.5), 1, max_qty))
            tv = round(price * qty, 2)
            t = _market_time(rng)
            trades.append(
                {
                    "trade_id": f"TRD-{tid:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": None,
                    "ticker": focus_ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": str(rng.choice(["BUY", "SELL"])),
                    "quantity": qty,
                    "price_usd": price,
                    "trade_value_usd": tv,
                    "order_type": "MARKET",
                    "time_to_execution_ms": _execution_ms(rng),
                    "is_off_hours": False,
                    "is_round_value": _is_round_value(tv),
                }
            )
            tid += 1

        for _ in range(n_other):
            ticker = str(rng.choice(other_tickers))
            sec_row = securities[securities["ticker"] == ticker].iloc[0]
            price = _trade_price(sec_row["avg_price_usd"], rng)
            qty = int(np.clip(rng.lognormal(np.log(50), 0.8), 1, 500))
            tv = round(price * qty, 2)
            t = _market_time(rng)
            trades.append(
                {
                    "trade_id": f"TRD-{tid:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": None,
                    "ticker": ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": str(rng.choice(["BUY", "SELL"])),
                    "quantity": qty,
                    "price_usd": price,
                    "trade_value_usd": tv,
                    "order_type": "MARKET",
                    "time_to_execution_ms": _execution_ms(rng),
                    "is_off_hours": False,
                    "is_round_value": _is_round_value(tv),
                }
            )
            tid += 1

    return trades


def _inject_off_hours_clustering(
    account_id: str,
    securities: pd.DataFrame,
    business_days: list[date],
    rng: np.random.Generator,
    trade_id_start: int,
) -> list[dict[str, Any]]:
    """Inject off-hours clustering: 75–85% of trades placed outside normal market hours.

    AML signal: off_hours_trade_pct far above population mean (~0.05),
    weekend_trade_pct elevated.
    """
    trades: list[dict[str, Any]] = []
    tid = trade_id_start
    all_tickers = securities["ticker"].tolist()
    all_dates = business_days.copy()

    # Also include some weekend dates for this pattern
    current = _SIM_START
    while current <= _SIM_END:
        if current.weekday() >= 5:
            all_dates.append(current)
        current += timedelta(days=1)
    all_dates.sort()

    idx = rng.choice(len(all_dates), size=min(100, len(all_dates)), replace=False)
    active_days = [all_dates[int(i)] for i in idx]

    for d in sorted(active_days):
        n_trades = int(rng.integers(2, 8))
        for _ in range(n_trades):
            # 80% off-hours
            is_off = rng.random() < 0.80
            ticker = str(rng.choice(all_tickers))
            sec_row = securities[securities["ticker"] == ticker].iloc[0]
            price = _trade_price(sec_row["avg_price_usd"], rng)
            qty = int(np.clip(rng.lognormal(np.log(200), 0.8), 1, 5000))
            tv = round(price * qty, 2)
            t = _market_time(rng, off_hours=is_off)
            trades.append(
                {
                    "trade_id": f"TRD-{tid:07d}",
                    "account_id": account_id,
                    "counterparty_account_id": None,
                    "ticker": ticker,
                    "trade_date": d.isoformat(),
                    "trade_time": t.strftime("%H:%M:%S"),
                    "trade_direction": str(rng.choice(["BUY", "SELL"])),
                    "quantity": qty,
                    "price_usd": price,
                    "trade_value_usd": tv,
                    "order_type": str(rng.choice(["MARKET", "LIMIT"])),
                    "time_to_execution_ms": _execution_ms(rng),
                    "is_off_hours": is_off,
                    "is_round_value": _is_round_value(tv),
                }
            )
            tid += 1

    return trades


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_trades(
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
    target_n: int = 100_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic trades and a labeled anomaly reference table.

    Returns (trades_df, injected_anomalies_df). The injected_anomalies_df is
    for validation only — it is never read by any model or feature module.
    """
    rng = np.random.default_rng(random_state)
    business_days = _business_days(_SIM_START, _SIM_END)

    # --- Select anomaly accounts (3–5% of accounts, ~15 per pattern) ---
    n_anomaly_per_pattern = 15
    n_anomaly_total = n_anomaly_per_pattern * len(ANOMALY_PATTERNS)
    anomaly_account_ids = list(
        rng.choice(accounts["account_id"].tolist(), size=n_anomaly_total, replace=False)
    )
    anomaly_labels: list[dict[str, Any]] = []
    anomaly_set: set[str] = set(anomaly_account_ids)

    pattern_assignments: dict[str, str] = {}
    for i, pattern in enumerate(ANOMALY_PATTERNS):
        for acct in anomaly_account_ids[
            i * n_anomaly_per_pattern : (i + 1) * n_anomaly_per_pattern
        ]:
            pattern_assignments[acct] = pattern
            anomaly_labels.append({"account_id": acct, "anomaly_pattern": pattern})

    # --- Assign activity levels to normal accounts ---
    activity = _assign_activity_levels(accounts, rng)

    # Scale activity levels so total normal trades ≈ 80% of target
    normal_accounts = [a for a in accounts["account_id"] if a not in anomaly_set]
    total_activity = sum(activity[a] * len(business_days) / 21 for a in normal_accounts)
    normal_target = int(target_n * 0.80)
    scale_factor = normal_target / max(total_activity, 1)

    # --- Generate normal trades ---
    all_trades: list[dict[str, Any]] = []
    tid_counter = 1

    for acct_id in normal_accounts:
        scaled_rate = activity[acct_id] * scale_factor
        trades = _generate_normal_trades(
            acct_id, scaled_rate, accounts, securities, business_days, rng, tid_counter
        )
        all_trades.extend(trades)
        tid_counter += len(trades)

    # --- Generate anomaly trades ---
    for acct_id, pattern in pattern_assignments.items():
        if pattern == "wash_trading":
            trades = _inject_wash_trading(
                acct_id, accounts, securities, business_days, rng, tid_counter
            )
        elif pattern == "velocity_spike":
            trades = _inject_velocity_spike(acct_id, securities, business_days, rng, tid_counter)
        elif pattern == "smurfing":
            trades = _inject_smurfing(acct_id, securities, business_days, rng, tid_counter)
        elif pattern == "illiquid_concentration":
            trades = _inject_illiquid_concentration(
                acct_id, securities, business_days, rng, tid_counter
            )
        elif pattern == "off_hours_clustering":
            trades = _inject_off_hours_clustering(
                acct_id, securities, business_days, rng, tid_counter
            )
        else:
            raise ValueError(f"Unknown anomaly pattern: {pattern}")
        all_trades.extend(trades)
        tid_counter += len(trades)

    trades_df = pd.DataFrame(all_trades)

    # Reassign trade_ids sequentially now that we have the full set
    trades_df = trades_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    trades_df["trade_id"] = [f"TRD-{i:07d}" for i in range(1, len(trades_df) + 1)]

    injected_df = pd.DataFrame(anomaly_labels)

    return trades_df, injected_df


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    holdout_dir = Path("data/holdout")
    raw_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    print("Loading accounts and securities...")
    accts = pd.read_csv(raw_dir / "accounts.csv")
    secs = pd.read_csv(raw_dir / "securities.csv")

    print("Generating trades...")
    trades_df, injected_df = generate_trades(accts, secs)

    # 80/20 time-based split: earliest 80% of trade dates → training, rest → holdout
    trades_df = trades_df.sort_values("trade_date").reset_index(drop=True)
    split_idx = int(len(trades_df) * 0.80)
    train_trades = trades_df.iloc[:split_idx]
    holdout_trades = trades_df.iloc[split_idx:]

    train_trades.to_csv(raw_dir / "trades.csv", index=False)
    holdout_trades.to_csv(holdout_dir / "new_trades.csv", index=False)
    injected_df.to_csv(raw_dir / "injected_anomalies.csv", index=False)

    print(f"Training trades:  {len(train_trades):,}")
    print(f"Holdout trades:   {len(holdout_trades):,}")
    print(f"Injected accounts: {len(injected_df)}")
    print(injected_df["anomaly_pattern"].value_counts().to_string())
