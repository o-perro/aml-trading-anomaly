"""Compute trade velocity and frequency features per account.

All rolling windows are anchored to the most recent trade date in the dataset,
so features are always computed relative to the same reference point.
"""

from datetime import timedelta

import numpy as np
import pandas as pd


def _add_datetime(trades: pd.DataFrame) -> pd.DataFrame:
    """Combine trade_date and trade_time into a single trade_datetime column."""
    return trades.assign(
        trade_datetime=pd.to_datetime(
            trades["trade_date"].astype(str) + " " + trades["trade_time"].astype(str)
        )
    )


def _burst_event_count(
    times: list[pd.Timestamp],
    window_minutes: int = 30,
    threshold: int = 5,
) -> int:
    """Count non-overlapping windows where threshold+ trades occurred within window_minutes."""
    if len(times) < threshold:
        return 0
    sorted_times = sorted(times)
    count = 0
    i = 0
    while i < len(sorted_times):
        window_end = sorted_times[i] + timedelta(minutes=window_minutes)
        j = i
        while j < len(sorted_times) and sorted_times[j] <= window_end:
            j += 1
        if j - i >= threshold:
            count += 1
            i = j  # skip past this burst to avoid double-counting
        else:
            i += 1
    return count


def compute_velocity_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Return one row per account with trade velocity and frequency features."""
    trades = _add_datetime(trades.copy())
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])

    ref_date = trades["trade_date"].max()
    cutoff_30d = ref_date - timedelta(days=30)
    cutoff_7d = ref_date - timedelta(days=7)

    t30 = trades[trades["trade_date"] > cutoff_30d]
    t7 = trades[trades["trade_date"] > cutoff_7d]

    all_accounts = trades["account_id"].unique()

    # --- trades_per_day and trade_value_per_day ---
    # Count active days (days with at least one trade) per account per window
    def _trades_per_active_day(window: pd.DataFrame) -> pd.Series:
        daily = window.groupby(["account_id", "trade_date"]).size().reset_index(name="n")
        trades_per_day = daily.groupby("account_id")["n"].mean()
        return trades_per_day

    def _value_per_day(window: pd.DataFrame) -> pd.Series:
        daily = window.groupby(["account_id", "trade_date"])["trade_value_usd"].sum()
        return daily.groupby("account_id").mean()

    tpd_30 = _trades_per_active_day(t30).rename("trades_per_day_30d")
    tpd_7 = _trades_per_active_day(t7).rename("trades_per_day_7d")
    vpd_30 = _value_per_day(t30).rename("trade_value_per_day_30d")
    vpd_7 = _value_per_day(t7).rename("trade_value_per_day_7d")

    # --- max trades in any single hour ---
    trades["trade_hour_bucket"] = trades["trade_datetime"].dt.floor("h")
    max_in_hour = (
        trades.groupby(["account_id", "trade_hour_bucket"])
        .size()
        .groupby("account_id")
        .max()
        .rename("max_trades_in_1hr")
    )

    # --- time between consecutive trades (seconds) ---
    trades_sorted = trades.sort_values(["account_id", "trade_datetime"])
    trades_sorted["prev_datetime"] = trades_sorted.groupby("account_id")["trade_datetime"].shift(1)
    trades_sorted["gap_sec"] = (
        trades_sorted["trade_datetime"] - trades_sorted["prev_datetime"]
    ).dt.total_seconds()

    inter_trade = trades_sorted.dropna(subset=["gap_sec"]).groupby("account_id")["gap_sec"]
    avg_gap = inter_trade.mean().rename("avg_time_between_trades_sec")
    min_gap = inter_trade.min().rename("min_time_between_trades_sec")

    # --- burst event count ---
    burst_counts: dict[str, int] = {}
    for acct_id, grp in trades.groupby("account_id"):
        burst_counts[str(acct_id)] = _burst_event_count(grp["trade_datetime"].tolist())
    burst_series = pd.Series(burst_counts, name="burst_event_count")

    # --- velocity ratio: 7d pace vs 30d pace ---
    velocity_ratio = (tpd_7 / tpd_30.replace(0, np.nan)).rename("velocity_ratio_7d_vs_30d")

    # --- assemble ---
    feature_parts = [
        tpd_30,
        tpd_7,
        vpd_30,
        vpd_7,
        max_in_hour,
        avg_gap,
        min_gap,
        burst_series,
        velocity_ratio,
    ]
    result = pd.DataFrame(index=pd.Index(all_accounts, name="account_id"))
    for part in feature_parts:
        result = result.join(part, how="left")

    result = result.fillna(0).reset_index()
    result = result.rename(columns={"index": "account_id"})

    return result
