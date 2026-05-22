"""Compute behavioral self-baseline deviation features per account.

These features compare each account to its own historical behavior rather
than to the population. This makes them powerful for catching accounts that
are acting differently from their own past — regardless of whether they are
active traders or quiet ones.
"""

from datetime import timedelta

import numpy as np
import pandas as pd


def _holding_periods(account_trades: pd.DataFrame) -> tuple[float, float]:
    """Return (avg_minutes, min_minutes) for matched buy/sell pairs using FIFO matching."""
    holding_times: list[float] = []

    dt_col = pd.to_datetime(
        account_trades["trade_date"]
        .astype(str)
        .str.cat(account_trades["trade_time"].astype(str), sep=" ")
    )
    account_trades = account_trades.copy()
    account_trades["_dt"] = dt_col

    for ticker, group in account_trades.groupby("ticker"):
        group = group.sort_values("_dt")
        buys = group[group["trade_direction"] == "BUY"]["_dt"].tolist()
        sells = group[group["trade_direction"] == "SELL"]["_dt"].tolist()

        for buy_dt in buys:
            future_sells = [s for s in sells if s > buy_dt]
            if future_sells:
                sell_dt = min(future_sells)
                holding_times.append((sell_dt - buy_dt).total_seconds() / 60)

    if not holding_times:
        return np.nan, np.nan
    return float(np.mean(holding_times)), float(np.min(holding_times))


def compute_behavioral_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Return one row per account with self-baseline behavioral deviation features."""
    trades = trades.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])

    ref_date = trades["trade_date"].max()
    cutoff_30d = ref_date - timedelta(days=30)

    # "Recent" = last 30 days. "Historical" = everything before that.
    t_recent = trades[trades["trade_date"] > cutoff_30d]
    t_hist = trades[trades["trade_date"] <= cutoff_30d]

    all_accounts = trades["account_id"].unique()

    # --- value z-score vs own history ---
    # Compare this month's total trade value to own historical monthly averages
    hist_monthly = (
        t_hist.assign(month=t_hist["trade_date"].dt.to_period("M"))
        .groupby(["account_id", "month"])["trade_value_usd"]
        .sum()
        .reset_index()
    )
    hist_stats = hist_monthly.groupby("account_id")["trade_value_usd"].agg(["mean", "std"])
    recent_value = t_recent.groupby("account_id")["trade_value_usd"].sum()

    value_zscore = (
        (recent_value - hist_stats["mean"]) / hist_stats["std"].replace(0, np.nan)
    ).rename("value_zscore_vs_self")

    # --- velocity z-score vs own history ---
    hist_daily_counts = (
        t_hist.groupby(["account_id", "trade_date"]).size().reset_index(name="daily_count")
    )
    hist_vel_stats = hist_daily_counts.groupby("account_id")["daily_count"].agg(["mean", "std"])
    recent_velocity = (
        t_recent.groupby(["account_id", "trade_date"]).size().groupby("account_id").mean()
    )
    velocity_zscore = (
        (recent_velocity - hist_vel_stats["mean"]) / hist_vel_stats["std"].replace(0, np.nan)
    ).rename("velocity_zscore_vs_self")

    # --- new ticker features ---
    # Tickers the account has ever traded before the recent window
    hist_tickers = t_hist.groupby("account_id")["ticker"].apply(set).rename("hist_tickers")
    recent_trades_with_hist = t_recent.join(hist_tickers, on="account_id", how="left")
    recent_trades_with_hist["hist_tickers"] = recent_trades_with_hist["hist_tickers"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    recent_trades_with_hist["is_new_ticker"] = recent_trades_with_hist.apply(
        lambda row: row["ticker"] not in row["hist_tickers"], axis=1
    )

    new_ticker_count = (
        recent_trades_with_hist[recent_trades_with_hist["is_new_ticker"]]
        .groupby("account_id")["ticker"]
        .nunique()
        .rename("new_ticker_count_30d")
    )
    new_ticker_pct = (
        recent_trades_with_hist.groupby("account_id")["is_new_ticker"]
        .mean()
        .rename("new_ticker_pct_30d")
    )

    # --- off-hours trade percentage ---
    off_hours_pct = (
        trades.groupby("account_id")["is_off_hours"].mean().rename("off_hours_trade_pct")
    )

    # --- weekend trade percentage ---
    trades["is_weekend"] = trades["trade_date"].dt.dayofweek >= 5
    weekend_pct = trades.groupby("account_id")["is_weekend"].mean().rename("weekend_trade_pct")

    # --- holding period (avg and min minutes between buy and next sell) ---
    holding_results: dict[str, tuple[float, float]] = {}
    for acct_id, grp in trades.groupby("account_id"):
        holding_results[str(acct_id)] = _holding_periods(grp)

    avg_holding = pd.Series(
        {k: v[0] for k, v in holding_results.items()},
        name="avg_holding_period_minutes",
    )
    min_holding = pd.Series(
        {k: v[1] for k, v in holding_results.items()},
        name="min_holding_period_minutes",
    )

    # --- assemble ---
    feature_parts = [
        value_zscore,
        velocity_zscore,
        new_ticker_count,
        new_ticker_pct,
        off_hours_pct,
        weekend_pct,
        avg_holding,
        min_holding,
    ]
    result = pd.DataFrame(index=pd.Index(all_accounts, name="account_id"))
    for part in feature_parts:
        result = result.join(part, how="left")

    # Z-scores default to 0 (no deviation) for accounts with insufficient history.
    # Count and pct features default to 0. Holding periods stay NaN (imputed later).
    result[["value_zscore_vs_self", "velocity_zscore_vs_self"]] = result[
        ["value_zscore_vs_self", "velocity_zscore_vs_self"]
    ].fillna(0)
    zero_fill_cols = [
        "new_ticker_count_30d",
        "new_ticker_pct_30d",
        "off_hours_trade_pct",
        "weekend_trade_pct",
    ]
    result[zero_fill_cols] = result[zero_fill_cols].fillna(0)

    result = result.reset_index()
    result = result.rename(columns={"index": "account_id"})

    return result
