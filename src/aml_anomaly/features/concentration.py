"""Compute ticker and counterparty concentration features per account.

Concentration features measure how narrowly an account's trading activity
is focused — on a single stock, a single counterparty, or illiquid securities.
Extreme concentration is a common AML signal.
"""

from datetime import timedelta

import numpy as np
import pandas as pd


def compute_concentration_features(
    trades: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per account with ticker and trade concentration features."""
    trades = trades.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])

    ref_date = trades["trade_date"].max()
    cutoff_30d = ref_date - timedelta(days=30)
    t30 = trades[trades["trade_date"] > cutoff_30d]

    all_accounts = trades["account_id"].unique()

    # Join security metadata onto trades (need is_illiquid and market_cap_tier)
    sec_meta = securities[["ticker", "is_illiquid", "market_cap_tier", "avg_daily_volume"]]
    trades_enriched = trades.merge(sec_meta, on="ticker", how="left")
    t30_enriched = t30.merge(sec_meta, on="ticker", how="left")

    # --- ticker concentration ---
    # For each account, what % of total trade value went to the single top ticker?
    ticker_value = (
        t30_enriched.groupby(["account_id", "ticker"])["trade_value_usd"].sum().reset_index()
    )
    account_total = ticker_value.groupby("account_id")["trade_value_usd"].sum()
    ticker_value = ticker_value.join(account_total.rename("account_total"), on="account_id")
    ticker_value["ticker_pct"] = ticker_value["trade_value_usd"] / ticker_value["account_total"]

    top1 = (
        ticker_value.sort_values("ticker_pct", ascending=False)
        .groupby("account_id")["ticker_pct"]
        .first()
        .rename("top_ticker_concentration_pct")
    )

    top3 = (
        ticker_value.sort_values("ticker_pct", ascending=False)
        .groupby("account_id")
        .apply(lambda x: x["ticker_pct"].head(3).sum(), include_groups=False)
        .rename("top_3_ticker_concentration_pct")
    )

    # --- illiquid and micro-cap concentration ---
    illiquid_value = (
        t30_enriched[t30_enriched["is_illiquid"] == True]  # noqa: E712
        .groupby("account_id")["trade_value_usd"]
        .sum()
        .rename("illiquid_value")
    )
    micro_value = (
        t30_enriched[t30_enriched["market_cap_tier"] == "Micro"]
        .groupby("account_id")["trade_value_usd"]
        .sum()
        .rename("micro_value")
    )
    total_30d = t30_enriched.groupby("account_id")["trade_value_usd"].sum().rename("total_30d")

    illiquid_pct = (illiquid_value / total_30d.replace(0, np.nan)).rename(
        "illiquid_stock_trade_pct"
    )
    micro_pct = (micro_value / total_30d.replace(0, np.nan)).rename("micro_cap_trade_pct")

    # --- trade size vs average daily volume ---
    # For each trade, compute quantity as % of that stock's avg daily volume
    trades_enriched["size_vs_adv"] = trades_enriched["quantity"] / trades_enriched[
        "avg_daily_volume"
    ].replace(0, np.nan)
    adv_max = (
        trades_enriched.groupby("account_id")["size_vs_adv"].max().rename("trade_size_vs_adv_max")
    )
    adv_avg = (
        trades_enriched.groupby("account_id")["size_vs_adv"].mean().rename("trade_size_vs_adv_avg")
    )

    # --- buy/sell ratio over 30 days ---
    buy_value = (
        t30[t30["trade_direction"] == "BUY"]
        .groupby("account_id")["trade_value_usd"]
        .sum()
        .rename("buy_value")
    )
    sell_value = (
        t30[t30["trade_direction"] == "SELL"]
        .groupby("account_id")["trade_value_usd"]
        .sum()
        .rename("sell_value")
    )
    buy_sell_ratio = (buy_value / sell_value.replace(0, np.nan)).rename("buy_sell_ratio_30d")

    # --- round value trade percentage ---
    round_pct = (
        trades.groupby("account_id")["is_round_value"].mean().rename("round_value_trade_pct")
    )

    # --- assemble ---
    feature_parts = [
        top1,
        top3,
        illiquid_pct,
        micro_pct,
        adv_max,
        adv_avg,
        buy_sell_ratio,
        round_pct,
    ]
    result = pd.DataFrame(index=pd.Index(all_accounts, name="account_id"))
    for part in feature_parts:
        result = result.join(part, how="left")

    result = result.fillna(0).reset_index()
    result = result.rename(columns={"index": "account_id"})

    return result
