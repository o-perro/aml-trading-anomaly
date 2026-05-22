"""Compute counterparty network and relationship features per account.

Network features look at who an account trades with, not just what it trades.
Unusual counterparty patterns — trading exclusively with one account, appearing
in circular chains, repeated same-day reversals — are strong AML signals.
"""

from datetime import timedelta

import pandas as pd


def _detect_circular_trades(cp_trades: pd.DataFrame) -> set[str]:
    """Return account IDs that appear in a 3-party circular trading chain (A→B→C→A)."""
    adjacency: dict[str, set[str]] = {}
    for _, row in cp_trades.iterrows():
        a = str(row["account_id"])
        b = str(row["counterparty_account_id"])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    circular: set[str] = set()
    for a in list(adjacency.keys()):
        for b in adjacency.get(a, set()):
            for c in adjacency.get(b, set()):
                if c != a and c != b and a in adjacency.get(c, set()):
                    circular.update([a, b, c])
    return circular


def compute_network_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Return one row per account with counterparty network features."""
    trades = trades.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])

    ref_date = trades["trade_date"].max()
    cutoff_30d = ref_date - timedelta(days=30)

    t30 = trades[trades["trade_date"] > cutoff_30d].copy()
    t30_cp = t30[t30["counterparty_account_id"].notna()].reset_index(drop=True)

    all_accounts = trades["account_id"].unique()

    # If there are no counterparty trades at all, return all zeros — avoids
    # groupby failures on empty DataFrames with unpredictable column retention.
    all_cp_trades_full = trades[trades["counterparty_account_id"].notna()]
    if len(all_cp_trades_full) == 0:
        result = pd.DataFrame({"account_id": all_accounts})
        for col in [
            "unique_counterparties_30d",
            "top_counterparty_concentration_pct",
            "new_counterparty_count_30d",
            "same_day_reversal_count",
            "circular_trade_flag",
            "shared_counterparty_ticker_count",
        ]:
            result[col] = 0
        return result

    # --- unique counterparties in last 30 days ---
    unique_cp = (
        t30_cp.groupby("account_id")["counterparty_account_id"]
        .nunique()
        .rename("unique_counterparties_30d")
    )

    # --- top counterparty concentration ---
    cp_counts = (
        t30_cp.groupby(["account_id", "counterparty_account_id"])
        .size()
        .reset_index(name="cp_count")
    )
    account_cp_total = cp_counts.groupby("account_id")["cp_count"].sum().reset_index(name="total")
    cp_counts = cp_counts.merge(account_cp_total, on="account_id", how="left")
    cp_counts["cp_pct"] = cp_counts["cp_count"] / cp_counts["total"]

    top_cp_pct = (
        cp_counts.sort_values("cp_pct", ascending=False)
        .groupby("account_id")["cp_pct"]
        .first()
        .rename("top_counterparty_concentration_pct")
    )

    # --- new counterparties this month ---
    hist_mask = (trades["trade_date"] <= cutoff_30d) & (trades["counterparty_account_id"].notna())
    hist_cp_dict: dict[str, set[str]] = (
        trades[hist_mask].groupby("account_id")["counterparty_account_id"].apply(set).to_dict()
    )
    t30_cp["is_new_cp"] = [
        str(cp) not in hist_cp_dict.get(str(acct), set())
        for acct, cp in zip(t30_cp["account_id"], t30_cp["counterparty_account_id"])
    ]
    new_cp_count = (
        t30_cp[t30_cp["is_new_cp"]]
        .groupby("account_id")["counterparty_account_id"]
        .nunique()
        .rename("new_counterparty_count_30d")
    )

    # --- same-day reversals ---
    trades_cp = trades.copy().reset_index(drop=True)
    grp = trades_cp.groupby(["account_id", "trade_date", "ticker"])["trade_direction"]
    has_buy = grp.transform(lambda x: "BUY" in x.values)
    has_sell = grp.transform(lambda x: "SELL" in x.values)
    trades_cp["_is_reversal"] = has_buy & has_sell
    reversal_count = (
        trades_cp[trades_cp["_is_reversal"]]
        .groupby(["account_id", "trade_date", "ticker"])
        .first()
        .reset_index()[["account_id"]]
        .groupby("account_id")
        .size()
        .rename("same_day_reversal_count")
    )

    # --- circular trade flag ---
    circular_accounts = _detect_circular_trades(all_cp_trades_full)
    circular_flag = pd.Series(
        {acct: int(acct in circular_accounts) for acct in all_accounts},
        name="circular_trade_flag",
    )

    # --- shared counterparty ticker count ---
    shared = (
        t30_cp.groupby(["account_id", "counterparty_account_id", "ticker"])
        .size()
        .reset_index(name="n")
    )
    shared_ticker_count = (
        shared[shared["n"] > 1]
        .groupby("account_id")["ticker"]
        .nunique()
        .rename("shared_counterparty_ticker_count")
    )

    # --- assemble ---
    feature_parts = [
        unique_cp,
        top_cp_pct,
        new_cp_count,
        reversal_count,
        circular_flag,
        shared_ticker_count,
    ]
    result = pd.DataFrame(index=pd.Index(all_accounts, name="account_id"))
    for part in feature_parts:
        result = result.join(part, how="left")

    result = result.fillna(0).reset_index()
    result = result.rename(columns={"index": "account_id"})

    return result
