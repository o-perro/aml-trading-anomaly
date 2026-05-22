"""Export anomaly scores and analyst reports to CSV and Excel.

Produces two output files:

  outputs/anomaly_scores.csv      - all accounts with raw scores and features
  outputs/flagged_accounts.xlsx   - analyst-ready Excel workbook with 4 tabs:

    Tab 1 — Summary          : one row per flagged account, sorted by anomaly rank
    Tab 2 — Feature Detail   : ranked feature contributions per flagged account
    Tab 3 — Supporting Trades: relevant individual trades per flagged account
    Tab 4 — Population Benchmarks: population mean and std per feature
"""

from pathlib import Path
from typing import Any

import pandas as pd

from aml_anomaly.reporting.flags import (
    FEATURE_LABELS,
    generate_narrative,
    get_ranked_features,
)


def export_anomaly_scores(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write all accounts with anomaly scores and feature values to CSV."""
    full = scores.merge(features, on="account_id", how="left")
    full.to_csv(output_path, index=False)
    print(f"  Anomaly scores: {len(full):,} accounts → {output_path}")


def _build_summary_tab(
    flagged_scores: pd.DataFrame,
    accounts: pd.DataFrame,
    features: pd.DataFrame,
    population_stats: dict[str, Any],
) -> pd.DataFrame:
    """One row per flagged account with profile, scores, and plain-English narrative."""
    rows = []
    for _, score_row in flagged_scores.iterrows():
        acct_id = str(score_row["account_id"])
        account_info = accounts[accounts["account_id"] == acct_id]

        narrative = generate_narrative(acct_id, features, flagged_scores, population_stats)

        row: dict[str, Any] = {
            "anomaly_rank": score_row["anomaly_rank"],
            "account_id": acct_id,
            "anomaly_score": round(float(score_row["anomaly_score"]), 4),
            "if_score": round(float(score_row["if_score"]), 4),
            "lof_score": round(float(score_row["lof_score"]), 4),
            "narrative": narrative,
        }

        if len(account_info) > 0:
            acct = account_info.iloc[0]
            row["account_type"] = acct.get("account_type", "")
            row["state"] = acct.get("state", "")
            row["risk_tier"] = acct.get("risk_tier", "")
            row["is_pep"] = acct.get("is_pep", "")
            row["account_age_days"] = acct.get("account_age_days", "")

        rows.append(row)

    return pd.DataFrame(rows).sort_values("anomaly_rank")


def _build_feature_detail_tab(
    flagged_scores: pd.DataFrame,
    features: pd.DataFrame,
    population_stats: dict[str, Any],
) -> pd.DataFrame:
    """Ranked feature contributions for every flagged account."""
    all_rows = []
    for _, score_row in flagged_scores.iterrows():
        acct_id = str(score_row["account_id"])
        ranked = get_ranked_features(acct_id, features, flagged_scores, population_stats)
        if len(ranked) == 0:
            continue
        ranked.insert(0, "account_id", acct_id)
        ranked.insert(1, "anomaly_rank", int(score_row["anomaly_rank"]))
        all_rows.append(ranked.reset_index())

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def _build_supporting_trades_tab(
    flagged_scores: pd.DataFrame,
    trades: pd.DataFrame,
    features: pd.DataFrame,
    population_stats: dict[str, Any],
    max_trades_per_account: int = 20,
) -> pd.DataFrame:
    """The most relevant individual trades for each flagged account.

    Selects trades that are most likely to have driven the anomalous features —
    off-hours trades, round-value trades, and same-counterparty trades are
    prioritized. Falls back to most recent trades if no specific signals exist.
    """
    all_rows = []

    display_cols = [
        "trade_id",
        "account_id",
        "trade_date",
        "trade_time",
        "ticker",
        "trade_direction",
        "quantity",
        "trade_value_usd",
        "counterparty_account_id",
        "is_off_hours",
        "is_round_value",
    ]
    available_cols = [c for c in display_cols if c in trades.columns]

    for _, score_row in flagged_scores.iterrows():
        acct_id = str(score_row["account_id"])
        acct_trades = trades[trades["account_id"] == acct_id].copy()
        if len(acct_trades) == 0:
            continue

        # Score each trade for relevance — prioritize the most suspicious ones
        acct_trades["_relevance"] = 0
        if "is_off_hours" in acct_trades.columns:
            acct_trades["_relevance"] += acct_trades["is_off_hours"].astype(int) * 2
        if "is_round_value" in acct_trades.columns:
            acct_trades["_relevance"] += acct_trades["is_round_value"].astype(int) * 2
        if "counterparty_account_id" in acct_trades.columns:
            acct_trades["_relevance"] += acct_trades["counterparty_account_id"].notna().astype(int)

        top_trades = (
            acct_trades.sort_values(["_relevance", "trade_value_usd"], ascending=[False, False])
            .head(max_trades_per_account)[available_cols]
            .copy()
        )
        top_trades.insert(0, "anomaly_rank", int(score_row["anomaly_rank"]))
        all_rows.append(top_trades)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def _build_population_benchmarks_tab(
    population_stats: dict[str, Any],
) -> pd.DataFrame:
    """Population mean and standard deviation for every feature."""
    rows = []
    for col, stats in population_stats.items():
        rows.append(
            {
                "feature": col,
                "plain_english_label": FEATURE_LABELS.get(col, col),
                "population_mean": round(stats["mean"], 4),
                "population_std": round(stats["std"], 4),
            }
        )
    return pd.DataFrame(rows).sort_values("feature")


def export_analyst_report(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    population_stats: dict[str, Any],
    output_path: Path,
) -> None:
    """Write the multi-tab analyst Excel workbook for all flagged accounts."""
    flagged = scores[scores["anomaly_flag"] == 1].sort_values("anomaly_rank")

    print(f"  Building analyst report for {len(flagged)} flagged accounts...")

    summary = _build_summary_tab(flagged, accounts, features, population_stats)
    feature_detail = _build_feature_detail_tab(flagged, features, population_stats)
    supporting_trades = _build_supporting_trades_tab(flagged, trades, features, population_stats)
    benchmarks = _build_population_benchmarks_tab(population_stats)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        feature_detail.to_excel(writer, sheet_name="Feature Detail", index=False)
        supporting_trades.to_excel(writer, sheet_name="Supporting Trades", index=False)
        benchmarks.to_excel(writer, sheet_name="Population Benchmarks", index=False)

    print(f"  Flagged accounts report → {output_path}")
    print(f"    Tab 1 — Summary:              {len(summary)} accounts")
    print(f"    Tab 2 — Feature Detail:        {len(feature_detail)} rows")
    print(f"    Tab 3 — Supporting Trades:     {len(supporting_trades)} trades")
    print(f"    Tab 4 — Population Benchmarks: {len(benchmarks)} features")
