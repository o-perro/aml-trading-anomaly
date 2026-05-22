"""Generate plain-English flag narratives and ranked feature contribution tables.

Every flagged account gets two layers of output:

  Layer 1 — Plain-English narrative (2-4 sentences, no variable names, no jargon)
  Layer 2 — Ranked feature contributions table (top 5 features by deviation from
             population mean, in standard deviations)

The FEATURE_LABELS dictionary is the single source of truth for plain-English
feature names across the entire reporting layer. All narratives and ranked tables
pull from here — never duplicate label strings elsewhere.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Central feature label dictionary
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    # Velocity & frequency
    "trades_per_day_30d": "Average trades per day (30-day)",
    "trades_per_day_7d": "Average trades per day (7-day)",
    "trade_value_per_day_30d": "Average daily trade value (30-day)",
    "trade_value_per_day_7d": "Average daily trade value (7-day)",
    "max_trades_in_1hr": "Most trades in a single hour",
    "avg_time_between_trades_sec": "Average time between trades (seconds)",
    "min_time_between_trades_sec": "Shortest gap between any two trades (seconds)",
    "burst_event_count": "Times with 5+ trades in a 30-minute window",
    "velocity_ratio_7d_vs_30d": "Recent trading pace vs. 30-day average",
    # Concentration
    "top_ticker_concentration_pct": "% of activity in single stock",
    "top_3_ticker_concentration_pct": "% of activity in top 3 stocks",
    "illiquid_stock_trade_pct": "% of trades in thinly-traded stocks",
    "trade_size_vs_adv_max": "Largest trade as % of stock's daily volume",
    "trade_size_vs_adv_avg": "Average trade size as % of stock's daily volume",
    "micro_cap_trade_pct": "% of trades in micro-cap stocks",
    "buy_sell_ratio_30d": "Buy-to-sell ratio (30-day)",
    "round_value_trade_pct": "% of trades with round dollar values",
    # Behavioral self-baseline
    "value_zscore_vs_self": "Trade value vs. own 90-day history",
    "velocity_zscore_vs_self": "Trade frequency vs. own 90-day history",
    "new_ticker_count_30d": "New stocks traded in past 30 days",
    "new_ticker_pct_30d": "% of recent trades in newly traded stocks",
    "off_hours_trade_pct": "% of trades outside market hours",
    "weekend_trade_pct": "% of trades on weekends",
    "avg_holding_period_minutes": "Average holding period (minutes)",
    "min_holding_period_minutes": "Shortest holding period (minutes)",
    # Network & counterparty
    "unique_counterparties_30d": "Unique counterparties (30-day)",
    "top_counterparty_concentration_pct": "% of trades with single counterparty",
    "new_counterparty_count_30d": "New counterparties this month",
    "same_day_reversal_count": "Same-day buy/sell reversals",
    "circular_trade_flag": "Circular trading chain detected",
    "shared_counterparty_ticker_count": "Stocks repeatedly traded with same counterparty",
    # Account profile
    "account_age_days": "Account age (days)",
    "risk_tier": "KYC risk tier",
    "is_pep": "Politically Exposed Person",
    "account_type_encoded": "Account type",
}


# ---------------------------------------------------------------------------
# Ranked feature contributions
# ---------------------------------------------------------------------------


def get_ranked_features(
    account_id: str,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    population_stats: dict[str, dict[str, float]],
    top_n: int = 5,
) -> pd.DataFrame:
    """Return a ranked table of the top_n features driving this account's anomaly score.

    Rank is determined by how far the account's value deviates from the population
    mean in standard deviations (σ). Higher absolute deviation = higher rank.
    """
    account_row = features[features["account_id"] == account_id]
    if len(account_row) == 0:
        return pd.DataFrame()

    account_vals = account_row.iloc[0]
    feature_cols = [c for c in features.columns if c in population_stats]

    rows = []
    for col in feature_cols:
        if col not in FEATURE_LABELS:
            continue
        val = account_vals.get(col, np.nan)
        if pd.isna(val):
            continue
        mean = population_stats[col]["mean"]
        std = population_stats[col]["std"]
        deviation = (val - mean) / std if std > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "account_value": round(float(val), 4),
                "population_mean": round(mean, 4),
                "deviation_sigma": round(deviation, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .assign(abs_dev=lambda x: x["deviation_sigma"].abs())
        .sort_values("abs_dev", ascending=False)
        .drop(columns=["abs_dev"])
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index = result.index + 1
    result.index.name = "rank"
    return result


# ---------------------------------------------------------------------------
# Plain-English narrative generation
# ---------------------------------------------------------------------------


def _phrase(
    col: str,
    val: float,
    mean: float,
    std: float,
) -> str | None:
    """Return a plain-English phrase for a single feature value, or None if not notable."""
    if std == 0 or abs(val - mean) / std < 1.5:
        return None

    if col == "velocity_ratio_7d_vs_30d" and val > 1.5:
        return f"trading activity surged to {val:.1f}× its normal pace over the past 7 days"

    if col == "value_zscore_vs_self" and val > 2:
        return (
            f"total trade value was {val:.1f} standard deviations above its own historical average"
        )

    if col == "off_hours_trade_pct" and val > 0.15:
        return f"{val:.0%} of trades were placed outside normal market hours (9:30am–4:00pm ET)"

    if col == "weekend_trade_pct" and val > 0.05:
        return f"{val:.0%} of trades were placed on weekends"

    if col == "same_day_reversal_count" and val >= 2:
        return f"executed {int(val)} same-day buy/sell reversals on the same stock"

    if col == "buy_sell_ratio_30d" and 0.85 <= val <= 1.15 and mean > 0.5:
        return (
            "buy and sell activity were nearly equal in volume"
            " — a pattern consistent with wash trading"
        )

    if col == "top_ticker_concentration_pct" and val > 0.5:
        return f"{val:.0%} of total trade value was concentrated in a single stock"

    if col == "illiquid_stock_trade_pct" and val > 0.3:
        return f"{val:.0%} of trade value was directed at thinly-traded, illiquid stocks"

    if col == "micro_cap_trade_pct" and val > 0.3:
        return f"{val:.0%} of trade value was in micro-cap stocks"

    if col == "round_value_trade_pct" and val > 0.3:
        return (
            f"{val:.0%} of trades had suspiciously round dollar values"
            " — a potential structuring signal"
        )

    if col == "burst_event_count" and val >= 2:
        return f"placed 5 or more trades within a 30-minute window on {int(val)} separate occasions"

    if col == "top_counterparty_concentration_pct" and val > 0.7:
        return f"{val:.0%} of trades were directed at a single counterparty"

    if col == "circular_trade_flag" and val == 1:
        return (
            "appears in a circular trading chain where the same stock"
            " passed through multiple accounts"
        )

    if col == "trade_size_vs_adv_max" and val > 0.1:
        return (
            f"placed at least one trade representing {val:.0%} of that stock's average daily volume"
        )

    return None


def generate_narrative(
    account_id: str,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    population_stats: dict[str, dict[str, float]],
) -> str:
    """Return a 2–4 sentence plain-English explanation of why this account was flagged.

    Covers only the top contributing features. Uses no variable names or
    statistical jargon — written for an AML analyst, not a data scientist.
    """
    account_row = features[features["account_id"] == account_id]
    score_row = scores[scores["account_id"] == account_id]

    if len(account_row) == 0:
        return "No feature data available for this account."

    account_vals = account_row.iloc[0]
    anomaly_rank = int(score_row.iloc[0]["anomaly_rank"]) if len(score_row) > 0 else 0

    # Compute deviations for all features and sort by abs deviation
    deviations: list[tuple[str, float, float, float, float]] = []
    for col, stats in population_stats.items():
        val = account_vals.get(col, np.nan)
        if pd.isna(val):
            continue
        mean = stats["mean"]
        std = stats["std"]
        if std > 0:
            dev = abs(val - mean) / std
            deviations.append((col, float(val), mean, std, dev))

    deviations.sort(key=lambda x: x[4], reverse=True)
    top_features = deviations[:8]  # consider top 8, select best phrases

    phrases = []
    for col, val, mean, std, _ in top_features:
        phrase = _phrase(col, val, mean, std)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= 4:
            break

    if not phrases:
        return (
            f"This account (anomaly rank #{anomaly_rank}) displayed an unusual combination "
            "of behavioral signals across multiple feature dimensions."
        )

    # Build 2–4 sentences from the phrases
    sentences: list[str] = []

    if len(phrases) == 1:
        sentences.append(f"This account's {phrases[0]}.")
    elif len(phrases) == 2:
        sentences.append(f"This account's {phrases[0]}, and {phrases[1]}.")
    else:
        sentences.append(f"This account's {phrases[0]}. Additionally, {phrases[1]}.")
        if len(phrases) >= 3:
            sentences.append(f"The account also {phrases[2]}.")
        if len(phrases) >= 4:
            sentences.append(f"Further, {phrases[3]}.")

    sentences.append(
        f"This account ranks #{anomaly_rank} out of all scored accounts by anomaly score."
    )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Convenience loader for population stats
# ---------------------------------------------------------------------------


def load_population_stats(models_dir: Path = Path("models")) -> dict[str, Any]:
    """Load the population feature statistics saved during training."""
    with open(models_dir / "population_stats.json") as f:
        return json.load(f)  # type: ignore[no-any-return]
