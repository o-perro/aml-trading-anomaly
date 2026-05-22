"""Run the full AML anomaly detection pipeline end-to-end.

Stages:
  1. Generate synthetic securities, accounts, and trades
  2. Split trades into training (80%) and holdout (20%) sets
  3. Train models — feature engineering, preprocessing, Isolation Forest,
     Local Outlier Factor (LOF), ensemble scoring
  4. Score holdout — apply fitted models to unseen data without retraining
"""

from pathlib import Path

import pandas as pd

from aml_anomaly.data_gen.accounts import generate_accounts
from aml_anomaly.data_gen.securities import generate_securities
from aml_anomaly.data_gen.trades import generate_trades
from aml_anomaly.models.score import score_trades
from aml_anomaly.models.train import train

RAW_DIR = Path("data/raw")
HOLDOUT_DIR = Path("data/holdout")
FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

RANDOM_STATE = 42


def _ensure_dirs() -> None:
    for d in [RAW_DIR, HOLDOUT_DIR, FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_data_generation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n[1/4] Generating synthetic data...")

    securities = generate_securities(n=200, random_state=RANDOM_STATE)
    securities.to_csv(RAW_DIR / "securities.csv", index=False)
    print(f"  Securities: {len(securities)} rows → data/raw/securities.csv")

    accounts = generate_accounts(n=2000, random_state=RANDOM_STATE)
    accounts.to_csv(RAW_DIR / "accounts.csv", index=False)
    print(f"  Accounts:   {len(accounts)} rows → data/raw/accounts.csv")

    trades, injected = generate_trades(accounts, securities, random_state=RANDOM_STATE)
    injected.to_csv(RAW_DIR / "injected_anomalies.csv", index=False)
    print(f"  Injected anomaly accounts: {len(injected)}")

    return securities, accounts, trades, injected  # type: ignore[return-value]


def run_train_holdout_split(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[2/4] Splitting trades into training and holdout sets...")

    # Random split so anomaly trades (concentrated in recent dates) appear in both sets.
    # A time-based split would push all recently-injected anomalies into holdout only,
    # leaving the training feature matrix with no anomalous behavior to learn from.
    trades_shuffled = trades.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(trades_shuffled) * 0.80)
    train_trades = trades_shuffled.iloc[:split_idx]
    holdout_trades = trades_shuffled.iloc[split_idx:]

    train_trades.to_csv(RAW_DIR / "trades.csv", index=False)
    holdout_trades.to_csv(HOLDOUT_DIR / "new_trades.csv", index=False)
    print(f"  Training trades:  {len(train_trades):,} → data/raw/trades.csv")
    print(f"  Holdout trades:   {len(holdout_trades):,} → data/holdout/new_trades.csv")

    return train_trades, holdout_trades


def run_training(
    train_trades: pd.DataFrame,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[3/4] Training models...")
    scores = train(train_trades, accounts, securities, models_dir=MODELS_DIR)
    scores.to_csv(OUTPUTS_DIR / "training_scores.csv", index=False)
    print(f"  Training scores saved → outputs/training_scores.csv")
    return scores


def run_holdout_scoring(
    holdout_trades: pd.DataFrame,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[4/4] Scoring holdout data...")
    scores = score_trades(holdout_trades, accounts, securities, models_dir=MODELS_DIR)
    scores.to_csv(OUTPUTS_DIR / "anomaly_scores.csv", index=False)
    print(f"  Holdout scores saved → outputs/anomaly_scores.csv")
    return scores


if __name__ == "__main__":
    _ensure_dirs()

    securities, accounts, trades, injected = run_data_generation()
    train_trades, holdout_trades = run_train_holdout_split(trades)
    training_scores = run_training(train_trades, accounts, securities)
    holdout_scores = run_holdout_scoring(holdout_trades, accounts, securities)

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print(f"  Total trades:         {len(trades):,}")
    print(f"  Injected anomalies:   {len(injected)}")
    print(f"  Holdout accounts scored: {len(holdout_scores):,}")
    print(f"  Holdout accounts flagged: {holdout_scores['anomaly_flag'].sum()}")
    print("\nNext step: open notebooks/03_pca_analysis.ipynb")
