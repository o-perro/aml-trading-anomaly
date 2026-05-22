"""Assemble all feature modules into a single modeling DataFrame.

This is the single entry point for feature engineering. Notebooks and the
training/scoring pipeline should call build_feature_matrix() rather than
importing individual feature modules directly.
"""

from pathlib import Path

import pandas as pd

from aml_anomaly.features.behavioral import compute_behavioral_features
from aml_anomaly.features.concentration import compute_concentration_features
from aml_anomaly.features.network import compute_network_features
from aml_anomaly.features.velocity import compute_velocity_features


def build_feature_matrix(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per account with all engineered features ready for modeling.

    Calls all four feature modules, merges on account_id, then appends account
    profile features (age, risk tier, PEP flag, account type encoding).
    """
    print("  Computing velocity features...")
    velocity = compute_velocity_features(trades)

    print("  Computing concentration features...")
    concentration = compute_concentration_features(trades, securities)

    print("  Computing behavioral features...")
    behavioral = compute_behavioral_features(trades)

    print("  Computing network features...")
    network = compute_network_features(trades)

    # Merge all feature sets on account_id
    features = (
        velocity.merge(concentration, on="account_id", how="outer")
        .merge(behavioral, on="account_id", how="outer")
        .merge(network, on="account_id", how="outer")
    )

    # Add account profile features
    account_profile = accounts[
        [
            "account_id",
            "account_age_days",
            "risk_tier",
            "is_pep",
            "account_type",
        ]
    ].copy()

    # Encode account_type as ordinal: Retail=1, Institutional=2, Broker-Dealer=3
    type_encoding = {"Retail": 1, "Institutional": 2, "Broker-Dealer": 3}
    account_profile["account_type_encoded"] = account_profile["account_type"].map(type_encoding)
    account_profile = account_profile.drop(columns=["account_type"])
    account_profile["is_pep"] = account_profile["is_pep"].astype(int)

    features = features.merge(account_profile, on="account_id", how="left")

    # Ensure all accounts from the accounts table are present, even with no trades
    all_accounts = accounts[["account_id"]].copy()
    features = all_accounts.merge(features, on="account_id", how="left")

    print(f"  Feature matrix: {len(features)} accounts × {len(features.columns) - 1} features")
    return features


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    trades = pd.read_csv(raw_dir / "trades.csv")
    accounts = pd.read_csv(raw_dir / "accounts.csv")
    securities = pd.read_csv(raw_dir / "securities.csv")

    print("Building feature matrix...")
    features = build_feature_matrix(trades, accounts, securities)

    output_path = features_dir / "feature_matrix.csv"
    features.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(features.describe().round(3).to_string())
