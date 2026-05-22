"""Score new trade data against fitted model objects without retraining.

Scoring mode: loads all fitted objects saved by train.py and applies them
to a new batch of trades. The model's definition of 'normal' never changes —
the same scaler, PCA, Isolation Forest, and Local Outlier Factor (LOF) that
were fitted on training data are applied directly to the new data.

This is the production-representative flow: new trades arrive, get featurized
using the same pipeline, and are scored against the already-fitted model.

CLI usage:
    uv run python -m aml_anomaly.models.score \\
        --trades data/holdout/new_trades.csv \\
        --output outputs/scored_holdout.xlsx
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from aml_anomaly.features.pipeline import build_feature_matrix
from aml_anomaly.models.dimensionality import preprocess_for_scoring
from aml_anomaly.models.ensemble import build_anomaly_scores
from aml_anomaly.models.isolation_forest import score_isolation_forest
from aml_anomaly.models.lof import score_lof

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RAW_DIR = Path("data/raw")
CONTAMINATION = 0.04


def _load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def score_trades(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    contamination: float = CONTAMINATION,
) -> pd.DataFrame:
    """Score a new batch of trades using the already-fitted model objects.

    Returns a DataFrame with one row per account and anomaly scores/flags.
    """
    print("Building feature matrix from new trades...")
    features = build_feature_matrix(trades, accounts, securities)

    print("Loading fitted model objects...")
    scaler: StandardScaler = _load(models_dir / "scaler.pkl")
    pca: PCA = _load(models_dir / "pca.pkl")
    if_model: IsolationForest = _load(models_dir / "isolation_forest.pkl")
    lof_model: LocalOutlierFactor = _load(models_dir / "lof.pkl")

    with open(models_dir / "feature_cols.json") as f:
        feature_cols: list[str] = json.load(f)
    with open(models_dir / "medians.json") as f:
        medians: dict[str, float] = json.load(f)

    print("Applying preprocessing (no refitting)...")
    X_pca = preprocess_for_scoring(features, feature_cols, medians, scaler, pca)

    print("Scoring with Isolation Forest...")
    if_scores_raw, _ = score_isolation_forest(if_model, X_pca)

    print("Scoring with Local Outlier Factor (LOF)...")
    lof_scores_raw, _ = score_lof(lof_model, X_pca)

    print("Computing ensemble scores...")
    scores = build_anomaly_scores(
        features["account_id"],
        if_scores_raw,
        lof_scores_raw,
        contamination=contamination,
    )

    flagged = scores["anomaly_flag"].sum()
    print("\nScoring complete.")
    print(f"  Accounts scored: {len(scores):,}")
    print(f"  Flagged (top {contamination:.0%}): {flagged}")

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Score new trades against fitted models.")
    parser.add_argument("--trades", required=True, help="Path to new trades CSV")
    parser.add_argument("--output", required=True, help="Path to output Excel file")
    args = parser.parse_args()

    print("Loading data...")
    trades = pd.read_csv(args.trades)
    accounts = pd.read_csv(RAW_DIR / "accounts.csv")
    securities = pd.read_csv(RAW_DIR / "securities.csv")

    scores = score_trades(trades, accounts, securities)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(str(output_path).replace(".xlsx", ".csv"), index=False)
    print(f"Scores saved to {output_path}")
