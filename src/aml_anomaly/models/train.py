"""Fit and save all model objects (scaler, PCA, Isolation Forest, Local Outlier Factor (LOF)).

Writes fitted objects to disk so the scoring pipeline can apply them without refitting.

Training mode: run once on the full training dataset. Learns what normal
looks like across the account population, then saves every fitted object
so the scoring pipeline can apply them to new data without refitting.

Saved artifacts:
  models/scaler.pkl           - fitted StandardScaler
  models/pca.pkl              - fitted PCA
  models/isolation_forest.pkl - fitted Isolation Forest
  models/lof.pkl              - fitted Local Outlier Factor (LOF)
  models/feature_cols.json    - feature columns that survived null dropping
  models/medians.json         - per-column medians used for imputation
  models/population_stats.json - mean and std per feature for deviation scoring
"""

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from aml_anomaly.features.pipeline import build_feature_matrix
from aml_anomaly.models.dimensionality import preprocess_for_training
from aml_anomaly.models.ensemble import build_anomaly_scores
from aml_anomaly.models.isolation_forest import run_isolation_forest
from aml_anomaly.models.lof import run_lof

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
CONTAMINATION = 0.04


def _save(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def train(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    securities: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    contamination: float = CONTAMINATION,
) -> pd.DataFrame:
    """Run the full training pipeline and save all fitted objects to models_dir.

    Returns the scored training DataFrame so results can be inspected immediately.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Building feature matrix...")
    features = build_feature_matrix(trades, accounts, securities)
    features.to_csv(Path("data/features/feature_matrix.csv"), index=False)

    print("Preprocessing (null drop → impute → scale → PCA)...")
    X_pca, feature_cols, medians, scaler, pca = preprocess_for_training(features)
    print(f"  Input features: {len(feature_cols)}")
    print(f"  PCA components retained: {pca.n_components_}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    account_ids = features["account_id"]

    print("Fitting Isolation Forest...")
    if_results, if_model = run_isolation_forest(X_pca, account_ids, contamination)

    print("Fitting Local Outlier Factor (LOF)...")
    lof_results, lof_model = run_lof(X_pca, account_ids, contamination)

    print("Computing ensemble scores...")
    scores = build_anomaly_scores(
        account_ids,
        if_results["if_score_raw"].to_numpy(),
        lof_results["lof_score_raw"].to_numpy(),
        contamination=contamination,
    )

    # Population statistics — used by the reporting module to compute
    # how many standard deviations each flagged account deviates from the mean
    feature_data = features[feature_cols].fillna(medians)
    pop_stats = {
        col: {"mean": float(feature_data[col].mean()), "std": float(feature_data[col].std())}
        for col in feature_cols
    }

    print("Saving model artifacts...")
    _save(scaler, models_dir / "scaler.pkl")
    _save(pca, models_dir / "pca.pkl")
    _save(if_model, models_dir / "isolation_forest.pkl")
    _save(lof_model, models_dir / "lof.pkl")

    with open(models_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(models_dir / "medians.json", "w") as f:
        json.dump(medians, f)
    with open(models_dir / "population_stats.json", "w") as f:
        json.dump(pop_stats, f)

    flagged = scores["anomaly_flag"].sum()
    print("\nTraining complete.")
    print(f"  Accounts scored: {len(scores):,}")
    print(f"  Flagged (top {contamination:.0%}): {flagged}")

    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw_dir = Path("data/raw")

    print("Loading training data...")
    trades = pd.read_csv(raw_dir / "trades.csv")
    accounts = pd.read_csv(raw_dir / "accounts.csv")
    securities = pd.read_csv(raw_dir / "securities.csv")

    scores = train(trades, accounts, securities)

    output_path = Path("outputs/training_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")
