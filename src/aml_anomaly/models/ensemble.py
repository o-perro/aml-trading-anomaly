"""Combine Isolation Forest and Local Outlier Factor (LOF) scores into a composite anomaly score.

Each model's raw score is normalized to [0, 1] before combining so that
neither model dominates due to differences in scale. The ensemble formula is:

    anomaly_score = (normalized_IF_score × 0.6) + (normalized_LOF_score × 0.4)

Isolation Forest gets higher weight (0.6) because it is more robust to
high-dimensional data and less sensitive to the choice of n_neighbors.
Local Outlier Factor (LOF) gets 0.4 weight and adds value by catching local
outliers that blend into the global population but stand out in their peer group.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize a score array to [0, 1] using min-max scaling."""
    scaler = MinMaxScaler()
    result: np.ndarray = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    return result


def compute_ensemble_scores(
    if_scores_raw: np.ndarray,
    lof_scores_raw: np.ndarray,
    if_weight: float = 0.6,
    lof_weight: float = 0.4,
) -> np.ndarray:
    """Combine normalized Isolation Forest and Local Outlier Factor (LOF) scores."""
    if_norm = normalize_scores(if_scores_raw)
    lof_norm = normalize_scores(lof_scores_raw)
    return (if_norm * if_weight) + (lof_norm * lof_weight)


def build_anomaly_scores(
    account_ids: pd.Series,
    if_scores_raw: np.ndarray,
    lof_scores_raw: np.ndarray,
    contamination: float = 0.04,
) -> pd.DataFrame:
    """Build the final scored DataFrame with ranks and binary flags.

    Columns produced:
      if_score       - normalized Isolation Forest score [0, 1]
      lof_score      - normalized Local Outlier Factor (LOF) score [0, 1]
      anomaly_score  - weighted composite score [0, 1], higher = more anomalous
      anomaly_rank   - rank from most to least anomalous (1 = most anomalous)
      anomaly_flag   - 1 if account is in the top contamination% by anomaly_score
    """
    if_norm = normalize_scores(if_scores_raw)
    lof_norm = normalize_scores(lof_scores_raw)
    ensemble = compute_ensemble_scores(if_scores_raw, lof_scores_raw)

    n_flagged = max(1, int(len(account_ids) * contamination))

    df = pd.DataFrame(
        {
            "account_id": account_ids.values,
            "if_score": if_norm,
            "lof_score": lof_norm,
            "anomaly_score": ensemble,
        }
    )
    df["anomaly_rank"] = df["anomaly_score"].rank(ascending=False, method="first").astype(int)
    df["anomaly_flag"] = (df["anomaly_rank"] <= n_flagged).astype(int)

    return df.sort_values("anomaly_rank").reset_index(drop=True)
