"""Local Outlier Factor (LOF) anomaly detection model — secondary model (40% weight).

Local Outlier Factor (LOF) measures how isolated an account is relative to its
nearest neighbors in feature space. An account in a sparse region scores higher
than one surrounded by similar accounts, even if it is not a global outlier.

This makes Local Outlier Factor (LOF) especially powerful for detecting accounts
that are unusual relative to their peer group — a retail investor who suddenly
trades like an institutional one, for example.

Parameters follow the PRD specification:
  n_neighbors=20, contamination=0.04, metric='euclidean'
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def fit_lof(
    X_pca: np.ndarray,
    contamination: float = 0.04,
    n_neighbors: int = 20,
) -> LocalOutlierFactor:
    """Fit a Local Outlier Factor (LOF) model on the PCA-transformed feature matrix."""
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric="euclidean",
        novelty=True,  # novelty=True allows scoring new data at inference time
    )
    model.fit(X_pca)
    return model


def score_lof(
    model: LocalOutlierFactor,
    X_pca: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score accounts and return (raw_scores, binary_flags).

    raw_scores: continuous — higher means more anomalous.
    binary_flags: 1 if Local Outlier Factor (LOF) predicts anomaly, 0 if normal.
    """
    # score_samples returns negative LOF scores — more negative = more anomalous.
    # We negate so higher = more anomalous, matching the Isolation Forest convention.
    raw_scores = -model.score_samples(X_pca)
    binary_flags = np.where(model.predict(X_pca) == -1, 1, 0)
    return raw_scores, binary_flags


def run_lof(
    X_pca: np.ndarray,
    account_ids: pd.Series,
    contamination: float = 0.04,
) -> tuple[pd.DataFrame, LocalOutlierFactor]:
    """Fit Local Outlier Factor (LOF) and return a scored DataFrame plus the fitted model."""
    model = fit_lof(X_pca, contamination=contamination)
    raw_scores, flags = score_lof(model, X_pca)

    results = pd.DataFrame(
        {
            "account_id": account_ids.values,
            "lof_score_raw": raw_scores,
            "lof_flag": flags,
        }
    )
    return results, model
