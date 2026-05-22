"""Isolation Forest anomaly detection model — primary model (60% weight in ensemble).

Isolation Forest works by randomly partitioning the feature space with decision
trees. Anomalies are isolated in fewer splits than normal observations because
they sit alone in sparse regions of the space. The fewer splits needed to isolate
an account, the more anomalous it is.

Parameters follow the PRD specification:
  n_estimators=200, contamination=0.04, max_samples='auto', random_state=42
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    X_pca: np.ndarray,
    contamination: float = 0.04,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest on the PCA-transformed feature matrix."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
    )
    model.fit(X_pca)
    return model


def score_isolation_forest(
    model: IsolationForest,
    X_pca: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score accounts and return (raw_scores, binary_flags).

    raw_scores: continuous — higher means more anomalous (we negate the
    model's internal score so the convention is consistent: high = bad).
    binary_flags: 1 if the model predicts anomaly, 0 if normal.
    """
    # score_samples returns negative scores — more negative = more anomalous.
    # We negate so higher = more anomalous, consistent with the ensemble convention.
    raw_scores = -model.score_samples(X_pca)
    binary_flags = np.where(model.predict(X_pca) == -1, 1, 0)
    return raw_scores, binary_flags


def run_isolation_forest(
    X_pca: np.ndarray,
    account_ids: pd.Series,
    contamination: float = 0.04,
) -> tuple[pd.DataFrame, IsolationForest]:
    """Fit Isolation Forest and return a scored DataFrame plus the fitted model."""
    model = fit_isolation_forest(X_pca, contamination=contamination)
    raw_scores, flags = score_isolation_forest(model, X_pca)

    results = pd.DataFrame(
        {
            "account_id": account_ids.values,
            "if_score_raw": raw_scores,
            "isolation_forest_flag": flags,
        }
    )
    return results, model
