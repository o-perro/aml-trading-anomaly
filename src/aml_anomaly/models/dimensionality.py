"""Preprocessing pipeline: null handling, standard scaling, and PCA reduction.

This module prepares the raw feature matrix for anomaly detection:
  1. Drop features with >20% nulls (too sparse to impute reliably)
  2. Impute remaining nulls with the column median
  3. Standard scale all features (required for PCA and Local Outlier Factor (LOF))
  4. Apply Principal Component Analysis (PCA) — retain components explaining 95% of variance

All fitted objects (scaler, PCA) are returned so they can be saved to disk
and reapplied to new data without refitting.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

NULL_DROP_THRESHOLD = 0.20
PCA_VARIANCE_TARGET = 0.95


def drop_high_null_features(
    df: pd.DataFrame,
    threshold: float = NULL_DROP_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where more than threshold fraction of values are null.

    Returns the trimmed DataFrame and the list of dropped column names.
    """
    null_rates = df.isnull().mean()
    cols_to_drop = null_rates[null_rates > threshold].index.tolist()
    if cols_to_drop:
        logger.info("Dropping %d high-null features: %s", len(cols_to_drop), cols_to_drop)
    return df.drop(columns=cols_to_drop), cols_to_drop


def impute_medians(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill remaining nulls with each column's median.

    Returns the imputed DataFrame and a dict of {column: median_value} so the
    same medians can be applied to new data at scoring time.
    """
    medians = df.median().to_dict()
    return df.fillna(medians), {str(k): float(v) for k, v in medians.items()}


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler and return scaled data plus the fitted scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def fit_pca(
    X_scaled: np.ndarray,
    variance_target: float = PCA_VARIANCE_TARGET,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA retaining enough components to explain variance_target of total variance.

    Returns the transformed data and the fitted PCA object.
    """
    # First fit with all components to determine how many we need
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative_variance, variance_target) + 1)
    n_components = min(n_components, X_scaled.shape[1])

    logger.info(
        "PCA: retaining %d components (%.1f%% variance explained)",
        n_components,
        cumulative_variance[n_components - 1] * 100,
    )

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def preprocess_for_training(
    features: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, float], StandardScaler, PCA]:
    """Run the full preprocessing pipeline on the training feature matrix.

    Steps: drop high-null → impute medians → scale → PCA.

    Returns:
        X_pca: transformed array ready for model fitting
        feature_cols: the feature column names that survived null dropping
        medians: column medians used for imputation
        scaler: fitted StandardScaler
        pca: fitted PCA
    """
    meta_cols = ["account_id"]
    feature_cols = [c for c in features.columns if c not in meta_cols]
    X = features[feature_cols].copy()

    X, dropped = drop_high_null_features(X)
    feature_cols = [c for c in feature_cols if c not in dropped]

    X, medians = impute_medians(X)
    X_array = X.to_numpy()

    X_scaled, scaler = fit_scaler(X_array)
    X_pca, pca = fit_pca(X_scaled)

    return X_pca, feature_cols, medians, scaler, pca


def preprocess_for_scoring(
    features: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float],
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """Apply the already-fitted preprocessing objects to new data.

    Never refits — uses the scaler and PCA from training to keep the
    definition of 'normal' stable across scoring runs.
    """
    X = features[feature_cols].copy()
    X = X.fillna(medians)
    X_scaled = scaler.transform(X.to_numpy())
    result: np.ndarray = pca.transform(X_scaled)
    return result
