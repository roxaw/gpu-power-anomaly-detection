"""
Isolation Forest anomaly detector.

This module provides functions to train an Isolation Forest model on
engineered features and apply it to detect anomalies in GPU power data.

The Isolation Forest is an unsupervised outlier detection algorithm that
isolates anomalies by recursively partitioning the feature space. We expose
two functions:
- train_isolation_forest: fit the model on provided feature columns.
- detect_isolation_forest: score and label each row as anomaly or normal.
"""

from typing import List, Optional

import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    contamination: float = 0.05,
    random_state: Optional[int] = None,
) -> IsolationForest:
    """
    Train an Isolation Forest model on the specified feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature columns.
    feature_cols : list of str
        Names of the columns used for training.
    contamination : float, default 0.05
        Expected proportion of anomalies in the training data.
    random_state : int or None, default None
        Seed for the random number generator.

    Returns
    -------
    sklearn.ensemble.IsolationForest
        The fitted Isolation Forest model.
    """
    x = df[feature_cols].copy()
    x = x.bfill().ffill().fillna(0.0).values

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples="auto",
    )

    model.fit(x)
    return model


def detect_isolation_forest(
    df: pd.DataFrame,
    model: IsolationForest,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Apply a fitted Isolation Forest model to score and label anomalies.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature columns.
    model : sklearn.ensemble.IsolationForest
        Fitted Isolation Forest model.
    feature_cols : list of str
        Names of the columns used for scoring.

    Returns
    -------
    pandas.DataFrame
        A copy of df with two additional columns:
        - 'iforest_score': anomaly score, where higher means more anomalous.
        - 'iforest_anomaly': boolean flag indicating anomalies.
    """
    result = df.copy()

    x = result[feature_cols].copy()
    x = x.bfill().ffill().fillna(0.0).values

    decision_scores = model.decision_function(x)
    result["iforest_score"] = -decision_scores

    labels = model.predict(x)
    result["iforest_anomaly"] = labels == -1

    return result