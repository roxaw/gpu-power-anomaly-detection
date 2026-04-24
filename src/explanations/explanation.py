"""
Explanation functions for anomalies.

This module defines simple functions that transform raw detector outputs and
feature values into human-readable explanations. Each function accepts a
DataFrame row as a pandas Series and returns a string.
"""

import pandas as pd


def explain_threshold(row: pd.Series, z_thresh: float = 3.0) -> str:
    """
    Explain a threshold-based anomaly.

    The explanation highlights whether the power is above or below the rolling
    mean, the deviation in watts, and the z-score compared to the threshold.

    Parameters
    ----------
    row : pandas.Series
        A row from the feature DataFrame containing 'power', 'rolling_mean',
        'dev_from_mean', and 'z_score'.
    z_thresh : float, default 3.0
        The z-score threshold used for detection.

    Returns
    -------
    str
        A human-readable explanation.
    """
    power = row.get("power")
    mean = row.get("rolling_mean")
    dev = row.get("dev_from_mean")
    z_score = row.get("z_score")

    direction = "above" if dev > 0 else "below"

    explanation = (
        f"Power {power:.2f} W is {abs(dev):.2f} W {direction} the rolling mean "
        f"of {mean:.2f} W (z-score {z_score:.2f}, threshold {z_thresh:.2f})."
    )

    return explanation


def explain_isolation_forest(row: pd.Series) -> str:
    """
    Explain an Isolation Forest anomaly.

    The explanation includes the anomaly score and highlights key feature
    values such as delta, deviation from mean, and z-score.

    Parameters
    ----------
    row : pandas.Series
        A row from the feature DataFrame containing 'iforest_score', 'delta',
        'dev_from_mean', and 'z_score'.

    Returns
    -------
    str
        A human-readable explanation.
    """
    score = row.get("iforest_score")
    delta = row.get("delta")
    dev = row.get("dev_from_mean")
    z_score = row.get("z_score")

    explanation = (
        f"Isolation Forest anomaly score {score:.3f}. "
        f"Delta: {delta:.2f} W, deviation from mean: {dev:.2f} W, "
        f"z-score: {z_score:.2f}."
    )

    return explanation