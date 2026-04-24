"""
Threshold-based anomaly detector.

This module provides a simple rule-based detector that flags anomalies when
the power reading deviates from its rolling baseline beyond a z-score
threshold. Optionally, absolute power bounds can be specified to flag
extreme values.

The detector assumes the input DataFrame has at least a 'power' column.
If rolling statistics and z-score are not present, it automatically
computes them using the feature builder.
"""

from typing import Optional

import pandas as pd


def detect_threshold(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    power_min: Optional[float] = None,
    power_max: Optional[float] = None,
    window: int = 10,
) -> pd.DataFrame:
    """
    Detect anomalies in a power time series using z-score and absolute thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'power' column and optionally rolling
        statistics and 'z_score'.
    z_thresh : float, default 3.0
        Z-score threshold for flagging anomalies. Points with |z_score| > z_thresh
        are marked as anomalies.
    power_min : float or None, default None
        Minimum acceptable power. Values below this are flagged as anomalies.
        If None, no lower bound is enforced.
    power_max : float or None, default None
        Maximum acceptable power. Values above this are flagged as anomalies.
        If None, no upper bound is enforced.
    window : int, default 10
        Rolling window size used if rolling statistics are not already in df.

    Returns
    -------
    pandas.DataFrame
        A copy of df with two additional columns:
        - 'threshold_score': absolute z-score (severity).
        - 'threshold_anomaly': boolean indicating threshold-based anomalies.
    """
    result = df.copy()

    if "z_score" not in result.columns:
        from src.features.builder import build_features

        result = build_features(result, window=window)

    result["threshold_score"] = result["z_score"].abs()
    result["threshold_anomaly"] = result["threshold_score"] > z_thresh

    if power_min is not None:
        result.loc[result["power"] < power_min, "threshold_anomaly"] = True

    if power_max is not None:
        result.loc[result["power"] > power_max, "threshold_anomaly"] = True

    return result