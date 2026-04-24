"""
Feature builder for GPU power anomaly detection.

This module defines functions to compute engineered features from the raw
power time series. The goal is to provide interpretable inputs for both the
threshold-based and Isolation Forest detectors.

Features computed:
- prev_power: previous power reading (shifted by one).
- delta: difference between current power and previous power.
- pct_change: percent change relative to the previous value.
- rolling_mean: rolling mean of power over a specified window.
- rolling_std: rolling standard deviation over the window.
- dev_from_mean: deviation of current power from its rolling mean.
- z_score: deviation divided by rolling standard deviation (optional).
"""

import pandas as pd


def build_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute time-series features from a DataFrame containing a 'power' column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least a 'power' column.
    window : int, default 10
        Window size for rolling statistics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional feature columns.
    """
    features = df.copy()

    features["prev_power"] = features["power"].shift(1)
    features["delta"] = features["power"] - features["prev_power"]
    features["pct_change"] = features["delta"] / features["prev_power"].abs()

    features["rolling_mean"] = (
        features["power"]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    features["rolling_std"] = (
        features["power"]
        .rolling(window=window, min_periods=1)
        .std(ddof=0)
    )

    features["dev_from_mean"] = features["power"] - features["rolling_mean"]

    std_nonzero = features["rolling_std"].replace(0, pd.NA)
    features["z_score"] = features["dev_from_mean"] / std_nonzero

    return features