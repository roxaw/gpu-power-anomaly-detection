"""
Evaluation utilities for the NAB dataset.

This module provides functions to load a time-series dataset such as NAB,
compute features, apply detectors, and compute simple evaluation metrics.

Assumptions:
- Input CSV should have at least 'timestamp' and 'value' or 'power' columns.
- An optional 'label' column, where 0 means normal and 1 means anomaly, can be
  used for computing precision, recall, and F1.
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.detectors.isolation_forest import (
    detect_isolation_forest,
    train_isolation_forest,
)
from src.detectors.threshold import detect_threshold
from src.features.builder import build_features


def load_nab_dataset(file_path: str, value_col: str = "value") -> pd.DataFrame:
    """
    Load a NAB-like CSV file into a DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    value_col : str, default "value"
        Name of the column containing the power or metric values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'timestamp' and 'power' columns, plus any other columns.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.rename(columns={value_col: "power"})
    return df


def _compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """
    Compute precision, recall, and F1 for binary anomaly labels.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_threshold(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    window: int = 10,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Apply the threshold detector and compute evaluation metrics if labels exist.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'power' and optionally a label column.
    z_thresh : float, default 3.0
        Z-score threshold.
    window : int, default 10
        Rolling window size.
    label_col : str or None, default None
        Name of the ground-truth label column.

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame with detector outputs.
    metrics : dict or None
        Dictionary with precision, recall, and F1, or None if labels unavailable.
    """
    df_feat = build_features(df, window=window)
    result = detect_threshold(df_feat, z_thresh=z_thresh, window=window)

    metrics = None
    if label_col:
        y_true = result[label_col].astype(bool)
        y_pred = result["threshold_anomaly"]
        metrics = _compute_metrics(y_true, y_pred)

    return result, metrics


def evaluate_isolation_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    contamination: float = 0.05,
    window: int = 10,
    label_col: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Apply the Isolation Forest detector and compute metrics if labels exist.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'power' and optionally a label column.
    feature_cols : list of str
        List of feature columns to use for Isolation Forest.
    contamination : float, default 0.05
        Expected proportion of anomalies.
    window : int, default 10
        Rolling window size.
    label_col : str or None, default None
        Name of the ground-truth label column.
    random_state : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame with detector outputs.
    metrics : dict or None
        Dictionary with precision, recall, and F1, or None if labels unavailable.
    """
    df_feat = build_features(df, window=window)
    model = train_isolation_forest(
        df_feat,
        feature_cols,
        contamination=contamination,
        random_state=random_state,
    )
    result = detect_isolation_forest(df_feat, model, feature_cols)

    metrics = None
    if label_col:
        y_true = result[label_col].astype(bool)
        y_pred = result["iforest_anomaly"]
        metrics = _compute_metrics(y_true, y_pred)

    return result, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate detectors on a NAB dataset")
    parser.add_argument("csv_path", help="Path to the NAB CSV file")
    parser.add_argument(
        "--label-col",
        default=None,
        help="Name of the label column, where 0 is normal and 1 is anomaly",
    )
    parser.add_argument(
        "--z-thresh",
        type=float,
        default=3.0,
        help="Z-score threshold for threshold detector",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Contamination for Isolation Forest",
    )
    args = parser.parse_args()

    df_nab = load_nab_dataset(args.csv_path)

    print("Evaluating threshold detector...")
    _, thresh_metrics = evaluate_threshold(
        df_nab,
        z_thresh=args.z_thresh,
        window=args.window,
        label_col=args.label_col,
    )
    print("Threshold metrics:", thresh_metrics)

    print("\nEvaluating Isolation Forest detector...")
    features_list = [
        "power",
        "prev_power",
        "delta",
        "pct_change",
        "rolling_mean",
        "rolling_std",
        "dev_from_mean",
        "z_score",
    ]
    _, if_metrics = evaluate_isolation_forest(
        df_nab,
        features_list,
        contamination=args.contamination,
        window=args.window,
        label_col=args.label_col,
    )
    print("Isolation Forest metrics:", if_metrics)