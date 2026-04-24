"""
Alert manager for anomaly detection.

This module defines a utility function to generate alert objects whenever a
detector flags an anomaly. Each alert includes contextual information that
can later be displayed in a dashboard or logged to disk.

Fields captured in each alert:
- timestamp: datetime of the data point.
- detector: name of the detector that fired.
- severity: numeric score indicating anomaly strength.
- reason: human-readable explanation for the anomaly.
- raw_value: raw power reading at the anomaly.
- features: dictionary of all feature values for that row.
"""

from typing import Any, Dict, List

import pandas as pd

from src.explanations.explanation import (
    explain_isolation_forest,
    explain_threshold,
)


def generate_alerts(
    df: pd.DataFrame,
    anomaly_col: str,
    score_col: str,
    detector_name: str,
) -> List[Dict[str, Any]]:
    """
    Generate alerts from a DataFrame of detection results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the detection results and feature columns.
    anomaly_col : str
        Name of the boolean column indicating whether each row is an anomaly.
    score_col : str
        Name of the numeric column providing the anomaly score or severity.
    detector_name : str
        Identifier for the detector.

    Returns
    -------
    list of dict
        A list of alert dictionaries, one per anomaly row.
    """
    alerts = []

    for _, row in df[df[anomaly_col]].iterrows():
        if detector_name == "threshold":
            reason = explain_threshold(row)
        elif detector_name == "isolation_forest":
            reason = explain_isolation_forest(row)
        else:
            reason = f"{detector_name} score {row[score_col]:.2f}"

        alert = {
            "timestamp": row["timestamp"],
            "detector": detector_name,
            "severity": row[score_col],
            "reason": reason,
            "raw_value": row["power"],
            "features": row.to_dict(),
        }
        alerts.append(alert)

    return alerts