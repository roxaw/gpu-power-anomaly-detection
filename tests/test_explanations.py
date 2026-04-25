import pandas as pd

from src.explanations.explanation import (
    explain_isolation_forest,
    explain_threshold,
)


def test_explain_threshold_contains_keywords():
    row = pd.Series(
        {
            "power": 210.0,
            "rolling_mean": 200.0,
            "dev_from_mean": 10.0,
            "z_score": 2.0,
        }
    )

    explanation = explain_threshold(row, z_thresh=2.5)

    assert (
        "above the rolling mean" in explanation
        or "below the rolling mean" in explanation
    )
    assert "z-score" in explanation


def test_explain_isolation_forest_contains_keywords():
    row = pd.Series(
        {
            "iforest_score": 0.15,
            "delta": -5.0,
            "dev_from_mean": -3.0,
            "z_score": -1.2,
        }
    )

    explanation = explain_isolation_forest(row)

    assert "Isolation Forest anomaly score" in explanation
    assert "Delta" in explanation
    assert "z-score" in explanation