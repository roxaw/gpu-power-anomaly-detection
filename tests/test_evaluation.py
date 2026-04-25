from src.data.generator import generate_power_series
from src.data.injector import inject_anomalies
from src.evaluation.nab_evaluation import (
    evaluate_isolation_forest,
    evaluate_threshold,
)


def test_evaluate_functions_return_metrics():
    df = generate_power_series(length=30, seed=0)

    anomalies = [
        {"type": "spike", "index": 5, "amplitude": 50},
        {"type": "drift", "index": 20, "length": 3, "amplitude": 20},
    ]
    df = inject_anomalies(df, anomalies, seed=1)

    df["label"] = df["anomaly"].astype(int)

    _, thresh_metrics = evaluate_threshold(
        df,
        z_thresh=2.5,
        window=5,
        label_col="label",
    )

    assert isinstance(thresh_metrics, dict)
    assert set(thresh_metrics.keys()) == {"precision", "recall", "f1"}

    for value in thresh_metrics.values():
        assert 0.0 <= value <= 1.0

    feature_cols = [
        "power",
        "prev_power",
        "delta",
        "pct_change",
        "rolling_mean",
        "rolling_std",
        "dev_from_mean",
        "z_score",
    ]

    _, iforest_metrics = evaluate_isolation_forest(
        df,
        feature_cols,
        contamination=0.1,
        window=5,
        label_col="label",
        random_state=42,
    )

    assert isinstance(iforest_metrics, dict)
    assert set(iforest_metrics.keys()) == {"precision", "recall", "f1"}

    for value in iforest_metrics.values():
        assert 0.0 <= value <= 1.0