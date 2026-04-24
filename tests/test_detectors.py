from src.data.generator import generate_power_series
from src.data.injector import inject_anomalies
from src.features.builder import build_features
from src.detectors.threshold import detect_threshold
from src.detectors.isolation_forest import (
    train_isolation_forest,
    detect_isolation_forest,
)


def test_threshold_detector_adds_output_columns():
    df = generate_power_series(length=30, seed=3)
    df = inject_anomalies(
        df,
        [{"type": "spike", "index": 10, "amplitude": 80}],
        seed=3,
    )
    df_feat = build_features(df, window=5)

    result = detect_threshold(df_feat, z_thresh=2.5, power_max=250)

    assert "threshold_score" in result.columns
    assert "threshold_anomaly" in result.columns


def test_isolation_forest_detector_adds_output_columns():
    df = generate_power_series(length=40, seed=4)
    df = inject_anomalies(
        df,
        [{"type": "spike", "index": 10, "amplitude": 80}],
        seed=4,
    )
    df_feat = build_features(df, window=5)

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

    model = train_isolation_forest(
        df_feat,
        feature_cols,
        contamination=0.1,
        random_state=4,
    )
    result = detect_isolation_forest(df_feat, model, feature_cols)

    assert "iforest_score" in result.columns
    assert "iforest_anomaly" in result.columns