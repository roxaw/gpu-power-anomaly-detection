from src.data.generator import generate_power_series
from src.data.injector import inject_anomalies
from src.features.builder import build_features
from src.detectors.isolation_forest import (
    train_isolation_forest,
    detect_isolation_forest,
)
from src.alerts.manager import generate_alerts


def test_generate_alerts_returns_alert_dictionaries():
    df = generate_power_series(length=40, seed=5)
    df = inject_anomalies(
        df,
        [{"type": "spike", "index": 10, "amplitude": 80}],
        seed=5,
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
        random_state=5,
    )
    df_if = detect_isolation_forest(df_feat, model, feature_cols)

    alerts = generate_alerts(
        df_if,
        "iforest_anomaly",
        "iforest_score",
        "isolation_forest",
    )

    assert isinstance(alerts, list)

    if alerts:
        first = alerts[0]
        assert "timestamp" in first
        assert "detector" in first
        assert "severity" in first
        assert "reason" in first
        assert "raw_value" in first
        assert "features" in first