from src.data.generator import generate_power_series
from src.features.builder import build_features


def test_build_features_adds_expected_columns():
    df = generate_power_series(length=20, seed=2)
    result = build_features(df, window=5)

    expected_cols = [
        "prev_power",
        "delta",
        "pct_change",
        "rolling_mean",
        "rolling_std",
        "dev_from_mean",
        "z_score",
    ]

    for col in expected_cols:
        assert col in result.columns

    assert len(result) == len(df)