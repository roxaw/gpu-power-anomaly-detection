from src.data.generator import generate_power_series


def test_generate_power_series_shape_and_columns():
    df = generate_power_series(length=10, seed=42)

    assert len(df) == 10
    assert "timestamp" in df.columns
    assert "power" in df.columns
    assert df["power"].notna().all()