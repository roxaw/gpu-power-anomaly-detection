from src.data.generator import generate_power_series
from src.data.injector import inject_anomalies


def test_inject_spike_marks_anomaly():
    df = generate_power_series(length=20, seed=1)
    original_power = df.loc[5, "power"]

    result = inject_anomalies(
        df,
        [{"type": "spike", "index": 5, "amplitude": 50}],
        seed=1,
    )

    assert "anomaly" in result.columns
    assert result.loc[5, "anomaly"] is True or result.loc[5, "anomaly"] == True
    assert result.loc[5, "power"] == original_power + 50