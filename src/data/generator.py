"""
Synthetic GPU power generator.

This module provides a function to generate a time series representing GPU
power usage under normal conditions. Later modules will use this series as
a baseline for injecting anomalies.

Design notes:
- We simulate a baseline power consumption that fluctuates around a constant
  value, optionally with a linear trend.
- Random Gaussian noise is added to mimic real fluctuations.
- The function returns a pandas DataFrame with timestamp and power columns.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

def generate_power_series(
    length: int = 1000,
    start_time: Optional[datetime] = None,
    interval_seconds: int = 1,
    base_power: float = 200.0,
    noise_level: float = 5.0,
    trend: float = 0.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a synthetic GPU power time series representing normal operation.

    Parameters
    ----------
    length : int, default 1000
        Number of data points to generate.
    start_time : datetime or None, default None
        Starting timestamp. If None, uses the current UTC time.
    interval_seconds : int, default 1
        Sampling interval between points in seconds.
    base_power : float, default 200.0
        Baseline power usage (in watts).
    noise_level : float, default 5.0
        Standard deviation of random noise around the baseline.
    trend : float, default 0.0
        Linear trend to add per sample (in watts). Positive values create an
        upward drift; negative values create a downward drift.
    seed : int or None, default None
        Random seed for reproducibility. If None, randomness is not seeded.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - 'timestamp': datetime objects
        - 'power': simulated power usage values
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Determine start time
    start = start_time or datetime.utcnow()

    # Generate timestamps
    timestamps = [
        start + timedelta(seconds=i * interval_seconds) for i in range(length)
    ]

    # Generate baseline power with noise and optional trend
    noise = np.random.normal(loc=0.0, scale=noise_level, size=length)
    # Trend increases linearly over the series (trend * i)
    trend_component = np.linspace(0, trend * length, num=length)
    power = base_power + trend_component + noise

    # Construct DataFrame
    df = pd.DataFrame({'timestamp': timestamps, 'power': power})

    return df