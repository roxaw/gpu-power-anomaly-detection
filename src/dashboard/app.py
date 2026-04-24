"""
Streamlit dashboard for real-time GPU power anomaly monitoring.

This app visualises a synthetic GPU power time series, overlays anomaly markers
from both the threshold and Isolation Forest detectors, and lists the generated
alerts. Users can tweak a few parameters (number of points, rolling window,
thresholds) via the sidebar.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.manager import generate_alerts
from src.data.generator import generate_power_series
from src.data.injector import inject_anomalies
from src.detectors.isolation_forest import (
    detect_isolation_forest,
    train_isolation_forest,
)
from src.detectors.threshold import detect_threshold
from src.features.builder import build_features

def main():
    st.set_page_config(page_title="GPU Power Anomaly Dashboard", layout="wide")
    st.title("GPU Power Anomaly Dashboard")

    # Sidebar configuration
    st.sidebar.header("Simulation Settings")
    n_points = st.sidebar.slider("Number of points", min_value=100, max_value=2000, value=300, step=50)
    window = st.sidebar.slider("Rolling window size", min_value=3, max_value=50, value=10, step=1)
    z_thresh = st.sidebar.slider("Threshold z-score", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    contamination = st.sidebar.slider("IF contamination", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
    show_threshold = st.sidebar.checkbox("Show Threshold Detector", value=True)
    show_iforest = st.sidebar.checkbox("Show Isolation Forest Detector", value=True)

    # Generate synthetic baseline series
    df_base = generate_power_series(length=n_points, seed=42)

    # Inject a few anomalies at fixed positions for demonstration
    anomalies = [
        {'type': 'spike', 'index': int(n_points * 0.3), 'amplitude': 50},
        {'type': 'drop', 'index': int(n_points * 0.5), 'amplitude': 40},
        {'type': 'drift', 'index': int(n_points * 0.7), 'length': 5, 'amplitude': 20}
    ]
    df_inj = inject_anomalies(df_base, anomalies, seed=123)

    # Build features
    df_feat = build_features(df_inj, window=window)

    # Prepare the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_feat["timestamp"],
        y=df_feat["power"],
        mode="lines",
        name="Power",
        line=dict(color="blue")
    ))

    alerts = []

    # Apply threshold detector
    if show_threshold:
        df_thresh = detect_threshold(df_feat, z_thresh=z_thresh, window=window)
        thresh_anoms = df_thresh[df_thresh["threshold_anomaly"]]
        fig.add_trace(go.Scatter(
            x=thresh_anoms["timestamp"],
            y=thresh_anoms["power"],
            mode="markers",
            name="Threshold Anomalies",
            marker=dict(color="red", symbol="circle", size=6)
        ))
        alerts.extend(generate_alerts(df_thresh, "threshold_anomaly", "threshold_score", "threshold"))

    # Apply Isolation Forest detector
    if show_iforest:
        feature_cols = ['power', 'prev_power', 'delta', 'pct_change', 'rolling_mean',
                        'rolling_std', 'dev_from_mean', 'z_score']
        model = train_isolation_forest(df_feat, feature_cols, contamination=contamination, random_state=42)
        df_if = detect_isolation_forest(df_feat, model, feature_cols)
        if_anoms = df_if[df_if["iforest_anomaly"]]
        fig.add_trace(go.Scatter(
            x=if_anoms["timestamp"],
            y=if_anoms["power"],
            mode="markers",
            name="Isolation Forest Anomalies",
            marker=dict(color="orange", symbol="x", size=6)
        ))
        alerts.extend(generate_alerts(df_if, "iforest_anomaly", "iforest_score", "isolation_forest"))

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Display alerts table
    st.subheader("Active Alerts")
    if alerts:
        # Create a DataFrame for display (drop the 'features' column for brevity)
        alerts_df = pd.DataFrame(alerts).drop(columns=['features'])
        st.dataframe(alerts_df)
    else:
        st.write("No alerts to display.")

if __name__ == "__main__":
    main()