# Real-Time Anomaly Detection for GPU Power Usage in AI Data Centers

This project implements a lean, modular system for detecting anomalies in GPU power usage in AI data centers. It is built for a course project and follows a strict plan focused on a **vertical slice** rather than a full observability platform.

## Objectives

- **Synthetic data generator** – create a GPU power time series with normal behaviour and injected anomalies.
- **Feature builder** – compute simple features (current value, delta, percent change, rolling statistics) from each timestamp.
- **Detectors** – implement a rule-based threshold baseline and an Isolation Forest detector.
- **Alert manager** – create alerts when anomalies occur, capturing timestamp, detector, severity, reason, raw value and features.
- **Dashboard** – build a simple Streamlit dashboard with a live power chart, anomaly markers, active alerts table, and an explanation panel.
- **Phase 2 evaluation** – evaluate the detectors on the NAB benchmark dataset.

I want to avoid heavy infrastructure (e.g., Kafka, Spark, Prometheus) and large observability stacks. The system is designed to be small, interpretable, and executable on a laptop.

## Build Plan

The implementation follows a build plan:

1. **Repo creation and skeleton** (this step).
2. Local environment setup and virtual environment.
3. Implement synthetic data generator and anomaly injector.
4. Feature engineering module.
5. Threshold detector implementation.
6. Isolation Forest implementation.
7. Alert manager.
8. Dashboard.
9. Explanation layer.
10. NAB evaluation.
11. Final polish and reporting.

## Development Setup

Using a Python virtual environment to keep dependencies isolated from your system installation.

1. **Create a virtual environment** (requires Python 3.8+):
   ```bash
   python3 -m venv .venv


## Usage

### Run the Streamlit dashboard

Activate the virtual environment first:

```powershell
.venv\Scripts\activate

## NAB Evaluation Result

For Phase 2 evaluation, the project was tested on the NAB file:

`realKnownCause/cpu_utilization_asg_misconfiguration.csv`

This file is infrastructure telemetry representing AWS CPU utilization across a cluster. Since NAB stores anomaly labels separately as anomaly windows, the anomaly window labels were converted into a point-wise binary `label` column for simple precision, recall, and F1 evaluation.

Command used:

```powershell
python -m src.evaluation.nab_evaluation data\nab\cpu_utilization_asg_misconfiguration_labeled.csv --label-col label --z-thresh 2.5 --window 10 --contamination 0.05

| Detector           | Precision | Recall |     F1 |
| ------------------ | --------: | -----: | -----: |
| Threshold detector |    0.0376 | 0.0667 | 0.0481 |
| Isolation Forest   |    0.3223 | 0.1941 | 0.2423 |
