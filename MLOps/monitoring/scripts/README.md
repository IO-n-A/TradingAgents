# MLOps Monitoring Scripts

This directory houses scripts used for various monitoring tasks within the MLOps lifecycle. These scripts can be scheduled to run periodically or triggered by events to assess data quality, model performance, data drift, and other relevant aspects of the MLOps system.

## Files

*   **`check_data_drift.py`**:
    *   **Objective:** To detect and quantify drift between a reference dataset (e.g., the dataset used for training a model) and a current dataset (e.g., recent production data or new batch data).
    *   **Functionality:**
        *   Takes paths to a reference dataset and a current dataset as input.
        *   Utilizes libraries like Evidently AI, Deepchecks, or custom statistical methods to compare the distributions and characteristics of the two datasets.
        *   Can check for various types of drift:
            *   **Covariate drift:** Changes in the distribution of input features.
            *   **Label drift:** Changes in the distribution of the target variable (if applicable).
            *   **Concept drift:** Changes in the relationship between input features and the target variable (harder to detect directly without labeled current data and model predictions).
        *   Outputs a report (e.g., HTML, JSON) summarizing the drift analysis, including drift scores for individual features and overall dataset drift.
        *   Can be configured to log results (e.g., to MLflow) or trigger alerts if significant drift is detected.
    *   **Usage:**
        ```bash
        python MLOps/monitoring/scripts/check_data_drift.py \
          --reference_data_path path/to/reference_data.csv \
          --current_data_path path/to/current_data.csv \
          --report_path MLOps/results/monitoring/data_drift_report.html
        ```
    *   **Dependencies:** Pandas, PyYAML. For advanced drift detection, libraries like `evidently` or `deepchecks` would be added.

*   **(Future Scripts - Examples)**
    *   **`check_model_degradation.py`**:
        *   Monitors the performance of a deployed model on recent production data.
        *   Compares current performance metrics (e.g., accuracy, F1-score, Sharpe ratio) against baseline metrics (e.g., from training or initial deployment).
        *   Alerts if performance degrades below a defined threshold.
    *   **`validate_data_schema.py`**:
        *   Checks if incoming data adheres to an expected schema (column names, data types, value ranges).
        *   Useful for data ingestion pipelines to catch data quality issues early.
    *   **`monitor_prediction_distribution.py`**:
        *   Tracks the distribution of predictions made by a deployed model.
        *   Significant shifts in prediction distribution might indicate issues with the model or changes in the input data.

## Integration

*   **Scheduling:** These scripts can be scheduled to run at regular intervals (e.g., daily, weekly) using tools like cron, Apache Airflow, Kubeflow Pipelines, or GitHub Actions.
*   **Triggering:** They can also be triggered as part of a CI/CD pipeline or an MLOps orchestration workflow (e.g., after a new batch of data is ingested, or before a model is promoted to production).
*   **Outputs & Alerting:**
    *   The scripts should output structured results (e.g., JSON, logs).
    *   Key findings (e.g., "drift_detected: true", "model_accuracy_degraded: true") can be exposed as metrics for Prometheus to scrape, enabling alerts via Alertmanager.
    *   Reports can be stored as artifacts in MLflow or other artifact stores.

## Development Guidelines for Monitoring Scripts

*   **Modularity:** Design scripts to be modular and focused on a specific monitoring task.
*   **Configurability:** Allow key parameters (data paths, thresholds, report paths) to be configured via command-line arguments or configuration files.
*   **Logging:** Implement comprehensive logging to track script execution and any issues.
*   **Error Handling:** Include robust error handling to manage unexpected situations gracefully.
*   **Dependencies:** Clearly define dependencies in a `requirements.txt` or similar.

These monitoring scripts are vital for maintaining the health, reliability, and performance of the MLOps system over time.