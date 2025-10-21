# MLOps Pipeline Orchestration

This directory is intended to hold scripts, configurations, or documentation related to orchestrating the various MLOps pipelines defined in the `MLOps/pipelines/` subdirectories. Pipeline orchestration involves defining the sequence of execution for different pipeline stages, managing dependencies between them, handling failures, and scheduling runs.

## Objective

The goal of orchestration is to automate the end-to-end MLOps workflow, from data ingestion to model deployment and monitoring, in a reliable and repeatable manner.

## Initial Approach: Master Shell Script

As a starting point, and for simplicity, the orchestration might be managed by a **master shell script** (e.g., `run_mlops_workflow.sh`). This script would:

1.  **Define Execution Order:** Call the individual pipeline scripts in the correct sequence:
    *   `MLOps/pipelines/data_ingestion/data_ingestion_pipeline.py`
    *   `MLOps/pipelines/feature_engineering/build_features.py`
    *   `MLOps/pipelines/sentiment_model/train_sentiment_model.py` (potentially conditional or less frequent)
    *   `MLOps/pipelines/rl_agent/train_rl_agent.py`
    *   `MLOps/deployment/backtesting/run_backtest.py` (after RL agent training)
    *   Monitoring scripts (e.g., `MLOps/monitoring/scripts/check_data_drift.py`) at appropriate points.

2.  **Manage DVC Commands:** Include DVC commands (`dvc add`, `dvc commit`, `dvc push`, `dvc pull`) at the appropriate stages to version data and ensure pipelines use the correct data versions.
    *   Example: After `data_ingestion_pipeline.py` finishes, run `dvc add data/raw/raw_financial_data.csv data/raw/raw_news_data.csv && dvc commit -m "Update raw data"`.
    *   Before `build_features.py` runs, ensure `dvc pull` is executed if running in an environment that doesn't have the latest data.

3.  **Pass Configurations:** Manage how configuration paths or parameters are passed to each pipeline script. This could involve setting environment variables or passing command-line arguments.

4.  **Basic Error Handling:** Check exit codes of each script and stop the workflow if a critical step fails.

5.  **Logging:** Consolidate logs or ensure individual pipeline logs are stored in a structured way.

### Example `run_mlops_workflow.sh` (Conceptual)

```bash
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting MLOps Workflow..."

# --- Configuration ---
GLOBAL_CONFIG="MLOps/config/common/global_vars.yaml"
DATA_SOURCES_CONFIG="MLOps/config/data_sources.yaml"
ENV_CONFIG="MLOps/config/environments/stock_trading_env_config.yaml"
SENTIMENT_MODEL_PARAMS="MLOps/config/sentiment_models/llama3_8b_lora_params.yaml"
RL_AGENT_PARAMS="MLOps/config/rl_agents/ppo_stocktrading_params.yaml" # Assuming one for now

RAW_FIN_DATA="raw_financial_data.csv"
RAW_NEWS_DATA="raw_news_data.csv"
PROCESSED_DATA="processed_data_with_features.csv"

# Ensure DVC is initialized (dvc init) - typically done once per project.

# --- 1. Data Ingestion ---
echo "Running Data Ingestion Pipeline..."
python MLOps/pipelines/data_ingestion/data_ingestion_pipeline.py \
    --global_config_path "$GLOBAL_CONFIG" \
    --data_sources_config_path "$DATA_SOURCES_CONFIG" \
    --financial_data_filename "$RAW_FIN_DATA" \
    --news_data_filename "$RAW_NEWS_DATA"

# dvc add MLOps/config/common/global_vars.yaml MLOps/config/data_sources.yaml # Track configs
# dvc add data/raw/"$RAW_FIN_DATA" data/raw/"$RAW_NEWS_DATA"
# dvc commit -m "Ingested new raw data"
# dvc push # If using remote DVC storage

# --- 2. Feature Engineering ---
echo "Running Feature Engineering Pipeline..."
# dvc pull data/raw/"$RAW_FIN_DATA" data/raw/"$RAW_NEWS_DATA" # Ensure latest raw data
python MLOps/pipelines/feature_engineering/build_features.py \
    --global_config_path "$GLOBAL_CONFIG" \
    --env_config_path "$ENV_CONFIG" \
    --sentiment_model_params_path "$SENTIMENT_MODEL_PARAMS" \
    --raw_financial_data_filename "$RAW_FIN_DATA" \
    --raw_news_data_filename "$RAW_NEWS_DATA" \
    --processed_data_filename "$PROCESSED_DATA"

# dvc add data/processed/"$PROCESSED_DATA"
# dvc commit -m "Engineered features from raw data"
# dvc push

# --- 3. Train Sentiment Model (Optional / Conditional) ---
# echo "Running Sentiment Model Training Pipeline..."
# python MLOps/pipelines/sentiment_model/train_sentiment_model.py --params_path "$SENTIMENT_MODEL_PARAMS" ...
# dvc add MLOps/model_registry/sentiment_models/... # Add trained model
# dvc commit -m "Trained new sentiment model"
# dvc push

# --- 4. Train RL Agent ---
echo "Running RL Agent Training Pipeline..."
# dvc pull data/processed/"$PROCESSED_DATA" # Ensure latest processed data
python MLOps/pipelines/rl_agent/train_rl_agent.py \
    --global_config_path "$GLOBAL_CONFIG" \
    --agent_params_path "$RL_AGENT_PARAMS" \
    --processed_data_filename "$PROCESSED_DATA"
    # Output model path will be handled by the script, potentially logged to MLflow

# dvc add MLOps/model_registry/rl_agents/... # Add trained model if not using MLflow registry exclusively
# dvc commit -m "Trained new RL agent"
# dvc push

# --- 5. Backtest RL Agent ---
echo "Running RL Agent Backtesting..."
# This would need the path to the latest trained model, perhaps from MLflow or a known location
# LATEST_MODEL_PATH="MLOps/model_registry/rl_agents/ppo_latest/model.zip" # Example
# python MLOps/deployment/backtesting/run_backtest.py \
#     --model_path "$LATEST_MODEL_PATH" \
#     --data_path data/processed/"$PROCESSED_DATA" \ # Or a specific test split
#     ...

# --- 6. Data Drift Check (Example) ---
# echo "Running Data Drift Check..."
# python MLOps/monitoring/scripts/check_data_drift.py \
#     --reference_data_path data/processed/training_snapshot.csv \ # Snapshot from training
#     --current_data_path data/processed/"$PROCESSED_DATA" \ # Current batch
#     ...

echo "MLOps Workflow Finished."
```

## Future: Advanced Orchestration Tools

For more complex workflows, robust error handling, parallel execution, scheduling, and better monitoring of pipeline states, dedicated orchestration tools should be considered:

*   **Apache Airflow:**
    *   Define workflows as Directed Acyclic Graphs (DAGs) in Python.
    *   Rich UI for monitoring, scheduling, and managing pipelines.
    *   Extensive library of operators for various tasks (Bash, Python, Docker, DVC, etc.).
*   **Kubeflow Pipelines (KFP):**
    *   Designed for orchestrating ML workflows on Kubernetes.
    *   Components can be containerized, promoting reproducibility and scalability.
    *   Integrates well with other Kubeflow components (e.g., Katib for hyperparameter tuning, KFServing for model serving).
*   **Prefect:**
    *   A modern Python-based workflow management system, often seen as a more flexible alternative to Airflow for certain use cases.
*   **Dagster:**
    *   A data orchestrator for the full development lifecycle, with a focus on data assets and local development experience.
*   **GitHub Actions / GitLab CI/CD:**
    *   Can be used for simpler CI/CD-driven orchestration, especially if the workflow is triggered by code changes or scheduled events.

The choice of an advanced orchestrator depends on the project's scale, complexity, existing infrastructure (e.g., Kubernetes availability), and team familiarity. The initial shell script approach provides a functional starting point that can be migrated to a more sophisticated tool later.