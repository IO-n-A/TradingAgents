# MLOps Experiment Tracking

This directory is dedicated to experiment tracking for model training, primarily using MLflow. Effective experiment tracking is crucial for reproducibility, comparison of different model versions, and understanding model performance.

## Files

*   **`mlflow_utils.py`**:
    *   A Python utility module containing helper functions to interact with an MLflow tracking server.
    *   **Key functionalities:**
        *   Setting up the MLflow tracking URI and experiment name.
        *   Starting and ending MLflow runs (including nested runs).
        *   Logging parameters (`mlflow.log_params`).
        *   Logging metrics (`mlflow.log_metrics`).
        *   Logging artifacts such as files or directories (`mlflow.log_artifact`, `mlflow.log_artifacts`). This can include configuration files, evaluation plots, or custom outputs.
        *   Logging trained models (`mlflow.<flavor>.log_model` or `mlflow.pyfunc.log_model`), which can also register them in the MLflow Model Registry.
        *   Logging tags (`mlflow.set_tags`) to categorize or add metadata to runs.
    *   This utility will be imported and used by the model training pipelines (e.g., `MLOps/pipelines/sentiment_model/train_sentiment_model.py` and `MLOps/pipelines/rl_agent/train_rl_agent.py`).

*   **(Optional) `MLOps/config/common/mlflow_config.yaml`**:
    *   A configuration file (can be placed in `MLOps/config/common/`) to specify the MLflow tracking URI and a default experiment name. `mlflow_utils.py` can be designed to read from this file.
    *   Example content:
        ```yaml
        tracking_uri: "http://localhost:5000" # Or your remote MLflow server, or "file:./mlruns"
        default_experiment_name: "FinAI_Algo_Default_Experiment"
        ```

## MLflow Setup

1.  **Install MLflow:**
    ```bash
    pip install mlflow
    ```

2.  **Start MLflow Tracking Server (Optional but Recommended for Centralized Tracking):**
    *   **Local Server:**
        A common way to start a local server, storing metadata and artifacts in a local `./mlruns` directory:
        ```bash
        mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
        ```
        This command tells MLflow to:
        *   `--host 127.0.0.1`: Listen on the local machine.
        *   `--port 5000`: Use port 5000.
        *   `--backend-store-uri ./mlruns`: Store run metadata (parameters, metrics, tags) in a local directory named `mlruns`.
        *   `--default-artifact-root ./mlruns`: Store artifacts (like models, data files) also in the `mlruns` directory.
        (For production, consider a more robust database like PostgreSQL or MySQL for `--backend-store-uri` and cloud storage like S3 or Azure Blob Storage for `--default-artifact-root`.)
    *   **Using Local Filesystem (Simpler, for individual use):**
        If no tracking server URI is specified (neither by environment variable nor explicitly in code), MLflow will default to logging to a local `mlruns` directory in the directory where the script is run. `mlflow.set_tracking_uri("file:./mlruns_project")` can be used to specify a particular project-level `mlruns` directory.

3.  **Configure Tracking URI for Scripts:**
    *   **Recommended Method: Environment Variable:**
        Set the `MLFLOW_TRACKING_URI` environment variable to point to your MLflow server. Scripts using the standard `mlflow` library or the `MLflowTracker` utility will automatically use this URI.
        *   Example (Bash/Zsh): `export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"`
        *   Example (Windows CMD): `set MLFLOW_TRACKING_URI=http://127.0.0.1:5000`
        *   Example (PowerShell): `$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"`
    *   **URI Precedence in `mlflow_utils.py`:**
        The `MLflowTracker` class in [`mlflow_utils.py`](./mlflow_utils.py:1) determines the tracking URI with the following order of precedence:
        1.  `MLFLOW_TRACKING_URI` environment variable.
        2.  `tracking_uri` parameter passed to the `MLflowTracker` constructor (e.g., from a configuration file).
        3.  Default local `./mlruns` directory (relative to the project root).
    *   Scripts directly using `mlflow` library functions (e.g., `mlflow.set_experiment()`, `mlflow.start_run()`) will also prioritize the `MLFLOW_TRACKING_URI` environment variable if set.

4.  **Access MLflow UI:**
    Open your browser and navigate to `http://localhost:5000` (or the address of your tracking server).

## Integration with Training Pipelines

The training scripts (`train_sentiment_model.py`, `train_rl_agent.py`) will:
1.  Import functions from `mlflow_utils.py`.
2.  Call `setup_mlflow()` or `start_mlflow_run()` at the beginning of the training process, specifying an appropriate experiment name (e.g., "SentimentModelTraining" or "RLAgentTraining_PPO").
3.  Inside the training loop or process:
    *   Log hyperparameters using `log_params()`.
    *   Log training/validation metrics (e.g., loss, accuracy, reward) at different epochs/steps using `log_metrics()`.
    *   Log any important configuration files or plots as artifacts using `log_artifact()`.
4.  After training is complete:
    *   Log the trained model using `log_model()`, potentially registering it.
    *   Log final evaluation metrics.
5.  Call `end_mlflow_run()` to conclude the run.

## What to Track

*   **Parameters:** All hyperparameters used for training (e.g., learning rate, batch size, network architecture for RL, LoRA config for sentiment models).
*   **Metrics:** Key performance indicators (e.g., training/validation loss, accuracy, F1-score for sentiment models; cumulative reward, Sharpe ratio, episode length for RL agents).
*   **Artifacts:**
    *   Configuration files used for the run.
    *   Plots (e.g., learning curves, evaluation metrics over time).
    *   The trained model itself (using `mlflow.log_model`).
    *   Data samples or feature importance plots.
    *   Log files from the training run.
*   **Tags:** Useful for organizing runs (e.g., `model_type: PPO`, `dataset_version: v1.2`, `status: experimental`).
*   **Source Code:** MLflow can automatically log the Git commit hash if the project is a Git repository, ensuring code versioning for reproducibility.

By systematically tracking experiments, the project can maintain a clear history of model development, compare different approaches, and easily reproduce or deploy the best-performing models.