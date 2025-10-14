# MLOps Model Registry

This directory and its associated processes define how trained models (both sentiment analysis models and Reinforcement Learning agents) are stored, versioned, and managed within the FinAI_algo project. A model registry is essential for tracking model lineage, promoting models through different stages (e.g., development, staging, production), and ensuring reproducibility.

## Objectives

*   **Centralized Storage:** Provide a defined location for storing trained model artifacts.
*   **Versioning:** Keep track of different versions of the same model, allowing rollback or comparison.
*   **Metadata Tracking:** Store relevant metadata about each model version (e.g., training date, source experiment/run ID, performance metrics, data version used for training).
*   **Accessibility:** Make it easy for deployment scripts (backtesting, paper trading, live trading) and other services to discover and load specific model versions.

## Implementation Strategy

The primary approach will be to leverage **MLflow Model Registry**. When models are logged using `mlflow_utils.log_model()` with a `registered_model_name`, MLflow handles the versioning and storage (either on the local filesystem if `file:` URI is used, or on a remote server/blob storage if configured).

Alternatively, a simpler file-based versioning system can be used initially if an MLflow server is not immediately available, but MLflow is the preferred long-term solution.

### 1. Using MLflow Model Registry (Preferred)

*   **Process:**
    1.  During model training (in `train_sentiment_model.py` or `train_rl_agent.py`), after a model is trained and evaluated, it is logged using `mlflow_utils.log_model(..., registered_model_name="MySentimentModel")` or `mlflow_utils.log_model(..., registered_model_name="MyRLAgent_PPO")`.
    2.  This action, if the `registered_model_name` is new, creates a new registered model in MLflow. If the name exists, it creates a new version of that model.
    3.  Models can then be managed (e.g., transitioning stages like "Staging", "Production", "Archived") through the MLflow UI or API.
*   **Storage Location:**
    *   The actual model artifacts are stored in the artifact store configured for the MLflow tracking server (e.g., `./mlruns_server_artifacts/` locally, or S3/Azure Blob/GCS remotely).
    *   This `MLOps/model_registry/` directory itself might not store the physical model files if a dedicated MLflow server is used, but it will contain this README and potentially scripts or configurations related to model registration or management if needed.
*   **Accessing Models:**
    *   Deployment scripts can load models using MLflow's model loading utilities, specifying the model name and version (e.g., `mlflow.pyfunc.load_model("models:/MyRLAgent_PPO/Production")`).

### 2. Simple Directory-Based Versioning (Fallback/Complementary)

If a full MLflow server setup is not immediately feasible, or for local development, a directory structure within this `MLOps/model_registry/` folder can be used:

*   **Structure:**
    ```
    MLOps/model_registry/
    ├── sentiment_models/
    │   ├── <model_name_version1>/  (e.g., llama3_8b_lora_v1.0.0_20231201103000)
    │   │   ├── adapter_model.bin
    │   │   ├── adapter_config.json
    │   │   └── metadata.json  (training date, source data hash, metrics, MLflow run ID if available)
    │   └── <model_name_version2>/
    │       └── ...
    ├── rl_agents/
    │   ├── <agent_name_version1>/ (e.g., ppo_stocktrading_v1.0.0_20231202150000)
    │   │   ├── model.zip (for Stable Baselines3) or agent.pth (for ElegantRL)
    │   │   └── metadata.json (training date, source data hash, env config, metrics, MLflow run ID)
    │   └── <agent_name_version2>/
    │       └── ...
    ```
*   **Naming Convention:**
    *   `<model_type>/<model_name_or_architecture>_<version_string>_<timestamp_or_identifier>/`
    *   Example: `sentiment_models/llama3_8b_lora_v1.0.1_20240115142200/`
    *   Example: `rl_agents/ppo_dowjones_v0.2.0_20240116103015/`
*   **`metadata.json`:** A crucial file in each versioned model directory, containing:
    *   `model_name`: e.g., "llama3_8b_lora_sentiment"
    *   `model_version`: e.g., "v1.0.1"
    *   `training_timestamp`: ISO format date-time.
    *   `mlflow_run_id`: (If MLflow was used for the training run, even if not for registry).
    *   `source_data_version`: Hash or identifier for the training dataset (e.g., DVC hash).
    *   `hyperparameters_config_path`: Path to the config file used for training.
    *   `key_performance_metrics`: e.g., {"accuracy": 0.85, "f1_score": 0.83} or {"sharpe_ratio": 1.5, "cumulative_return": 0.25}.
    *   `model_artifact_files`: List of main model files within the directory.
    *   `notes`: Any relevant notes about this model version.

## Model Artifacts to Store

*   **Sentiment Models (e.g., LoRA fine-tuned LLMs):**
    *   LoRA adapter weights (e.g., `adapter_model.bin` or `pytorch_model.bin` if saved that way by Hugging Face `save_pretrained`).
    *   Adapter configuration (e.g., `adapter_config.json`).
    *   Tokenizer files (if not automatically handled by loading the base model + adapter).
    *   The `metadata.json` file.
*   **RL Agents (e.g., from Stable Baselines3, ElegantRL):**
    *   The serialized agent model file (e.g., `.zip` for Stable Baselines3, `.pth` for ElegantRL).
    *   Associated normalization statistics for observations/rewards if applicable and saved separately.
    *   The `metadata.json` file, which should also reference the environment configuration used for training.

## Workflow Integration

1.  **Training Pipelines (`MLOps/pipelines/.../train_*.py`):**
    *   After successful training and evaluation, the pipeline script will be responsible for:
        *   Saving the model artifacts in the standard format.
        *   Creating the `metadata.json` file.
        *   If using MLflow Model Registry: Calling `mlflow_utils.log_model()` with `registered_model_name`.
        *   If using directory-based: Copying the model artifacts and metadata to the appropriate versioned directory under `MLOps/model_registry/`.
2.  **Deployment Scripts (`MLOps/deployment/.../*.py`):**
    *   These scripts will load models by:
        *   If using MLflow Model Registry: Querying MLflow for the desired model name and version/stage (e.g., "MyRLAgent_PPO/Production").
        *   If using directory-based: Reading a configuration that specifies the path to the desired model version within `MLOps/model_registry/`, or having a mechanism to select the "latest" or a specific version.

This README provides a guideline. The exact implementation details, especially for the directory-based approach, will evolve as the project progresses. The strong recommendation is to utilize the MLflow Model Registry for its comprehensive features.