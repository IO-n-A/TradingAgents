# MLOps RL Agent Training Pipeline

This directory contains the scripts and configurations for the Reinforcement Learning (RL) agent training pipeline of the FinAI_algo project. The primary goal of this pipeline is to train RL agents for financial trading tasks using the feature-engineered data.

## Files

*   **`__init__.py`**: (If this directory becomes a sub-package).
*   **`train_rl_agent.py`**:
    *   The main script that orchestrates the RL agent training process.
    *   **Functionality:**
        1.  **Loads Configurations:** Reads settings from:
            *   `MLOps/config/common/global_vars.yaml` (for data paths, MLflow settings).
            *   `MLOps/config/rl_agents/<agent_type>_params.yaml` (e.g., `ppo_stocktrading_params.yaml` for agent-specific hyperparameters like learning rate, network architecture, batch size).
            *   `MLOps/config/environments/stock_trading_env_config.yaml` (for environment parameters like initial amount, transaction costs, technical indicators to use, sentiment feature flag).
        2.  **Loads Processed Data:**
            *   Reads the feature-engineered dataset (e.g., from `data/processed/processed_data_with_features.csv`). This file is assumed to be tracked by DVC and produced by the feature engineering pipeline.
        3.  **Initializes Environment:**
            *   Sets up the FinRL trading environment (e.g., `StockTradingEnv`) using the processed data and environment configurations. This includes defining the observation space (based on available features, including sentiment if enabled) and action space.
        4.  **Initializes Agent:**
            *   Instantiates the DRL agent (e.g., PPO, A2C, DDPG from Stable Baselines3 or ElegantRL, as specified in the agent parameters config) with its hyperparameters and the trading environment.
        5.  **Experiment Tracking Setup (MLflow):**
            *   Initializes an MLflow run using utilities from `MLOps/experiment_tracking/mlflow_utils.py`.
            *   Logs hyperparameters, environment configurations, and any relevant tags (e.g., agent type, dataset version).
        6.  **Trains Agent:**
            *   Runs the agent's training loop for a specified number of timesteps or episodes.
            *   During training, logs metrics (e.g., cumulative reward, episode length, loss values if applicable) to MLflow at regular intervals.
        7.  **Saves Trained Model:**
            *   Saves the trained agent model to a specified path, typically within `MLOps/model_registry/rl_agents/` or lets MLflow handle model artifact logging.
            *   The model saving format depends on the RL library used (e.g., `.zip` for Stable Baselines3).
        8.  **Logs Model to MLflow:**
            *   Uses `mlflow_utils.log_model()` to log the trained model to MLflow, potentially registering it in the MLflow Model Registry.
        9.  **(Optional) Initial Evaluation/Backtest:**
            *   May perform a quick backtest on a validation split of the data to get initial performance metrics and log them to MLflow. A more thorough backtest is typically done by the dedicated `MLOps/deployment/backtesting/run_backtest.py` script.
    *   **Execution:**
        ```bash
        python MLOps/pipelines/rl_agent/train_rl_agent.py \
          --global_config_path MLOps/config/common/global_vars.yaml \
          --agent_params_path MLOps/config/rl_agents/ppo_stocktrading_params.yaml \
          --processed_data_filename processed_data_with_features.csv \
          # --output_model_dir MLOps/model_registry/rl_agents/ppo_run1 # Optional if MLflow handles it
        ```

## Key Components & Libraries

*   **RL Libraries:** Stable Baselines3, ElegantRL (as integrated within FinRL).
*   **Environment:** Custom FinRL trading environments (e.g., `StockTradingEnv`), configured to use sentiment features.
*   **Experiment Tracking:** MLflow, using `MLOps/experiment_tracking/mlflow_utils.py`.
*   **Data:** Processed, feature-engineered data from the `MLOps/pipelines/feature_engineering/` stage.

## Output

*   **Trained RL Agent Model:** Saved model files (e.g., `.zip`, `.pth`) stored locally (e.g., in `MLOps/model_registry/rl_agents/`) and/or as an MLflow artifact.
*   **MLflow Run:** An MLflow run containing:
    *   Logged hyperparameters.
    *   Logged training metrics (rewards, losses, etc.).
    *   The trained model artifact.
    *   Configuration files as artifacts.
    *   (Optionally) Initial evaluation results or plots.

## Dependencies

*   Pandas, NumPy
*   PyYAML
*   FinRL (and its dependencies like Stable Baselines3, ElegantRL, OpenAI Gym)
*   MLflow
*   DVC (for accessing DVC-tracked processed data)

## Future Enhancements

*   Hyperparameter optimization (e.g., using Optuna, Ray Tune, integrated with MLflow).
*   Support for more RL algorithms and custom agent architectures.
*   Curriculum learning or more sophisticated training strategies.
*   Automated model evaluation and comparison within the pipeline.
*   Distributed training for larger datasets or more complex agents.