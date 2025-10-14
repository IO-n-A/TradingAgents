# MLOps/pipelines/training/finrl_us_equity_momentum_train.py
# This script orchestrates the training of FinRL agents for the US equity momentum strategy.
# It includes environment setup, agent selection, training, and MLflow tracking.

import argparse
import datetime
import logging
import os
import time

import mlflow
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.config import INDICATORS, DATA_SAVE_DIR, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAINED_MODEL_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values (can be overridden by a config file)
DEFAULT_CONFIG = {
    "data_path_template": "data/processed/features/strategy_1/{date}/processed_data.csv",
    "model_output_path_template": "models/finrl/strategy_1/{agent_name}_{timestamp}",
    "mlflow_experiment_name": "FinRL_US_Equity_Momentum",
    "environment": {
        "initial_amount": 1000000,
        "hmax": 100,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "tech_indicator_list": INDICATORS, # From finrl.config
        "turbulence_threshold": None, # Example, can be set if turbulence data is available
        "risk_free_rate": 0.0, # Example, for Sharpe ratio calculation in env if needed
        "buy_cost_pct": 0.001, # Specific buy cost
        "sell_cost_pct": 0.001, # Specific sell cost
        "state_space": len(INDICATORS) + 3, # open, high, low, close, volume + indicators + holdings + cash
        "stock_dim": 0, # Will be determined by data
        "action_space": 0, # Will be determined by data (number of stocks)
        "mode": "train",
        "model_name": "", # Will be set
        "iteration": "" # Will be set
    },
    "agent_config": {
        "PPO": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "verbose": 0
        },
        "A2C": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.0,
            "verbose": 0
        },
        "DDPG": {
            "policy": "MlpPolicy",
            "learning_rate": 0.001,
            "batch_size": 100,
            "buffer_size": 50000,
            "gamma": 0.99,
            "tau": 0.005,
            "verbose": 0
        }
        # Add other agents like SAC, TD3 if needed
    },
    "training_params": {
        "total_timesteps": 20000, # Short for example, increase for real training
        "log_interval": 1, # For SB3 logger
        "tb_log_name_template": "{agent_name}"
    }
}

SUPPORTED_AGENTS = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3
}

class MLflowCallback(BaseCallback):
    """
    A custom callback for logging metrics and artifacts to MLflow.
    This callback logs evaluation metrics and the model at the end of training.
    """
    def __init__(self, verbose=0):
        super(MLflowCallback, self).__init__(verbose)
        self.rollout_count = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        Logs initial parameters or setup if needed.
        """
        logger.info("Training started. MLflow callback active.")
        # You can log initial model architecture or other params here if needed
        # mlflow.log_param("model_architecture", str(self.model.policy))

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout.
        Logs metrics available in self.logger.get_log_dict().
        """
        self.rollout_count += 1
        # Log general SB3 metrics
        if self.logger:
            for key, value in self.logger.get_latest_values().items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"sb3/{key}", value, step=self.num_timesteps)
        
        # Log environment-specific metrics if available (e.g., from info dict)
        # This requires the environment to return these metrics in the info dict
        # For example, if env returns 'sharpe_ratio' in info:
        # infos = self.locals.get("infos", [{}])
        # if infos and isinstance(infos, list) and len(infos) > 0:
        #     if "sharpe_ratio" in infos[0]:
        #         mlflow.log_metric("env/sharpe_ratio", infos[0]["sharpe_ratio"], step=self.num_timesteps)
        #     if "cumulative_return" in infos[0]:
        #          mlflow.log_metric("env/cumulative_return", infos[0]["cumulative_return"], step=self.num_timesteps)


    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        Can be used for more granular logging if needed, but _on_rollout_end is often sufficient.
        """
        # Example: Log reward every N steps
        # if self.num_timesteps % 1000 == 0:
        #     reward = self.locals.get("rewards")
        #     if reward is not None and len(reward) > 0:
        #         mlflow.log_metric("reward_mean", np.mean(reward), step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        Logs final model and any summary metrics.
        """
        logger.info("Training finished. Logging final model to MLflow.")
        # The model is saved via agent.save() and then logged as an artifact separately.


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads the processed feature data from the given path.
    This function loads data that has been processed, including engineered momentum and sentiment features.
    Input: data_path (str) - Path to the CSV file.
    Output: pd.DataFrame - Loaded data.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        data_df = pd.read_csv(data_path)
        # Ensure standard column names if necessary, e.g., 'date', 'tic', 'close'
        # data_df['date'] = pd.to_datetime(data_df['date'])
        # data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        logger.info(f"Data loaded successfully from {data_path}. Shape: {data_df.shape}")
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise
    print(f"Data loaded from {data_path}. The first few rows are:\n{data_df.head()}")
    print(f"The loaded data has {data_df.shape[0]} rows and {data_df.shape[1]} columns.")
    return data_df


def setup_environment(df: pd.DataFrame, env_config: dict) -> StockTradingEnv:
    """
    Configures and instantiates a stock trading environment.
    This function sets up the FinRL stock trading environment with the provided data and configuration.
    Input: df (pd.DataFrame) - DataFrame with financial and feature data.
               env_config (dict) - Configuration dictionary for the environment.
    Output: StockTradingEnv - Instantiated stock trading environment.
    """
    logger.info("Setting up stock trading environment.")
    
    # Determine stock_dim and action_space from data
    stock_dimension = len(df.tic.unique())
    env_config["stock_dim"] = stock_dimension
    env_config["action_space"] = stock_dimension 
    
    # Ensure all tech indicators are present in the DataFrame
    missing_indicators = [col for col in env_config["tech_indicator_list"] if col not in df.columns]
    if missing_indicators:
        logger.warning(f"Missing indicators in DataFrame: {missing_indicators}. They will be ignored or may cause errors.")
        # Optionally, remove missing indicators from the list or handle as needed
        # env_config["tech_indicator_list"] = [ind for ind in env_config["tech_indicator_list"] if ind in df.columns]

    # The state space needs to be calculated based on actual features used.
    # Base state: holdings (stock_dim) + cash (1) = stock_dim + 1
    # Features: OHLCV (5 per stock) + tech_indicators (len(tech_indicator_list) per stock)
    # For simplicity, FinRL's StockTradingEnv often flattens this.
    # Let's use the env_config's state_space if provided, or calculate.
    # The default StockTradingEnv calculates state space internally based on features.
    # We will pass the df and let the environment handle it.

    try:
        env_kwargs = {
            "df": df,
            "stock_dim": env_config["stock_dim"],
            "hmax": env_config["hmax"],
            "initial_amount": env_config["initial_amount"],
            "transaction_cost_pct": env_config["transaction_cost_pct"],
            "reward_scaling": env_config["reward_scaling"],
            "state_space": env_config["state_space"], # This might be overridden by env based on df
            "action_space": env_config["action_space"],
            "tech_indicator_list": env_config["tech_indicator_list"],
            "turbulence_threshold": env_config.get("turbulence_threshold"), # Use .get for optional keys
            "risk_free_rate": env_config.get("risk_free_rate", 0.0),
            "buy_cost_pct": env_config.get("buy_cost_pct", [env_config["transaction_cost_pct"]] * env_config["stock_dim"]),
            "sell_cost_pct": env_config.get("sell_cost_pct", [env_config["transaction_cost_pct"]] * env_config["stock_dim"]),
            "mode": env_config.get("mode", "train"),
            "model_name": env_config.get("model_name", ""),
            "iteration": env_config.get("iteration", "")
        }
        
        # The environment expects a list of costs if they are per stock
        if isinstance(env_kwargs["buy_cost_pct"], float):
            env_kwargs["buy_cost_pct"] = [env_kwargs["buy_cost_pct"]] * env_kwargs["stock_dim"]
        if isinstance(env_kwargs["sell_cost_pct"], float):
            env_kwargs["sell_cost_pct"] = [env_kwargs["sell_cost_pct"]] * env_kwargs["stock_dim"]


        env_train = StockTradingEnv(**env_kwargs)
        # Wrap in DummyVecEnv for SB3
        env_train_sb3 = DummyVecEnv([lambda: env_train])
        logger.info("Stock trading environment set up successfully.")
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise
    
    print(f"Environment setup complete. Observation space: {env_train.observation_space}, Action space: {env_train.action_space}")
    print(f"State space as per env: {env_train.state_space}, Stock dimension: {env_train.stock_dim}")
    return env_train_sb3


def train_agent(env, agent_name: str, agent_config: dict, training_params: dict, model_output_path: str, tb_log_path: str):
    """
    Selects, configures, and trains a reinforcement learning agent.
    This function initializes the chosen RL agent, trains it using the provided environment and parameters,
    and saves the trained model. MLflow logging for hyperparameters and metrics is handled via callback.
    Input: env (DummyVecEnv) - The training environment.
           agent_name (str) - Name of the agent to train (e.g., "PPO", "A2C").
           agent_config (dict) - Configuration for the agent.
           training_params (dict) - Parameters for the training loop.
           model_output_path (str) - Path to save the trained model.
           tb_log_path (str) - Path for TensorBoard logs.
    Output: Trained agent model.
    """
    logger.info(f"Starting training for agent: {agent_name}")

    if agent_name not in SUPPORTED_AGENTS:
        logger.error(f"Agent {agent_name} is not supported. Supported agents: {list(SUPPORTED_AGENTS.keys())}")
        raise ValueError(f"Agent {agent_name} is not supported.")

    agent_class = SUPPORTED_AGENTS[agent_name]
    specific_agent_params = agent_config.get(agent_name, {})
    
    # Log agent hyperparameters to MLflow
    for param, value in specific_agent_params.items():
        mlflow.log_param(f"agent_{param}", value)
    mlflow.log_param("agent_name", agent_name)
    mlflow.log_param("total_timesteps", training_params["total_timesteps"])

    # Setup SB3 logger for TensorBoard
    # new_logger = sb3_configure_logger(tb_log_path, ["stdout", "tensorboard"]) # Removed CSV and JSON for cleaner logs

    try:
        model = agent_class(
            env=env,
            tensorboard_log=tb_log_path, # Pass the path for SB3 to create subdirs
            verbose=specific_agent_params.pop("verbose", 1), # Remove verbose to avoid passing it twice if in policy_kwargs
            **specific_agent_params # Pass other params like policy, learning_rate, etc.
        )
        # model.set_logger(new_logger)

        # Train the agent
        logger.info(f"Training {agent_name} for {training_params['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=training_params["total_timesteps"],
            log_interval=training_params["log_interval"],
            tb_log_name=training_params["tb_log_name_template"].format(agent_name=agent_name), # SB3 will append _1, _2 etc.
            callback=MLflowCallback() # Add our custom MLflow callback
        )
        logger.info(f"Training completed for {agent_name}.")

        # Save the model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model_save_file = os.path.join(model_output_path, f"{agent_name}_model.zip")
        model.save(model_save_file)
        logger.info(f"Trained model saved to {model_save_file}")
        
        # Log model artifact to MLflow
        mlflow.log_artifact(model_save_file, artifact_path="model")
        # Log TensorBoard logs directory
        # MLflow automatically logs artifacts from the directory if specified with log_artifacts
        # For SB3, tensorboard logs are usually in tb_log_path/tb_log_name_template_run_id
        # We can log the entire tb_log_path or find the specific run folder.
        # For simplicity, let's assume tb_log_path contains relevant logs for this run.
        # However, SB3 creates subfolders like PPO_1, PPO_2.
        # A more robust way is to find the latest folder or pass it from the callback.
        # For now, logging the parent tb_log_path.
        if os.path.exists(tb_log_path):
             mlflow.log_artifacts(tb_log_path, artifact_path="tensorboard_logs")


    except Exception as e:
        logger.error(f"Error during agent training or saving: {e}")
        raise
    
    print(f"Agent {agent_name} trained and saved. Model artifact logged to MLflow.")
    return model


def main(args):
    """
    Main function to orchestrate the FinRL training pipeline.
    This function handles argument parsing, configuration loading, data loading,
    environment setup, agent training, and MLflow integration.
    Input: args (argparse.Namespace) - Command-line arguments.
    Output: None
    """
    start_time = time.time()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            # Deep merge YAML config into default config (simple dict update for now)
            # More sophisticated merge might be needed for nested dicts if partial overrides are common
            for key, value in yaml_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
            logger.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {args.config_file}: {e}. Using default config.")

    # Override config with command-line arguments if provided
    agent_name = args.agent_name if args.agent_name else "PPO" # Default to PPO
    current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    data_path = args.data_path if args.data_path else config["data_path_template"].format(date=current_date_str)
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_base = args.model_output_path if args.model_output_path else config["model_output_path_template"].format(agent_name=agent_name, timestamp=timestamp_str)
    
    mlflow_experiment_name = args.mlflow_experiment_name if args.mlflow_experiment_name else config["mlflow_experiment_name"]
    
    # Ensure model output directory exists (DVC compatible structure)
    # Example: models/finrl/strategy_1/PPO_20250608_190000/PPO_model.zip
    os.makedirs(model_output_base, exist_ok=True)

    # Setup MLflow
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run(run_name=f"{agent_name}_train_{timestamp_str}") as run:
        mlflow_run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {mlflow_run_id}")
        mlflow.log_param("script_args_data_path", args.data_path if args.data_path else "default_template")
        mlflow.log_param("script_args_model_output_path", args.model_output_path if args.model_output_path else "default_template")
        mlflow.log_param("script_args_agent_name", agent_name)
        mlflow.log_param("script_args_config_file", args.config_file if args.config_file else "None")
        mlflow.log_param("effective_data_path", data_path)
        mlflow.log_param("effective_model_output_base", model_output_base)

        # Load data
        logger.info(f"Loading data from: {data_path}")
        processed_df = load_data(data_path)

        # Setup environment
        env_config = config.get("environment", {})
        # Update env_config with dynamic values if needed
        env_config["model_name"] = agent_name # Pass agent name to env if it uses it
        env_config["iteration"] = mlflow_run_id # Pass run_id as iteration for tracking
        
        logger.info("Setting up environment...")
        train_env = setup_environment(processed_df, env_config)

        # Train agent
        agent_config_params = config.get("agent_config", {})
        training_params_config = config.get("training_params", {})
        
        # Define TensorBoard log path relative to MLflow run for better organization
        # Or use a global TENSORBOARD_LOG_DIR from finrl.config
        tb_log_path = os.path.join(TENSORBOARD_LOG_DIR, mlflow_run_id) # Store TB logs per run
        os.makedirs(tb_log_path, exist_ok=True)
        logger.info(f"TensorBoard logs will be stored in: {tb_log_path}")

        logger.info(f"Training agent {agent_name}...")
        trained_model = train_agent(
            env=train_env,
            agent_name=agent_name,
            agent_config=agent_config_params,
            training_params=training_params_config,
            model_output_path=model_output_base, # This is the directory
            tb_log_path=tb_log_path
        )
        
        # Log any final summary metrics if not covered by callback
        # Example: final_portfolio_value = train_env.envs[0].final_asset_value # Access underlying env
        # mlflow.log_metric("final_portfolio_value_example", final_portfolio_value)

        logger.info("Training pipeline completed.")
        mlflow.log_param("status", "completed")

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total execution time: {duration:.2f} seconds.")
    print(f"Script finished. Total execution time: {duration:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinRL US Equity Momentum Training Pipeline")
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--data_path", type=str, help="Path to the processed input data CSV file (e.g., data/processed/features/strategy_1/YYYY-MM-DD/processed_data.csv). Overrides config.")
    parser.add_argument("--model_output_path", type=str, help="Base directory to save the trained model artifacts (e.g., models/finrl/strategy_1/AGENT_TIMESTAMP). Overrides config.")
    parser.add_argument("--agent_name", type=str, choices=list(SUPPORTED_AGENTS.keys()), help="Name of the RL agent to train (e.g., PPO, A2C). Overrides config.")
    parser.add_argument("--mlflow_experiment_name", type=str, help="Name of the MLflow experiment. Overrides config.")
    
    args = parser.parse_args()
    main(args)