# MLOps/pipelines/rl_agent/train_rl_agent.py

import logging
import logging.config
import os
import sys
import yaml
import pandas as pd
from datetime import datetime

# Ensure the main project directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# FinRL imports
from FinRL.finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from FinRL.finrl.meta.env_portfolio_allocation.env_portfolio_meta_example import PortfolioMetaExampleEnv # FinRL-Meta Placeholder Env
from FinRL.finrl.agents.stablebaselines3.models import DRLAgent
from FinRL.finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor # For data splitting if needed
from FinRL.finrl.plot import backtest_stats # For potential result saving

# Import for ElegantRL Adaptor
from core.agent_adaptors.elegantrl_adaptor import ElegantRLAdaptor

# MLOps utilities (to be created)
# from MLOps.experiment_tracking.mlflow_utils import start_mlflow_run, log_params, log_metrics, log_model # Placeholder

# Configure logging
# Ensure logging_config.py is correctly located and configured
try:
    logging_config_path = os.path.join(project_root, 'config', 'logging_config.py')
    if os.path.exists(logging_config_path):
        logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
    else:
        # Fallback basic logging if config file is missing
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.warning(f"Logging config file not found at {logging_config_path}. Using basicConfig.")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.error(f"Error configuring logging from file: {e}. Using basicConfig.")

logger = logging.getLogger(__name__)

# Placeholder for MLflow utils
class MLflowUtilsPlaceholder:
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        # logger.info for placeholder init can be noisy if logger is not fully set up when this class is defined.
        # It's better to log when methods are actually called.

    def start_mlflow_run(self, experiment_name, run_name):
        logger.info(f"MLP_PH: Starting run '{run_name}' in experiment '{experiment_name}'.")
        class MockRun:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockRun()

    def log_params(self, params_dict):
        logger.info(f"MLP_PH: Logging parameters: {params_dict}")

    def log_metrics(self, metrics_dict, step=None):
        logger.info(f"MLP_PH: Logging metrics: {metrics_dict} at step {step if step else 'N/A'}")

    def log_model(self, model, artifact_path):
        logger.info(f"MLP_PH: Logging model to artifact path: {artifact_path}")

    def log_artifact(self, local_path, artifact_path=None):
        logger.info(f"MLP_PH: Logging artifact from {local_path} to {artifact_path if artifact_path else ''}")

mlflow_utils = MLflowUtilsPlaceholder()


def load_config(config_path: str) -> dict:
    logger.debug(f"File: MLOps/pipelines/rl_agent/train_rl_agent.py, Function: load_config, Purpose: Loads a YAML configuration file, Output: Dictionary with loaded configuration.")
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise

# Function: load_processed_data
# Description: Loads processed, feature-engineered data from a CSV file into a pandas DataFrame.
#              It expects a 'date' column which is converted to datetime objects.
# Input:
#   data_path: String, the path to the CSV file containing the processed data.
# Output: A pandas DataFrame with the loaded data.
# Dependencies: pandas (pd), logging.
def load_processed_data(processed_data_dir: str) -> pd.DataFrame:
    logger.debug(f"File: MLOps/pipelines/rl_agent/train_rl_agent.py, Function: load_processed_data, Purpose: Loads the latest processed feature-engineered data from a directory, Output: Pandas DataFrame.")
    """Loads the latest processed feature-engineered data from the specified directory."""
    logger.info(f"Searching for latest processed data in directory: {processed_data_dir}")
    try:
        if not os.path.exists(processed_data_dir):
            logger.error(f"Processed data directory not found: {processed_data_dir}")
            raise FileNotFoundError(f"Directory not found: {processed_data_dir}")

        data_files = sorted(
            [f for f in os.listdir(processed_data_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(processed_data_dir, f)) and f.startswith("processed_data_")],
            reverse=True # Assuming lexicographical sort on timestamped name gives latest
        )
        if not data_files:
            logger.error(f"No 'processed_data_*.csv' files found in {processed_data_dir}")
            raise FileNotFoundError(f"No suitable processed data files found in {processed_data_dir}")

        latest_data_file = os.path.join(processed_data_dir, data_files[0])
        logger.info(f"Loading latest processed data from: {latest_data_file}")
        df = pd.read_csv(latest_data_file)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Processed data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError: # Specific re-raise
        raise
    except Exception as e:
        logger.error(f"Error loading processed data from {processed_data_dir}: {e}", exc_info=True)
        raise

# Function: split_data
# Description: Splits a DataFrame into training and trading (testing/validation) sets based on date ranges.
#              The resulting DataFrames are sorted by date and ticker.
# Input:
#   df: pandas DataFrame, the input data containing a 'date' column.
#   train_start_date: String, start date for the training set.
#   train_end_date: String, end date for the training set.
#   trade_start_date: String, start date for the trading set.
#   trade_end_date: String, end date for the trading set.
# Output: A tuple containing two pandas DataFrames: (train_df, trade_df).
# Dependencies: pandas (pd), logging.
def split_data(df: pd.DataFrame, train_start_date: str, train_end_date: str,
               trade_start_date: str, trade_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug(f"File: MLOps/pipelines/rl_agent/train_rl_agent.py, Function: split_data, Purpose: Splits DataFrame into training and trading sets based on dates, Output: Tuple (train_df, trade_df).")
    """Splits data into training and trading sets."""
    logger.info(f"Splitting data: Train ({train_start_date} - {train_end_date}), Trade ({trade_start_date} - {trade_end_date})")
    
    train_df = df[(df["date"] >= train_start_date) & (df["date"] < train_end_date)]
    trade_df = df[(df["date"] >= trade_start_date) & (df["date"] < trade_end_date)]
    
    # FinRL's DataProcessor often sorts and sets index
    train_df = train_df.sort_values(["date", "tic"]).reset_index(drop=True)
    trade_df = trade_df.sort_values(["date", "tic"]).reset_index(drop=True)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Trading data shape: {trade_df.shape}")
    return train_df, trade_df


# Function: train_rl_agent
# Description: Trains a Deep Reinforcement Learning (DRL) agent for stock trading using the FinRL library.
#              It sets up the trading environment (`StockTradingEnv`), initializes the DRL agent
#              (e.g., PPO) with specified parameters, trains the model, and saves the trained model.
# Input:
#   train_df: pandas DataFrame, the training data.
#   agent_params: Dictionary, parameters for the DRL agent (e.g., agent_name, learning_rate, total_timesteps).
#   env_params: Dictionary, parameters for the stock trading environment.
#   model_output_dir: String, directory to save the trained model.
# Output: String, the path to the saved trained model file.
# Dependencies: logging, os, datetime, pandas (pd),
#               FinRL.finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv,
#               FinRL.finrl.agents.stablebaselines3.models.DRLAgent,
#               mlflow_utils (global placeholder).
def train_rl_agent(train_df: pd.DataFrame, agent_params: dict, env_params: dict, model_output_dir: str, global_vars: dict):
    logger.debug(f"File: MLOps/pipelines/rl_agent/train_rl_agent.py, Function: train_rl_agent, Purpose: Trains a DRL agent using FinRL, saves the model, Output: Path to saved model file.")
    """
    Trains an RL agent using FinRL.
    """
    logger.info("Starting RL agent training.")

    # Determine which environment to use based on configuration
    # This could be a parameter in env_params or a global setting
    environment_type = env_params.get("environment_type", "StockTradingEnv") # Default to existing env

    # Prepare environment arguments
    stock_dimension = len(train_df.tic.unique())
    # Ensure tech_indicator_list from env_params is used, and filter against train_df columns
    configured_tech_indicators = env_params.get('tech_indicator_list', [])
    available_indicators = [col for col in configured_tech_indicators if col in train_df.columns]
    if len(available_indicators) != len(configured_tech_indicators):
        logger.warning(f"Some tech indicators specified in env_config ({configured_tech_indicators}) are not in the training data. Using available ones: {available_indicators}.")
    
    # Sentiment features from env_params
    sentiment_feature_list = env_params.get('sentiment_feature_list', [])
    available_sentiment_features = [col for col in sentiment_feature_list if col in train_df.columns]
    if sentiment_feature_list and len(available_sentiment_features) != len(sentiment_feature_list):
         logger.warning(f"Some sentiment features specified in env_config ({sentiment_feature_list}) are not in the training data. Using available ones: {available_sentiment_features}.")


    # Common parameters for most environments
    base_env_args = {
        "df": train_df,
        "initial_amount": env_params.get('initial_amount', 1000000),
        "buy_cost_pct": env_params.get('buy_cost_pct', 0.001), # Use specific keys
        "sell_cost_pct": env_params.get('sell_cost_pct', 0.001),# Use specific keys
        "reward_scaling": env_params.get('reward_scaling', 1.0),
        "tech_indicator_list": available_indicators, # Use filtered list
        "sentiment_feature_list": available_sentiment_features, # Pass available sentiment features
    }

    if environment_type == "PortfolioMetaExampleEnv":
        logger.info(f"Initializing PortfolioMetaExampleEnv.")
        portfolio_env_args = {
            **base_env_args,
            "state_space_lookback": env_params.get('state_space_lookback', 30),
            "custom_env_config": env_params.get('custom_portfolio_config', {})
        }
        portfolio_env_args = {k: v for k, v in portfolio_env_args.items() if v is not None}
        env_train = PortfolioMetaExampleEnv(**portfolio_env_args)
        logger.info(f"PortfolioMetaExampleEnv initialized. Obs space: {env_train.observation_space.shape}, Action space: {env_train.action_space.shape}")

    elif environment_type == "StockTradingEnv":
        logger.info(f"Initializing StockTradingEnv.")
        stock_trading_env_args = {
            **base_env_args,
            "stock_dim": stock_dimension,
            "hmax": env_params.get('hmax', 100),
            # state_space and action_space are determined by the env itself based on stock_dim and features
            "state_space": stock_dimension, # Placeholder, env calculates actual
            "action_space": stock_dimension, # Placeholder, env calculates actual
            "num_stock_shares": [0] * stock_dimension, # Default initial shares
            "turbulence_threshold": env_params.get('turbulence_threshold'), # Can be None
            "risk_indicator_col": env_params.get('risk_indicator_col', 'turbulence'),
            "make_plots": env_params.get('make_plots', False),
            "print_verbosity": env_params.get('print_verbosity', 500),
        }
        stock_trading_env_args = {k: v for k, v in stock_trading_env_args.items() if v is not None}
        env_train = StockTradingEnv(**stock_trading_env_args)
        logger.info(f"StockTradingEnv initialized. Obs space: {env_train.observation_space.shape}, Action space: {env_train.action_space.shape}")
    else:
        logger.error(f"Unsupported environment_type: {environment_type}. Choose 'StockTradingEnv' or 'PortfolioMetaExampleEnv'.")
        raise ValueError(f"Unsupported environment_type: {environment_type}")

    framework = agent_params.get("framework", "finrl").lower() # Default to finrl
    model_name = agent_params.get("agent_name", "ppo").lower()
    total_timesteps = agent_params.get("total_timesteps", 20000)
    model_save_path = None

    if framework == "elegantrl":
        logger.info(f"Initializing ElegantRLAdaptor for agent: {model_name.upper()}")
        # Placeholder for ElegantRL agent instantiation and training
        # env_config for ElegantRL might need different structure than current_env_params
        # For now, pass current_env_params as a placeholder for env_config
        # and agent_params for agent_specific_params
        
        # The ElegantRLAdaptor will use the env_train object directly
        # to derive necessary parameters like state_dim, action_dim, etc.
        # and agent_params for other configurations.

        adaptor = ElegantRLAdaptor(
            agent_name=model_name.upper(), # ElegantRL might expect uppercase, e.g., PPO
            env=env_train,                 # Pass the actual environment instance
            agent_params=agent_params      # Pass all agent_params for now
        )
        logger.info(f"Training ElegantRL agent: {model_name.upper()} for {total_timesteps} timesteps.")
        adaptor.train_agent(total_timesteps=total_timesteps)
        
        # Save the model using adaptor
        model_filename = f"ELEGANT_{model_name.upper()}_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth" # .pth or other ElegantRL format
        model_save_path = os.path.join(model_output_dir, model_filename)
        adaptor.save_model(path=model_save_path) # save_model expects a directory, or full path. Adjust if needed.
                                                 # For now, assuming it handles the full path.
        logger.info(f"Trained ElegantRL agent model saved to: {model_save_path}")
        
        # Log model with MLflow (placeholder for ElegantRL model object)
        # mlflow_utils.log_model(adaptor.agent, artifact_path=f"rl_models/elegantrl_{model_name}") # adaptor.agent is placeholder
        logger.info(f"Placeholder: MLflow logging for ElegantRL model {model_name} (adaptor.agent).")


    elif framework == "finrl":
        # Initialize DRL agent (FinRL)
        agent = DRLAgent(env=env_train)
        logger.info(f"Training FinRL DRL agent: {model_name.upper()}")

        # Model training parameters for FinRobot/SB3
        model_kwargs = {
            "learning_rate": agent_params.get("learning_rate", 0.0001),
            "n_steps": agent_params.get("n_steps", 2048),
            "batch_size": agent_params.get("batch_size", 64),
            "ent_coef": agent_params.get("ent_coef", 0.01),
            "gamma": agent_params.get("gamma", 0.99),
            "verbose": agent_params.get("verbose", 0),
        }
        if "net_dims" in agent_params:
            model_kwargs["policy_kwargs"] = dict(net_arch=[dict(pi=agent_params["net_dims"], vf=agent_params["net_dims"])])

        model = agent.get_model(model_name, model_kwargs=model_kwargs)
        
        # Training
        trained_model = agent.train_model(model=model,
                                          tb_log_name=model_name,
                                          total_timesteps=total_timesteps)
        
        # Save the model
        model_filename = f"FINRL_{model_name.upper()}_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model_save_path = os.path.join(model_output_dir, model_filename)
        trained_model.save(model_save_path)
        logger.info(f"Trained FinRL agent model saved to: {model_save_path}")
        
        # Log model with MLflow
        mlflow_utils.log_model(trained_model, artifact_path=f"rl_models/finrl_{model_name}")
    else:
        logger.error(f"Unsupported RL framework: {framework}. Choose 'finrl' or 'elegantrl'.")
        raise ValueError(f"Unsupported RL framework: {framework}")
    
    return model_save_path


# Function: main
# Description: Main orchestrator for the RL agent training pipeline.
#              It loads configurations, processed data, splits the data,
#              trains the RL agent, and logs results/artifacts using (placeholder) MLflow.
# Input: None (reads configuration and data paths from predefined locations).
# Output: None. Executes the RL agent training pipeline.
# Dependencies: logging, os, sys, datetime, load_config (local function),
#               load_processed_data (local function), split_data (local function),
#               train_rl_agent (local function), mlflow_utils (global placeholder).
# Globals: project_root.
def main():
    logger.debug(f"File: MLOps/pipelines/rl_agent/train_rl_agent.py, Function: main, Purpose: Main orchestrator for the RL agent training pipeline, Output: None (executes pipeline).")
    logger.info("MLOps RL Agent Training Pipeline started.")

    # Load configurations
    global_vars_config_path = os.path.join(project_root, "MLOps", "config", "common", "global_vars.yaml")
    # Assuming a default PPO config, can be made dynamic
    rl_agent_config_path = os.path.join(project_root, "MLOps", "config", "rl_agents", "ppo_stocktrading_params.yaml")
    data_sources_config_path = os.path.join(project_root, "MLOps", "config", "data_sources.yaml")
    
    global_vars_config = load_config(global_vars_config_path)
    rl_agent_config = load_config(rl_agent_config_path)
    data_sources_config = load_config(data_sources_config_path) # Load data_sources.yaml
    
    # Load environment config specified within agent config
    env_config_file = rl_agent_config.get("environment_config_path", "MLOps/config/environments/stock_trading_env_config.yaml")
    env_config_path = os.path.join(project_root, env_config_file)
    env_config = load_config(env_config_path)

    # Define input and output paths
    # Get processed data directory from global_vars_config
    paths_config = global_vars_config.get("paths", {})
    processed_data_dir_rel = paths_config.get("processed_data_dir", "data/processed")
    processed_data_abs_dir = os.path.join(project_root, processed_data_dir_rel)
    
    model_output_dir_rel = paths_config.get("model_registry_dir", "MLOps/model_registry") # More generic
    model_output_dir = os.path.join(project_root, model_output_dir_rel, "rl_agents")
    os.makedirs(model_output_dir, exist_ok=True)

    # MLflow experiment setup (placeholder)
    experiment_name = "FinAI_RL_Training"
    run_name = f"{rl_agent_config.get('agent_name', 'DRL_Agent')}_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow_utils.start_mlflow_run(experiment_name, run_name):
        mlflow_utils.log_params(rl_agent_config)
        mlflow_utils.log_params({"environment_config": env_config})
        mlflow_utils.log_params({"global_vars": global_vars_config})
        mlflow_utils.log_params({"data_sources_config_data_paths": data_sources_config.get("data_paths", {})})


        try:
            # 1. Load Processed Data
            processed_df = load_processed_data(processed_data_abs_dir)

            # 2. Split Data
            # Dates from global_vars_config['date_ranges']
            date_ranges_config = global_vars_config.get('date_ranges', {})
            train_start_date = date_ranges_config.get('training_start_date', '2020-01-01')
            train_end_date = date_ranges_config.get('training_end_date', '2023-12-31')
            # Trade/test dates are used by split_data but the trade_df is not used in this training script
            trade_start_date = date_ranges_config.get('backtesting_start_date', '2024-01-01')
            trade_end_date = date_ranges_config.get('backtesting_end_date', '2024-06-30')
            
            mlflow_utils.log_params({
                "train_start_date": train_start_date, "train_end_date": train_end_date,
                "trade_start_date": trade_start_date, "trade_end_date": trade_end_date
            })

            train_df, _ = split_data(processed_df, train_start_date, train_end_date, trade_start_date, trade_end_date)
            
            if train_df.empty:
                logger.error("Training data is empty after splitting. Check date ranges and processed data.")
                raise ValueError("Training data is empty after splitting.")

            # 3. Train RL Agent
            model_path = train_rl_agent(train_df, rl_agent_config, env_config, model_output_dir, global_vars_config)
            
            # Log some results (e.g., path to saved model)
            mlflow_utils.log_artifact(model_path, "trained_model_files")
            # In a real scenario, you'd capture training metrics (e.g., mean reward) from the DRLAgent/SB3 callbacks
            # and log them using mlflow_utils.log_metrics()

            logger.info(f"MLOps RL Agent Training Pipeline finished successfully. Model saved at {model_path}")

        except Exception as e:
            logger.error(f"MLOps RL Agent Training Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()