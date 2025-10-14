# MLOps/deployment/backtesting/run_backtest.py

import logging
import logging.config
import os
import sys
import yaml
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt # For saving plots

# Ensure the main project directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# FinRL imports
from FinRobot.finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from FinRobot.finrl.agents.stablebaselines3.models import DRLAgent as FinRL_DRLAgent # Renamed to avoid clash if we define local DRLAgent
from FinRobot.finrl.plot import backtest_stats, backtest_plot, get_daily_return # For plotting and stats
from FinRobot.finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor # For baseline data

# MLOps utilities (using placeholder for now)
class MLflowUtilsPlaceholder:
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        # logger.info for placeholder init can be noisy if logger is not fully set up when this class is defined.

    def start_mlflow_run(self, experiment_name, run_name):
        logger.info(f"MLP_PH: Starting run '{run_name}' in experiment '{experiment_name}'.")
        class MockRun:
            def __init__(self):
                self.info = type('info', (object,), {'run_id': 'mock_run_id'}) # Mock run_id
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockRun()

    def log_params(self, params_dict):
        logger.info(f"MLP_PH: Logging parameters: {params_dict}")

    def log_metrics(self, metrics_dict, step=None):
        logger.info(f"MLP_PH: Logging metrics: {metrics_dict} at step {step if step else 'N/A'}")

    def log_artifact(self, local_path, artifact_path=None):
        logger.info(f"MLP_PH: Logging artifact from {local_path} to {artifact_path if artifact_path else ''}")

mlflow_utils = MLflowUtilsPlaceholder()

# Configure logging
# Ensure logging_config.py is correctly located and configured
try:
    logging_config_path = os.path.join(project_root, 'config', 'logging_config.py')
    if os.path.exists(logging_config_path):
        logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
    else:
        # Fallback basic logging if config file is missing
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.warning(f"Logging config file not found at {logging_config_path}. Using basicConfig.")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.error(f"Error configuring logging from file: {e}. Using basicConfig.")

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    logger.debug(f"Loading YAML configuration from: {config_path}")
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
def load_processed_data(data_path: str) -> pd.DataFrame:
    logger.debug(f"Loading processed feature-engineered data from CSV: {data_path}")
    """Loads processed feature-engineered data."""
    logger.info(f"Loading processed data for backtesting from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Processed data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Processed data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading processed data: {e}", exc_info=True)
        raise

# Function: get_trade_data
# Description: Filters a DataFrame to include only data within the specified trading/backtesting period.
#              The data is sorted by date and ticker.
# Input:
#   df: pandas DataFrame, the input data containing a 'date' column.
#   trade_start_date: String, the start date of the trading period (inclusive).
#   trade_end_date: String, the end date of the trading period (exclusive).
# Output: A pandas DataFrame filtered for the trading period.
# Dependencies: pandas (pd), logging.
def get_trade_data(df: pd.DataFrame, trade_start_date: str, trade_end_date: str) -> pd.DataFrame:
    logger.debug(f"Filtering DataFrame for trading period: {trade_start_date} to {trade_end_date}")
    """Filters data for the trading/backtesting period."""
    logger.info(f"Filtering data for trading period: {trade_start_date} - {trade_end_date}")
    trade_df = df[(df["date"] >= trade_start_date) & (df["date"] < trade_end_date)]
    trade_df = trade_df.sort_values(["date", "tic"]).reset_index(drop=True)
    logger.info(f"Trading data shape: {trade_df.shape}")
    if trade_df.empty:
        logger.error("Trade data is empty after filtering. Check date ranges.")
        raise ValueError("Trade data is empty.")
    return trade_df

# Function: run_backtesting
# Description: Runs the backtesting process for a trained DRL agent.
#              This involves setting up the trading environment, loading the trained model,
#              running predictions (trading simulation), saving results (account values, actions, plots),
#              and calculating performance statistics.
# Input:
#   trade_df: pandas DataFrame, the data for the trading period.
#   trained_model_path: String, path to the saved trained DRL model.
#   agent_params: Dictionary, parameters for the DRL agent (e.g., agent_name).
#   env_params: Dictionary, parameters for the stock trading environment.
#   results_output_dir: String, directory to save backtesting results.
# Output: A tuple containing paths to the saved account value CSV, actions CSV, and plot image.
# Dependencies: logging, os, datetime, matplotlib.pyplot, pandas (pd),
#               FinRL.finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv,
#               FinRL.finrl.agents.stablebaselines3.models.DRLAgent,
#               FinRL.finrl.plot (backtest_stats, backtest_plot, get_daily_return),
#               FinRL.finrl.meta.data_processors.processor_yahoofinance.YahooFinanceProcessor (optional for baseline),
#               mlflow_utils (global placeholder).
def run_backtesting(
    trade_df: pd.DataFrame,
    trained_model_path: str,
    agent_params: dict, # For agent name
    env_params: dict,
    results_output_dir: str,
    backtesting_config: dict # Pass the full backtesting config for more params
):
    logger.debug(f"Initiating backtesting function for model: {trained_model_path}")
    """
    Runs backtesting for a trained RL agent.
    """
    logger.info(f"Starting backtesting for model: {trained_model_path}")

    stock_dimension = len(trade_df.tic.unique())
    
    # Get tech and sentiment features from env_params (which should be loaded from environment_config_file)
    configured_tech_indicators = env_params.get('tech_indicator_list', [])
    available_tech_indicators = [col for col in configured_tech_indicators if col in trade_df.columns]
    if len(available_tech_indicators) != len(configured_tech_indicators):
        logger.warning(f"Some tech indicators specified in env_config ({configured_tech_indicators}) are not in the trade data. Using available: {available_tech_indicators}.")

    configured_sentiment_features = env_params.get('sentiment_feature_list', [])
    available_sentiment_features = [col for col in configured_sentiment_features if col in trade_df.columns]
    if configured_sentiment_features and len(available_sentiment_features) != len(configured_sentiment_features):
        logger.warning(f"Some sentiment features specified in env_config ({configured_sentiment_features}) are not in the trade data. Using available: {available_sentiment_features}.")

    # Prepare environment arguments
    env_kwargs = {
        "df": trade_df,
        "stock_dim": stock_dimension,
        "hmax": env_params.get('hmax', 100),
        "initial_amount": env_params.get('initial_amount', 1000000),
        "num_stock_shares": [0] * stock_dimension, # Start with no shares for backtesting
        "buy_cost_pct": env_params.get('buy_cost_pct', 0.001),
        "sell_cost_pct": env_params.get('sell_cost_pct', 0.001),
        "reward_scaling": env_params.get('reward_scaling', 1.0),
        "state_space": stock_dimension, # Env calculates actual
        "action_space": stock_dimension, # Env calculates actual
        "tech_indicator_list": available_tech_indicators,
        "sentiment_feature_list": available_sentiment_features,
        "turbulence_threshold": env_params.get('turbulence_threshold'), # Can be None
        "risk_indicator_col": env_params.get('risk_indicator_col', 'turbulence'),
        "make_plots": False,
        "print_verbosity": 0,
        "mode": "trade", # Set mode to trade for backtesting
        "initial": True # Start fresh for backtest
    }
    env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
    
    env_trade = StockTradingEnv(**env_kwargs)
    
    # Load the trained DRL model using FinRL's DRLAgent static method for prediction
    # This is similar to how FinRobot/finrl/test.py loads models for testing.
    # DRL_prediction_load_from_file in FinRL_DRLAgent handles model loading and prediction loop.
    
    drl_lib = backtesting_config.get("drl_lib", "stable_baselines3")
    model_name_for_load = backtesting_config.get("model_name", "ppo") # e.g. "ppo", "sac"
    deterministic_pred = backtesting_config.get("deterministic_prediction", True)

    logger.info(f"Attempting to load model {model_name_for_load} from {trained_model_path} using {drl_lib} conventions.")

    if drl_lib == "stable_baselines3":
        # FinRL_DRLAgent.DRL_prediction_load_from_file returns episode_total_assets
        # We need account_value and actions DataFrame.
        # The DRL_prediction method (non-static) in FinRL_DRLAgent is more suitable if we load model first.
        
        # Let's adapt the DRL_prediction logic here for clarity and control
        try:
            model_to_predict = FinRL_DRLAgent.MODELS[model_name_for_load].load(trained_model_path, env=env_trade)
            logger.info(f"SB3 Model {trained_model_path} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SB3 model {trained_model_path}: {e}", exc_info=True)
            raise
        
        # Manually run the prediction loop to get account_value and actions
        # This mirrors parts of FinRL_DRLAgent.DRL_prediction
        test_env_instance, _ = env_trade.get_sb_env() # Get the DummyVecEnv
        obs = test_env_instance.reset()
        
        for _ in range(len(env_trade.df.index.unique())):
            action, _ = model_to_predict.predict(obs, deterministic=deterministic_pred)
            obs, _, dones, _ = test_env_instance.step(action)
            if dones[0]:
                break
        
        df_account_value = env_trade.save_asset_memory()
        df_actions = env_trade.save_action_memory()

    elif drl_lib == "elegantrl":
        logger.warning("ElegantRL backtesting from a saved model path is not fully implemented in this script yet.")
        logger.warning("Using placeholder data for ElegantRL backtest results.")
        # Placeholder for ElegantRL prediction - this would need ElegantRL's specific loading and prediction logic
        # For now, create dummy DataFrames
        dates_for_dummy = pd.to_datetime(trade_df['date'].unique())
        df_account_value = pd.DataFrame({
            'date': dates_for_dummy,
            'account_value': env_params.get('initial_amount', 1000000) * (1 + (pd.Series(range(len(dates_for_dummy))) * 0.001)) # Dummy upward trend
        })
        df_actions = pd.DataFrame(index=dates_for_dummy[:-1]) # Actions up to T-1
        for tic in trade_df['tic'].unique():
            df_actions[tic] = 0 # Dummy no actions
        df_actions.reset_index(inplace=True)
        df_actions.rename(columns={'index':'date'}, inplace=True)

    else:
        logger.error(f"Unsupported DRL library for backtesting: {drl_lib}")
        raise ValueError(f"Unsupported DRL library: {drl_lib}")
    
    # Save account value and actions
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    account_value_filename = os.path.join(results_output_dir, f"account_value_{agent_name}_{timestamp_str}.csv")
    actions_filename = os.path.join(results_output_dir, f"actions_{agent_name}_{timestamp_str}.csv")
    df_account_value.to_csv(account_value_filename, index=False)
    df_actions.to_csv(actions_filename, index=False)
    logger.info(f"Backtesting account values saved to: {account_value_filename}")
    logger.info(f"Backtesting actions saved to: {actions_filename}")

    # Calculate and log performance statistics
    logger.info("Calculating backtesting performance statistics...")
    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name='account_value') # FinRL default
    # perf_stats_all_pd = pd.DataFrame.from_dict(perf_stats_all, orient='index', columns=['value'])
    
    # Log metrics to MLflow (placeholder)
    # Flatten the perf_stats_all if it's a nested dict/Series for MLflow
    flat_stats = {}
    if isinstance(perf_stats_all, pd.Series):
        flat_stats = perf_stats_all.to_dict()
    elif isinstance(perf_stats_all, pd.DataFrame): # if it returns a DataFrame
        # Potentially select specific rows/columns or flatten
        flat_stats = perf_stats_all.iloc[0].to_dict() if not perf_stats_all.empty else {}


    mlflow_utils.log_metrics(flat_stats)
    logger.info(f"Backtesting Performance Stats:\n{perf_stats_all}")

    # Generate and save plots
    plot_filename = os.path.join(results_output_dir, f"backtest_plot_{agent_name}_{timestamp_str}.png")
    
    # Need baseline data for comparison plot
    baseline_ticker = backtesting_config.get("baseline_ticker_for_pyfolio", "^GSPC")
    baseline_returns = None
    
    if baseline_ticker:
        try:
            logger.info(f"Fetching baseline data for: {baseline_ticker}")
            baseline_df = YahooFinanceProcessor().download_data(
                start_date=trade_df['date'].min().strftime('%Y-%m-%d'),
                end_date=trade_df['date'].max().strftime('%Y-%m-%d'),
                ticker_list=[baseline_ticker],
                time_interval="1D"
            )
            if not baseline_df.empty and 'close' in baseline_df.columns:
                # Ensure the column used for get_daily_return matches the ticker name for clarity in pyfolio
                baseline_df = baseline_df.rename(columns={'close': baseline_ticker})
                baseline_returns = get_daily_return(baseline_df, value_col_name=baseline_ticker)
                logger.info(f"Successfully fetched and processed baseline data for {baseline_ticker}.")
            else:
                logger.warning(f"Baseline data for {baseline_ticker} is empty or missing 'close' column.")
        except ImportError:
            logger.warning("YahooFinanceProcessor not found. Baseline data for plotting will be skipped.")
        except Exception as e:
            logger.warning(f"Could not fetch baseline data for {baseline_ticker}: {e}. Plot will not include baseline.", exc_info=True)

    # Generate pyfolio plot if enabled
    if backtesting_config.get("generate_pyfolio_plots", True):
        try:
            plt.figure(figsize=(15,10)) # Create a figure context for backtest_plot
            backtest_plot( # This function from finrl.plot uses pyfolio.create_full_tear_sheet
                account_value=df_account_value, # Pass the DataFrame directly
                baseline_df=baseline_df if baseline_returns is not None else None, # Pass baseline_df if available
                baseline_ticker=baseline_ticker if baseline_returns is not None else None
            )
            plt.savefig(plot_filename)
            plt.close() # Close the plot to free memory
            logger.info(f"Backtesting plot saved to: {plot_filename}")
            mlflow_utils.log_artifact(plot_filename, "backtesting_plots")
        except Exception as e:
            logger.error(f"Error generating or saving backtest plot: {e}", exc_info=True)
    else:
        logger.info("Pyfolio plot generation skipped as per configuration.")
    plt.savefig(plot_filename)
    plt.close() # Close the plot to free memory
    logger.info(f"Backtesting plot saved to: {plot_filename}")
    mlflow_utils.log_artifact(plot_filename, "backtesting_plots")

    logger.info("Backtesting finished.")
    return account_value_filename, actions_filename, plot_filename


# Function: main
# Description: The main entry point for the backtesting script.
#              It orchestrates the loading of configurations, data, and the trained model,
#              then initiates the backtesting process and logs results using (placeholder) MLflow.
# Input: None (reads paths and configurations from files or predefined locations).
# Output: None. Executes the backtesting pipeline.
# Dependencies: logging, os, sys, load_config (local function), load_processed_data (local function),
#               get_trade_data (local function), run_backtesting (local function),
#               mlflow_utils (global placeholder), datetime.
# Globals: project_root.
def main():
    logger.debug(f"Main orchestrator for the backtesting pipeline started.")
    logger.info("MLOps Backtesting Pipeline started.")

    # Load configurations
    global_vars_config = load_config(os.path.join(project_root, "MLOps", "config", "common", "global_vars.yaml"))
    backtesting_pipeline_config = load_config(os.path.join(project_root, "MLOps", "config", "backtesting_config.yaml"))
    
    # Load the specific RL agent config (e.g., PPO, SAC) if needed for agent_name, though model_name in backtesting_config should suffice
    # For simplicity, we'll rely on model_name from backtesting_config for DRLAgent.
    # rl_agent_config_path = os.path.join(project_root, "MLOps", "config", "rl_agents",
    #                                     f"{backtesting_pipeline_config.get('model_name', 'ppo').lower()}_stocktrading_params.yaml") # Construct path
    # rl_agent_config = load_config(rl_agent_config_path) # This might not be strictly needed if model_name is enough

    env_config_file_rel = backtesting_pipeline_config.get("environment_config_file", "MLOps/config/environments/stock_trading_env_config.yaml")
    env_config = load_config(os.path.join(project_root, env_config_file_rel))

    # Define input paths
    processed_data_dir_rel = global_vars_config.get("paths", {}).get("processed_data_dir", "data/processed")
    processed_data_abs_dir = os.path.join(project_root, processed_data_dir_rel)
    
    # Find the latest processed data file
    try:
        processed_files = sorted(
            [f for f in os.listdir(processed_data_abs_dir) if f.startswith("processed_market_sentiment_data_") and f.endswith(".csv")],
            reverse=True
        )
        if not processed_files:
            raise FileNotFoundError(f"No 'processed_market_sentiment_data_*.csv' files found in {processed_data_abs_dir}")
        latest_processed_data_file = os.path.join(processed_data_abs_dir, processed_files[0])
        logger.info(f"Using latest processed data file: {latest_processed_data_file}")
    except Exception as e:
        logger.error(f"Error finding latest processed data file: {e}", exc_info=True)
        sys.exit(1)

    trained_model_path_rel = backtesting_pipeline_config.get("trained_model_path", "models/rl_agents/placeholder_model.zip")
    trained_model_path_abs = os.path.join(project_root, trained_model_path_rel)

    if not os.path.exists(trained_model_path_abs):
        logger.error(f"Trained model for backtesting not found at: {trained_model_path_abs}. Please train a model first or update path in backtesting_config.yaml.")
        sys.exit(1)

    # Define output directory for backtesting results
    results_base_dir_rel = global_vars_config.get("paths", {}).get("results_dir", "MLOps/results")
    results_subdir = backtesting_pipeline_config.get("results_subdir", "backtesting_runs")
    current_run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    specific_results_output_dir = os.path.join(project_root, results_base_dir_rel, results_subdir,
                                             f"{backtesting_pipeline_config.get('backtest_run_name', 'DefaultBacktest')}_{current_run_timestamp}")
    os.makedirs(specific_results_output_dir, exist_ok=True)

    # MLflow experiment setup
    experiment_name = "FinAI_RL_Backtesting"
    run_name = f"{backtesting_pipeline_config.get('backtest_run_name', 'DRL_Backtest')}_{current_run_timestamp}"

    with mlflow_utils.start_mlflow_run(experiment_name, run_name):
        mlflow_utils.log_params({"trained_model_path": trained_model_path_rel}) # Log relative path
        mlflow_utils.log_params({"backtesting_config": backtesting_pipeline_config})
        mlflow_utils.log_params({"environment_config_used": env_config})
        mlflow_utils.log_params({"processed_data_file_used": latest_processed_data_file})


        try:
            # 1. Load Processed Data
            processed_df = load_processed_data(latest_processed_data_file)

            # 2. Get Trade Data for backtesting period
            date_ranges_config = global_vars_config.get('date_ranges', {})
            trade_start_date = date_ranges_config.get('backtesting_start_date', '2024-01-01')
            trade_end_date = date_ranges_config.get('backtesting_end_date', '2024-06-30')
            mlflow_utils.log_params({"backtest_trade_start_date": trade_start_date, "backtest_trade_end_date": trade_end_date})
            
            trade_df = get_trade_data(processed_df, trade_start_date, trade_end_date)

            # 3. Run Backtesting
            # Pass relevant parts of configs
            agent_params_for_backtest = {
                "agent_name": backtesting_pipeline_config.get("model_name", "ppo"),
                "drl_lib": backtesting_pipeline_config.get("drl_lib", "stable_baselines3")
            }

            run_backtesting(
                trade_df,
                trained_model_path_abs,
                agent_params_for_backtest, # Contains agent_name and drl_lib
                env_config, # Contains env parameters like tech_indicator_list, sentiment_feature_list
                specific_results_output_dir,
                backtesting_pipeline_config # Pass the full backtesting config
            )
            
            # DVC tracking for results directory
            if backtesting_pipeline_config.get("dvc_tracking", {}).get("enabled", False):
                try:
                    dvc_add_target = os.path.join(results_base_dir_rel, results_subdir) # Track the specific run's parent dir or the whole subdir
                    logger.info(f"Attempting to run 'dvc add {dvc_add_target}' for backtest results.")
                    subprocess.run(["dvc", "add", dvc_add_target], check=True, cwd=project_root)
                    logger.info(f"DVC tracking updated for: {dvc_add_target}")
                    dvc_results_file = dvc_add_target.strip(os.sep) + ".dvc"
                    if os.path.exists(os.path.join(project_root, dvc_results_file)):
                         mlflow_utils.log_artifact(os.path.join(project_root, dvc_results_file), "dvc_metadata_results")
                except Exception as e:
                    logger.error(f"Failed to run 'dvc add' for backtest results {dvc_add_target}: {e}", exc_info=True)

            logger.info(f"MLOps Backtesting Pipeline finished successfully. Results in {specific_results_output_dir}")
            logger.info(f"MLOps Backtesting Pipeline finished successfully. Results in {backtest_results_dir}")

        except Exception as e:
            logger.error(f"MLOps Backtesting Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()