# MLOps/pipelines/data_ingestion/data_ingestion_pipeline.py
"""
Data ingestion pipeline script for FinRL.

This script orchestrates the downloading of stock data using YahooDownloader
and news data using NewsDataDownloader. It loads configurations for API keys
and data sources, fetches the data, and saves it to specified raw data directories
with timestamping.
"""

import os
import sys
import yaml
import logging
import pandas as pd
from datetime import datetime
import subprocess
import mlflow
import mlflow.entities # For RunStatus
import time

# --- Configuration Loading ---
# Add project root to sys.path to allow importing FinRL and config modules
# This assumes the script is run from the MLOps/pipelines/data_ingestion directory or project root
try:
    # Navigating up to the project root (FinAI_algo)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    # Correctly locate FinRL relative to the new PROJECT_ROOT
    FINRL_PATH = os.path.join(PROJECT_ROOT, "FinRL")
    if FINRL_PATH not in sys.path:
        sys.path.append(FINRL_PATH)

except Exception as e:
    print(f"Error adjusting sys.path: {e}. Ensure script is run from a valid location.")
    # Fallback for sys.path if dynamic adjustment fails (e.g. if script is moved)
    # This might require manual adjustment if the script's location changes significantly
    if os.getcwd() not in sys.path: # Add current working directory as a last resort
        sys.path.append(os.getcwd())
    # Attempt to add common relative paths for FinRL if direct structure is not found
    if "../../../FinRL" not in sys.path: # Relative to MLOps/pipelines/data_ingestion
        sys.path.append("../../../FinRL")


# Import FinRL components after path adjustment
try:
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.data_processors.news_downloader import NewsDataDownloader
except ImportError as e:
    print(f"Failed to import FinRL modules: {e}. Check sys.path and FinRL installation.")
    # Provide more specific guidance if possible
    print(f"Current sys.path: {sys.path}")
    print(f"Expected FinRL location: {FINRL_PATH if 'FINRL_PATH' in locals() else 'Not determined'}")
    sys.exit(1)


# --- Global Variables & Configuration ---
CONFIG_DIR = os.path.join(PROJECT_ROOT, "MLOps", "config")
API_KEYS_PATH = os.path.join(PROJECT_ROOT, "config", "api_keys.yaml")
DATA_SOURCES_CONFIG_PATH = os.path.join(CONFIG_DIR, "data_sources.yaml")

# --- Logger Setup ---
# Basic logger setup, to be enhanced by logging_config.py if available
# As per coding standards, logger should be configured centrally.
# This is a local fallback/setup for this specific pipeline script.
pipeline_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)

def load_yaml_config(file_path: str) -> dict:
    """
    Loads a YAML configuration file.
    This function loads a YAML file from the given path and returns its content as a dictionary.
    It handles potential file not found errors.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        pipeline_logger.info(f"Successfully loaded YAML configuration from: {file_path}")
        return config
    except FileNotFoundError:
        pipeline_logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        pipeline_logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    print(f"Loaded YAML config from {file_path}. The content is: {config}")
    return config

def setup_logging(config: dict):
    """
    Sets up logging for the pipeline based on configuration.
    This function configures the logging level and output file for the pipeline.
    It uses settings from the provided configuration dictionary.
    """
    log_config = config.get("logging", {})
    log_level_str = log_config.get("level", "INFO").upper()
    log_file_rel_path = log_config.get("log_file", "data_ingestion_pipeline.log")

    # Ensure log directory exists (relative to this script's dir)
    log_dir = os.path.join(os.path.dirname(__file__), "logs") # Store logs within data_ingestion folder
    os.makedirs(log_dir, exist_ok=True)
    log_file_abs_path = os.path.join(log_dir, os.path.basename(log_file_rel_path))


    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs if script is re-run in same session
    for handler in pipeline_logger.handlers[:]:
        pipeline_logger.removeHandler(handler)

    # Add new handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    pipeline_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_abs_path, mode='a') # Append mode
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)
    
    pipeline_logger.setLevel(log_level)
    pipeline_logger.info(f"Logging setup complete. Level: {log_level_str}. Log file: {log_file_abs_path}")
    print(f"Logging setup. Level: {log_level_str}. File: {log_file_abs_path}")


def get_run_parameters(config: dict, run_mode: str = "test") -> tuple[list[str], str, str]:
    """
    Retrieves ticker list, start date, and end date based on run mode.
    This function selects the appropriate parameters (ticker list, start/end dates) from the
    configuration based on whether the pipeline is run in 'test' or 'prod' mode.
    It helps in managing different datasets for development and production.
    """
    common_params = config.get("common_params", {})
    if run_mode == "test":
        tickers = common_params.get("test_ticker_list", ["AAPL"])
        start_date = common_params.get("test_start_date", "2023-01-01")
        end_date = common_params.get("test_end_date", "2023-01-07")
        pipeline_logger.info("Running in TEST mode with test parameters.")
    else: # prod or default
        tickers = common_params.get("prod_ticker_list", ["AAPL", "MSFT"]) # Default to something if not set
        start_date = common_params.get("prod_start_date", "2020-01-01")
        # For end_date in prod, you might want it to be dynamic (e.g., today)
        # For now, using a fixed placeholder or the one from config
        end_date = common_params.get("prod_end_date", datetime.now().strftime("%Y-%m-%d"))
        pipeline_logger.info("Running in PRODUCTION mode with production parameters.")
    
    print(f"Run parameters for mode '{run_mode}': Tickers: {tickers}, Start: {start_date}, End: {end_date}")
    return tickers, start_date, end_date


def download_stock_data(config: dict, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Downloads stock data using YahooDownloader.
    This function initializes the YahooDownloader with the provided configuration and parameters,
    fetches historical stock data, and returns it as a pandas DataFrame.
    It handles potential errors during the download process.
    """
    yahoo_config = config.get("yahoo_finance", {})
    if not yahoo_config.get("enabled", False):
        pipeline_logger.info("Yahoo Finance data download is disabled in config.")
        print("Yahoo Finance download disabled.")
        return None

    downloader = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=tickers
    )
    try:
        pipeline_logger.info(f"Fetching stock data for tickers: {tickers} from {start_date} to {end_date} via YahooDownloader.")
        stock_df = downloader.fetch_data(auto_adjust=yahoo_config.get("auto_adjust_prices", False))
        pipeline_logger.info(f"Successfully fetched {len(stock_df)} rows of stock data.")
        print(f"Fetched {len(stock_df)} stock data rows.")
        return stock_df
    except Exception as e:
        pipeline_logger.error(f"Error downloading stock data: {e}", exc_info=True)
        print(f"Error downloading stock data: {e}")
        return None

def download_news_data(config: dict, api_keys: dict, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Downloads news data using NewsDataDownloader for all enabled sources.
    This function iterates through configured news sources, initializes NewsDataDownloader for each,
    fetches news articles, and concatenates them into a single pandas DataFrame.
    It relies on API keys loaded separately.
    """
    news_config = config.get("news_data", {})
    if not news_config.get("enabled", False):
        pipeline_logger.info("News data download is disabled in config.")
        print("News data download disabled.")
        return None

    all_news_df = pd.DataFrame()
    configured_sources = news_config.get("sources", [])

    for source_details in configured_sources:
        source_name = source_details.get("source_name")
        if not source_details.get("enabled", False):
            pipeline_logger.info(f"News source '{source_name}' is disabled in config. Skipping.")
            print(f"News source '{source_name}' disabled.")
            continue

        pipeline_logger.info(f"Attempting to download news from source: {source_name}")
        # NewsDataDownloader tries to load API keys internally from a global API_KEYS or passed one.
        # We ensure it has access to the keys loaded by this pipeline.
        # The NewsDataDownloader itself has logic to pick the correct key based on `source_name`.
        # No need to pass api_keys[source_name.upper() + "_API_KEY"] directly unless its internal logic is insufficient.
        
        # Extract kwargs for the specific source from data_sources.yaml
        source_kwargs = {k: v for k, v in source_details.items() if k not in ["source_name", "enabled"]}

        try:
            # The NewsDataDownloader will use its internal logic to find the API key
            # from the globally loaded API_KEYS (if its path logic works) or if passed.
            # For robustness, we can pass the specific key if the internal loading is problematic.
            # However, the current NewsDataDownloader tries to load from `config/api_keys.yaml` itself.
            # Let's rely on its internal mechanism first.
            # If specific key passing is needed:
            # specific_api_key = None
            # if source_name == "finnhub": specific_api_key = api_keys.get("FINNHUB_API_KEY")
            # elif source_name == "newsapi": specific_api_key = api_keys.get("NEWS_API_ORG_KEY")
            # ... and so on for other keys.
            
            downloader = NewsDataDownloader(
                source=source_name,
                ticker_list=tickers,
                start_date=start_date,
                end_date=end_date,
                # api_key=specific_api_key, # Optionally pass key directly
                **source_kwargs # Pass other params like api_call_delay
            )
            news_df = downloader.fetch_news()
            if news_df is not None and not news_df.empty:
                pipeline_logger.info(f"Successfully fetched {len(news_df)} news items from {source_name}.")
                print(f"Fetched {len(news_df)} news items from {source_name}.")
                all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)
            else:
                pipeline_logger.warning(f"No news data returned from {source_name}.")
                print(f"No news data from {source_name}.")
        except Exception as e:
            pipeline_logger.error(f"Error downloading news data from {source_name}: {e}", exc_info=True)
            print(f"Error with news source {source_name}: {e}")

    if not all_news_df.empty:
        pipeline_logger.info(f"Total news items fetched from all enabled sources: {len(all_news_df)}")
        print(f"Total news items: {len(all_news_df)}")
        return all_news_df.sort_values(by=['timestamp', 'ticker']).reset_index(drop=True)
    else:
        pipeline_logger.warning("No news data fetched from any enabled source.")
        print("No news data fetched overall.")
        return None

def save_data(df: pd.DataFrame, base_path: str, filename_prefix: str, timestamp_format: str, tickers: list[str]):
    """
    Saves a DataFrame to a CSV file with a timestamp and ticker information in the filename.
    This function constructs a filename incorporating the provided prefix, current timestamp,
    and a summary of tickers. It creates the necessary directories and saves the DataFrame.
    It ensures data is stored in an organized manner.
    """
    if df is None or df.empty:
        pipeline_logger.info(f"No data to save for prefix {filename_prefix}.")
        print(f"No data to save for {filename_prefix}.")
        return

    os.makedirs(base_path, exist_ok=True)
    timestamp_str = datetime.now().strftime(timestamp_format)
    
    # Create a condensed ticker string for the filename
    if len(tickers) > 3:
        ticker_str = f"{'_'.join(tickers[:2])}_etal"
    else:
        ticker_str = '_'.join(tickers)
    ticker_str = ticker_str.replace(r"/", "_") # Sanitize for filename

    filename = f"{filename_prefix}_{ticker_str}_{timestamp_str}.csv"
    full_path = os.path.join(base_path, filename)

    try:
        df.to_csv(full_path, index=False)
        pipeline_logger.info(f"Data successfully saved to: {full_path}")
        print(f"Data saved to: {full_path}")
        dvc_file_path = None

        # Add the directory to DVC to track the new/updated file
        try:
            pipeline_logger.info(f"Attempting to run 'dvc add {base_path}' to track changes.")
            result = subprocess.run(["dvc", "add", base_path], check=True, capture_output=True, text=True)
            pipeline_logger.info(f"Successfully ran 'dvc add {base_path}'. Output: {result.stdout.strip()}")
            print(f"DVC tracking updated for: {base_path}")
            # Construct the .dvc file path (e.g., data/raw/stock_data.dvc)
            dvc_file_path = base_path.strip(os.sep) + ".dvc" # os.sep ensures correct handling of trailing slashes
            if not os.path.exists(dvc_file_path):
                 # if base_path was like "data/raw/stock_data/" dvc creates "data/raw/stock_data.dvc"
                 # if base_path was like "data/raw/stock_data" dvc creates "data/raw/stock_data.dvc"
                 # let's try to find it if it's not immediately obvious
                 alt_dvc_file_path = os.path.join(os.path.dirname(base_path), os.path.basename(base_path) + ".dvc")
                 if os.path.exists(alt_dvc_file_path):
                     dvc_file_path = alt_dvc_file_path
                 else:
                    pipeline_logger.warning(f"Could not find DVC file at {dvc_file_path} or {alt_dvc_file_path} after 'dvc add'. Artifact logging might be incomplete.")
                    dvc_file_path = None # Reset if not found
        except subprocess.CalledProcessError as e:
            pipeline_logger.error(f"Failed to run 'dvc add {base_path}'. Error: {e.stderr.strip()}")
            print(f"Error updating DVC tracking for {base_path}: {e.stderr.strip()}")
        except FileNotFoundError:
            pipeline_logger.error("DVC command not found. Ensure DVC is installed and in your system's PATH.")
            print("DVC command not found. Please install DVC and ensure it's in PATH.")

    except Exception as e:
        pipeline_logger.error(f"Error saving data to {full_path}: {e}", exc_info=True)
        print(f"Error saving data to {full_path}: {e}")
    return dvc_file_path


def main(run_mode: str = "test"):
    """
    Main function to run the data ingestion pipeline.
    This function orchestrates the entire data ingestion process: loading configurations,
    setting up logging, fetching stock and news data, and saving the results.
    It uses the specified run_mode ('test' or 'prod') to determine parameters.
    """
    MLFLOW_EXPERIMENT_NAME = "Data Ingestion"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    pipeline_start_time = time.time()

    try:
        data_sources_config = load_yaml_config(DATA_SOURCES_CONFIG_PATH)
        api_keys_config = load_yaml_config(API_KEYS_PATH) # NewsDownloader also tries to load this
    except Exception as e:
        pipeline_logger.critical(f"Failed to load critical configuration. Exiting. Error: {e}")
        print(f"Critical config load error: {e}")
        # Log failure to MLflow if possible, though unlikely if config load fails early
        with mlflow.start_run(run_name=f"data_ingestion_pipeline_FAIL_config_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.set_tag("mlflow.runName", run.data.tags.get("mlflow.runName")) # Ensure name is set
            mlflow.log_param("run_mode", run_mode)
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error_type", "ConfigurationLoadError")
            mlflow.set_tag("error_message", str(e))
        return

    setup_logging(data_sources_config) # Setup logging based on data_sources_config
    
    run_name = f"data_ingestion_{run_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("mlflow.runName", run_name) # Set the name tag for UI
        pipeline_logger.info(f"--- Starting Data Ingestion Pipeline (MLflow Run ID: {run.info.run_id}) ---")
        print(f"--- Starting Data Ingestion Pipeline (MLflow Run ID: {run.info.run_id}) ---")

        mlflow.log_param("run_mode", run_mode)
        mlflow.log_params({
            "project_root": PROJECT_ROOT,
            "data_sources_config_path": DATA_SOURCES_CONFIG_PATH,
            "api_keys_path": API_KEYS_PATH
        })

        # Log relevant parts of data_sources_config
        if data_sources_config:
            mlflow.log_dict(data_sources_config.get("common_params", {}), "common_params.json")
            mlflow.log_dict(data_sources_config.get("yahoo_finance", {}), "yahoo_finance_config.json")
            mlflow.log_dict(data_sources_config.get("news_data", {}), "news_data_config.json")
            mlflow.log_dict(data_sources_config.get("data_output", {}), "data_output_config.json")


        tickers, start_date, end_date = get_run_parameters(data_sources_config, run_mode)
        mlflow.log_param("tickers_count", len(tickers))
        mlflow.log_param("tickers_list", ", ".join(tickers) if len(tickers) < 10 else ", ".join(tickers[:10]) + "...") # Log first few for brevity
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        
        output_config = data_sources_config.get("data_output", {})
        raw_stock_dir_rel = output_config.get("raw_stock_data_dir", "data/raw/stock_data")
        raw_news_dir_rel = output_config.get("raw_news_data_dir", "data/raw/news_data")
        raw_stock_dir = os.path.join(PROJECT_ROOT, raw_stock_dir_rel)
        raw_news_dir = os.path.join(PROJECT_ROOT, raw_news_dir_rel)
        ts_format = output_config.get("timestamp_format", "%Y%m%d_%H%M%S")

        # Download Stock Data
        pipeline_logger.info("--- Processing Stock Data ---")
        stock_dl_start_time = time.time()
        stock_df = download_stock_data(data_sources_config, tickers, start_date, end_date)
        stock_dl_time = time.time() - stock_dl_start_time
        mlflow.log_metric("stock_download_duration_sec", round(stock_dl_time, 2))

        stock_dvc_file = None
        if stock_df is not None and not stock_df.empty:
            mlflow.log_metric("stock_data_rows", stock_df.shape[0])
            mlflow.log_metric("stock_data_cols", stock_df.shape[1])
            if 'tic' in stock_df.columns:
                mlflow.log_metric("stock_data_unique_tickers", stock_df['tic'].nunique())
            
            stock_dvc_file = save_data(
                stock_df,
                raw_stock_dir,
                data_sources_config.get("yahoo_finance", {}).get("filename_prefix", "yahoodata"),
                ts_format,
                tickers
            )
            if stock_dvc_file and os.path.exists(stock_dvc_file):
                mlflow.log_artifact(stock_dvc_file, artifact_path="dvc_metadata")
                pipeline_logger.info(f"Logged stock DVC metadata: {stock_dvc_file}")
            elif stock_dvc_file:
                 pipeline_logger.warning(f"Stock DVC file {stock_dvc_file} was reported by save_data but not found for logging.")
        else:
            pipeline_logger.warning("Stock data download returned no data or failed.")
            mlflow.log_metric("stock_data_rows", 0)
            mlflow.log_metric("stock_data_cols", 0)

        # Download News Data
        pipeline_logger.info("--- Processing News Data ---")
        news_dl_start_time = time.time()
        news_df = download_news_data(data_sources_config, api_keys_config, tickers, start_date, end_date)
        news_dl_time = time.time() - news_dl_start_time
        mlflow.log_metric("news_download_duration_sec", round(news_dl_time, 2))

        news_dvc_file = None
        if news_df is not None and not news_df.empty:
            mlflow.log_metric("news_data_rows", news_df.shape[0])
            mlflow.log_metric("news_data_cols", news_df.shape[1])
            if 'ticker' in news_df.columns: # Assuming 'ticker' column for news
                mlflow.log_metric("news_data_unique_tickers", news_df['ticker'].nunique())
            
            news_dvc_file = save_data(
                news_df,
                raw_news_dir,
                data_sources_config.get("news_data", {}).get("filename_prefix", "newsdata"),
                ts_format,
                tickers
            )
            if news_dvc_file and os.path.exists(news_dvc_file):
                mlflow.log_artifact(news_dvc_file, artifact_path="dvc_metadata")
                pipeline_logger.info(f"Logged news DVC metadata: {news_dvc_file}")
            elif news_dvc_file:
                pipeline_logger.warning(f"News DVC file {news_dvc_file} was reported by save_data but not found for logging.")

        else:
            pipeline_logger.warning("News data download returned no data or failed.")
            mlflow.log_metric("news_data_rows", 0)
            mlflow.log_metric("news_data_cols", 0)

        pipeline_duration = time.time() - pipeline_start_time
        mlflow.log_metric("total_pipeline_duration_sec", round(pipeline_duration, 2))
        mlflow.set_tag("status", "COMPLETED")
        pipeline_logger.info("--- Data Ingestion Pipeline Finished ---")
        print("--- Data Ingestion Pipeline Finished ---")

    # Ensure run ends even if exceptions occur within the 'with' block that aren't caught by it
    # (though 'with mlflow.start_run()' should handle this for typical exceptions)
    current_run = mlflow.active_run()
    if current_run:
        run_status = mlflow.tracking.MlflowClient().get_run(current_run.info.run_id).info.status
        if run_status == mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.RUNNING):
            mlflow.end_run(status="KILLED") # Or another appropriate status
            pipeline_logger.warning(f"MLflow run {current_run.info.run_id} was still running and has been forcibly ended.")


if __name__ == "__main__":
    # Example: Run in test mode
    # To run in production mode, call main("prod")
    # Consider using argparse for command-line arguments to set run_mode
    
    # Quick check for paths before running main
    pipeline_logger.info(f"PROJECT_ROOT determined as: {PROJECT_ROOT}")
    pipeline_logger.info(f"DATA_SOURCES_CONFIG_PATH: {DATA_SOURCES_CONFIG_PATH}")
    pipeline_logger.info(f"API_KEYS_PATH: {API_KEYS_PATH}")
    
    # Check if config files exist
    if not os.path.exists(DATA_SOURCES_CONFIG_PATH):
        pipeline_logger.error(f"CRITICAL: data_sources.yaml not found at {DATA_SOURCES_CONFIG_PATH}. Pipeline cannot run.")
    elif not os.path.exists(API_KEYS_PATH):
        pipeline_logger.error(f"CRITICAL: api_keys.yaml not found at {API_KEYS_PATH}. Pipeline may fail for news data.")
    else:
        main(run_mode="test") # Default to test mode

    # To run with command line argument for mode:
    # import argparse
    # parser = argparse.ArgumentParser(description="Run FinRL Data Ingestion Pipeline.")
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default="test",
    #     choices=["test", "prod"],
    #     help="Run mode for the pipeline (test or prod)."
    # )
    # args = parser.parse_args()
    # main(run_mode=args.mode)