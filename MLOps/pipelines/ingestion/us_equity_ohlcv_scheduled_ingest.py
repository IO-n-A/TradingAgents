# MLOps/pipelines/ingestion/us_equity_ohlcv_scheduled_ingest.py
# This script collects daily end-of-day (EOD) Open, High, Low, Close, Volume (OHLCV)
# data, along with VIX, for a predefined list of US equity tickers from Yahoo Finance.
# It implements batching and delays to manage API rate limits and prevent throttling.
# The collected data is saved in a structured format suitable for DVC versioning. 

import datetime
import logging
import os
import time
from typing import List, Dict, Any
import argparse # Added for command-line arguments

import pandas as pd
import yaml
import yfinance as yf

# Determine project root and add to sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
try:
    from config.logging_config import setup_logging, SUCCESS_LEVEL_NUM # Assuming this sets up root logger
    setup_logging() # Initialize with default level
    logger = logging.getLogger(__name__)
    logger.info("Custom logging initialized successfully for us_equity_ohlcv_scheduled_ingest.py.")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    SUCCESS_LEVEL_NUM = 25 # Define it for consistency if import fails
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
    logging.Logger.success = success
    logger.warning(
        "Could not import custom logging_config. Using basic logging for us_equity_ohlcv_scheduled_ingest.py."
    )


# Constants
# These will be overridden by config file values if loaded successfully
DEFAULT_US_EQUITY_TICKERS: List[str] = ["AAPL", "MSFT"] # Minimal default
VIX_TICKER: str = "^VIX"
BATCH_SIZE: int = 50
DELAY_BETWEEN_BATCHES: int = 60
DATA_OUTPUT_BASE_DIR: str = "data/raw/ohlcv/us_equity"
DATE_FORMAT: str = "%Y-%m-%d"
DEFAULT_CONFIG_PATH = os.path.join(project_root, "MLOps/config/data_sources.yaml")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                logger.warning(f"Configuration file '{config_path}' is empty.")
                return {}
            logger.info(f"Configuration loaded successfully from '{config_path}'.")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        return {}

# API keys loading function (remains for good practice, though yfinance doesn't need it for public data)
def load_api_keys(keys_path: str = os.path.join(project_root, "config/api_keys.yaml")) -> Dict[str, Any]:
    """
    Loads API keys from a YAML configuration file.
    This function is a placeholder as yfinance does not typically require API keys
    for public data. However, it's good practice for managing other potential API interactions.

    Args:
        keys_path (str): Path to the API keys YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing API keys or other configurations.
                        Returns an empty dictionary if the file is not found or is empty.
    """
    # This function loads API keys from the specified YAML file.
    # It handles potential file errors and returns a dictionary of keys.
    # For yfinance, keys are not strictly needed, but this structure is for good practice.
    try:
        with open(keys_path, "r") as f:
            api_keys = yaml.safe_load(f)
            if api_keys is None:
                logger.warning(f"API keys file '{keys_path}' is empty.")
                return {}
            logger.info(f"API keys loaded successfully from '{keys_path}'.")
            return api_keys
    except FileNotFoundError:
        logger.warning(
            f"API keys file '{keys_path}' not found. Proceeding without API keys."
        )
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing API keys file '{keys_path}': {e}")
        return {}
    # print("API keys loading attempted. For yfinance, keys are not typically required for public data access.")
    # print(f"If '{keys_path}' was found and parsed, keys would be available; otherwise, an empty dict is returned.")


def fetch_ohlcv_data(
    tickers: List[str], start_date: str, end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetches OHLCV data for a list of tickers from Yahoo Finance.
    This function downloads historical market data for the given tickers between
    the specified start and end dates. It returns a dictionary where keys are
    ticker symbols and values are pandas DataFrames with the OHLCV data.

    Args:
        tickers (List[str]): A list of stock ticker symbols.
        start_date (str): The start date for data collection (YYYY-MM-DD).
        end_date (str): The end date for data collection (YYYY-MM-DD).

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping tickers to their OHLCV data.
                                 Returns an empty dictionary if fetching fails for all.
    """
    # This function downloads OHLCV data for the provided list of tickers.
    # It uses yfinance.download and handles potential errors during the download process.
    # The output is a dictionary mapping each ticker to its corresponding DataFrame.
    ohlcv_data: Dict[str, pd.DataFrame] = {}
    logger.info(f"Fetching OHLCV data for {len(tickers)} tickers from {start_date} to {end_date}.")
    try:
        # yfinance can download multiple tickers at once.
        # The result is a Panel for multiple tickers, or DataFrame for a single ticker.
        # If grouping by ticker, the columns will be MultiIndex.
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, actions=False)
        if data.empty:
            logger.warning(f"No data returned for tickers: {tickers} from {start_date} to {end_date}.")
            return {}

        if len(tickers) == 1:
            if not data.empty:
                ohlcv_data[tickers[0]] = data
        else:
            for ticker in tickers:
                # Check if ticker data is present (some might fail)
                if ticker in data.columns.levels[0]:
                    ticker_df = data[ticker].copy()
                    # Drop rows where all OHLCV values are NaN (e.g., non-trading days for a specific ticker)
                    ticker_df.dropna(how='all', subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
                    if not ticker_df.empty:
                        ohlcv_data[ticker] = ticker_df
                else:
                    logger.warning(f"No data found for ticker: {ticker} in the downloaded batch.")
        logger.info(f"Successfully fetched data for {len(ohlcv_data)} out of {len(tickers)} requested tickers.")
    except Exception as e:
        logger.error(f"Error downloading data for tickers {tickers}: {e}", exc_info=True)
        # In case of a general error, return what has been collected so far or an empty dict.
    # print(f"OHLCV data fetching completed for tickers: {', '.join(tickers)}.")
    # print(f"Returned a dictionary with {len(ohlcv_data)} tickers' dataframes if successful.")
    return ohlcv_data


def save_data_to_csv(
    data: Dict[str, pd.DataFrame], base_path: str, current_date_str: str
) -> None:
    """
    Saves the collected OHLCV data to CSV files in a structured directory.
    This function iterates through the dictionary of DataFrames. For each ticker,
    it saves its data to a CSV file within a subdirectory named after the current date.
    The directory structure is `base_path/YYYY-MM-DD/TICKER.csv`.

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of ticker symbols to DataFrames.
        base_path (str): The base directory for saving data (e.g., 'data/raw/ohlcv/us_equity').
        current_date_str (str): The current date as a string (YYYY-MM-DD) for the subdirectory.
    """
    # This function saves the provided ticker data into CSV files.
    # It creates a date-specific subdirectory under the base_path and stores each ticker's data there.
    # Error handling for file operations is included.
    output_dir = os.path.join(base_path, current_date_str)
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Error creating directory {output_dir}: {e}", exc_info=True)
        return  # Cannot proceed if directory creation fails

    for ticker, df in data.items():
        if df.empty:
            logger.info(f"Skipping save for {ticker} as its DataFrame is empty.")
            continue
        # Sanitize ticker name for filename (e.g., ^VIX -> VIX.csv)
        safe_ticker_filename = ticker.replace("^", "") + ".csv"
        file_path = os.path.join(output_dir, safe_ticker_filename)
        try:
            df.to_csv(file_path)
            logger.info(f"Successfully saved data for {ticker} to {file_path}")
        except IOError as e:
            logger.error(f"Error saving data for {ticker} to {file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving data for {ticker} to {file_path}: {e}", exc_info=True)
    # print(f"Data saving process completed. CSV files were written to '{output_dir}'.")
    # print(f"Each ticker's data is in its own CSV file within this directory.")


def main() -> None:
    """
    Main function to orchestrate the EOD OHLCV data ingestion process.
    This function sets up dates, loads tickers from config, processes them in batches,
    fetches data using yfinance, and saves it to the specified directory structure.
    """
    parser = argparse.ArgumentParser(description="Fetch EOD OHLCV data for US Equities and VIX.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the data sources YAML configuration file."
    )
    args = parser.parse_args()

    logger.info(f"Starting EOD US Equity OHLCV and VIX data ingestion process using config: {args.config_path}.")
    
    config = load_config_from_yaml(args.config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    strategy_1_config = config.get("strategy_1_data_config", {})
    us_equity_tickers = strategy_1_config.get("us_equity_tickers", DEFAULT_US_EQUITY_TICKERS)
    vix_ticker = strategy_1_config.get("vix_ticker", VIX_TICKER)
    
    # Get other params from config if available, else use constants
    batch_size = config.get("yahoo_finance", {}).get("batch_size", BATCH_SIZE)
    delay_between_batches = config.get("yahoo_finance", {}).get("delay_between_batches", DELAY_BETWEEN_BATCHES)
    data_output_base_dir = config.get("data_paths", {}).get("raw_ohlcv_us_equity_dir", DATA_OUTPUT_BASE_DIR)


    today = datetime.date.today()
    end_date_dt = today
    start_date_dt = end_date_dt - datetime.timedelta(days=4) # Fetch a bit more to ensure data availability
    
    current_date_str = end_date_dt.strftime(DATE_FORMAT)
    start_date_str = start_date_dt.strftime(DATE_FORMAT)
    end_date_str = end_date_dt.strftime(DATE_FORMAT)

    logger.info(f"Targeting EOD data for date: {current_date_str} (fetching window: {start_date_str} to {end_date_str})")

    all_tickers_to_fetch = us_equity_tickers + [vix_ticker]
    logger.info(f"Total tickers to process (including VIX): {len(all_tickers_to_fetch)}")
    logger.debug(f"Tickers: {all_tickers_to_fetch}")

    num_batches = (len(all_tickers_to_fetch) + batch_size - 1) // batch_size
    logger.info(f"Processing in {num_batches} batches of size {batch_size}.")

    all_fetched_data: Dict[str, pd.DataFrame] = {}

    for i in range(num_batches):
        batch_start_index = i * batch_size
        batch_end_index = batch_start_index + batch_size
        current_batch_tickers = all_tickers_to_fetch[batch_start_index:batch_end_index]

        logger.info(f"Processing Batch {i+1}/{num_batches}: {current_batch_tickers}")
        
        batch_data = fetch_ohlcv_data(
            tickers=current_batch_tickers,
            start_date=start_date_str,
            end_date=end_date_str,
        )
        
        processed_batch_data: Dict[str, pd.DataFrame] = {}
        for ticker, df in batch_data.items():
            if not df.empty:
                target_datetime = pd.to_datetime(current_date_str)
                if target_datetime in df.index:
                    single_day_df = df.loc[[target_datetime]].copy()
                    processed_batch_data[ticker] = single_day_df
                    logger.debug(f"Filtered data for {ticker} to date {current_date_str}.")
                elif not df.empty:
                    latest_available_df = df.iloc[[-1]].copy()
                    actual_date_str = latest_available_df.index[0].strftime(DATE_FORMAT)
                    logger.warning(f"Data for {current_date_str} not found for {ticker}. Using latest available: {actual_date_str}")
                    processed_batch_data[ticker] = latest_available_df
                else:
                    logger.info(f"Empty dataframe for {ticker} after attempting to filter by date.")
            else:
                logger.info(f"Empty dataframe for {ticker} before date filtering.")


        all_fetched_data.update(processed_batch_data)

        if i < num_batches - 1:
            logger.info(f"Waiting for {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)

    if not all_fetched_data:
        logger.warning("No data was fetched for any ticker after processing all batches. Exiting.")
        return

    save_data_to_csv(all_fetched_data, data_output_base_dir, current_date_str)

    logger.success("EOD US Equity OHLCV and VIX data ingestion process completed.")


if __name__ == "__main__":
    main()