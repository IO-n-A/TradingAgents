# core/price_feature_engineering.py
# This script loads market data, adds technical indicators, saves the augmented data,
# and logs its completion to the backlog.

import logging
import pandas as pd
import subprocess
import datetime
from typing import Union # Added for type hinting compatibility
from finrl.meta.data_processor import DataProcessor # Using DataProcessor as it's more general

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Define constants
INPUT_DATA_PATH = "data/market_data_ohlcv_vix_turbulence.csv"
OUTPUT_DATA_PATH = "data/market_data_with_indicators.csv"
BACKLOG_FILE_PATH = "log/backlog.md"
TIME_ID_SCRIPT_PATH = "helpers/get_time_id.py"

# Technical indicators to add
# Based on the task and common usage with stockstats (used by DataProcessor)
# SMA, EMA, RSI, MACD
# stockstats typically uses 'macd', 'rsi_PERIOD', 'close_PERIOD_sma', 'close_PERIOD_ema'
# We will use common periods.
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "rsi_14",  # Relative Strength Index 14 days
    "close_20_sma",  # Simple Moving Average 20 days
    "close_50_sma",  # Simple Moving Average 50 days (added for variety)
    "ema_20",  # Exponential Moving Average 20 days (stockstats might create 'close_20_ema')
]


def load_market_data(file_path: str) -> Union[pd.DataFrame, None]:
    """
    Loads market data from a CSV file into a pandas DataFrame.
    It expects the CSV to contain OHLCV, VIX, and turbulence data.
    The function logs the loading process and returns the DataFrame or None on error.
    """
    # This function loads market data from the specified CSV file.
    # It uses pandas to read the CSV and logs success or any errors encountered.
    # The function returns a pandas DataFrame containing the market data, or None if loading fails.
    try:
        logger.info(f"Loading market data from {file_path}...")
        df = pd.read_csv(file_path)
        # Ensure 'date' column is in datetime format if not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}: {e}")
        return None
    # print(f"Market data loaded from {file_path}. DataFrame shape: {df.shape if df is not None else 'Error'}.")
    # print(f"This data includes OHLCV, VIX, and turbulence information for analysis.")


def add_technical_indicators_to_data(
    data_df: pd.DataFrame, indicators: list[str]
) -> Union[pd.DataFrame, None]:
    """
    Adds specified technical indicators to the DataFrame using FinRL's DataProcessor.
    The input DataFrame should have 'tic', 'date', 'open', 'high', 'low', 'close', 'volume'.
    It logs the process and returns the DataFrame with added indicators or None on error.
    """
    # This function takes a DataFrame and a list of indicator names.
    # It uses FinRL's DataProcessor to calculate and add these technical indicators.
    # The function returns the augmented DataFrame, or None if an error occurs during processing.
    if data_df is None or data_df.empty:
        logger.error("Input DataFrame is empty or None. Cannot add indicators.")
        return None
    
    required_cols = ['tic', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        logger.error(f"DataFrame is missing required columns for DataProcessor: {missing_cols}")
        # Attempt to rename common alternatives if they exist
        rename_map = {
            'symbol': 'tic', 'time': 'date', 'adj close': 'adj_close'
            # Add other common alternative names if necessary
        }
        # Check if 'adj_close' exists and 'close' does not, then use 'adj_close' as 'close'
        if 'adj_close' in data_df.columns and 'close' not in data_df.columns:
            data_df = data_df.rename(columns={'adj_close': 'close'})
            logger.info("Renamed 'adj_close' to 'close'.")
            # Re-check missing columns
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if not missing_cols:
                 logger.info("All required columns now present after renaming 'adj_close'.")
            else:
                logger.error(f"Still missing columns after attempting rename: {missing_cols}")
                return None
        elif missing_cols: # if still missing after adj_close check
             logger.error(f"DataFrame is missing required columns for DataProcessor: {missing_cols}")
             return None


    try:
        logger.info(f"Adding technical indicators: {', '.join(indicators)}")
        processor = DataProcessor(data_source="yahoofinance", start_date=data_df['date'].min(), end_date=data_df['date'].max()) # Changed "custom" to "yahoofinance"
        
        # DataProcessor.add_technical_indicator expects a list of DataFrames, one per tic
        # However, the method itself processes a single df if passed correctly.
        # The internal call to stockstats works on a grouped by 'tic' df.
        # Let's ensure the data is sorted by tic and date for stockstats.
        data_df = data_df.sort_values(by=['tic', 'date'])
        
        # The add_technical_indicator method in FinRL's DataProcessor
        # typically takes the dataframe and the list of tech_indicator_list.
        # It processes the dataframe in place or returns a new one.
        # Let's assume it returns a new DataFrame or modifies it.
        # The FinRL DataProcessor's add_technical_indicator method processes data grouped by 'tic'.
        # It expects the main dataframe to be passed.
        
        temp_df = data_df.copy()
        if 'date' in temp_df.columns and 'timestamp' not in temp_df.columns:
            logger.info("Renaming 'date' to 'timestamp' for DataProcessor.add_technical_indicator.")
            temp_df = temp_df.rename(columns={'date': 'timestamp'})
            # Ensure 'timestamp' is datetime, as 'date' was converted in load_market_data
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        elif 'date' not in temp_df.columns and 'timestamp' not in temp_df.columns:
            logger.error("Neither 'date' nor 'timestamp' column found before calling add_technical_indicator.")
            return None

        augmented_df = processor.add_technical_indicator(temp_df, indicators)
        
        # Rename 'timestamp' back to 'date' if it was changed and exists in the result
        if 'timestamp' in augmented_df.columns and 'date' not in augmented_df.columns:
            logger.info("Renaming 'timestamp' back to 'date' after adding indicators.")
            augmented_df = augmented_df.rename(columns={'timestamp': 'date'})
            
        # Clean up NaN values introduced by indicators (common for initial periods)
        # augmented_df = processor.clean_data(augmented_df) # clean_data might drop too much if not careful
        # A more gentle approach for NaNs from indicators:
        # For indicators, NaNs at the beginning are expected. We can fill them or leave them.
        # Let's fill with 0 or forward fill after an initial period.
        # For simplicity, we'll rely on how DataProcessor handles it, often ffill/bfill.
        # If clean_data is too aggressive, we might need a custom NaN handling.
        # The prompt mentions DataProcessor usually handles missing data.
        
        logger.info(f"Successfully added technical indicators. New shape: {augmented_df.shape}")
        return augmented_df
    except Exception as e:
        logger.error(f"An error occurred while adding technical indicators: {e}")
        return None
    # print(f"Technical indicators ({', '.join(indicators)}) added. DataFrame shape: {augmented_df.shape if augmented_df is not None else 'Error'}.")
    # print(f"The DataFrame now includes calculated values for SMA, EMA, RSI, and MACD.")


def save_augmented_data(data_df: pd.DataFrame, file_path: str) -> bool:
    """
    Saves the DataFrame with augmented data to a new CSV file.
    It logs the saving process and returns True on success, False on error.
    """
    # This function saves the provided DataFrame to a CSV file at the specified path.
    # It logs the outcome of the save operation.
    # The function returns True if the data is saved successfully, and False otherwise.
    if data_df is None or data_df.empty:
        logger.error("DataFrame is empty or None. Nothing to save.")
        return False
    try:
        logger.info(f"Saving augmented data to {file_path}...")
        data_df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved augmented data to {file_path}.")
        return True
    except Exception as e:
        logger.error(f"An error occurred while saving data to {file_path}: {e}")
        return False
    # print(f"Augmented data saved to {file_path}. Success: {True if 'try' else False}.") # This logic is flawed for print
    # print(f"The file contains the original market data plus the newly added technical indicators.")


def log_completion_to_backlog(summary_message: str, script_path: str, backlog_path: str) -> None:
    """
    Logs the script's completion to the backlog.md file.
    It retrieves a timestamp and unique ID using a helper script,
    then prepends this information along with the summary to the backlog file.
    """
    # This function logs a summary of the script's actions to a backlog file.
    # It first calls a helper script to get a unique ID and timestamp.
    # Then, it prepends this information and the summary message to the specified backlog file.
    try:
        logger.info("Executing get_time_id.py to fetch timestamp and ID...")
        process = subprocess.run(
            ["python", script_path], capture_output=True, text=True, check=True
        )
        time_id_output = process.stdout.strip()
        # Expected output format from helpers/get_time_id.py: "[YYYY-MM-DD HH:MM:SS.ffffff] - [Handle: IO-n_A XY-Z]"
        # We need to parse this to get a "YYYY-MM-DD_HHMMSS" style timestamp and the handle as ID.
        try:
            # Extract timestamp string (e.g., "YYYY-MM-DD HH:MM:SS.ffffff")
            ts_str_with_brackets = time_id_output.split("] - [Handle:")[0].strip("[")
            # Convert to datetime object
            dt_obj = datetime.datetime.strptime(ts_str_with_brackets, '%Y-%m-%d %H:%M:%S.%f')
            # Format to desired string
            timestamp = dt_obj.strftime('%Y-%m-%d_%H%M%S')
            
            # Extract handle string (e.g., "IO-n_A XY-Z")
            unique_id = time_id_output.split("[Handle: ")[1].strip("]")
        except Exception as e:
            logger.warning(f"Could not parse timestamp and ID from '{time_id_output}' due to: {e}. Using current time and full output as ID.")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            unique_id = time_id_output # Fallback

        log_entry = f"## {timestamp} - ID: {unique_id}\n\n"
        log_entry += f"**Script:** `core/price_feature_engineering.py`\n\n"
        log_entry += f"{summary_message}\n\n---\n\n"

        logger.info(f"Prepending completion log to {backlog_path}...")
        with open(backlog_path, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(log_entry + content)
        logger.info("Successfully logged completion to backlog.")
    except FileNotFoundError:
        logger.error(f"Error: The helper script {script_path} was not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_path}: {e}. Output: {e.stderr}")
    except Exception as e:
        logger.error(f"An error occurred while logging to backlog: {e}")
    # print(f"Completion logged to {backlog_path} with timestamp and ID.")
    # print(f"The log includes a summary of the data processing and saving operations.")


def main():
    """
    Main function to orchestrate the feature engineering process.
    It loads data, adds indicators, saves the result, and logs completion.
    This function serves as the entry point and controls the overall workflow.
    """
    # This is the main execution block for the script.
    # It coordinates loading data, adding indicators, saving, and logging.
    # It ensures the entire process runs smoothly from start to finish.
    logger.info("Starting price feature engineering process...")

    market_data_df = load_market_data(INPUT_DATA_PATH)

    if market_data_df is not None:
        print(f"Market data loaded from {INPUT_DATA_PATH}. DataFrame shape: {market_data_df.shape if market_data_df is not None else 'Error'}.")
        print(f"This data includes OHLCV, VIX, and turbulence information for analysis.")
        
        augmented_data_df = add_technical_indicators_to_data(
            market_data_df, TECHNICAL_INDICATORS_LIST
        )

        if augmented_data_df is not None:
            print(f"Technical indicators ({', '.join(TECHNICAL_INDICATORS_LIST)}) added. DataFrame shape: {augmented_data_df.shape if augmented_data_df is not None else 'Error'}.")
            print(f"The DataFrame now includes calculated values for the specified indicators.")

            save_success = save_augmented_data(augmented_data_df, OUTPUT_DATA_PATH)

            if save_success:
                print(f"Augmented data saved to {OUTPUT_DATA_PATH}. Success: {save_success}.")
                print(f"The file contains the original market data plus the newly added technical indicators.")

                summary = (
                    "Loaded market data with OHLCV, VIX, and turbulence. "
                    f"Calculated and added {', '.join(TECHNICAL_INDICATORS_LIST)} technical indicators. "
                    f"Saved the augmented dataset to {OUTPUT_DATA_PATH}."
                )
                log_completion_to_backlog(summary, TIME_ID_SCRIPT_PATH, BACKLOG_FILE_PATH)
                print(f"Completion logged to {BACKLOG_FILE_PATH} with timestamp and ID.")
                print(f"The log includes a summary of the data processing and saving operations.")
            else:
                logger.error("Failed to save augmented data. Backlog will not be updated for this run.")
                print(f"Failed to save augmented data to {OUTPUT_DATA_PATH}. Check logs for errors.")
                print(f"Backlog update skipped due to save failure.")
        else:
            logger.error("Failed to add technical indicators. Process halted.")
            print(f"Failed to add technical indicators. Check logs for errors.")
            print(f"Process halted before saving and backlog update.")
    else:
        logger.error("Failed to load market data. Process halted.")
        print(f"Failed to load market data from {INPUT_DATA_PATH}. Check logs for errors.")
        print(f"Process halted before indicator addition, saving, and backlog update.")

    logger.info("Price feature engineering process finished.")
    # print("Price feature engineering process has concluded.")
    # print("Check logs for details of the execution and any potential issues.")


if __name__ == "__main__":
    main()