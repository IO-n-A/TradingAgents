# core/combine_market_and_sentiment_data.py
# This script loads feature-engineered market data and news sentiment data.
# It then aggregates sentiment scores to a daily level, merges them with the market data,
# and saves the final combined dataset to a CSV file.

import logging
import pandas as pd
from datetime import datetime
import subprocess
import os
from typing import Tuple, Dict, Any

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Define file paths
MARKET_DATA_INPUT_PATH = "data/market_data_with_indicators.csv"
NEWS_SENTIMENT_INPUT_PATH = "data/market_news_with_sentiment.csv"
COMBINED_DATA_OUTPUT_PATH = "data/combined_market_data_and_sentiment.csv"
BACKLOG_FILE_PATH = "log/backlog.md"
TIME_ID_SCRIPT_PATH = "helpers/get_time_id.py"


# This function loads a CSV file into a pandas DataFrame.
# It takes a file path as input.
# It returns a pandas DataFrame containing the data from the CSV file.
def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For other pandas related errors.
    """
    logger.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"File {file_path} is empty.")
        print(f"Successfully loaded data from {file_path}. The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Error: No data in file {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

# This function aggregates daily sentiment scores from news data.
# It takes a DataFrame with news articles, their publication dates, and sentiment scores.
# It returns a DataFrame with daily aggregated sentiment scores (mean of positive, negative, neutral).
def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates sentiment scores to a daily level.

    The 'published_at' column is parsed to date-only.
    Mean 'positive', 'negative', and 'neutral' scores are calculated per day.

    Args:
        news_df (pd.DataFrame): DataFrame containing news articles with 'published_at',
                                'positive', 'negative', and 'neutral' columns.

    Returns:
        pd.DataFrame: DataFrame with 'date' and aggregated daily sentiment scores
                      ('daily_positive_mean', 'daily_negative_mean', 'daily_neutral_mean').
    """
    logger.info("Aggregating daily sentiment scores...")
    if not isinstance(news_df, pd.DataFrame):
        logger.error("Input news_df must be a pandas DataFrame.")
        raise TypeError("Input news_df must be a pandas DataFrame.")
    if 'published_at' not in news_df.columns:
        logger.error("'published_at' column not found in news_df.")
        raise ValueError("'published_at' column not found in news_df.")
    if not all(col in news_df.columns for col in ['positive', 'negative', 'neutral']):
        logger.error("Sentiment score columns ('positive', 'negative', 'neutral') not found.")
        raise ValueError("Sentiment score columns ('positive', 'negative', 'neutral') not found.")

    # Ensure 'published_at' is in datetime format and extract date
    try:
        news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date
    except Exception as e:
        logger.error(f"Error converting 'published_at' to datetime: {e}")
        raise

    # Aggregate sentiment scores
    daily_sentiment = news_df.groupby('date').agg(
        daily_positive_mean=('positive', 'mean'),
        daily_negative_mean=('negative', 'mean'),
        daily_neutral_mean=('neutral', 'mean')
    ).reset_index()

    # Convert 'date' column in daily_sentiment to datetime64[ns] for merging
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

    print(f"Successfully aggregated daily sentiment. The resulting DataFrame has {daily_sentiment.shape[0]} days of sentiment data.")
    return daily_sentiment

# This function merges market data with aggregated daily sentiment scores.
# It takes two DataFrames: market data (ticker-specific) and daily sentiment data (market-wide).
# It returns a merged DataFrame, joining on the 'date' column.
def merge_market_and_sentiment_data(
    market_data_df: pd.DataFrame,
    daily_sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges market data with daily aggregated sentiment scores.

    The merge is performed on the 'date' column. Market data 'date' column
    is converted to datetime if not already.

    Args:
        market_data_df (pd.DataFrame): DataFrame with ticker-specific market data,
                                       including a 'date' column.
        daily_sentiment_df (pd.DataFrame): DataFrame with market-wide daily
                                           aggregated sentiment scores, including a 'date' column.

    Returns:
        pd.DataFrame: The merged DataFrame containing market data and sentiment scores.
    """
    logger.info("Merging market data with daily sentiment scores...")
    if 'date' not in market_data_df.columns:
        logger.error("'date' column not found in market_data_df.")
        raise ValueError("'date' column not found in market_data_df.")
    if 'date' not in daily_sentiment_df.columns:
        logger.error("'date' column not found in daily_sentiment_df.")
        raise ValueError("'date' column not found in daily_sentiment_df.")

    # Ensure 'date' columns are in datetime format for proper merging
    try:
        market_data_df['date'] = pd.to_datetime(market_data_df['date'])
        # daily_sentiment_df['date'] is already converted in aggregate_daily_sentiment
    except Exception as e:
        logger.error(f"Error converting 'date' column to datetime: {e}")
        raise

    # Perform a left merge to keep all market data and add sentiment where dates match
    combined_df = pd.merge(market_data_df, daily_sentiment_df, on='date', how='left')

    print(f"Successfully merged market and sentiment data. The combined DataFrame has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")
    return combined_df

# This function saves a pandas DataFrame to a CSV file.
# It takes a DataFrame and a file path as input.
# It does not return anything but writes the DataFrame to the specified file.
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the CSV file will be saved.
    """
    logger.info(f"Saving data to {file_path}...")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Successfully saved data to {file_path}. The file contains {df.shape[0]} rows.")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

# This function retrieves a timestamp and ID by executing an external Python script.
# It takes the path to the script as input.
# It returns a tuple containing the timestamp string and the ID string.
def get_timestamp_id(script_path: str) -> Tuple[str, str]:
    """Executes a script to get a timestamp and ID.

    Args:
        script_path (str): Path to the Python script that outputs timestamp and ID.

    Returns:
        Tuple[str, str]: A tuple containing (timestamp_str, id_str).
                         Returns ("N/A", "N/A") if the script fails.
    """
    logger.info(f"Executing {script_path} to get timestamp and ID...")
    try:
        result = subprocess.run(
            ["python", script_path], capture_output=True, text=True, check=True
        )
        parts = result.stdout.strip().split("_")
        if len(parts) == 2:
            timestamp_str, id_str = parts[0], parts[1]
            print(f"Successfully retrieved timestamp '{timestamp_str}' and ID '{id_str}'.")
            return timestamp_str, id_str
        else:
            logger.warning(f"Unexpected output format from {script_path}: {result.stdout.strip()}")
            print(f"Could not parse timestamp and ID from script output: {result.stdout.strip()}. Using default values.")
            return "N/A", "N/A"
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_path}: {e}. Output: {e.stderr}")
        print(f"Failed to execute {script_path}. Error: {e}. Using default values for timestamp and ID.")
        return "N/A", "N/A"
    except FileNotFoundError:
        logger.error(f"Script {script_path} not found.")
        print(f"Script {script_path} not found. Using default values for timestamp and ID.")
        return "N/A", "N/A"

# This function logs a summary of the completed action to the backlog file.
# It takes a timestamp, an ID, and a summary message.
# It prepends the formatted entry to the specified backlog file.
def log_to_backlog(timestamp_str: str, id_str: str, summary: str, backlog_path: str) -> None:
    """Prepends a summary of the completed action to the backlog file.

    Args:
        timestamp_str (str): The timestamp string.
        id_str (str): The ID string.
        summary (str): A 3-sentence summary of the completed action.
        backlog_path (str): Path to the backlog.md file.
    """
    logger.info(f"Logging completion to {backlog_path}...")
    entry = f"## {timestamp_str} - ID: {id_str}\n\n{summary}\n\n---\n\n"
    try:
        # Ensure the directory for backlog.md exists
        os.makedirs(os.path.dirname(backlog_path), exist_ok=True)
        
        if os.path.exists(backlog_path):
            with open(backlog_path, "r+", encoding="utf-8") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(entry + content)
        else:
            with open(backlog_path, "w", encoding="utf-8") as f:
                f.write(entry)
        print(f"Successfully logged completion to {backlog_path}. The entry has been prepended.")
    except Exception as e:
        logger.error(f"Error writing to backlog file {backlog_path}: {e}")
        print(f"Failed to write to backlog file {backlog_path}. Error: {e}.")

# This is the main function that orchestrates the data combination process.
# It loads market and news data, aggregates sentiment, merges them, and saves the result.
# Finally, it logs the completion of this task to the backlog.
def main() -> None:
    """Main function to run the data combination and sentiment aggregation process."""
    script_name = os.path.basename(__file__)
    logger.info(f"Starting {script_name}: Combine Market Data and Aggregated Sentiment.")
    
    summary_for_backlog = ""
    operation_successful = False

    try:
        # 1. Load input data
        logger.info(f"Attempting to load market data from {MARKET_DATA_INPUT_PATH}")
        market_data_df = load_data(MARKET_DATA_INPUT_PATH)
        logger.info(f"Market data loaded successfully. Shape: {market_data_df.shape}")

        logger.info(f"Attempting to load news sentiment data from {NEWS_SENTIMENT_INPUT_PATH}")
        news_sentiment_df = load_data(NEWS_SENTIMENT_INPUT_PATH) # load_data logs a general warning if df.empty
        logger.info(f"News sentiment data loaded. Shape: {news_sentiment_df.shape}. Is empty: {news_sentiment_df.empty}")

        # Check if news_sentiment_df is empty (no rows, only headers as per problem description)
        if news_sentiment_df.empty:
            logger.warning(
                f"Input news sentiment data from {NEWS_SENTIMENT_INPUT_PATH} is empty (may contain headers only). "
                "Sentiment features will not be merged."
            )
            # Save original market data directly
            save_data(market_data_df, COMBINED_DATA_OUTPUT_PATH)
            logger.info(
                f"Saved original market data (from {MARKET_DATA_INPUT_PATH}) directly to {COMBINED_DATA_OUTPUT_PATH} "
                "as news sentiment data was empty."
            )
            summary_for_backlog = (
                f"Script {script_name}: Loaded market data from {MARKET_DATA_INPUT_PATH}. "
                f"News sentiment data from {NEWS_SENTIMENT_INPUT_PATH} was found to be empty. "
                f"No sentiment aggregation or merge was performed. "
                f"The original market data was saved directly to {COMBINED_DATA_OUTPUT_PATH}."
            )
            operation_successful = True
        else:
            # 2. Aggregate sentiment (only if news_sentiment_df is not empty)
            logger.info("News sentiment data is not empty. Proceeding with sentiment aggregation.")
            daily_sentiment_df = aggregate_daily_sentiment(news_sentiment_df)
            logger.info(f"Daily sentiment aggregated. Shape: {daily_sentiment_df.shape}")

            # 3. Merge data
            logger.info("Proceeding with merging market data and aggregated sentiment data.")
            combined_df = merge_market_and_sentiment_data(market_data_df, daily_sentiment_df)
            logger.info(f"Market and sentiment data merged. Shape: {combined_df.shape}")

            # 4. Save output
            save_data(combined_df, COMBINED_DATA_OUTPUT_PATH)
            logger.info(f"Successfully created and saved combined dataset with sentiment to {COMBINED_DATA_OUTPUT_PATH}")
            summary_for_backlog = (
                f"Script {script_name}: Loaded market data from {MARKET_DATA_INPUT_PATH} and news sentiment data from {NEWS_SENTIMENT_INPUT_PATH}. "
                "Aggregated news sentiment to daily scores. "
                f"Merged daily sentiment with market data and saved the combined dataset to {COMBINED_DATA_OUTPUT_PATH}."
            )
            operation_successful = True

        # 6. Backlog Logging (common for both successful cases)
        if operation_successful and summary_for_backlog:
            timestamp, task_id = get_timestamp_id(TIME_ID_SCRIPT_PATH)
            log_to_backlog(timestamp, task_id, summary_for_backlog, BACKLOG_FILE_PATH)
            logger.info("Backlog updated successfully with operation summary.")

    except FileNotFoundError as fnf_error:
        logger.critical(f"Critical Error in {script_name}: Input file not found: {fnf_error}. Aborting.", exc_info=True)
        summary_for_backlog = (
            f"Script {script_name} failed: Critical error - Input file not found: {fnf_error}. "
            "Processing halted before data combination could occur. "
            "No output file was generated due to this error."
        )
        timestamp, task_id = get_timestamp_id(TIME_ID_SCRIPT_PATH)
        log_to_backlog(timestamp, task_id, summary_for_backlog, BACKLOG_FILE_PATH)
        logger.info("Backlog updated with file not found failure information.")
        print(f"{script_name} failed. Critical Error: Input file not found: {fnf_error}. Check logs. Failure logged to backlog.")
        return
    except pd.errors.EmptyDataError as ede_error:
        logger.critical(f"Critical Error in {script_name}: Input file is completely empty (no data, no headers): {ede_error}. Aborting.", exc_info=True)
        summary_for_backlog = (
            f"Script {script_name} failed: Critical error - Input file is completely empty: {ede_error}. "
            "Processing halted. No output file was generated."
        )
        timestamp, task_id = get_timestamp_id(TIME_ID_SCRIPT_PATH)
        log_to_backlog(timestamp, task_id, summary_for_backlog, BACKLOG_FILE_PATH)
        logger.info("Backlog updated with empty data error failure information.")
        print(f"{script_name} failed. Critical Error: Input file completely empty: {ede_error}. Check logs. Failure logged to backlog.")
        return
    except Exception as e:
        logger.critical(f"An unexpected error occurred in {script_name} during the data combination process: {e}", exc_info=True)
        summary_for_backlog = (
            f"Script {script_name} encountered an unexpected error: {type(e).__name__} - {e}. "
            f"Processing was halted. The state of the output file {COMBINED_DATA_OUTPUT_PATH} is uncertain. "
            f"Please check the logs for detailed error information."
        )
        timestamp, task_id = get_timestamp_id(TIME_ID_SCRIPT_PATH)
        log_to_backlog(timestamp, task_id, summary_for_backlog, BACKLOG_FILE_PATH)
        logger.info("Backlog updated with unexpected error failure information.")
        print(f"{script_name} failed. Unexpected Error: {e}. Check logs for details. Failure logged to backlog.")
        return

    if operation_successful:
        print(f"{script_name} executed successfully. Output saved to {COMBINED_DATA_OUTPUT_PATH} and backlog updated.")
    else:
        # This path implies an issue not caught by an exception or an earlier return,
        # or if operation_successful was not set true due to an unhandled logic path.
        print(f"{script_name} did not complete successfully. Please check logs for details. Backlog may contain error info if an error was logged.")
    return


if __name__ == "__main__":
    main()