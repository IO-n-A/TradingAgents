# MLOps/orchestration/daily_data_collection_orchestrator.py
# This script orchestrates the daily data collection process by scheduling and
# executing the `us_equity_ohlcv_scheduled_ingest.py` and
# `market_news_scheduled_ingest.py` scripts. It ensures adherence to API
# constraints and logs the overall process. 

import argparse
import datetime
import logging
import subprocess
import sys
import time
import os
from typing import Dict, Any, Optional

# Determine project root and add to sys.path
# The script is in MLOps/orchestration/, so project_root is two levels up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to ensure it's checked first

# Attempt to import custom logging setup
try:
    from config.logging_config import setup_logging, SUCCESS_LEVEL_NUM
except ImportError:
    # Fallback basic logging if custom config is not found
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    SUCCESS_LEVEL_NUM = 25 # Define it for consistency if import fails
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
    logging.Logger.success = success
    logging.getLogger(__name__).warning(
        "Could not import custom logging_config. Using basic logging."
    )

logger = logging.getLogger(__name__)

# --- Configuration ---
# Paths to the ingestion scripts
OHLCV_INGEST_SCRIPT_PATH: str = os.path.join(project_root, "MLOps/pipelines/ingestion/us_equity_ohlcv_scheduled_ingest.py")
NEWS_INGEST_SCRIPT_PATH: str = os.path.join(project_root, "MLOps/pipelines/ingestion/market_news_scheduled_ingest.py")

# API constraint related (though actual enforcement is within individual scripts)
# These are more for awareness at the orchestrator level if needed for complex scheduling.
# For now, we assume individual scripts manage their specific limits.
# NEWSAPI_MAX_REQUESTS_PER_DAY: int = 100 (Defined in market_news_scheduled_ingest.py)
# FINNHUB_MAX_REQUESTS_PER_MINUTE: int = 60 (Defined in market_news_scheduled_ingest.py)
# YFINANCE_BATCH_DELAY: int = 60 (Defined in us_equity_ohlcv_scheduled_ingest.py)

# Orchestrator specific delays
DELAY_BETWEEN_MAIN_TASKS: int = 120  # 2 minutes delay between running OHLCV and News ingestion

# --- Helper Functions ---

def initialize_script_logging():
    """Initializes logging for the orchestrator script using the custom setup."""
    # This function sets up the logging configuration for the orchestrator.
    # It uses the centralized setup_logging from config.logging_config for consistency.
    # This ensures all logs from the orchestrator follow the standard format.
    try:
        from config.logging_config import setup_logging
        setup_logging() # Initialize with default level (INFO)
        logger.info(
            "Custom logging initialized successfully for daily_data_collection_orchestrator.py.",
             extra={'filename_summary': __name__}
        )
    except ImportError:
        logger.warning(
            "Failed to initialize custom orchestrator logging. Basic logging remains active.",
            extra={'filename_summary': __name__}
        )
    # print("Orchestrator logging initialization attempted.")
    # print("If config.logging_config.setup_logging was found, custom logging is active.")


def run_ingestion_script(
    script_path: str, script_args: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Executes a given data ingestion script as a subprocess.
    This function constructs the command to run the Python script, including any
    necessary arguments. It captures and logs the output (stdout and stderr)
    of the script and returns True if the script ran successfully (exit code 0).

    Args:
        script_path (str): The file path to the Python script to execute.
        script_args (Optional[Dict[str, Any]]): A dictionary of command-line arguments
                                                 for the script (e.g., {"--source": "newsapi"}).

    Returns:
        bool: True if the script executed successfully (exit code 0), False otherwise.
    """
    # This function executes a specified ingestion script using subprocess.
    # It builds the command, runs the script, logs its output, and checks the return code.
    # This allows the orchestrator to manage and monitor individual ingestion tasks.
    command = [sys.executable, script_path] # sys.executable ensures using the same python interpreter
    if script_args:
        for arg, value in script_args.items():
            command.append(arg)
            if isinstance(value, list): # For nargs='+'
                command.extend(value)
            elif value is not None : # Handle cases where value might be boolean flags or just values
                command.append(str(value))


    logger.info(f"Executing script: {' '.join(command)}", extra={'filename_summary': __name__})
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        if stdout:
            logger.info(f"Output from {script_path}:\n{stdout.strip()}", extra={'filename_summary': script_path})
        if stderr:
            # Log stderr as warning or error based on return code
            if process.returncode != 0:
                logger.error(f"Errors from {script_path}:\n{stderr.strip()}", extra={'filename_summary': script_path})
            else:
                logger.warning(f"Stderr (possibly warnings) from {script_path}:\n{stderr.strip()}", extra={'filename_summary': script_path})
        
        if process.returncode == 0:
            logger.success(f"Script {script_path} executed successfully.", extra={'filename_summary': __name__})
            # print(f"Script {script_path} completed with success.")
            # print("Its logs should provide details of the data fetched and saved.")
            return True
        else:
            logger.error(
                f"Script {script_path} failed with exit code {process.returncode}.",
                extra={'filename_summary': __name__, 'data_payload': {'exit_code': process.returncode}}
            )
            # print(f"Script {script_path} failed. Check the logs for detailed error messages from the script.")
            return False
    except FileNotFoundError:
        logger.error(f"Script not found: {script_path}. Cannot execute.", exc_info=True, extra={'filename_summary': __name__})
        # print(f"Error: The script at {script_path} was not found. Please check the path.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running {script_path}: {e}", exc_info=True, extra={'filename_summary': __name__})
        # print(f"An unexpected error occurred while trying to run {script_path}. See logs for details.")
        return False


# --- Main Orchestration Logic ---
def main():
    """
    Main function to orchestrate the daily data collection tasks.
    It defines the sequence of operations: first fetching OHLCV data,
    then fetching news data. It includes delays and basic status tracking.
    Command-line arguments can specify which news source to use for the news ingestion part.
    """
    # This is the main execution block for the orchestrator.
    # It initializes logging, parses arguments, and then sequentially runs the ingestion scripts.
    # It includes logic for choosing the news source and managing delays.

    initialize_script_logging() # Setup custom logging for the orchestrator

    parser = argparse.ArgumentParser(
        description="Orchestrates daily data collection for US equities OHLCV and market news."
    )
    parser.add_argument(
        "--news_source",
        type=str,
        choices=["newsapi", "finnhub"],
        default="newsapi", # Default to NewsAPI as primary
        help="The news API source to use for market_news_scheduled_ingest.py ('newsapi' or 'finnhub').",
    )
    # Add other orchestrator-level arguments if needed in the future, e.g., specific dates.
    args = parser.parse_args()

    logger.info("Starting daily data collection orchestration.", extra={'filename_summary': __name__})
    start_time = time.time()

    # --- Task 1: Ingest US Equity OHLCV Data ---
    logger.info("Initiating Task 1: US Equity OHLCV Data Ingestion.", extra={'filename_summary': __name__})
    ohlcv_success = run_ingestion_script(OHLCV_INGEST_SCRIPT_PATH)
    if ohlcv_success:
        logger.success("Task 1 (OHLCV Ingestion) completed successfully.", extra={'filename_summary': __name__})
    else:
        logger.error("Task 1 (OHLCV Ingestion) failed. Check logs for details.", extra={'filename_summary': __name__})
        # Decide if failure is critical. For now, we'll proceed to news.
        # In a more robust system, this might trigger alerts or stop the pipeline.

    logger.info(f"Waiting for {DELAY_BETWEEN_MAIN_TASKS} seconds before starting news ingestion...", extra={'filename_summary': __name__})
    time.sleep(DELAY_BETWEEN_MAIN_TASKS)

    # --- Task 2: Ingest Market News Data ---
    logger.info(f"Initiating Task 2: Market News Data Ingestion using source: {args.news_source}.", extra={'filename_summary': __name__})
    news_script_args = {"--source": args.news_source}
    # If news_source is 'newsapi', default queries from market_news_scheduled_ingest.py will be used
    # unless overridden here or by its own internal logic.
    # If news_source is 'finnhub', default category will be used.
    
    news_success = run_ingestion_script(NEWS_INGEST_SCRIPT_PATH, news_script_args)
    if news_success:
        logger.success(f"Task 2 (News Ingestion from {args.news_source}) completed successfully.", extra={'filename_summary': __name__})
    else:
        logger.error(f"Task 2 (News Ingestion from {args.news_source}) failed. Check logs.", extra={'filename_summary': __name__})

    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(
        f"Daily data collection orchestration finished in {total_duration:.2f} seconds.",
        extra={'filename_summary': __name__, 'data_payload': {'duration_seconds': total_duration}}
    )
    
    # Final summary print
    # print("\n--- Orchestration Summary ---")
    # ohlcv_status = "SUCCESS" if ohlcv_success else "FAILED"
    # news_status = "SUCCESS" if news_success else "FAILED"
    # print(f"1. OHLCV Data Ingestion: {ohlcv_status}")
    # print(f"2. News Data Ingestion (Source: {args.news_source}): {news_status}")
    # print(f"Total execution time: {total_duration:.2f} seconds.")
    # print("Refer to the detailed logs for more information on each step.")


if __name__ == "__main__":
    main()