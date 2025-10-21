# data_ingestion_script.py
# This script orchestrates the download, processing, and saving of financial market data.
# It fetches OHLCV data, VIX, and turbulence index for a combined list of NASDAQ 100 and S&P 500 tickers.
# The output is a CSV file containing the processed data.

import datetime
import logging
import os
import pandas as pd
import subprocess # Added for executing helper script
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.config_tickers import NAS_100_TICKER, SP_500_TICKER

# --- New Markdown Logging Setup ---
SCRIPT_RUN_TIMESTAMP = None
SCRIPT_RUN_HANDLE = None
MARKDOWN_LOG_FILE_PATH = os.path.join("FinAI_algo", "log", "data-log.md")
SCRIPT_MODULE_NAME = "FinAI_algo/core/data_ingestion_script.py"

def _ensure_log_directory_exists():
    """Ensures the directory for the markdown log file exists."""
    log_dir = os.path.dirname(MARKDOWN_LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            # Fallback to console if directory creation fails
            print(f"CRITICAL - Error creating log directory {log_dir}: {e}")
            # Potentially raise or exit if logging is critical
            return False
    return True

def log_to_markdown(level: str, message: str, function_name: str = None):
    """Writes a formatted log entry to the markdown log file."""
    global SCRIPT_RUN_TIMESTAMP, SCRIPT_RUN_HANDLE
    
    if not _ensure_log_directory_exists():
        # If directory creation failed, don't attempt to write to file
        # A message would have already been printed to console by _ensure_log_directory_exists
        return

    timestamp_to_log = SCRIPT_RUN_TIMESTAMP if SCRIPT_RUN_TIMESTAMP else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    handle_to_log = SCRIPT_RUN_HANDLE if SCRIPT_RUN_HANDLE else "UNKNOWN_HANDLE"
    
    module_func_part = f"Module: [{SCRIPT_MODULE_NAME}]"
    if function_name:
        module_func_part += f"::{function_name}"
    
    log_entry = f"[{timestamp_to_log}] - [Handle: {handle_to_log}] - {level.upper()} - {module_func_part} - Message: {message}\n"
    
    try:
        with open(MARKDOWN_LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except IOError as e:
        # Fallback to console if file writing fails
        print(f"ERROR - Failed to write to markdown log file {MARKDOWN_LOG_FILE_PATH}: {e}")
        print(f"Fallback log: {log_entry.strip()}")


def initialize_script_run_id():
    """Executes get_time_id.py to fetch and set the script run timestamp and handle."""
    global SCRIPT_RUN_TIMESTAMP, SCRIPT_RUN_HANDLE
    helper_script_path = os.path.join("FinAI_algo", "helpers", "get_time_id.py")
    
    try:
        # Ensure the helper script exists and is executable if needed (though python scripts usually don't need +x on Windows)
        if not os.path.exists(helper_script_path):
            error_msg = f"Helper script {helper_script_path} not found."
            log_to_markdown("ERROR", error_msg, "initialize_script_run_id")
            print(f"ERROR - {error_msg}") # Also print to console
            SCRIPT_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            SCRIPT_RUN_HANDLE = "FALLBACK_HANDLE_SCRIPT_NOT_FOUND"
            return

        process = subprocess.run(
            ["python", helper_script_path],
            capture_output=True,
            text=True,
            check=False, # Don't raise exception for non-zero exit, handle manually
            encoding='utf-8' # Specify encoding
        )
        
        if process.returncode == 0:
            output = process.stdout.strip()
            # Expected format: "YYYY-MM-DD HH:MM:SS.ffffff GeneratedHandle"
            parts = output.split(" ", 2) # Split at most twice
            if len(parts) == 3: # Check if we have three parts: date, time, handle
                timestamp_str_candidate = f"{parts[0]} {parts[1]}"
                # Validate timestamp format (basic check)
                try:
                    datetime.datetime.strptime(timestamp_str_candidate, "%Y-%m-%d %H:%M:%S.%f")
                    SCRIPT_RUN_TIMESTAMP = timestamp_str_candidate
                    SCRIPT_RUN_HANDLE = parts[2]
                    # Initial log to markdown confirming successful ID fetch
                    log_to_markdown("INFO", f"Successfully fetched run_id. Timestamp: {SCRIPT_RUN_TIMESTAMP}, Handle: {SCRIPT_RUN_HANDLE}", "initialize_script_run_id")
                except ValueError:
                    error_msg = f"Helper script returned timestamp in unexpected format: {output}"
                    log_to_markdown("ERROR", error_msg, "initialize_script_run_id")
                    print(f"ERROR - {error_msg}")
                    SCRIPT_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    SCRIPT_RUN_HANDLE = "FALLBACK_HANDLE_PARSE_ERROR"
            else:
                error_msg = f"Helper script output format unexpected: '{output}'. Expected 'YYYY-MM-DD HH:MM:SS.ffffff GeneratedHandle'."
                log_to_markdown("ERROR", error_msg, "initialize_script_run_id")
                print(f"ERROR - {error_msg}")
                SCRIPT_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                SCRIPT_RUN_HANDLE = "FALLBACK_HANDLE_FORMAT_ERROR"
        else:
            error_msg = f"Helper script {helper_script_path} failed. Return code: {process.returncode}. Stderr: {process.stderr.strip()}"
            log_to_markdown("ERROR", error_msg, "initialize_script_run_id")
            print(f"ERROR - {error_msg}")
            SCRIPT_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            SCRIPT_RUN_HANDLE = "FALLBACK_HANDLE_EXEC_ERROR"
            
    except FileNotFoundError:
        error_msg = f"Python interpreter or helper script {helper_script_path} not found."
        log_to_markdown("CRITICAL", error_msg, "initialize_script_run_id")
        print(f"CRITICAL - {error_msg}")
        SCRIPT_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        SCRIPT_RUN_HANDLE = "FALLBACK_HANDLE_FILENOTFOUND"
    except Exception as e:
        error_msg = f"An unexpected error occurred while running helper script: {e}"
        log_to_markdown("CRITICAL", error_msg, "initialize_script_run_id")
        print(f"CRITICAL - {error_msg}")
        SCRIPT_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        SCRIPT_RUN_HANDLE = "FALLBACK_HANDLE_UNEXPECTED_ERROR"

# --- End New Markdown Logging Setup ---


# Configure basic logging (for console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# GET_UNIQUE_TICKERS
def get_unique_tickers(list1: list[str], list2: list[str]) -> list[str]:
    """Combines two ticker lists and returns a unique, sorted list."""
    log_to_markdown("INFO", "Starting ticker combination and deduplication.", "get_unique_tickers")
    
    combined_tickers = list(set(list1 + list2))
    combined_tickers.sort()
    
    num_input_tickers = len(list1) + len(list2)
    num_unique_tickers = len(combined_tickers)
    
    log_message = f"Processed {num_input_tickers} input tickers. Returned {num_unique_tickers} unique tickers."
    logger.info(f"Combined and deduplicated tickers. Total unique tickers: {num_unique_tickers}") # Keep console log
    log_to_markdown("SUCCESS", log_message, "get_unique_tickers")
    # print(f"The function get_unique_tickers processed {num_input_tickers} input tickers. " # Replaced by markdown log
    #       f"It returned {num_unique_tickers} unique tickers.")
    return combined_tickers

# GET_DATE_RANGE
def get_date_range() -> tuple[str, str]:
    """Calculates the date range for the last 6 months."""
    log_to_markdown("INFO", "Calculating date range for the last 6 months.", "get_date_range")
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=6 * 30)  # Approximate 6 months
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    log_message = f"Calculated date range: {start_date_str} to {end_date_str}."
    logger.info(f"Calculated date range: {start_date_str} to {end_date_str}") # Keep console log
    log_to_markdown("SUCCESS", log_message, "get_date_range")
    # print(f"The function get_date_range determined the data fetching period. " # Replaced by markdown log
    #       f"The period is from {start_date_str} to {end_date_str}.")
    return start_date_str, end_date_str

# PROCESS_FINANCIAL_DATA
def process_financial_data():
    """
    Fetches, processes, and saves daily OHLCV, VIX, and turbulence data.
    """
    logger.info("Starting financial data processing.") # Keep console log
    log_to_markdown("INFO", "Starting financial data processing.", "process_financial_data")

    processor = YahooFinanceProcessor()
    time_interval = "1d"  # Daily data

    # 1. Ticker Selection
    log_to_markdown("INFO", "Starting ticker selection: Combining NASDAQ 100 and S&P 500 tickers.", "process_financial_data")
    unique_tickers = get_unique_tickers(NAS_100_TICKER, SP_500_TICKER)
    if not unique_tickers:
        error_msg = "Ticker list is empty after combining sources. Aborting."
        logger.error(error_msg) # Keep console log
        log_to_markdown("ERROR", error_msg, "process_financial_data")
        # print("The process_financial_data function aborted. The ticker list was empty after combining sources.") # Replaced
        return None, 0, 0, 0, "", "" # Ensure consistent return for main block
    log_to_markdown("SUCCESS", f"Ticker selection complete. Total unique tickers: {len(unique_tickers)}.", "process_financial_data")

    # 2. Data Fetching Date Range
    log_to_markdown("INFO", "Defining data fetching date range (last 6 months).", "process_financial_data")
    start_date_str, end_date_str = get_date_range()
    log_to_markdown("SUCCESS", f"Date range defined: {start_date_str} to {end_date_str}.", "process_financial_data")

    processor.start = start_date_str
    processor.end = end_date_str
    processor.time_interval = time_interval
    log_to_markdown("INFO", f"YahooFinanceProcessor attributes set: start_date={start_date_str}, end_date={end_date_str}, time_interval={time_interval}.", "process_financial_data")

    # 3. Data Fetching (OHLCV)
    log_msg_ohlcv_attempt = f"Attempting to fetch OHLCV data for {len(unique_tickers)} tickers from {start_date_str} to {end_date_str}."
    logger.info(log_msg_ohlcv_attempt) # Keep console log
    log_to_markdown("INFO", log_msg_ohlcv_attempt, "process_financial_data")
    
    market_df, failed_tickers = processor.scrap_data(
        stock_names=unique_tickers,
        start_date=start_date_str,
        end_date=end_date_str,
    )

    successfully_fetched_tickers = [t for t in unique_tickers if t not in failed_tickers]

    if market_df.empty:
        error_msg = f"Failed to download OHLCV data for any ticker. Total attempted: {len(unique_tickers)}. All failed."
        logger.error(error_msg) # Keep console log
        log_to_markdown("ERROR", error_msg, "process_financial_data")
        if failed_tickers:
            failed_list_msg = f"List of all tickers that failed to download: {failed_tickers}"
            logger.info(failed_list_msg) # Keep console log
            log_to_markdown("INFO", failed_list_msg, "process_financial_data")
        # print("The process_financial_data function aborted. OHLCV data download resulted in an empty DataFrame for all tickers.") # Replaced
        return None, len(unique_tickers), 0, len(failed_tickers), start_date_str, end_date_str
    
    success_msg_ohlcv = f"Successfully downloaded OHLCV data for {len(successfully_fetched_tickers)} tickers. Shape: {market_df.shape}."
    logger.info(success_msg_ohlcv) # Keep console log
    log_to_markdown("SUCCESS", success_msg_ohlcv, "process_financial_data")
    
    if successfully_fetched_tickers:
        log_to_markdown("INFO", f"List of successfully fetched tickers ({len(successfully_fetched_tickers)}): {successfully_fetched_tickers}", "process_financial_data")
    if failed_tickers:
        warning_msg_failed = f"List of tickers that failed to download ({len(failed_tickers)}): {failed_tickers}"
        logger.warning(warning_msg_failed) # Keep console log
        log_to_markdown("WARNING", warning_msg_failed, "process_financial_data")
    # print(f"The OHLCV data download is complete. Data for {len(successfully_fetched_tickers)} tickers fetched. {len(failed_tickers)} tickers failed.") # Replaced

    if 'date' in market_df.columns and 'timestamp' not in market_df.columns:
        market_df = market_df.rename(columns={'date': 'timestamp'})
        rename_msg = "Renamed 'date' column to 'timestamp' for compatibility with VIX/turbulence functions."
        logger.info(rename_msg) # Keep console log
        log_to_markdown("INFO", rename_msg, "process_financial_data")

    # 4. Additional Features: Add VIX (using ^VIX)
    vix_column_name = "VIX"
    vix_attempt_msg = f"Attempting to add VIX data (using ^VIX as source, column name '{vix_column_name}')."
    logger.info(vix_attempt_msg) # Keep console log
    log_to_markdown("INFO", vix_attempt_msg, "process_financial_data")
    try:
        market_df = processor.add_vix(data=market_df)
        if vix_column_name not in market_df.columns:
            vix_warn_msg = f"'{vix_column_name}' column not found after add_vix. VIX data might be missing or failed to merge."
            logger.warning(vix_warn_msg) # Keep console log
            log_to_markdown("WARNING", vix_warn_msg, "process_financial_data")
        elif market_df[vix_column_name].isnull().all():
            vix_warn_msg_nan = f"'{vix_column_name}' column is present but contains all NaN values. VIX data might not have aligned or was unavailable."
            logger.warning(vix_warn_msg_nan) # Keep console log
            log_to_markdown("WARNING", vix_warn_msg_nan, "process_financial_data")
        else:
            vix_success_msg = f"Successfully processed VIX data. Shape after VIX: {market_df.shape}."
            logger.info(vix_success_msg) # Keep console log
            log_to_markdown("SUCCESS", vix_success_msg, "process_financial_data")
        # print(f"VIX data addition attempt complete. The DataFrame shape is now {market_df.shape}.") # Replaced
    except Exception as e:
        vix_error_msg = f"Error adding VIX data: {e}. Proceeding without VIX."
        logger.error(vix_error_msg) # Keep console log
        log_to_markdown("ERROR", vix_error_msg, "process_financial_data")
        # print(f"An error occurred while adding VIX data: {e}. The process will continue without VIX.") # Replaced

    # 5. Additional Features: Add Turbulence Index (conditionally)
    turbulence_attempt_msg = "Attempting to calculate and add turbulence index."
    logger.info(turbulence_attempt_msg) # Keep console log
    log_to_markdown("INFO", turbulence_attempt_msg, "process_financial_data")
    if vix_column_name in market_df.columns and not market_df[vix_column_name].isnull().all():
        try:
            market_df = processor.add_turbulence(data=market_df)
            if 'turbulence' not in market_df.columns:
                turb_warn_msg = "Turbulence column not found after add_turbulence. Turbulence data might be missing or failed to merge."
                logger.warning(turb_warn_msg) # Keep console log
                log_to_markdown("WARNING", turb_warn_msg, "process_financial_data")
            elif market_df['turbulence'].isnull().all():
                turb_warn_nan_msg = "Turbulence column is present but contains all NaN values."
                logger.warning(turb_warn_nan_msg) # Keep console log
                log_to_markdown("WARNING", turb_warn_nan_msg, "process_financial_data")
            else:
                turb_success_msg = f"Successfully added turbulence index. Shape after turbulence: {market_df.shape}."
                logger.info(turb_success_msg) # Keep console log
                log_to_markdown("SUCCESS", turb_success_msg, "process_financial_data")
            # print(f"Turbulence index calculation and addition attempt complete. The DataFrame shape is now {market_df.shape}.") # Replaced
        except Exception as e:
            turb_error_msg = f"Error adding turbulence index: {e}. Proceeding without turbulence index."
            logger.error(turb_error_msg) # Keep console log
            log_to_markdown("ERROR", turb_error_msg, "process_financial_data")
            # print(f"An error occurred while adding the turbulence index: {e}. The process will continue without it.") # Replaced
    else:
        turb_skip_msg = f"Skipping turbulence index calculation because VIX data ('{vix_column_name}' column) is missing or all NaN."
        logger.warning(turb_skip_msg) # Keep console log
        log_to_markdown("WARNING", turb_skip_msg, "process_financial_data")
        # print(f"Turbulence index calculation skipped as VIX data ('{vix_column_name}') is not available.") # Replaced

    if 'timestamp' in market_df.columns and 'date' not in market_df.columns:
         market_df = market_df.rename(columns={'timestamp': 'date'})
         revert_msg = "Reverted 'timestamp' column back to 'date' for final CSV output."
         logger.info(revert_msg) # Keep console log
         log_to_markdown("INFO", revert_msg, "process_financial_data")

    # 6. Output
    output_dir = "data"
    output_filename = "market_data_ohlcv_vix_turbulence.csv"
    output_path = os.path.join(output_dir, output_filename)
    abs_output_path = os.path.abspath(output_path)

    log_to_markdown("INFO", f"Preparing to save data to {abs_output_path}.", "process_financial_data")
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            dir_create_msg = f"Created output directory: {output_dir}"
            logger.info(dir_create_msg) # Keep console log
            log_to_markdown("INFO", dir_create_msg, "process_financial_data")
        market_df.to_csv(output_path, index=False)
        save_success_msg = f"Successfully saved data to {abs_output_path}. Final DataFrame shape: {market_df.shape}."
        logger.info(save_success_msg) # Keep console log
        log_to_markdown("SUCCESS", save_success_msg, "process_financial_data")
        # print(f"The final processed data has been saved. The data is located at {abs_output_path}.") # Replaced
    except Exception as e:
        save_error_msg = f"Error saving data to CSV at {abs_output_path}: {e}"
        logger.error(save_error_msg) # Keep console log
        log_to_markdown("ERROR", save_error_msg, "process_financial_data")
        # print(f"An error occurred while saving the data to CSV: {e}.") # Replaced
        return None, len(unique_tickers), len(successfully_fetched_tickers), len(failed_tickers), start_date_str, end_date_str # Ensure consistent return

    proc_complete_msg = "Financial data processing function completed."
    logger.info(proc_complete_msg) # Keep console log
    log_to_markdown("SUCCESS", proc_complete_msg, "process_financial_data")
    # print("The financial data processing script has finished its execution.") # Replaced
    return abs_output_path, len(unique_tickers), len(successfully_fetched_tickers), len(failed_tickers), start_date_str, end_date_str


if __name__ == "__main__":
    # Initialize script run ID and timestamp FIRST for all subsequent markdown logs
    # This also creates the log directory if it doesn't exist.
    initialize_script_run_id()

    logger.info("Script execution started.") # Keep console log
    log_to_markdown("INFO", "Script execution started.", "main_execution")
    
    output_path_result, total_tickers_attempted, num_successful_tickers, num_failed_tickers, start_d_result, end_d_result = None, 0, 0, 0, "", ""
    
    try:
        result = process_financial_data()
        if result and result[0] is not None: # Check if path is not None
            output_path_result, total_tickers_attempted, num_successful_tickers, num_failed_tickers, start_d_result, end_d_result = result
            
            summary_message_parts = [
                f"Data saved to: {output_path_result}",
                f"Attempted: {total_tickers_attempted} tickers",
                f"Successfully processed: {num_successful_tickers} tickers",
                f"Failed to process: {num_failed_tickers} tickers",
                f"Date range: {start_d_result} to {end_d_result}"
            ]
            summary_message_md = "Script finished. " + ". ".join(summary_message_parts) + "."
            
            logger.info(f"Script finished. Data saved to: {output_path_result}. Attempted: {total_tickers_attempted}, Successful: {num_successful_tickers}, Failed: {num_failed_tickers}, Range: {start_d_result} to {end_d_result}.") # Console summary
            log_to_markdown("SUCCESS", summary_message_md, "main_execution")
            # print(f"Script execution finished. Output file: {output_path_result}. {summary_message_md}") # Replaced

            if num_failed_tickers > 0:
                failed_tickers_warn_md = f"{num_failed_tickers} tickers could not be processed. Check logs for specific ticker errors and previous WARNING messages in this log."
                logger.warning(f"{num_failed_tickers} tickers could not be processed. Check console logs for specific ticker errors.") # Keep console log
                log_to_markdown("WARNING", failed_tickers_warn_md, "main_execution")
                # print(f"Warning: {num_failed_tickers} tickers failed. See logs for details.") # Replaced
        else:
            # This case implies process_financial_data returned None or a result where path is None
            critical_error_msg = "Script execution failed to produce an output path or encountered a critical error early in process_financial_data. Check previous logs for details."
            if result: # if result is not None, but result[0] was None
                 _, total_tickers_attempted, num_successful_tickers, num_failed_tickers, start_d_result, end_d_result = result
                 critical_error_msg += f" Stats: Attempted={total_tickers_attempted}, Successful={num_successful_tickers}, Failed={num_failed_tickers}."

            logger.error(critical_error_msg) # Keep console log
            log_to_markdown("CRITICAL", critical_error_msg, "main_execution")
            # print("Script execution encountered a critical issue. Please check the logs for details.") # Replaced

    except Exception as e:
        unexpected_error_msg = f"An unexpected error occurred during script execution: {e}"
        logger.exception(unexpected_error_msg) # Keep console log (with traceback)
        log_to_markdown("CRITICAL", f"{unexpected_error_msg} - Traceback logged to console.", "main_execution")
        # print(f"Script execution failed with an unexpected error: {e}. Check logs.") # Replaced
    finally:
        log_to_markdown("INFO", "Script execution finished (final block).", "main_execution")