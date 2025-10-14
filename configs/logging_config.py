# config/logging_config.py
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
import random
import string

# 1. Introduce a new custom log level named SUCCESS
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    """
    Log 'message % args' with severity 'SUCCESS'.

    To pass exception information, use the keyword argument exc_info with
    a true value, e.g.

    logger.success("Operation completed successfully", exc_info=1)
    """
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

logging.Logger.success = success


# Define UTC+2 timezone for consistent timestamping
utc_plus_2 = timezone(timedelta(hours=2))


# 2. Modify the log_formatter
class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that includes an optional filename summary and data payload.
    - 'filename_display': Populated with 'File Summary: <summary>' if 'filename_summary' is in 'extra',
                          otherwise defaults to '(<filename>:<lineno>)'.
    - 'data_display': Populated with ' - Data: <payload>' if 'data_payload' is in 'extra'.
    - Prepends a custom timestamp (UTC+2) and a short random handle to each log message.
    
    Example usage:
    logger.info(
        "Processing complete", 
        extra={
            'filename_summary': 'data_processing_module', 
            'data_payload': {'items_processed': 100, 'status': 'OK'}
        }
    )
    logger.success("Deployment successful!", extra={'filename_summary': 'deployment_script'})
    """
    def __init__(self, fmt="[%(custom_timestamp)s] - [Handle: %(short_handle)s] - %(name)s%(filename_display)s - [%(levelname)s] - %(message)s%(data_display)s", style='%'):
        super().__init__(fmt=fmt, style=style)

    def format(self, record):
        # Generate custom timestamp (UTC+2, milliseconds precision)
        now_utc_plus_2 = datetime.now(utc_plus_2)
        record.custom_timestamp = now_utc_plus_2.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Generate short random handle (e.g., AB-C)
        random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
        another_random_part = ''.join(random.choices(string.ascii_uppercase, k=1))
        record.short_handle = f"{random_part}-{another_random_part}"

        # Ensure filename_display attribute exists and is populated
        if hasattr(record, 'filename_summary') and record.filename_summary:
            record.filename_display = f" (File Summary: {record.filename_summary})"
        elif record.filename: # record.filename and record.lineno are standard attributes
            record.filename_display = f" ({record.filename}:{record.lineno})"
        else:
            record.filename_display = "" # Default to empty string if no filename info

        # Ensure data_display attribute exists and is populated
        if hasattr(record, 'data_payload') and record.data_payload:
            record.data_display = f" - Data: {str(record.data_payload)}"
        else:
            record.data_display = "" # Default to empty string if no data payload
        
        return super().format(record)


def setup_logging(level=logging.INFO):
    """
    Sets up logging with a custom SUCCESS level and a formatter that includes
    a custom timestamp, a short handle, optional filename summaries, and data payloads.

    To include filename summary or data payload, pass them in the 'extra' dict:
    logger.info("Log message", extra={'filename_summary': 'my_module_tasks', 'data_payload': {'key': 'value'}})
    logger.success("Operation succeeded", extra={'filename_summary': 'critical_op'})

    Logs to console and always appends to 'log/backlog.md'.
    """
    # Use the new CustomFormatter
    log_formatter = CustomFormatter()
    
    # Get the root logger.
    # Remove any existing handlers to avoid duplicate messages if setup_logging is called multiple times.
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.setLevel(level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # File Handler - always log to log/backlog.md
    log_dir = "log"
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Error creating log directory {log_dir}: {e}")
            # If directory creation fails, we might not be able to log to file.
            # Console logging will still work.
            return 
    
    log_file_path = os.path.join(log_dir, "backlog.md")
            
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a') # Ensure append mode
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        
        # This message will also go to the file with the new format
        logging.getLogger(__name__).info(
            f"Logging to console and to file: {log_file_path}",
            extra={'filename_summary': 'logging_setup'}
        )
    except IOError as e:
        print(f"Error creating or writing to log file {log_file_path}: {e}")