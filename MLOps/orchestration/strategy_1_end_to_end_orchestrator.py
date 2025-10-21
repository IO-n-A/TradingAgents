# MLOps/orchestration/strategy_1_end_to_end_orchestrator.py
"""
Orchestrates the end-to-end execution of Strategy 1.
This script sequentially runs data collection, feature engineering (using a
specified LoRA-enhanced sentiment model), FinRL model training, and
FinGPT sentiment model fine-tuning for the next cycle.
It is configured via a main YAML file.
"""

import argparse
import datetime
import logging
import os
import subprocess
import sys
import yaml
from typing import Dict, Any, Optional

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # Add FileHandler if logging to file is desired
    ]
)
logger = logging.getLogger(__name__)

# This function executes a given script with specified arguments and configuration.
# It captures output, logs results, and handles errors.
# The function returns True if the script executed successfully, False otherwise.
def execute_script(
    script_path: str,
    config_path: Optional[str] = None,
    script_args: Optional[Dict[str, str]] = None,
    step_name: str = "Unnamed Step"
) -> bool:
    """
    Executes a Python script as a subprocess.

    Args:
        script_path (str): Relative path to the Python script to execute.
        config_path (Optional[str]): Relative path to the YAML config file for the script.
        script_args (Optional[Dict[str, str]]): Additional command-line arguments for the script.
        step_name (str): Name of the pipeline step for logging.

    Returns:
        bool: True if execution was successful, False otherwise.
    """
    # This function constructs a command to run a Python script with specified configurations and arguments.
    # It then executes this command, logs the output, and returns a status indicating success or failure.
    # This modular execution approach allows for clear tracking and management of each pipeline step.
    full_script_path = os.path.join(PROJECT_ROOT, script_path)
    if not os.path.exists(full_script_path):
        logger.error(f"[{step_name}] Script not found: {full_script_path}")
        print(f"[{step_name}] Script not found: {full_script_path}. The script path is invalid.")
        return False

    command = [sys.executable, full_script_path]
    if config_path:
        full_config_path = os.path.join(PROJECT_ROOT, config_path)
        if not os.path.exists(full_config_path):
            logger.warning(f"[{step_name}] Config file not found: {full_config_path}. Proceeding without it if script allows.")
            # Some scripts might not require a config or have defaults.
        else:
            command.extend(["--config", full_config_path])

    if script_args:
        for arg_name, arg_value in script_args.items():
            command.extend([f"--{arg_name}", str(arg_value)])

    logger.info(f"[{step_name}] Executing command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT)
        stdout, stderr = process.communicate()

        if stdout:
            logger.info(f"[{step_name}] STDOUT:\n{stdout}")
        if stderr:
            if process.returncode == 0: # Some scripts might output to stderr for info
                logger.info(f"[{step_name}] STDERR (Return Code 0):\n{stderr}")
            else:
                logger.error(f"[{step_name}] STDERR:\n{stderr}")
        
        if process.returncode != 0:
            logger.error(f"[{step_name}] Failed with return code {process.returncode}.")
            print(f"[{step_name}] Script execution failed with code {process.returncode}. Check logs for details.")
            return False
        
        logger.info(f"[{step_name}] Completed successfully.")
        print(f"[{step_name}] Script execution completed successfully. Output and errors (if any) are logged.")
        return True

    except FileNotFoundError:
        logger.error(f"[{step_name}] Error: The script '{full_script_path}' was not found.")
        print(f"[{step_name}] The specified script was not found. Please check the path.")
        return False
    except Exception as e:
        logger.error(f"[{step_name}] An unexpected error occurred: {e}", exc_info=True)
        print(f"[{step_name}] An unexpected error occurred during script execution. Details: {e}")
        return False

# This is the main orchestration function for Strategy 1.
# It reads a YAML configuration file, sets up logging, and sequentially executes the defined pipeline steps.
# It handles the flow of information, such as passing the LoRA adapter path to the feature engineering script.
def run_strategy_1_orchestration(main_config_path: str) -> None:
    """
    Main orchestration function for Strategy 1.
    Loads configuration and executes pipeline steps sequentially.

    Args:
        main_config_path (str): Path to the main YAML configuration file for the orchestrator.
    """
    # This function loads the main configuration, sets the log level, and iterates through defined pipeline steps.
    # For each enabled step, it calls `execute_script` with appropriate parameters, including the LoRA adapter path for feature engineering.
    # This ensures a controlled and configurable execution of the entire Strategy 1 pipeline.
    try:
        with open(main_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Main configuration file not found: {main_config_path}")
        print(f"The main configuration file {main_config_path} was not found. Orchestration cannot proceed.")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing main configuration file {main_config_path}: {e}")
        print(f"Error parsing the main configuration file. Please check its format. Details: {e}")
        return

    global_settings = config.get("global_settings", {})
    log_level = global_settings.get("log_level", "INFO").upper()
    logging.getLogger().setLevel(log_level) # Set root logger level
    logger.info(f"Global log level set to: {log_level}")

    current_lora_adapter_path = global_settings.get("current_lora_adapter_path")
    if not current_lora_adapter_path:
        logger.warning("`current_lora_adapter_path` not found in global_settings. Feature engineering might fail if it expects it.")
        # It's a warning because some feature engineering setups might not use LoRA initially or have a default.

    pipeline_steps = config.get("pipeline_steps", {})
    ordered_steps = [
        "daily_data_collection",
        "feature_engineering_strategy_1",
        "finrl_model_training",
        "fingpt_sentiment_finetune"
    ]

    logger.info("Starting Strategy 1 End-to-End Orchestration.")
    overall_success = True

    for step_key in ordered_steps:
        step_config = pipeline_steps.get(step_key)
        if not step_config:
            logger.warning(f"Configuration for step '{step_key}' not found. Skipping.")
            continue

        if not step_config.get("enabled", False):
            logger.info(f"Step '{step_key}' is disabled. Skipping.")
            continue

        logger.info(f"--- Starting step: {step_key} ---")
        
        script_path = step_config.get("script_path")
        step_config_path = step_config.get("config_path") # Config for the sub-script
        additional_args = step_config.get("args", {}) # For any other generic args

        if not script_path:
            logger.error(f"No script_path defined for step '{step_key}'. Skipping.")
            overall_success = False
            continue
        
        # Special handling for feature_engineering_strategy_1 to pass LoRA adapter path
        if step_key == "feature_engineering_strategy_1":
            if current_lora_adapter_path:
                # Ensure strategy_1_feature_pipeline.py accepts --lora_adapter_path
                additional_args["lora_adapter_path"] = current_lora_adapter_path
                logger.info(f"Passing lora_adapter_path='{current_lora_adapter_path}' to {step_key}")
            else:
                logger.warning(f"No current_lora_adapter_path available for {step_key}. "
                               "The feature engineering script might use a default or fail if one is required.")

        success = execute_script(
            script_path=script_path,
            config_path=step_config_path,
            script_args=additional_args,
            step_name=step_key
        )

        if not success:
            logger.error(f"Step '{step_key}' failed. Aborting further pipeline execution.")
            overall_success = False
            break # Stop orchestration on failure
        
        logger.info(f"--- Completed step: {step_key} ---")

    if overall_success:
        logger.info("Strategy 1 End-to-End Orchestration completed successfully.")
        print("Strategy 1 orchestration finished successfully. All enabled steps completed.")
    else:
        logger.error("Strategy 1 End-to-End Orchestration failed at one of the steps.")
        print("Strategy 1 orchestration failed. Please check the logs for details on the failed step.")
    
    # The orchestration process has concluded.
    # The final status (success or failure) has been logged, indicating whether all enabled steps ran to completion.
    # This provides a clear outcome for the automated pipeline execution.


# This is the entry point of the script.
# It parses command-line arguments, specifically the path to the main YAML configuration file.
# It then calls the main orchestration function.
if __name__ == "__main__":
    # This script entry point sets up argument parsing for the main configuration file path.
    # It then initiates the orchestration process by calling `run_strategy_1_orchestration`.
    # This allows the orchestrator to be invoked from the command line with a specified configuration.
    parser = argparse.ArgumentParser(description="Run Strategy 1 End-to-End Orchestration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the main YAML configuration file for the orchestrator (e.g., MLOps/config/orchestration/strategy_1_main_config.yaml)"
    )
    args = parser.parse_args()

    # Example: python MLOps/orchestration/strategy_1_end_to_end_orchestrator.py --config MLOps/config/orchestration/strategy_1_main_config.yaml
    run_strategy_1_orchestration(args.config)
    print(f"Orchestration process initiated with config: {args.config}. The process has now finished.")