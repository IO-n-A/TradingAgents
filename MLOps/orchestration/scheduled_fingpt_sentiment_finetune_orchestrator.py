# Purpose: Orchestrates the scheduled execution of the FinGPT sentiment fine-tuning pipeline.
# Inputs: Path to the pipeline execution configuration, and potentially schedule configurations (though actual scheduling is external).
# Outputs: Invokes the fine-tuning pipeline script and handles overall logging for the orchestration.

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml
from loguru import logger

# This script is designed to be called by a scheduler (e.g., cron, Airflow).
# It calls MLOps/pipelines/finetuning/execute_fingpt_sentiment_finetune.py to run the pipeline.
# It manages overall logging for the orchestration process.

def load_orchestrator_config(config_path: str) -> dict:
    """
    Loads the orchestrator's YAML configuration file.
    This configuration specifies paths and settings for the orchestration, including the pipeline config.
    It ensures the orchestrator knows which pipeline configuration to use.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded orchestrator configuration from {config_path}")
        # The orchestrator configuration has been loaded.
        # This will guide the execution of the fine-tuning pipeline.
        return config
    except FileNotFoundError:
        logger.error(f"Orchestrator configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML orchestrator configuration file {config_path}: {e}")
        sys.exit(1)


def run_pipeline_execution_script(
    pipeline_script_path: str,
    pipeline_config_path: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str
) -> bool:
    """
    Executes the MLOps pipeline script (execute_fingpt_sentiment_finetune.py).
    This function constructs the command to run the pipeline execution script.
    It passes necessary configurations like the pipeline's own config path and MLflow details.
    """
    cmd = [
        sys.executable, # Use the current Python interpreter
        pipeline_script_path,
        "--config_path", pipeline_config_path,
        "--mlflow_experiment_name", mlflow_experiment_name
    ]
    if mlflow_tracking_uri: # Only add if specified, otherwise execute_script will use its default
        cmd.extend(["--mlflow_tracking_uri", mlflow_tracking_uri])

    logger.info(f"Executing MLOps pipeline script with command: {' '.join(cmd)}")

    try:
        # Using Popen to stream output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(line.strip()) # Log each line from the subprocess
        
        process.wait() # Wait for the subprocess to complete

        if process.returncode == 0:
            logger.success(f"MLOps pipeline script {pipeline_script_path} executed successfully.")
            # The pipeline execution script completed without errors.
            # Fine-tuning and MLflow logging should be complete.
            return True
        else:
            logger.error(f"MLOps pipeline script {pipeline_script_path} failed with return code {process.returncode}.")
            # The pipeline execution script encountered an error.
            # Detailed logs should be available from the execution script itself.
            return False
    except Exception as e:
        logger.error(f"An exception occurred while running {pipeline_script_path}: {e}")
        return False


def main():
    # This main function drives the orchestration of the fine-tuning pipeline.
    # It loads its configuration, then calls the pipeline execution script.
    # It's intended to be the entry point for a scheduled job.

    parser = argparse.ArgumentParser(description="Orchestrate the scheduled FinGPT sentiment fine-tuning pipeline.")
    parser.add_argument("--orchestrator_config_path", type=str, required=True,
                        help="Path to the YAML configuration file for this orchestrator.")
    # Potentially add arguments for schedule-specific parameters if needed in the future.
    
    args = parser.parse_args()

    # Setup logging for the orchestrator
    # Basic loguru setup, can be expanded (e.g., log to file)
    log_file_name = f"orchestrator_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # Ensure logs directory exists if logging to file
    # For now, just console output via logger.add as default.
    # Example: logger.add(os.path.join("logs", "orchestrator", log_file_name), rotation="1 day")

    logger.info(f"Starting scheduled FinGPT sentiment fine-tuning orchestration at {datetime.now()}")
    logger.info(f"Using orchestrator configuration: {args.orchestrator_config_path}")

    orchestrator_config = load_orchestrator_config(args.orchestrator_config_path)

    pipeline_execution_script_relative_path = orchestrator_config.get("pipeline_execution_script_path", "MLOps/pipelines/finetuning/execute_fingpt_sentiment_finetune.py")
    # Construct absolute path based on this script's location or a known base directory
    # Assuming this orchestrator script is in MLOps/orchestration/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # FinAI_algo root
    pipeline_execution_script_path = os.path.join(base_dir, pipeline_execution_script_relative_path.replace("/", os.sep))
    pipeline_execution_script_path = os.path.abspath(pipeline_execution_script_path)


    pipeline_config_relative_path = orchestrator_config.get("pipeline_config_path")
    if not pipeline_config_relative_path:
        logger.error("`pipeline_config_path` not found in orchestrator configuration.")
        sys.exit(1)
    pipeline_config_path = os.path.join(base_dir, pipeline_config_relative_path.replace("/", os.sep))
    pipeline_config_path = os.path.abspath(pipeline_config_path)


    mlflow_config = orchestrator_config.get("mlflow_config", {})
    mlflow_tracking_uri = mlflow_config.get("tracking_uri") # Can be None
    mlflow_experiment_name = mlflow_config.get("experiment_name", "FinGPT_Sentiment_Finetuning_Scheduled")


    if not os.path.exists(pipeline_execution_script_path):
        logger.error(f"Pipeline execution script not found at {pipeline_execution_script_path}")
        sys.exit(1)
    if not os.path.exists(pipeline_config_path):
        logger.error(f"Pipeline configuration file not found at {pipeline_config_path}")
        sys.exit(1)

    logger.info(f"Pipeline execution script: {pipeline_execution_script_path}")
    logger.info(f"Pipeline configuration file: {pipeline_config_path}")
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri if mlflow_tracking_uri else 'Default (./mlruns)'}")
    logger.info(f"MLflow Experiment Name: {mlflow_experiment_name}")

    success = run_pipeline_execution_script(
        pipeline_script_path=pipeline_execution_script_path,
        pipeline_config_path=pipeline_config_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name
    )

    if success:
        logger.success("FinGPT sentiment fine-tuning orchestration completed successfully.")
        # The orchestration completed its task of running the pipeline.
        # All detailed results are within the pipeline's execution and MLflow.
    else:
        logger.error("FinGPT sentiment fine-tuning orchestration failed.")
        # The orchestration encountered an issue, likely from the pipeline script.
        # Check previous logs for specific error messages.
        sys.exit(1) # Exit with error if orchestration failed

if __name__ == "__main__":
    main()