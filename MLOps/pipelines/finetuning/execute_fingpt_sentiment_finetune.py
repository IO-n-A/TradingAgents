# Purpose: Executes the FinGPT sentiment fine-tuning process using the core training script.
# Inputs: Configuration for data paths, model paths, hyperparameters, MLflow tracking URI, and experiment name.
# Outputs: Invokes the fine-tuning script, logs results to MLflow, and ensures LoRA adapters are saved in a DVC-compatible structure.

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import mlflow
import yaml
from loguru import logger

# This script calls run_sentiment_lora_finetune_chatglm2.py and integrates its execution with MLflow.
# It handles parameter passing, MLflow logging (params, metrics, artifacts), and DVC-compatible output structure.

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.
    This function reads a YAML file specified by the path and returns its content as a dictionary.
    It's used to manage hyperparameters and paths for the fine-tuning process.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        # The configuration has been loaded from the YAML file.
        # This dictionary will be used to set up the fine-tuning run.
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)


def construct_dvc_output_path(base_output_dir: str, model_name_prefix: str) -> str:
    """
    Constructs a DVC-compatible output path with a version-like timestamp.
    The path will be in the format: base_output_dir/model_name_prefix_YYYYMMDD_HHMMSS.
    This helps in versioning the trained model artifacts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dvc_output_path = os.path.join(base_output_dir, f"{model_name_prefix}_{timestamp}")
    logger.info(f"Constructed DVC-compatible output path: {dvc_output_path}")
    # A unique, timestamped path for DVC has been created.
    # This ensures that each run's artifacts are stored separately.
    return dvc_output_path


def run_finetuning_script(
    script_path: str,
    train_jsonl_path: str,
    validation_jsonl_path: str,
    base_model_name: str,
    output_dir: str,
    hyperparameters: dict,
) -> bool:
    """
    Executes the core fine-tuning script (run_sentiment_lora_finetune_chatglm2.py) with the given parameters.
    This function builds the command-line arguments from the provided configuration and hyperparameters.
    It then runs the script as a subprocess and monitors its execution.
    """
    cmd = [
        sys.executable, # Use the current Python interpreter
        script_path,
        "--train_jsonl_path", train_jsonl_path,
        "--validation_jsonl_path", validation_jsonl_path,
        "--base_model_name", base_model_name,
        "--output_dir", output_dir,
    ]

    # Add hyperparameters from the config
    for key, value in hyperparameters.items():
        cmd.extend([f"--{key}", str(value)])

    logger.info(f"Executing fine-tuning script with command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        # Stream output
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(line.strip()) # Log each line from the subprocess
        
        process.wait() # Wait for the subprocess to complete

        if process.returncode == 0:
            logger.success(f"Fine-tuning script {script_path} executed successfully.")
            # The fine-tuning script completed without errors.
            # The LoRA adapter should now be saved in the output_dir.
            return True
        else:
            logger.error(f"Fine-tuning script {script_path} failed with return code {process.returncode}.")
            # The fine-tuning script encountered an error.
            # Further investigation of the logs from the script is needed.
            return False
    except Exception as e:
        logger.error(f"An exception occurred while running {script_path}: {e}")
        return False


def log_metrics_from_files(mlflow_client: mlflow.tracking.MlflowClient, run_id: str, output_dir: str) -> None:
    """
    Loads metrics from JSON files saved by the training script and logs them to MLflow.
    The training script is expected to save 'train_results.json' and 'all_results.json'.
    This function parses these files and logs the metrics to the active MLflow run.
    """
    metrics_files = {
        "train_results": os.path.join(output_dir, "train_results.json"),
        "all_results": os.path.join(output_dir, "all_results.json"), # Contains eval metrics
        "trainer_state": os.path.join(output_dir, "trainer_state.json") # Contains log history
    }

    for metrics_name, file_path in metrics_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    metrics_data = json.load(f)
                
                if metrics_name == "trainer_state" and "log_history" in metrics_data:
                    # Log evaluation metrics from log_history
                    for entry in metrics_data["log_history"]:
                        if "eval_loss" in entry: # Assuming eval metrics are prefixed with 'eval_'
                            step = entry.get("step", 0)
                            for key, value in entry.items():
                                if key.startswith("eval_"):
                                     mlflow_client.log_metric(run_id, key, value, step=int(step))
                            logger.info(f"Logged evaluation metrics from trainer_state.json at step {step} to MLflow.")
                else:
                    # For train_results.json and all_results.json (if it's flat)
                    # Or handle nested structures if necessary
                    for key, value in metrics_data.items():
                        if isinstance(value, (int, float)):
                            mlflow_client.log_metric(run_id, f"{metrics_name}_{key}", value)
                    logger.info(f"Logged metrics from {file_path} to MLflow.")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from {file_path}. Skipping metrics logging for this file.")
            except Exception as e:
                logger.warning(f"Error logging metrics from {file_path}: {e}")
        else:
            logger.warning(f"Metrics file {file_path} not found. Skipping.")
    # Metrics from the fine-tuning script's output files have been logged to MLflow.
    # This provides a record of the model's performance during training and evaluation.


def main():
    # This is the main execution function for the fine-tuning pipeline script.
    # It parses arguments, sets up MLflow, runs the fine-tuning, and logs artifacts/metrics.
    # Ensures integration with MLOps practices.
    parser = argparse.ArgumentParser(description="Execute FinGPT sentiment fine-tuning pipeline with MLflow integration.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file for fine-tuning.")
    parser.add_argument("--mlflow_tracking_uri", type=str, help="MLflow tracking URI. Defaults to local './mlruns' if not set.")
    parser.add_argument("--mlflow_experiment_name", type=str, default="FinGPT_Sentiment_Finetuning", help="Name of the MLflow experiment.")
    
    args = parser.parse_args()

    config = load_config(args.config_path)

    # MLflow setup
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Core script path (assuming it's relative to this script or in PYTHONPATH)
    # Adjust if run_sentiment_lora_finetune_chatglm2.py is in a different fixed location
    core_script_path = os.path.join(os.path.dirname(__file__), "..", "..", "FinGPT", "finetuning", "run_sentiment_lora_finetune_chatglm2.py")
    core_script_path = os.path.abspath(core_script_path) # Ensure absolute path

    if not os.path.exists(core_script_path):
        logger.error(f"Core fine-tuning script not found at {core_script_path}. Please ensure it's correctly placed.")
        sys.exit(1)

    # DVC-compatible output directory for this run's artifacts
    # The actual LoRA adapter will be saved by the core script *inside* a temporary directory,
    # which we then log as an artifact to MLflow. The DVC path is for the MLflow run's output.
    # For DVC tracking of the model itself, the orchestrator might handle `dvc add`.
    # Here, we ensure MLflow artifacts are organized.
    
    # Using a temporary directory for the core script's output, then logging to MLflow
    with tempfile.TemporaryDirectory() as temp_output_dir:
        logger.info(f"Using temporary directory for fine-tuning output: {temp_output_dir}")

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            logger.info(f"MLflow Experiment Name: {args.mlflow_experiment_name}")
            logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

            # Log configuration parameters
            mlflow.log_params(config.get("general_config", {}))
            mlflow.log_params(config.get("hyperparameters", {}))
            mlflow.log_artifact(args.config_path, "config") # Log the config file itself

            # Extract parameters for the core script
            general_conf = config.get("general_config", {})
            hyperparams_conf = config.get("hyperparameters", {})

            train_jsonl = general_conf.get("train_jsonl_path")
            valid_jsonl = general_conf.get("validation_jsonl_path")
            base_model = general_conf.get("base_model_name")
            
            # The output_dir for the core script will be the temp_output_dir
            # The final DVC path is conceptual for where MLflow might store it or where DVC tracks it.
            # The actual LoRA adapter will be in temp_output_dir.
            
            if not all([train_jsonl, valid_jsonl, base_model]):
                logger.error("Missing critical paths in general_config: train_jsonl_path, validation_jsonl_path, or base_model_name.")
                sys.exit(1)

            success = run_finetuning_script(
                script_path=core_script_path,
                train_jsonl_path=train_jsonl,
                validation_jsonl_path=valid_jsonl,
                base_model_name=base_model,
                output_dir=temp_output_dir, # Core script saves its output here
                hyperparameters=hyperparams_conf
            )

            if success:
                logger.info("Fine-tuning script completed. Logging artifacts and metrics to MLflow.")
                # Log trained LoRA adapter weights (the entire output_dir of the core script)
                mlflow.log_artifacts(temp_output_dir, artifact_path="lora_adapter_weights")
                
                # Log metrics
                log_metrics_from_files(mlflow.tracking.MlflowClient(), run_id, temp_output_dir)
                logger.success("Successfully logged artifacts and metrics to MLflow.")
            else:
                logger.error("Fine-tuning script failed. Check logs for details. No artifacts or metrics will be logged beyond parameters.")
                mlflow.set_tag("status", "failed")
                sys.exit(1) # Exit with error if fine-tuning failed
            
            mlflow.set_tag("status", "completed")
    
    logger.info("MLOps fine-tuning execution script finished.")
    # The fine-tuning execution, including MLflow logging, is complete.
    # Results can be viewed in the MLflow UI.

if __name__ == "__main__":
    main()