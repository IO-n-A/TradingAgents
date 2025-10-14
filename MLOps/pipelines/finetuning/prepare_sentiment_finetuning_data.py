"""
Prepare Sentiment Fine-tuning Data Pipeline

Orchestrates the data curation process for FinGPT fine-tuning.
It calls the core data curation script and can optionally split
the output into training and validation sets.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random # For shuffling

# Ensure the core module can be found if this script is run directly
# This assumes the script is in MLOps/pipelines/finetuning
# and the project root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Attempt to import train_test_split, handle if not available
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn is not installed. Data splitting functionality will be limited.")
    print("Please install it by running: pip install scikit-learn")


def run_curation_script(
    raw_news_dir: str,
    output_jsonl_path: str,
    config_path: str,
    python_executable: str = sys.executable
) -> bool:
    """
    Calls the core/data_curation/curate_news_for_sentiment_finetuning.py script.

    Args:
        raw_news_dir (str): Path to the raw news data directory.
        output_jsonl_path (str): Path for the output JSONL file.
        config_path (str): Path to the curation configuration YAML.
        python_executable (str): Path to the python interpreter.

    Returns:
        bool: True if the script ran successfully, False otherwise.
    """
    curation_script_path = os.path.join(
        PROJECT_ROOT,
        "core", "data_curation", "curate_news_for_sentiment_finetuning.py"
    )

    if not os.path.exists(curation_script_path):
        print(f"Error: Curation script not found at {curation_script_path}")
        return False
    if not os.path.exists(config_path):
        print(f"Error: Curation config file not found at {config_path}")
        return False
    if not os.path.isdir(raw_news_dir):
        print(f"Error: Raw news directory not found at {raw_news_dir}")
        return False

    command = [
        python_executable,
        curation_script_path,
        "--raw_news_dir", raw_news_dir,
        "--output_jsonl_path", output_jsonl_path,
        "--config_path", config_path
    ]

    print(f"Executing data curation script: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Curation script executed successfully.")
            print("Stdout:\n", stdout)
            return True
        else:
            print(f"Error running curation script. Return code: {process.returncode}")
            print("Stdout:\n", stdout)
            print("Stderr:\n", stderr)
            return False
    except Exception as e:
        print(f"An exception occurred while running the curation script: {e}")
        return False

def load_jsonl(file_path: str) -> List[Dict]:
    """Loads data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: File not found for loading: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        raise


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Saves data to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Successfully saved data to {file_path}")
    except IOError as e:
        print(f"Error writing to output file {file_path}: {e}")
        raise

def split_data(
    data: List[Dict],
    test_size: float = 0.1, # For validation set
    random_state: Optional[int] = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits data into training and validation sets.
    Uses sklearn.model_selection.train_test_split if available,
    otherwise uses a simple random split.
    """
    if not data:
        return [], []

    if SKLEARN_AVAILABLE:
        print(f"Using scikit-learn for splitting data (test_size={test_size}).")
        train_data, val_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        return train_data, val_data
    else:
        print(f"Warning: scikit-learn not available. Using basic random split (validation_proportion={test_size}).")
        # Basic random split if sklearn is not available
        shuffled_data = random.sample(data, len(data)) # Shuffle
        split_idx = int(len(shuffled_data) * (1 - test_size))
        train_data = shuffled_data[:split_idx]
        val_data = shuffled_data[split_idx:]
        return train_data, val_data


def main():
    """Main function to orchestrate the data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare sentiment fine-tuning data pipeline.")
    parser.add_argument(
        "--raw_news_dir",
        type=str,
        required=True,
        help="Path to the directory containing raw news articles."
    )
    parser.add_argument(
        "--output_jsonl_basepath", # e.g., data/processed/fingpt_finetune_datasets/us_equity_sentiment
        type=str,
        required=True,
        help="Base path and filename prefix for the output JSONL file (date and .jsonl will be appended). Example: 'data/processed/fingpt_finetune_datasets/us_equity_sentiment'"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the curation configuration YAML file (e.g., config/curation_config.yaml)."
    )
    parser.add_argument(
        "--split_data",
        action="store_true",
        help="If set, split the curated data into train.jsonl and validation.jsonl."
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the validation split (e.g., 0.1 for 10%). Only used if --split_data is set."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for data splitting to ensure reproducibility."
    )
    parser.add_argument(
        "--date_tag",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="Date tag for the output file (YYYYMMDD format). Defaults to current date."
    )

    args = parser.parse_args()

    print("Starting sentiment fine-tuning data preparation pipeline...")

    # Construct the full output path for the main curated file
    # e.g., data/processed/fingpt_finetune_datasets/us_equity_sentiment_20231026.jsonl
    main_output_jsonl_path = f"{args.output_jsonl_basepath}_{args.date_tag}.jsonl"
    output_dir = os.path.dirname(main_output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Step 1: Run the data curation script
    curation_successful = run_curation_script(
        args.raw_news_dir,
        main_output_jsonl_path,
        args.config_path
    )

    if not curation_successful:
        print("Data curation script failed. Exiting pipeline.")
        sys.exit(1)
    
    if not os.path.exists(main_output_jsonl_path) or os.path.getsize(main_output_jsonl_path) == 0:
        print(f"Curation script reported success, but output file {main_output_jsonl_path} is missing or empty. Exiting.")
        sys.exit(1)

    # Step 2: Optionally split the data
    if args.split_data:
        print(f"Splitting data into training and validation sets (validation size: {args.validation_size})...")
        try:
            all_data = load_jsonl(main_output_jsonl_path)
            if not all_data:
                print(f"No data found in {main_output_jsonl_path} to split. Skipping split.")
            else:
                train_data, val_data = split_data(
                    all_data,
                    test_size=args.validation_size,
                    random_state=args.random_seed
                )

                train_output_path = os.path.join(output_dir, f"train_{args.date_tag}.jsonl") # Or just train.jsonl
                validation_output_path = os.path.join(output_dir, f"validation_{args.date_tag}.jsonl") # Or just validation.jsonl
                
                # For DVC, often simpler names are preferred if the YYYYMMDD is part of the directory structure
                # For this task, the requirement is "us_equity_sentiment_YYYYMMDD.jsonl" for the main file,
                # and "train.jsonl" and "validation.jsonl" for splits.
                # Let's adjust to make train/validation files simpler if they are in the dated output_dir.
                # If output_jsonl_basepath was "data/processed/fingpt_finetune_datasets/us_equity_sentiment"
                # and date_tag is "20230101", then output_dir is "data/processed/fingpt_finetune_datasets"
                # main_output_jsonl_path is "data/processed/fingpt_finetune_datasets/us_equity_sentiment_20230101.jsonl"
                # We want train.jsonl and validation.jsonl in the *same directory* as the main output file.
                
                # Let's use the directory of the main output file for train/val files
                # And name them simply train.jsonl and validation.jsonl
                # This aligns better with common DVC practices for dataset versions.
                # The YYYYMMDD tag will be on the main file.
                
                train_output_path_final = os.path.join(os.path.dirname(main_output_jsonl_path), "train.jsonl")
                validation_output_path_final = os.path.join(os.path.dirname(main_output_jsonl_path), "validation.jsonl")

                save_jsonl(train_data, train_output_path_final)
                save_jsonl(val_data, validation_output_path_final)
                print(f"Training data saved to: {train_output_path_final} ({len(train_data)} records)")
                print(f"Validation data saved to: {validation_output_path_final} ({len(val_data)} records)")

        except Exception as e:
            print(f"An error occurred during data splitting: {e}")
            # Decide if this should be a fatal error for the pipeline
            # For now, we'll print the error and continue, as the main file is still generated.
    else:
        print("Data splitting was not requested.")

    print("Sentiment fine-tuning data preparation pipeline completed.")
    print(f"Main curated data (DVC target): {main_output_jsonl_path}")
    if args.split_data:
         print(f"Train split (DVC target): {os.path.join(os.path.dirname(main_output_jsonl_path), 'train.jsonl')}")
         print(f"Validation split (DVC target): {os.path.join(os.path.dirname(main_output_jsonl_path), 'validation.jsonl')}")


if __name__ == "__main__":
    # Example of how to run:
    # Assuming:
    # 1. `config/curation_config.yaml` exists.
    # 2. `core/data_curation/curate_news_for_sentiment_finetuning.py` exists.
    # 3. A directory `data/raw/sample_news_for_pipeline` exists with some text files.
    #
    # To create sample raw news data:
    # mkdir -p data/raw/sample_news_for_pipeline
    # echo "Positive news about AAPL growth" > data/raw/sample_news_for_pipeline/news1.txt
    # echo "Negative outlook for MSFT stocks" > data/raw/sample_news_for_pipeline/news2.txt
    # echo "Neutral market sentiment today" > data/raw/sample_news_for_pipeline/news3.txt
    #
    # python MLOps/pipelines/finetuning/prepare_sentiment_finetuning_data.py \
    #   --raw_news_dir data/raw/sample_news_for_pipeline \
    #   --output_jsonl_basepath data/processed/fingpt_finetune_datasets/us_equity_sentiment \
    #   --config_path config/curation_config.yaml \
    #   --split_data \
    #   --validation_size 0.2 \
    #   --date_tag testrun
    main()