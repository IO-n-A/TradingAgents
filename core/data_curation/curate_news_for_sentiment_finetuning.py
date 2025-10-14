"""
Curate News Data for Sentiment Fine-tuning

This script filters and prepares raw news data for FinGPT fine-tuning.
It reads news articles, applies an initial sentiment label using the
initial_labeler.py script, and formats the output into JSONL format.
"""

import argparse
import json
import os
import yaml
from datetime import datetime
from typing import Dict, Any, List

# Adjust the import path to correctly locate initial_labeler
# Assuming initial_labeler.py is in core/sentiment_analysis/
import sys
# Add the parent directory of 'core' to sys.path to allow absolute import from 'core'
# This assumes the script is run from a context where 'core' is a sibling directory
# or that the project root is in PYTHONPATH.
# For direct execution, this might need adjustment or running as a module.
# A more robust solution for larger projects would be proper packaging.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.sentiment_analysis.initial_labeler import get_initial_sentiment

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Configuration file is empty or invalid.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        raise
    except ValueError as e:
        print(f"Error in configuration content: {e}")
        raise

def read_news_content(file_path: str) -> str:
    """Reads the content of a single news file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return "" # Return empty string if file cannot be read

def process_news_files(
    raw_news_dir: str,
    config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Processes news files from the directory, applies initial sentiment,
    and formats them.

    Args:
        raw_news_dir (str): Path to the directory containing raw news files.
        config (Dict[str, Any]): Loaded curation configuration.

    Returns:
        List[Dict[str, str]]: A list of processed data entries in JSONL format.
    """
    processed_data = []
    instruction_template = config.get("jsonl_instruction_template", "Analyze sentiment:")
    sentiment_config = config.get("sentiment_labeling", {})
    positive_keywords = sentiment_config.get("positive_keywords", [])
    negative_keywords = sentiment_config.get("negative_keywords", [])
    
    # Placeholder for date filtering - currently not implemented as file naming/metadata for dates is undefined
    # start_date_str = config.get("date_range", {}).get("start_date")
    # end_date_str = config.get("date_range", {}).get("end_date")
    # start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
    # end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

    if not os.path.isdir(raw_news_dir):
        print(f"Error: Raw news directory not found or is not a directory: {raw_news_dir}")
        return []

    for filename in os.listdir(raw_news_dir):
        file_path = os.path.join(raw_news_dir, filename)
        if os.path.isfile(file_path):
            # Add date filtering logic here if news filenames or metadata contain dates
            # For example, if filename is YYYY-MM-DD_article.txt:
            # try:
            #     file_date_str = filename.split('_')[0]
            #     file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
            #     if start_date and file_date < start_date:
            #         continue
            #     if end_date and file_date > end_date:
            #         continue
            # except ValueError:
            #     print(f"Warning: Could not parse date from filename {filename}. Skipping date filter for this file.")
            #     pass # Or handle as an error

            news_content = read_news_content(file_path)
            if not news_content.strip(): # Skip empty or whitespace-only files
                print(f"Warning: File {filename} is empty or contains only whitespace. Skipping.")
                continue

            initial_label = get_initial_sentiment(news_content, positive_keywords, negative_keywords)
            
            processed_entry = {
                "instruction": instruction_template,
                "input": news_content,
                "output": initial_label
            }
            processed_data.append(processed_entry)
        else:
            print(f"Warning: Skipping non-file item in raw news directory: {filename}")
            
    return processed_data

def save_to_jsonl(data: List[Dict[str, str]], output_file_path: str) -> None:
    """Saves the processed data to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
        print(f"Successfully saved curated data to {output_file_path}")
    except IOError as e:
        print(f"Error writing to output file {output_file_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while saving to JSONL: {e}")
        raise

def main():
    """Main function to orchestrate the data curation process."""
    parser = argparse.ArgumentParser(description="Curate news data for sentiment fine-tuning.")
    parser.add_argument(
        "--raw_news_dir",
        type=str,
        required=True,
        help="Path to the directory containing raw news articles."
    )
    parser.add_argument(
        "--output_jsonl_path",
        type=str,
        required=True,
        help="Path to save the output JSONL file (e.g., data/processed/fingpt_finetune_datasets/us_equity_sentiment_YYYYMMDD.jsonl)."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the curation configuration YAML file (e.g., config/curation_config.yaml)."
    )
    args = parser.parse_args()

    print("Starting data curation process...")
    try:
        config = load_config(args.config_path)
        print(f"Configuration loaded from {args.config_path}")

        processed_data = process_news_files(args.raw_news_dir, config)
        
        if not processed_data:
            print("No data was processed. Exiting.")
            return

        save_to_jsonl(processed_data, args.output_jsonl_path)
        print("Data curation process completed.")

    except Exception as e:
        print(f"An error occurred during the curation process: {e}")
        # Potentially exit with a non-zero status code for pipeline integration
        # sys.exit(1)

if __name__ == "__main__":
    # Example of how to run (assuming files are in place):
    # python core/data_curation/curate_news_for_sentiment_finetuning.py \
    #   --raw_news_dir data/raw/sample_news \
    #   --output_jsonl_path data/processed/fingpt_finetune_datasets/sample_sentiment_data.jsonl \
    #   --config_path config/curation_config.yaml
    #
    # For this to run, you'd need:
    # 1. `config/curation_config.yaml` (created in a previous step)
    # 2. `core/sentiment_analysis/initial_labeler.py` (created in a previous step)
    # 3. A directory `data/raw/sample_news` with some text files.
    main()