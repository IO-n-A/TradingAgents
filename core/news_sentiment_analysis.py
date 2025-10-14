# core/news_sentiment_analysis.py

# Standard library imports
import logging
import os
import sys
import subprocess
from datetime import datetime
import yaml
import pandas as pd
from pandas.errors import EmptyDataError
import torch # For device detection
from typing import Optional, Union, List, Dict # Added for type hinting

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# FinRL and project-specific imports
try:
    from FinRL.finrl.meta.preprocessor.sentiment_text_preprocessor import SentimentTextPreprocessor
    from FinRL.finrl.sentiment_analyzer_service import SentimentAnalyzerService
except ImportError as e:
    print(f"Error importing FinRL modules: {e}. Ensure FinRL is correctly installed and paths are set.")
    sys.exit(1)

# Configure basic logging
# As per coding-standards.md, logger for modules that output to log/backlog.md
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Output logs to console
    ]
)

# Define file paths relative to the project root
NEWS_DATA_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "market_news_us.csv")
NEWS_DATA_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "market_news_with_sentiment.csv")
SENTIMENT_MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "MLOps", "config", "sentiment_models", "llama3_8b_lora_params.yaml")
API_KEYS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "api_keys.yaml")
BACKLOG_FILE_PATH = os.path.join(PROJECT_ROOT, "log", "backlog.md")
GET_TIME_ID_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "helpers", "get_time_id.py")


# Function: load_yaml_config
# Purpose: Loads a YAML configuration file. This function reads a YAML file from the given path and returns its content as a Python dictionary. It handles potential file not found errors and YAML parsing errors.
# Input: file_path (str): The path to the YAML configuration file.
# Output: dict: A dictionary containing the configuration from the YAML file, or None if an error occurs.
def load_yaml_config(file_path: str) -> Optional[dict]:
    """Loads a YAML configuration file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Optional[dict]: A dictionary containing the configuration, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML configuration from: {file_path}")
        print(f"File: core/news_sentiment_analysis.py, Function: load_yaml_config, Output: Configuration dictionary loaded. Path: {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        print(f"File: core/news_sentiment_analysis.py, Function: load_yaml_config, Output: Error - File not found. Path: {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {file_path}: {e}")
        print(f"File: core/news_sentiment_analysis.py, Function: load_yaml_config, Output: Error - YAML parsing failed. Path: {file_path}")
        return None

# Function: get_hugging_face_api_key
# Purpose: Retrieves the Hugging Face API key from the API keys configuration file. This function loads the API keys configuration and extracts the HUGGING_FACE_API_KEY. It's used for services that might require authentication with Hugging Face.
# Input: config_path (str): Path to the API keys YAML file.
# Output: Optional[str]: The Hugging Face API key if found, otherwise None.
def get_hugging_face_api_key(config_path: str) -> Optional[str]:
    """Retrieves the Hugging Face API key from the configuration file.

    Args:
        config_path (str): Path to the API keys YAML file.

    Returns:
        Optional[str]: The Hugging Face API key if found, otherwise None.
    """
    api_keys_config = load_yaml_config(config_path)
    if api_keys_config and "HUGGING_FACE_API_KEY" in api_keys_config:
        key = api_keys_config["HUGGING_FACE_API_KEY"]
        logger.info("Hugging Face API key retrieved.")
        print(f"File: core/news_sentiment_analysis.py, Function: get_hugging_face_api_key, Output: Hugging Face API key retrieved. Key presence: {'Yes' if key else 'No'}")
        return key
    logger.warning("Hugging Face API key not found in configuration.")
    print(f"File: core/news_sentiment_analysis.py, Function: get_hugging_face_api_key, Output: Hugging Face API key not found.")
    return None

# Function: run_get_time_id_script
# Purpose: Executes the get_time_id.py script to obtain a timestamp and ID. This function runs an external Python script and captures its standard output. This ID is used for logging and tracking purposes.
# Input: script_path (str): The path to the get_time_id.py script.
# Output: Optional[str]: The output from the script (timestamp_id) if successful, otherwise None.
def run_get_time_id_script(script_path: str) -> Optional[str]:
    """Executes the get_time_id.py script and returns its output.

    Args:
        script_path (str): The path to the get_time_id.py script.

    Returns:
        Optional[str]: The output from the script (timestamp_id) if successful, otherwise None.
    """
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
        timestamp_id = result.stdout.strip()
        logger.info(f"Successfully executed {script_path}. Timestamp ID: {timestamp_id}")
        print(f"File: core/news_sentiment_analysis.py, Function: run_get_time_id_script, Output: Timestamp ID obtained. ID: {timestamp_id}")
        return timestamp_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_path}: {e}. Output: {e.stderr}")
        print(f"File: core/news_sentiment_analysis.py, Function: run_get_time_id_script, Output: Error executing script. Error: {e}")
        return None
    except FileNotFoundError:
        logger.error(f"Script not found: {script_path}")
        print(f"File: core/news_sentiment_analysis.py, Function: run_get_time_id_script, Output: Error - Script not found. Path: {script_path}")
        return None

# Function: log_to_backlog
# Purpose: Prepends a summary message to the backlog.md file. This function constructs a backlog entry with a timestamp and ID, and the provided summary. It then writes this entry to the beginning of the specified backlog file.
# Input: summary (str): The 3-sentence summary of the completed action.
# Input: backlog_file_path (str): Path to the backlog.md file.
# Input: time_id_script_path (str): Path to the get_time_id.py script.
# Output: None
def log_to_backlog(summary: str, backlog_file_path: str, time_id_script_path: str) -> None:
    """Prepends a summary message with timestamp and ID to the backlog.md file.

    Args:
        summary (str): The 3-sentence summary of the completed action.
        backlog_file_path (str): Path to the backlog.md file.
        time_id_script_path (str): Path to the get_time_id.py script.
    """
    timestamp_id = run_get_time_id_script(time_id_script_path)
    if not timestamp_id:
        logger.error("Failed to get timestamp ID for backlog logging. Skipping backlog update.")
        print("File: core/news_sentiment_analysis.py, Function: log_to_backlog, Output: Failed to get timestamp ID, backlog update skipped.")
        return

    entry = f"## {timestamp_id}\n\n{summary}\n\n---\n\n"

    try:
        existing_content = ""
        if os.path.exists(backlog_file_path):
            with open(backlog_file_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        with open(backlog_file_path, 'w', encoding='utf-8') as f:
            f.write(entry + existing_content)
        logger.info(f"Successfully logged action to {backlog_file_path}")
        print(f"File: core/news_sentiment_analysis.py, Function: log_to_backlog, Output: Action logged to backlog. Path: {backlog_file_path}")
    except IOError as e:
        logger.error(f"Error writing to backlog file {backlog_file_path}: {e}")
        print(f"File: core/news_sentiment_analysis.py, Function: log_to_backlog, Output: Error writing to backlog file. Error: {e}")

# Function: process_news_sentiment
# Purpose: Main function to load news data, preprocess text, generate sentiment scores, and save results. This function orchestrates the entire news sentiment analysis pipeline. It handles data loading, preprocessor and analyzer initialization, text processing, sentiment prediction, and saving the augmented data.
# Input: None
# Output: bool: True if processing was successful, False otherwise.
def process_news_sentiment() -> bool:
    """
    Loads news data, preprocesses text, generates sentiment scores, and saves the augmented data.
    Also logs completion to the backlog.
    """
    logger.info("Starting news sentiment analysis process...")

    # 1. Load sentiment model configuration
    model_config = load_yaml_config(SENTIMENT_MODEL_CONFIG_PATH)
    if not model_config:
        logger.error("Failed to load sentiment model configuration. Aborting.")
        return False

    # 2. Load news data
    try:
        news_df = pd.read_csv(NEWS_DATA_INPUT_PATH)
        logger.info(f"Successfully loaded news data from: {NEWS_DATA_INPUT_PATH}. Shape: {news_df.shape}")

        if news_df.empty:
            logger.warning(f"Input news data file {NEWS_DATA_INPUT_PATH} contains no data rows. Sentiment analysis will be skipped.")
            
            original_columns = list(news_df.columns)
            sentiment_headers = ['preprocessed_text', 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'unknown_sentiment']
            
            final_output_columns = original_columns[:] # Create a copy
            for sh_col in sentiment_headers:
                if sh_col not in final_output_columns:
                    final_output_columns.append(sh_col)
            
            empty_output_df = pd.DataFrame(columns=final_output_columns)
            
            try:
                empty_output_df.to_csv(NEWS_DATA_OUTPUT_PATH, index=False)
                logger.info(f"Created empty output file with headers: {NEWS_DATA_OUTPUT_PATH}")
            except Exception as e_save:
                logger.error(f"Error creating empty output file {NEWS_DATA_OUTPUT_PATH}: {e_save}")
                log_to_backlog(
                    f"Attempted to process news sentiment. Input file {NEWS_DATA_INPUT_PATH} was empty. "
                    f"Failed to create empty output file {NEWS_DATA_OUTPUT_PATH} with headers due to error: {e_save}. "
                    "Sentiment analysis was skipped.",
                    BACKLOG_FILE_PATH,
                    GET_TIME_ID_SCRIPT_PATH
                )
                return False

            summary_empty = (
                f"Input news data file {NEWS_DATA_INPUT_PATH} was empty (no data rows). Sentiment analysis was skipped. "
                f"An empty output file {NEWS_DATA_OUTPUT_PATH} with headers ({', '.join(final_output_columns)}) was created. "
                "No news articles were processed."
            )
            log_to_backlog(summary_empty, BACKLOG_FILE_PATH, GET_TIME_ID_SCRIPT_PATH)
            logger.info("News sentiment analysis skipped due to empty input. Empty output file created.")
            print(f"File: core/news_sentiment_analysis.py, Function: process_news_sentiment, Output: True - Processed empty input. Empty output file created.")
            return True # Gracefully handled empty input

    except FileNotFoundError:
        logger.error(f"Input news data file not found: {NEWS_DATA_INPUT_PATH}. Aborting.")
        log_to_backlog(
            f"News sentiment analysis failed. Input file {NEWS_DATA_INPUT_PATH} was not found. "
            "No output file was created. "
            "Please ensure the preceding script `core/news_fetcher.py` runs successfully.",
            BACKLOG_FILE_PATH,
            GET_TIME_ID_SCRIPT_PATH
        )
        return False
    except EmptyDataError:
        logger.error(f"Input news data file {NEWS_DATA_INPUT_PATH} is empty or malformed (e.g., 0 bytes, no headers). Cannot determine original columns. Aborting.")
        log_to_backlog(
            f"News sentiment analysis failed. Input file {NEWS_DATA_INPUT_PATH} is empty or malformed (e.g., 0 bytes, no headers), "
            "making it impossible to determine original columns for the output. "
            "No output file was created. Please check `core/news_fetcher.py` to ensure it creates a valid CSV with headers, even if there's no data.",
            BACKLOG_FILE_PATH,
            GET_TIME_ID_SCRIPT_PATH
        )
        return False
    except Exception as e:
        logger.error(f"Error loading news data from {NEWS_DATA_INPUT_PATH}: {e}. Aborting.")
        log_to_backlog(
            f"News sentiment analysis failed due to an error loading input file {NEWS_DATA_INPUT_PATH}: {e}. "
            "No output file was created. ",
            BACKLOG_FILE_PATH,
            GET_TIME_ID_SCRIPT_PATH
        )
        return False

    # 3. Initialize SentimentTextPreprocessor
    # No custom rules specified in task, using defaults.
    text_preprocessor = SentimentTextPreprocessor()
    logger.info("SentimentTextPreprocessor initialized.")

    # 4. Initialize SentimentAnalyzerService
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Extract parameters for SentimentAnalyzerService from model_config
    base_model_path = model_config.get("base_model_name_or_path")
    lora_weights_path = model_config.get("peft_model_path") # Can be None
    tokenizer_path = model_config.get("tokenizer_name_or_path", base_model_path)
    
    # Handle quantization_config mapping to bitsandbytes_config
    bnb_config = None
    quant_config = model_config.get("quantization_config")
    if quant_config:
        bnb_config = {}
        if quant_config.get("load_in_8bit"):
            bnb_config["load_in_8bit"] = True
        elif quant_config.get("load_in_4bit"):
            bnb_config["load_in_4bit"] = True
            bnb_config["bnb_4bit_quant_type"] = quant_config.get("bnb_4bit_quant_type", "nf4")
            bnb_config["bnb_4bit_use_double_quant"] = quant_config.get("bnb_4bit_use_double_quant", True)
            # bnb_4bit_compute_dtype needs to be torch.dtype, service handles string conversion
            bnb_config["bnb_4bit_compute_dtype"] = quant_config.get("bnb_4bit_compute_dtype", "bfloat16" if device == "cuda" else "float32")

    # torch_dtype_str: Use training_parameters.fp16 or bf16 if available, else default.
    # The service defaults to "float16". We can make it more specific if config provides.
    torch_dtype_str = "float16" # Default
    if model_config.get("training_parameters", {}).get("fp16"):
        torch_dtype_str = "float16"
    elif model_config.get("training_parameters", {}).get("bf16"):
        torch_dtype_str = "bfloat16"
    
    if device == 'cpu' and bnb_config:
        logger.warning("BitsAndBytes quantization is typically for GPU. Disabling for CPU.")
        bnb_config = None
        torch_dtype_str = "float32"


    # Get Hugging Face API key (optional, depends on model accessibility)
    # hf_api_key = get_hugging_face_api_key(API_KEYS_CONFIG_PATH)
    # The SentimentAnalyzerService doesn't directly take an API key,
    # it relies on huggingface_hub login or environment variables.
    # For now, we assume the environment is set up if private models are used.

    try:
        sentiment_analyzer = SentimentAnalyzerService(
            model_name_or_path=base_model_path,
            lora_weights_path=lora_weights_path,
            tokenizer_name_or_path=tokenizer_path,
            device=device,
            bitsandbytes_config=bnb_config,
            torch_dtype_str=torch_dtype_str
        )
        logger.info("SentimentAnalyzerService initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SentimentAnalyzerService: {e}", exc_info=True)
        return False

    # 5. Process news articles
    processed_texts = []
    prompts = []

    # Assuming 'headline' and 'description' columns exist.
    # Task: "Ensure the preprocessor handles potentially missing headlines or descriptions gracefully."
    # Task: "prepare the news text (e.g., headlines and/or descriptions)"
    for index, row in news_df.iterrows():
        headline = str(row.get('headline', '')) if pd.notna(row.get('headline')) else ''
        description = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
        
        combined_text = ""
        if headline and description:
            combined_text = headline + ". " + description # Simple concatenation
        elif headline:
            combined_text = headline
        elif description:
            combined_text = description
        
        if not combined_text.strip(): # If both are empty or only whitespace
            logger.debug(f"Row {index}: Empty text after combining headline and description. Skipping prompt generation.")
            # We still need to maintain order for results, so add a placeholder or handle later
            processed_texts.append("") # Store empty cleaned text
            prompts.append("") # Store empty prompt, analyzer should handle
            continue

        cleaned_text = text_preprocessor.clean_text(combined_text)
        processed_texts.append(cleaned_text) # Store for potential output

        # Using default prompt format from SentimentTextPreprocessor
        prompt = text_preprocessor.format_prompt(cleaned_text)
        prompts.append(prompt)

    logger.info(f"Prepared {len(prompts)} prompts for sentiment analysis.")

    # Filter out empty prompts before sending to analyzer to avoid errors if it can't handle them
    valid_prompts_with_indices = [(i, p) for i, p in enumerate(prompts) if p.strip()]
    valid_indices = [item[0] for item in valid_prompts_with_indices]
    actual_prompts_to_analyze = [item[1] for item in valid_prompts_with_indices]

    all_sentiment_scores = [{'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'unknown': 1.0}] * len(prompts)

    if actual_prompts_to_analyze:
        try:
            # max_new_tokens for sentiment is usually small (e.g., "positive", "negative")
            batch_sentiment_scores = sentiment_analyzer.predict_sentiment_batch(actual_prompts_to_analyze, max_new_tokens=10)
            logger.info(f"Sentiment prediction completed for {len(batch_sentiment_scores)} valid prompts.")

            # Map results back to original list
            for i, scores in enumerate(batch_sentiment_scores):
                original_index = valid_indices[i]
                all_sentiment_scores[original_index] = scores
        except Exception as e:
            logger.error(f"Error during sentiment prediction batch: {e}", exc_info=True)
            # all_sentiment_scores will remain as default 'unknown' for all
    else:
        logger.info("No valid prompts to analyze after filtering empty ones.")


    # 6. Add sentiment scores to DataFrame
    # The _parse_output_to_sentiment_scores in SentimentAnalyzerService returns a dict
    # e.g., {'positive': 1.0, 'negative': 0.0, 'neutral': 0.0} or includes 'unknown'
    # We'll create columns for each potential key.
    df_scores = pd.DataFrame(all_sentiment_scores)
    
    # Add preprocessed text if kept (as per instruction "preprocessed text (if kept)")
    news_df['preprocessed_text'] = processed_texts 
    
    # Assign sentiment scores with user-specified names
    # Assumes df_scores (from pd.DataFrame(all_sentiment_scores)) will have 'positive', 'negative', 'neutral', 'unknown' columns
    news_df['positive_sentiment'] = df_scores['positive']
    news_df['negative_sentiment'] = df_scores['negative']
    news_df['neutral_sentiment'] = df_scores['neutral']
    news_df['unknown_sentiment'] = df_scores['unknown']

    # 7. Save augmented DataFrame
    try:
        news_df.to_csv(NEWS_DATA_OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved augmented news data with sentiment to: {NEWS_DATA_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error saving augmented news data to {NEWS_DATA_OUTPUT_PATH}: {e}")
        return False

    # 8. Log completion to backlog
    # 8. Log completion to backlog
    num_articles_input = news_df.shape[0]
    num_articles_analyzed = len(actual_prompts_to_analyze) # Defined around line 291

    summary = (
        f"Successfully processed news sentiment. Loaded {num_articles_input} articles from {NEWS_DATA_INPUT_PATH}. "
        f"Sentiment analysis performed on {num_articles_analyzed} non-empty articles after preprocessing. "
        f"Saved augmented data with sentiment scores to {NEWS_DATA_OUTPUT_PATH}."
    )
    log_to_backlog(summary, BACKLOG_FILE_PATH, GET_TIME_ID_SCRIPT_PATH)

    logger.info("News sentiment analysis process completed successfully.")
    print(f"File: core/news_sentiment_analysis.py, Function: process_news_sentiment, Output: True - Process completed successfully. Augmented data saved.")
    return True


if __name__ == "__main__":
    logger.info("Executing news_sentiment_analysis.py as a script.")
    success = process_news_sentiment()
    if success:
        logger.info("Script finished successfully.")
        sys.exit(0)
    else:
        logger.error("Script encountered errors.")
        sys.exit(1)