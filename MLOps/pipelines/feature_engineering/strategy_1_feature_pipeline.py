# MLOps/pipelines/feature_engineering/strategy_1_feature_pipeline.py
"""
Orchestrates the feature engineering steps for Strategy 1.
Loads raw data, calculates technical indicators, preprocesses news text,
generates sentiment scores, aggregates features, and saves them.
"""
import pandas as pd
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse # Added for command-line argument parsing
import yaml # Added for loading sentiment model config

# Assuming the core modules are in the parent directory or PYTHONPATH is set up
# For robust imports, especially when running scripts from different locations,
# consider adding the project root to sys.path or using a proper package structure.
import sys
# Add project root to sys.path, assuming this script is in MLOps/pipelines/feature_engineering/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.feature_engineering.technical_indicators_calculator import calculate_technical_indicators
from core.preprocess.text_preprocessor import TextPreprocessor
from core.sentiment_analysis.fingpt_sentiment_analyzer_service import FinGPTSentimentAnalyzerService, DEFAULT_MODEL_NAME

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s'
)

# Configuration constants (can be moved to a YAML file later)
CONFIG: Dict[str, Any] = {
    "base_raw_data_path": os.path.join(project_root, "data", "raw"),
    "base_processed_data_path": os.path.join(project_root, "data", "processed", "features", "strategy_1"),
    "ohlcv_source_dir_template": "ohlcv/us_equity/{date_str}", # e.g., ohlcv/us_equity/2023-01-01
    "news_source_dir_template": "news_articles/{date_str}",   # e.g., news_articles/2023-01-01
    "sentiment_model_name": DEFAULT_MODEL_NAME, # Can be overridden
    "output_file_name_tech_indicators": "technical_indicators.parquet",
    "output_file_name_processed_news": "processed_news_with_sentiment.parquet",
    "output_file_name_aggregated_features": "aggregated_features.parquet", # Example for combined features
    "date_format": "%Y-%m-%d",
    "text_preprocessor_lang": "english",
    "technical_indicators_params": { # Default parameters for indicators
        "sma_windows": [10, 20, 50],
        "ema_windows": [10, 20, 50],
        "rsi_window": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9
    }
}

# This function loads OHLCV data for a given date.
# It expects data to be in CSV or Parquet files within a date-specific directory.
# It returns a pandas DataFrame with the OHLCV data.
def load_ohlcv_data(date_str: str, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Loads raw OHLCV data for a specific date.
    Placeholder: Assumes data is in multiple CSV/Parquet files per ticker in a directory.
    For simplicity, this example will try to load a single aggregated file if it exists,
    or simulate loading by returning a dummy DataFrame.

    Args:
        date_str (str): The date string in YYYY-MM-DD format.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Optional[pd.DataFrame]: DataFrame with OHLCV data, or None if not found/error.
    """
    ohlcv_dir_path = os.path.join(
        config["base_raw_data_path"],
        config["ohlcv_source_dir_template"].format(date_str=date_str)
    )
    logger.info(f"Attempting to load OHLCV data from: {ohlcv_dir_path}")

    # In a real scenario, you would iterate through files, load them, and concatenate.
    # Example: Load a hypothetical aggregated file.
    # For this example, we'll create a dummy DataFrame if no actual file is found.
    # This part needs to be adapted to the actual data storage format.
    
    # Try to find a parquet or csv file. This is a simplified loader.
    # A more robust loader would scan for all relevant ticker files.
    potential_files = [f for f in os.listdir(ohlcv_dir_path) if f.endswith(('.parquet', '.csv'))] if os.path.exists(ohlcv_dir_path) else []
    
    if potential_files:
        file_path = os.path.join(ohlcv_dir_path, potential_files[0]) # Load the first found file
        try:
            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            logger.info(f"Loaded OHLCV data from {file_path}. Shape: {df.shape}")
            # Ensure standard columns: 'open', 'high', 'low', 'close', 'volume'.
            # May also have 'ticker', 'date', 'vix', 'turbulence'.
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"OHLCV data from {file_path} is missing one of required columns: {required_cols}")
                return None
            # The OHLCV data has been loaded successfully from the specified path.
            # It is now ready for technical indicator calculation.
            print(f"OHLCV data loaded from {file_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading OHLCV data from {file_path}: {e}", exc_info=True)
            return None
    else:
        logger.warning(f"No OHLCV data files found in {ohlcv_dir_path}. Returning dummy data for demonstration.")
        # Create dummy data if no file is found for demonstration
        dummy_data = {
            'date': pd.to_datetime([f'{date_str} 00:00:00']*20 + [f'{date_str} 00:00:00']*20),
            'ticker': ['AAPL']*20 + ['MSFT']*20, # Example for multiple tickers
            'open': [i + 0.5 for i in range(150, 170)] + [i + 0.5 for i in range(250, 270)],
            'high': [i + 1.0 for i in range(150, 170)] + [i + 1.0 for i in range(250, 270)],
            'low': [i - 0.5 for i in range(150, 170)] + [i - 0.5 for i in range(250, 270)],
            'close': list(range(150, 170)) + list(range(250, 270)),
            'volume': [1000000 + i*10000 for i in range(20)] + [1200000 + i*10000 for i in range(20)],
            'vix': [15 + i*0.1 for i in range(20)] + [16 + i*0.1 for i in range(20)], # Example VIX
            'turbulence': [0.01 + i*0.001 for i in range(20)] + [0.02 + i*0.001 for i in range(20)] # Example Turbulence
        }
        df = pd.DataFrame(dummy_data)
        df['date'] = pd.to_datetime(df['date'])
        # The dummy OHLCV data has been generated for demonstration purposes.
        # This data will be used for calculating technical indicators.
        print(f"Generated dummy OHLCV data for {date_str}. Shape: {df.shape}")
        return df

# This function loads raw news data for a given date.
# It expects data to be in JSON or CSV files within a date-specific directory.
# It returns a list of dictionaries, each representing a news article.
def load_raw_news_data(date_str: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Loads raw news data for a specific date.
    Placeholder: Assumes data is in JSON files (e.g., one file per source or aggregated).
    For simplicity, this example will try to load a single JSON file if it exists,
    or simulate loading by returning dummy news items.

    Args:
        date_str (str): The date string in YYYY-MM-DD format.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        List[Dict[str, Any]]: List of news articles (dictionaries with 'headline', 'summary', 'content', etc.).
    """
    news_dir_path = os.path.join(
        config["base_raw_data_path"],
        config["news_source_dir_template"].format(date_str=date_str)
    )
    logger.info(f"Attempting to load news data from: {news_dir_path}")
    
    # Try to find a JSON file. This is a simplified loader.
    potential_files = [f for f in os.listdir(news_dir_path) if f.endswith('.json')] if os.path.exists(news_dir_path) else []

    if potential_files:
        file_path = os.path.join(news_dir_path, potential_files[0]) # Load the first found JSON file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                news_items = json.load(f) # Expects a list of dicts
            if not isinstance(news_items, list):
                logger.error(f"News data in {file_path} is not a list of articles.")
                news_items = [] # Fallback to empty
            logger.info(f"Loaded {len(news_items)} news articles from {file_path}.")
            # The news data has been loaded successfully from the specified JSON file.
            # This data is now ready for text preprocessing and sentiment analysis.
            print(f"News data loaded from {file_path}. Number of articles: {len(news_items)}")
            return news_items
        except Exception as e:
            logger.error(f"Error loading news data from {file_path}: {e}", exc_info=True)
            return []
    else:
        logger.warning(f"No news data files found in {news_dir_path}. Returning dummy news for demonstration.")
        # Create dummy news if no file is found for demonstration
        dummy_news = [
            {"id": "news1", "ticker": "AAPL", "date": date_str, "headline": "Tech Giant AAPL Announces New Product Line", "summary": "AAPL's stock surged today after the announcement.", "content": "Full content of AAPL news..."},
            {"id": "news2", "ticker": "MSFT", "date": date_str, "headline": "MSFT Reports Strong Earnings for Q3", "summary": "Positive outlook for MSFT as earnings beat expectations.", "content": "Full content of MSFT news..."},
            {"id": "news3", "ticker": "GENL", "date": date_str, "headline": "Market Update: S&P 500 Reaches New Highs", "summary": "Overall market sentiment is positive.", "content": "General market news content..."}
        ]
        # The dummy news data has been generated for demonstration.
        # This data will be used for text preprocessing and sentiment analysis.
        print(f"Generated dummy news data for {date_str}. Number of articles: {len(dummy_news)}")
        return dummy_news

# This function saves the processed data to a specified path.
# It takes a pandas DataFrame and an output path as input.
# It saves the DataFrame as a Parquet file.
def save_processed_data(df: pd.DataFrame, output_path: str, file_name: str) -> None:
    """
    Saves the processed DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The directory to save the file in.
        file_name (str): The name of the file (e.g., 'technical_indicators.parquet').
    """
    if df.empty:
        logger.warning(f"DataFrame is empty. Skipping save for {file_name} in {output_path}.")
        # The DataFrame to be saved is empty.
        # No file will be created for this empty dataset.
        print(f"Skipped saving empty DataFrame: {file_name}")
        return

    os.makedirs(output_path, exist_ok=True)
    full_file_path = os.path.join(output_path, file_name)
    try:
        df.to_parquet(full_file_path, index=False)
        logger.info(f"Successfully saved data to {full_file_path}. Shape: {df.shape}")
        # The processed data has been successfully saved to the specified Parquet file.
        # This data is now versioned and ready for use in model training or further analysis.
        print(f"Data saved to {full_file_path}. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving data to {full_file_path}: {e}", exc_info=True)
        # An error occurred while trying to save the processed data.
        # The data could not be written to the disk.
        print(f"Failed to save data to {full_file_path}: {e}")


# This is the main function that orchestrates the feature engineering pipeline.
# It takes a date string as input to process data for a specific day.
# It loads data, computes features, and saves the results.
def run_feature_pipeline(date_to_process: str, pipeline_config: Dict[str, Any]) -> None:
    """
    Main function to run the feature engineering pipeline for a given date.

    Orchestrates loading data, calculating technical indicators, processing news,
    performing sentiment analysis, aggregating features, and saving results.

    Args:
        date_to_process (str): The date for which to process features (YYYY-MM-DD).
        pipeline_config (Dict[str, Any]): Configuration dictionary for the pipeline.
    """
    logger.info(f"Starting Strategy 1 feature engineering pipeline for date: {date_to_process}")

    # 1. Load raw OHLCV data
    # In a multi-ticker scenario, ohlcv_df might contain data for many tickers.
    # Technical indicators should be calculated per ticker.
    raw_ohlcv_df_all_tickers = load_ohlcv_data(date_to_process, pipeline_config)

    processed_ohlcv_dfs = []
    if raw_ohlcv_df_all_tickers is not None and not raw_ohlcv_df_all_tickers.empty:
        if 'ticker' in raw_ohlcv_df_all_tickers.columns:
            for ticker, group_df in raw_ohlcv_df_all_tickers.groupby('ticker'):
                logger.info(f"Calculating technical indicators for ticker: {ticker}...")
                # Ensure data is sorted by date if not already
                group_df = group_df.sort_values(by='date') if 'date' in group_df.columns else group_df
                tech_indicators_df_ticker = calculate_technical_indicators(
                    group_df, **pipeline_config["technical_indicators_params"]
                )
                if tech_indicators_df_ticker is not None and not tech_indicators_df_ticker.empty:
                    processed_ohlcv_dfs.append(tech_indicators_df_ticker)
            
            if processed_ohlcv_dfs:
                final_tech_indicators_df = pd.concat(processed_ohlcv_dfs, ignore_index=True)
            else:
                final_tech_indicators_df = pd.DataFrame() # Empty if no ticker data processed
        else: # Assume single ticker or already aggregated data
            logger.info("Calculating technical indicators for the loaded OHLCV data (assumed single series or pre-aggregated)...")
            final_tech_indicators_df = calculate_technical_indicators(
                raw_ohlcv_df_all_tickers, **pipeline_config["technical_indicators_params"]
            )
    else:
        logger.warning("No OHLCV data loaded. Skipping technical indicator calculation.")
        final_tech_indicators_df = pd.DataFrame()


    # 2. Load raw news data
    raw_news_items = load_raw_news_data(date_to_process, pipeline_config)
    processed_news_with_sentiment_list = []

    if raw_news_items:
        # 3. Initialize Text Preprocessor
        text_preprocessor = TextPreprocessor(language=pipeline_config["text_preprocessor_lang"])

        # 4. Initialize Sentiment Analyzer Service
        # This assumes the model defined by `sentiment_model_name` is available.
        # For actual FinGPT, this would involve loading ChatGLM2 + LoRA adapters.
        try:
            sentiment_model_config_path = pipeline_config.get("sentiment_model_config_path")
            sentiment_model_config = {}
            if sentiment_model_config_path and os.path.exists(sentiment_model_config_path):
                with open(sentiment_model_config_path, 'r') as f:
                    sentiment_model_config = yaml.safe_load(f)
                logger.info(f"Loaded sentiment model configuration from: {sentiment_model_config_path}")
            else:
                logger.warning(f"Sentiment model config path not provided or file not found: {sentiment_model_config_path}. Using defaults or pipeline_config values.")

            # Determine base model path: from YAML, then from pipeline_config, then default.
            base_model_path = sentiment_model_config.get("base_model_name_or_path", pipeline_config.get("sentiment_model_name", DEFAULT_MODEL_NAME))
            
            # Determine tokenizer path: from YAML, then base_model_path.
            tokenizer_path = sentiment_model_config.get("tokenizer_name_or_path", base_model_path)

            # Determine LoRA adapter path: priority to pipeline_config (from orchestrator/args), then YAML, then None.
            lora_path_from_pipeline_config = pipeline_config.get("lora_adapter_path")
            lora_path_from_yaml = sentiment_model_config.get("peft_model_path")
            
            final_lora_path = lora_path_from_pipeline_config if lora_path_from_pipeline_config is not None else lora_path_from_yaml

            # Get quantization settings from YAML, with defaults if not present
            quant_config = sentiment_model_config.get("quantization_config", {})
            load_4bit = quant_config.get("load_in_4bit", False) # Default to False if not specified
            bnb_4bit_quant = quant_config.get("bnb_4bit_quant_type", "nf4")
            bnb_4bit_double_quant = quant_config.get("bnb_4bit_use_double_quant", True)
            # Service expects string like "torch.float16", YAML might have "torch.float16" or just "float16"
            bnb_4bit_compute_dtype = quant_config.get("bnb_4bit_compute_dtype", "torch.float16")
            if not bnb_4bit_compute_dtype.startswith("torch."):
                 bnb_4bit_compute_dtype = f"torch.{bnb_4bit_compute_dtype}"


            # Get instruction template from YAML, with service default if not present
            instruction_template = sentiment_model_config.get("instruction_template", None) # Let service use its default if None

            logger.info(f"Initializing sentiment analyzer with: "
                        f"Base Model: {base_model_path}, "
                        f"Tokenizer: {tokenizer_path}, "
                        f"LoRA Adapter: {final_lora_path}, "
                        f"Load in 4-bit: {load_4bit}")

            service_args = {
                "base_model_name_or_path": base_model_path,
                "lora_adapter_path": final_lora_path,
                "tokenizer_name_or_path": tokenizer_path,
                "load_in_4bit": load_4bit,
                "bnb_4bit_quant_type": bnb_4bit_quant,
                "bnb_4bit_use_double_quant": bnb_4bit_double_quant,
                "bnb_4bit_compute_dtype_str": bnb_4bit_compute_dtype
            }
            if instruction_template: # Only pass if explicitly defined in config
                service_args["instruction_template"] = instruction_template
            
            sentiment_analyzer = FinGPTSentimentAnalyzerService(**service_args)

        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analyzer Service: {e}. Proceeding without sentiment.", exc_info=True)
            sentiment_analyzer = None # Allow pipeline to continue if sentiment model fails

        # Process news items
        for news_item in raw_news_items:
            # Ensure 'headline' or 'summary' or 'content' exists for processing
            text_to_process = news_item.get('headline', '') or news_item.get('summary', '') or news_item.get('content', '')
            if not text_to_process:
                logger.warning(f"News item (ID: {news_item.get('id', 'N/A')}) has no processable text. Skipping.")
                processed_text = ""
                sentiment_label = "n/a"
                sentiment_score = 0.0
            else:
                # 5. Preprocess news text
                processed_text = text_preprocessor.process_text(text_to_process)
                
                # 6. Get sentiment scores
                if sentiment_analyzer:
                    sentiment_result = sentiment_analyzer.predict_sentiment([processed_text])
                    if sentiment_result:
                        sentiment_label = sentiment_result[0]['label']
                        sentiment_score = sentiment_result[0]['score']
                    else:
                        sentiment_label = "error"
                        sentiment_score = 0.0
                else:
                    sentiment_label = "unavailable"
                    sentiment_score = 0.0

            news_item_copy = news_item.copy() # Avoid modifying original list of dicts
            news_item_copy['processed_text'] = processed_text
            news_item_copy['sentiment_label'] = sentiment_label
            news_item_copy['sentiment_score'] = sentiment_score
            processed_news_with_sentiment_list.append(news_item_copy)
        
        processed_news_df = pd.DataFrame(processed_news_with_sentiment_list)
    else:
        logger.warning("No news data loaded. Skipping news preprocessing and sentiment analysis.")
        processed_news_df = pd.DataFrame()

    # 7. Aggregate features (Conceptual - depends on strategy needs)
    # This step would combine technical_indicators_df and processed_news_df.
    # For example, if news is per ticker, merge on date and ticker.
    # If news is general market news, sentiment might be an aggregated daily score.
    # For this example, we'll save them separately and mention aggregation.
    logger.info("Feature aggregation step (conceptual): For now, saving indicators and news sentiment separately.")
    
    # Example: Aggregate daily sentiment per market/index (if applicable)
    # This is highly dependent on how news items are tagged (e.g., with tickers or market indices)
    # For now, we assume processed_news_df contains individual news sentiments.
    # An aggregation could be:
    # daily_sentiment_summary = processed_news_df.groupby('date')['sentiment_score'].mean().reset_index()
    # This daily_sentiment_summary could then be merged with daily technical indicators.

    # For this example, we'll just ensure the dataframes have a date column for potential merging.
    if 'date' not in final_tech_indicators_df.columns and not final_tech_indicators_df.empty:
         # If OHLCV data was per-day and didn't have a date column after processing (e.g. index was date)
        final_tech_indicators_df['date'] = pd.to_datetime(date_to_process)

    if 'date' not in processed_news_df.columns and not processed_news_df.empty:
        # News items usually have a date, but if not, assign the processing date
        processed_news_df['date'] = pd.to_datetime(date_to_process)

    # A simple merge example if both have 'date' and 'ticker' (if applicable)
    # This is a placeholder for a more sophisticated aggregation strategy.
    aggregated_features_df = pd.DataFrame()
    if not final_tech_indicators_df.empty and not processed_news_df.empty:
        # If news can be mapped to tickers (e.g., news_item has a 'ticker' field)
        if 'ticker' in final_tech_indicators_df.columns and 'ticker' in processed_news_df.columns:
            try:
                # Ensure date columns are of the same type for merging
                final_tech_indicators_df['date'] = pd.to_datetime(final_tech_indicators_df['date'])
                processed_news_df['date'] = pd.to_datetime(processed_news_df['date'])
                
                # Aggregate news sentiment per ticker per day (e.g., mean score)
                # This assumes one news item per row. If multiple news per ticker per day, aggregate first.
                # Example: news_sentiment_per_ticker_day = processed_news_df.groupby(['date', 'ticker'])['sentiment_score'].mean().reset_index()
                # For simplicity, let's assume processed_news_df is already suitable for a direct merge or needs specific logic.
                # This merge is highly speculative without knowing exact schemas.
                # aggregated_features_df = pd.merge(final_tech_indicators_df, news_sentiment_per_ticker_day, on=['date', 'ticker'], how='left')
                logger.info("Conceptual merge: Aggregated features would be created here if schemas align for merging.")
                # For now, we'll just set aggregated_features_df to tech indicators to have something.
                aggregated_features_df = final_tech_indicators_df.copy()
                # And add a general daily sentiment if available (e.g. mean of all news for the day)
                if not processed_news_df.empty:
                    aggregated_features_df['daily_avg_sentiment_score'] = processed_news_df['sentiment_score'].mean()

            except Exception as e:
                logger.error(f"Error during conceptual feature aggregation merge: {e}", exc_info=True)
                aggregated_features_df = final_tech_indicators_df # Fallback
        else: # If no common ticker, perhaps add general market sentiment to all rows
            if not processed_news_df.empty:
                daily_avg_sentiment = processed_news_df['sentiment_score'].mean()
                final_tech_indicators_df['daily_avg_sentiment_score'] = daily_avg_sentiment
            aggregated_features_df = final_tech_indicators_df.copy()
            logger.info("Aggregated features: Added daily average sentiment to technical indicators.")

    elif not final_tech_indicators_df.empty:
        aggregated_features_df = final_tech_indicators_df.copy()
    elif not processed_news_df.empty:
        # If only news features, this would be the aggregated_features_df
        # This case might need more thought based on strategy (e.g. sentiment only strategy?)
        aggregated_features_df = processed_news_df.copy()


    # 8. Save engineered features
    output_dir_for_date = os.path.join(pipeline_config["base_processed_data_path"], date_to_process)
    os.makedirs(output_dir_for_date, exist_ok=True)

    save_processed_data(final_tech_indicators_df, output_dir_for_date, pipeline_config["output_file_name_tech_indicators"])
    save_processed_data(processed_news_df, output_dir_for_date, pipeline_config["output_file_name_processed_news"])
    
    if not aggregated_features_df.empty:
         save_processed_data(aggregated_features_df, output_dir_for_date, pipeline_config["output_file_name_aggregated_features"])
    else:
        logger.warning("Aggregated features DataFrame is empty. Nothing to save for aggregated_features.")


    # The feature engineering pipeline for the given date has completed.
    # Processed technical indicators and news sentiment scores have been generated and saved.
    print(f"Feature engineering pipeline finished for date: {date_to_process}")
    logger.info(f"Strategy 1 feature engineering pipeline completed successfully for date: {date_to_process}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Strategy 1 Feature Engineering Pipeline.")
    parser.add_argument(
        "--date_to_process",
        type=str,
        default=(datetime.now() - pd.Timedelta(days=1)).strftime(CONFIG["date_format"]),
        help="The date to process features for, in YYYY-MM-DD format. Defaults to yesterday."
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None, # Default to None, so it's only used if provided
        help="Path to the LoRA adapter for the sentiment model."
    )
    parser.add_argument(
        "--sentiment_model_config_path",
        type=str,
        default=os.path.join(project_root, "MLOps", "config", "sentiment_models", "llama3_8b_lora_params.yaml"), # Default to the one I modified
        help="Path to the YAML configuration file for the sentiment model."
    )
    # Add more arguments here if needed, e.g., paths to specific config files

    args = parser.parse_args()

    # Update the main CONFIG with any command-line arguments if necessary,
    # or pass them separately. For lora_adapter_path, we'll add it to a copy.
    current_pipeline_config = CONFIG.copy()
    if args.lora_adapter_path:
        current_pipeline_config["lora_adapter_path"] = args.lora_adapter_path
        logger.info(f"Using LoRA adapter path from command line: {args.lora_adapter_path}")
    # No 'else' here, if not provided by CLI, it will be picked up from sentiment_model_config_path YAML or be None

    if args.sentiment_model_config_path:
        current_pipeline_config["sentiment_model_config_path"] = args.sentiment_model_config_path
        logger.info(f"Using sentiment model config path from command line: {args.sentiment_model_config_path}")
    else:
        logger.info(f"No sentiment model config path provided via command line. Using default: {current_pipeline_config.get('sentiment_model_config_path')}")
        # Ensure it's in config if not from args, using the default from parser
        if "sentiment_model_config_path" not in current_pipeline_config:
             current_pipeline_config["sentiment_model_config_path"] = os.path.join(project_root, "MLOps", "config", "sentiment_models", "llama3_8b_lora_params.yaml")


    # Create dummy raw data directories and files for the example date if running directly
    # This part might be skipped if the orchestrator ensures data exists
    # For direct execution testing:
    example_date_str = args.date_to_process
    
    # Check if data needs to be created (e.g. if not run by orchestrator which prepares data)
    # This dummy data creation might be removed if orchestrator always provides data
    dummy_ohlcv_dir = os.path.join(
        CONFIG["base_raw_data_path"],
        CONFIG["ohlcv_source_dir_template"].format(date_str=example_date_str)
    )
    dummy_news_dir = os.path.join(
        CONFIG["base_raw_data_path"],
        CONFIG["news_source_dir_template"].format(date_str=example_date_str)
    )
    
    if not os.path.exists(dummy_ohlcv_dir) or not os.listdir(dummy_ohlcv_dir):
        os.makedirs(dummy_ohlcv_dir, exist_ok=True)
        logger.info(f"Dummy OHLCV directory created/empty at {dummy_ohlcv_dir}, load_ohlcv_data will generate dummy data.")

    if not os.path.exists(dummy_news_dir) or not os.listdir(dummy_news_dir):
        os.makedirs(dummy_news_dir, exist_ok=True)
        dummy_news_content = [
            {"id": "dummy_news1", "ticker": "AAPL", "date": example_date_str, "headline": "AAPL Positive News from Example", "summary": "Stock is doing great."},
            {"id": "dummy_news2", "ticker": "MSFT", "date": example_date_str, "headline": "MSFT Negative Outlook from Example", "summary": "Concerns about future performance."}
        ]
        with open(os.path.join(dummy_news_dir, "dummy_news_data.json"), "w") as f:
            json.dump(dummy_news_content, f)
        logger.info(f"Created dummy news data in {dummy_news_dir}")


    logger.info(f"Running feature engineering pipeline for date: {args.date_to_process}")
    try:
        run_feature_pipeline(args.date_to_process, current_pipeline_config)
        logger.info(f"Pipeline run completed for {args.date_to_process}. Check logs and data/processed/features/strategy_1/{args.date_to_process}/")
    except Exception as e:
        logger.critical(f"Pipeline run failed for {args.date_to_process}: {e}", exc_info=True)
        # The main pipeline execution encountered a critical error.
        # This prevented the successful completion of the feature engineering process.
        print(f"Critical error in pipeline execution: {e}")