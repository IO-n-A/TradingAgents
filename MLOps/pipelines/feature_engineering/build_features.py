"""
MLOps Pipeline: Feature Engineering and Final Processing

This script performs data processing on raw data, including:
- Sentiment analysis on news data using FinGPT models.
- Merging sentiment scores with financial market data.
- Feature engineering using FinRL's FeatureEngineer (technical indicators, VIX, turbulence).
- Final data cleaning and preparation for RL agent training.
"""

import os
import sys
import yaml
import pandas as pd
import logging
import logging.config # For fileConfig
from datetime import datetime
import subprocess
# import mlflow # Keep for now, will use placeholder
# import mlflow.entities # For RunStatus
import time

# Ensure the main project directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from FinRobot.finrl.meta.preprocessor.preprocessors import FeatureEngineer
from FinRobot.finrl.config import INDICATORS as DEFAULT_INDICATORS # Assuming FinRobot.finrl.config exists
from FinRobot.finrl.meta.data_processor import DataProcessor
from FinRobot.finrl.meta.preprocessor.sentiment_text_preprocessor import SentimentTextPreprocessor
from FinRobot.finrl.sentiment_analyzer_service import SentimentAnalyzerService


# MLOps utilities (placeholder for now)
class MLflowUtilsPlaceholder:
    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri
        # Removed logger call from __init__ as logger might not be configured yet when placeholder is defined.
        # logger.info(f"MLflowUtilsPlaceholder: Initialized. Tracking URI: {self.tracking_uri} (Note: Actual MLflow logging is not implemented in placeholder).")

    def start_mlflow_run(self, experiment_name, run_name):
        logger.info(f"MLP: Start run '{run_name}' in experiment '{experiment_name}'.")
        class MockRun:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass # __exit__ should generally return None or a bool to suppress exceptions
        return MockRun()

    def log_params(self, params_dict):
        logger.info(f"MLP: Log params: {params_dict}")

    def log_metrics(self, metrics_dict, step=None):
        logger.info(f"MLP: Log metrics: {metrics_dict} at step {step if step else 'N/A'}")

    def log_artifact(self, local_path, artifact_path=None):
        logger.info(f"MLP: Log artifact: {local_path} to {artifact_path or ''}")

mlflow_utils = MLflowUtilsPlaceholder()

# Configure logging using the centralized configuration
try:
    logging_config_path = os.path.join(project_root, 'config', 'logging_config.py')
    if os.path.exists(logging_config_path):
        logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.warning(f"Logging config file not found at {logging_config_path}. Using basicConfig.")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.error(f"Error configuring logging from file: {e}. Using basicConfig.")

logger = logging.getLogger(__name__)


def load_pipeline_configs():
    """Loads all necessary configuration files for the feature engineering pipeline."""
    configs = {}
    try:
        configs['global_vars'] = load_config(os.path.join(project_root, "MLOps", "config", "common", "global_vars.yaml"))
        configs['feature_engineering'] = load_config(os.path.join(project_root, "MLOps", "config", "feature_engineering_config.yaml"))
        
        # Load the specific sentiment model config specified in feature_engineering_config.yaml
        sentiment_model_config_file = configs['feature_engineering'].get('sentiment_analysis', {}).get('sentiment_model_config_to_use')
        if sentiment_model_config_file:
            configs['sentiment_model_inference'] = load_config(os.path.join(project_root, "MLOps", "config", sentiment_model_config_file))
        else:
            configs['sentiment_model_inference'] = {} # Default if not specified
            logger.warning("sentiment_model_config_to_use not specified in feature_engineering_config. Sentiment analysis might use defaults or fail.")
        
        logger.info("All pipeline configurations loaded successfully.")
        return configs
    except Exception as e:
        logger.error(f"Error loading one or more pipeline configurations: {e}", exc_info=True)
        raise

def load_config(config_path: str) -> dict:
    """Loads a single YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise


def load_raw_data(global_vars_config: dict):
    """Loads raw stock data and news data based on paths in global_vars_config."""
    logger.info("Loading raw data...")
    paths_config = global_vars_config.get("paths", {})
    raw_data_dir = os.path.join(project_root, paths_config.get("raw_data_dir", "data/raw"))
    
    raw_stock_data_file = os.path.join(raw_data_dir, "raw_financial_data.csv")
    raw_news_data_file = os.path.join(raw_data_dir, "raw_news_data.csv")
    
    stock_df, news_df = None, None

    try:
        if not os.path.exists(raw_stock_data_file):
            logger.error(f"Raw stock data file not found: {raw_stock_data_file}")
        else:
            logger.info(f"Loading raw stock data from: {raw_stock_data_file}")
            stock_df = pd.read_csv(raw_stock_data_file)
            # Convert 'date' column to datetime objects if not already
            if 'date' in stock_df.columns:
                 stock_df['date'] = pd.to_datetime(stock_df['date'])
            logger.info(f"Successfully loaded stock data. Shape: {stock_df.shape}")
    except Exception as e:
        logger.error(f"Error loading raw stock data from {raw_stock_data_file}: {e}", exc_info=True)

    try:
        if not os.path.exists(raw_news_data_file):
            logger.warning(f"Raw news data file not found: {raw_news_data_file}. Proceeding without news data.")
        else:
            logger.info(f"Loading raw news data from: {raw_news_data_file}")
            news_df = pd.read_csv(raw_news_data_file)
            # Convert 'timestamp' to datetime objects
            if 'timestamp' in news_df.columns:
                news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            logger.info(f"Successfully loaded news data. Shape: {news_df.shape}")
    except Exception as e:
        logger.warning(f"Error loading raw news data from {raw_news_data_file}: {e}. Proceeding without news data.", exc_info=True)
        
    logger.info("Raw data loading attempt complete.")
    return stock_df, news_df


def analyze_sentiment_on_news(news_df: pd.DataFrame, sentiment_config: dict, fe_config: dict) -> pd.DataFrame:
    """Analyzes sentiment on news data and adds scores to the DataFrame."""
    if news_df is None or news_df.empty:
        logger.info("No news data to analyze sentiment for.")
        return news_df # Return original or None if it was None

    sa_config = fe_config.get('sentiment_analysis', {})
    if not sa_config.get('enabled', False):
        logger.info("Sentiment analysis is disabled in feature_engineering_config. Skipping.")
        return news_df

    logger.info("Starting sentiment analysis on news data...")
    
    # Get sentiment model parameters from the specific model's config file
    # (loaded as sentiment_config['sentiment_model_inference'])
    model_params = sentiment_config.get('general_config', {}) # From the loaded sentiment model's YAML
    lora_path = model_params.get('peft_model_to_load_for_inference') # Assuming this key exists for inference
    if not lora_path and 'lora_weights_path' in sentiment_config: # Fallback for older config
        lora_path = sentiment_config.get('lora_weights_path')


    # Ensure paths are absolute if they are relative and point to 'models/' directory
    base_model_path = model_params.get('base_model_name_or_path', 'default_base_model')
    if base_model_path.startswith("models/"):
        base_model_path = os.path.join(project_root, base_model_path)
    
    tokenizer_path = model_params.get('tokenizer_name', base_model_path)
    if tokenizer_path.startswith("models/"):
        tokenizer_path = os.path.join(project_root, tokenizer_path)

    if lora_path and lora_path.startswith("models/"):
         lora_path = os.path.join(project_root, lora_path)
    elif lora_path and not os.path.isabs(lora_path) and not lora_path.startswith("FinNLP"): # Check if it's a HF path
        # If it's a relative path not starting with models/ or FinNLP/, assume it's relative to project root
        # This might need adjustment based on where LoRA adapters are stored.
        # For now, let's assume it could be a direct path or a HF model hub ID.
        pass


    quant_config_dict = sentiment_config.get('hyperparameters', {}).get('quantization_config') # from sentiment model config
    if not quant_config_dict: # Fallback to fe_config if defined there (less ideal)
        quant_config_dict = sa_config.get('quantization_config')


    text_preprocessor = SentimentTextPreprocessor()
    sentiment_analyzer = SentimentAnalyzerService(
        model_name_or_path=base_model_path,
        lora_weights_path=lora_path,
        tokenizer_name_or_path=tokenizer_path,
        device=sa_config.get('device', 'auto'),
        bitsandbytes_config=quant_config_dict,
        torch_dtype_str=sa_config.get('torch_dtype_str', 'auto')
    )

    # Determine the text column to use for sentiment analysis
    text_column = 'full_text' if 'full_text' in news_df.columns and news_df['full_text'].notna().any() else 'summary'
    if text_column not in news_df.columns or news_df[text_column].isna().all():
        logger.warning(f"Neither 'full_text' nor 'summary' column found or usable in news_df. Skipping sentiment analysis.")
        return news_df
    
    logger.info(f"Using '{text_column}' column for sentiment analysis.")

    prompts = []
    instruction_template = sa_config.get('prompt_instruction_template', "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\nInput: {input}\nAnswer: ")
    
    for text_content in news_df[text_column]:
        if pd.isna(text_content) or not isinstance(text_content, str) or not text_content.strip():
            prompts.append(None) # Placeholder for empty/invalid text
            continue
        cleaned_text = text_preprocessor.clean_text(text_content)
        prompt = text_preprocessor.format_prompt(cleaned_text, instruction_template=instruction_template)
        prompts.append(prompt)

    # Filter out None prompts before batch prediction
    valid_prompts_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [p for p in prompts if p is not None]

    if not valid_prompts:
        logger.info("No valid text found in news data for sentiment analysis.")
        # Add empty sentiment columns if they don't exist
        for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_unknown']:
            if col not in news_df.columns: news_df[col] = 0.0 if col != 'sentiment_unknown' else 1.0
        return news_df

    sentiment_results = sentiment_analyzer.predict_sentiment_batch(
        valid_prompts,
        max_new_tokens=sa_config.get('prediction_max_new_tokens', 10)
    )

    # Initialize sentiment columns
    news_df['sentiment_positive'] = 0.0
    news_df['sentiment_negative'] = 0.0
    news_df['sentiment_neutral'] = 0.0
    news_df['sentiment_unknown'] = 1.0 # Default to unknown

    # Populate sentiment scores for valid prompts
    for i, result_idx in enumerate(valid_prompts_indices):
        scores = sentiment_results[i]
        news_df.loc[result_idx, 'sentiment_positive'] = scores.get('positive', 0.0)
        news_df.loc[result_idx, 'sentiment_negative'] = scores.get('negative', 0.0)
        news_df.loc[result_idx, 'sentiment_neutral'] = scores.get('neutral', 0.0)
        news_df.loc[result_idx, 'sentiment_unknown'] = scores.get('unknown', 0.0)
        if scores.get('unknown', 0.0) < 1.0: # if known sentiment
            news_df.loc[result_idx, 'sentiment_unknown'] = 0.0


    logger.info("Sentiment analysis complete. Scores added to news DataFrame.")
    return news_df


def add_technical_and_market_features(stock_df: pd.DataFrame, fe_config_params: dict) -> pd.DataFrame:
    """Adds technical indicators, VIX, and turbulence features to stock_df."""
    if stock_df is None or stock_df.empty:
        logger.warning("Stock data is None or empty, skipping technical/market feature engineering.")
        return stock_df
        
    logger.info("Adding technical indicators and market features (VIX, turbulence)...")
    
    tech_indicator_list = fe_config_params.get("tech_indicator_list", DEFAULT_INDICATORS)
    use_vix = fe_config_params.get("use_vix", True) # Default to True as per FinRL common practice
    use_turbulence = fe_config_params.get("use_turbulence", True) # Default to True

    # DataProcessor is used here for its existing methods to add VIX and turbulence
    # It requires a data_source for some internal logic, even if not downloading.
    dp = DataProcessor(data_source="yahoofinance") # Placeholder data_source

    # Add technical indicators using DataProcessor's method
    try:
        stock_df_with_tech = dp.add_technical_indicator(stock_df.copy(), tech_indicator_list)
        logger.info(f"Added technical indicators: {tech_indicator_list}")
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}", exc_info=True)
        stock_df_with_tech = stock_df.copy() # Proceed with original if tech ind. fail

    # Add VIX using DataProcessor
    if use_vix:
        try:
            stock_df_with_tech_vix = dp.add_vix(stock_df_with_tech)
            logger.info("Added VIX data.")
        except Exception as e:
            logger.error(f"Error adding VIX data: {e}. Proceeding without VIX.", exc_info=True)
            stock_df_with_tech_vix = stock_df_with_tech # Proceed without VIX
    else:
        stock_df_with_tech_vix = stock_df_with_tech
        logger.info("Skipping VIX data addition as per configuration.")

    # Add turbulence index using DataProcessor
    if use_turbulence:
        try:
            # Ensure 'date' column is in datetime format if not already
            if 'date' in stock_df_with_tech_vix.columns and not pd.api.types.is_datetime64_any_dtype(stock_df_with_tech_vix['date']):
                 stock_df_with_tech_vix['date'] = pd.to_datetime(stock_df_with_tech_vix['date'])

            final_df_with_turbulence = dp.add_turbulence(stock_df_with_tech_vix)
            logger.info("Added turbulence index.")
        except Exception as e:
            logger.error(f"Error adding turbulence index: {e}. Proceeding without turbulence.", exc_info=True)
            final_df_with_turbulence = stock_df_with_tech_vix # Proceed without turbulence
    else:
        final_df_with_turbulence = stock_df_with_tech_vix
        logger.info("Skipping turbulence index addition as per configuration.")
        
    return final_df_with_turbulence


def merge_sentiment_with_stock_data(stock_df_with_features: pd.DataFrame, news_df_with_sentiment: pd.DataFrame, fe_config: dict) -> pd.DataFrame:
    """Merges aggregated sentiment scores with the stock data."""
    if news_df_with_sentiment is None or news_df_with_sentiment.empty or 'sentiment_positive' not in news_df_with_sentiment.columns:
        logger.info("No sentiment data to merge, or sentiment columns not found. Returning stock data as is.")
        # Ensure standard sentiment columns exist even if no news, filled with neutral/unknown
        for col in ['daily_sentiment_positive_mean', 'daily_sentiment_negative_mean', 'daily_sentiment_neutral_mean', 'daily_sentiment_unknown_mean', 'daily_news_count']:
            if col not in stock_df_with_features.columns:
                stock_df_with_features[col] = 0.0 if 'count' not in col else 0
        return stock_df_with_features

    logger.info("Merging sentiment scores with stock data...")
    
    # Ensure 'date' column for merging
    if 'timestamp' in news_df_with_sentiment.columns:
        news_df_with_sentiment['date'] = pd.to_datetime(news_df_with_sentiment['timestamp']).dt.date
    elif 'date' in news_df_with_sentiment.columns:
        news_df_with_sentiment['date'] = pd.to_datetime(news_df_with_sentiment['date']).dt.date
    else:
        logger.error("News data requires a 'timestamp' or 'date' column for merging.")
        return stock_df_with_features # Return original if no date column

    if 'date' in stock_df_with_features.columns:
         stock_df_with_features['date'] = pd.to_datetime(stock_df_with_features['date']).dt.date
    else:
        logger.error("Stock data requires a 'date' column for merging.")
        return stock_df_with_features


    # Aggregate sentiment scores per day and ticker
    aggregation_strategy = fe_config.get('sentiment_analysis', {}).get('sentiment_aggregation_strategy', 'mean_probs')
    
    agg_functions = {}
    if aggregation_strategy == 'mean_probs':
        agg_functions = {
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'sentiment_unknown': 'mean',
            'headline': 'count' # Count of news articles
        }
    # Add other strategies if needed (e.g., 'majority_vote_label', 'sum_scores')
    else:
        logger.warning(f"Unsupported sentiment_aggregation_strategy: {aggregation_strategy}. Defaulting to mean probabilities.")
        agg_functions = {
            'sentiment_positive': 'mean', 'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean', 'sentiment_unknown': 'mean', 'headline': 'count'
        }

    required_sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_unknown']
    if not all(col in news_df_with_sentiment.columns for col in required_sentiment_cols):
        logger.error(f"News DataFrame is missing one or more required sentiment columns: {required_sentiment_cols}")
        return stock_df_with_features

    # Ensure 'tic' column exists for grouping
    if 'tic' not in news_df_with_sentiment.columns:
        logger.error("News DataFrame is missing 'tic' column for grouping.")
        # Attempt to merge without 'tic' if only one ticker is present in stock_df
        if 'tic' in stock_df_with_features.columns and stock_df_with_features['tic'].nunique() == 1:
            logger.info("Attempting to merge sentiment by date only as single ticker detected in stock_df.")
            daily_sentiment = news_df_with_sentiment.groupby('date').agg(agg_functions).reset_index()
            merge_on_cols = ['date']
        else: # Cannot reliably merge
            return stock_df_with_features
    else:
        daily_sentiment = news_df_with_sentiment.groupby(['date', 'tic']).agg(agg_functions).reset_index()
        merge_on_cols = ['date', 'tic']


    daily_sentiment.rename(columns={
        'sentiment_positive': 'daily_sentiment_positive_mean',
        'sentiment_negative': 'daily_sentiment_negative_mean',
        'sentiment_neutral': 'daily_sentiment_neutral_mean',
        'sentiment_unknown': 'daily_sentiment_unknown_mean',
        'headline': 'daily_news_count'
    }, inplace=True)

    # Merge with stock data
    final_df = pd.merge(stock_df_with_features, daily_sentiment, on=merge_on_cols, how='left')
    
    # Fill NaNs for sentiment columns (e.g., days with no news)
    sentiment_cols_to_fill = [
        'daily_sentiment_positive_mean', 'daily_sentiment_negative_mean',
        'daily_sentiment_neutral_mean', 'daily_sentiment_unknown_mean'
    ]
    for col in sentiment_cols_to_fill:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0.0) # Assuming 0 for missing sentiment scores
    if 'daily_news_count' in final_df.columns:
        final_df['daily_news_count'] = final_df['daily_news_count'].fillna(0).astype(int)
    
    # Fill 'daily_sentiment_unknown_mean' with 1.0 if other sentiments are 0 and news_count is 0
    # to signify no news rather than neutral news.
    if 'daily_sentiment_unknown_mean' in final_df.columns and 'daily_news_count' in final_df.columns:
        mask_no_news = final_df['daily_news_count'] == 0
        final_df.loc[mask_no_news, 'daily_sentiment_unknown_mean'] = 1.0
        for sent_col in ['daily_sentiment_positive_mean', 'daily_sentiment_negative_mean', 'daily_sentiment_neutral_mean']:
            if sent_col in final_df.columns:
                final_df.loc[mask_no_news, sent_col] = 0.0


    logger.info("Sentiment data merged with stock data.")
    return final_df


def save_processed_data(final_df: pd.DataFrame, global_vars_config: dict, fe_config: dict) -> str:
    """Saves the processed data and returns the .dvc file path if DVC is used."""
    if final_df is None or final_df.empty:
        logger.warning("Final dataframe is None or empty, skipping save.")
        return None

    paths_config = global_vars_config.get("paths", {})
    processed_data_dir_rel = paths_config.get("processed_data_dir", "data/processed")
    processed_data_dir_abs = os.path.join(project_root, processed_data_dir_rel)
    os.makedirs(processed_data_dir_abs, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"processed_market_sentiment_data_{timestamp}.csv"
    save_path = os.path.join(processed_data_dir_abs, file_name)
    dvc_file_path_relative_to_repo_root = None
    
    try:
        final_df.to_csv(save_path, index=False)
        logger.info(f"Processed data saved to {save_path}")

        if fe_config.get("dvc_tracking", {}).get("enabled", False):
            try:
                dvc_add_target = processed_data_dir_rel # Track the directory
                logger.info(f"Attempting to run 'dvc add {dvc_add_target}' to track changes.")
                result = subprocess.run(["dvc", "add", dvc_add_target], check=True, capture_output=True, text=True, cwd=project_root)
                logger.info(f"Successfully ran 'dvc add {dvc_add_target}'. Output: {result.stdout.strip()}")
                
                # Construct the .dvc file path relative to the repository root
                dvc_file_path_relative_to_repo_root = dvc_add_target.strip(os.sep) + ".dvc"
                if not os.path.exists(os.path.join(project_root, dvc_file_path_relative_to_repo_root)):
                    logger.warning(f"Could not find DVC file at {dvc_file_path_relative_to_repo_root} after 'dvc add'. DVC file logging might be incorrect.")
                    dvc_file_path_relative_to_repo_root = None # Reset if not found
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to run 'dvc add {dvc_add_target}'. Error: {e.stderr.strip()}")
            except FileNotFoundError:
                logger.error("DVC command not found. Ensure DVC is installed and in your system's PATH.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}", exc_info=True)
    return dvc_file_path_relative_to_repo_root


def main():
    """Main function to run the feature engineering pipeline."""
    pipeline_start_time = time.time()
    run_name = f"FeatureEngineering_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow_utils.start_mlflow_run(experiment_name="FinAI_FeatureEngineering", run_name=run_name) as run:
        mlflow_utils.log_params({"pipeline_run_name": run_name})
        logger.info(f"Starting MLOps Feature Engineering Pipeline (MLflow Run ID: {run.info.run_id if run else 'N/A'})...")
        
        try:
            configs = load_pipeline_configs()
            global_vars_config = configs['global_vars']
            fe_config = configs['feature_engineering']
            sentiment_model_config_for_inference = configs['sentiment_model_inference']

            mlflow_utils.log_params({"global_vars_config": global_vars_config})
            mlflow_utils.log_params({"feature_engineering_config": fe_config})
            mlflow_utils.log_params({"sentiment_model_inference_config_used": sentiment_model_config_for_inference})

            # --- Load Raw Data ---
            stock_df, news_df = load_raw_data(global_vars_config)
            if stock_df is None or stock_df.empty:
                raise ValueError("Raw stock data could not be loaded or is empty. Cannot proceed.")

            # --- Analyze Sentiment on News ---
            news_df_with_sentiment = analyze_sentiment_on_news(news_df, sentiment_model_config_for_inference, fe_config)

            # --- Add Technical and Market Features to Stock Data ---
            stock_df_with_tech_market_features = add_technical_and_market_features(
                stock_df,
                fe_config.get("feature_engineer_params", {})
            )
            if stock_df_with_tech_market_features is None or stock_df_with_tech_market_features.empty:
                raise ValueError("Stock data processing (technical/market features) resulted in empty data.")

            # --- Merge Sentiment with Stock Data ---
            final_df = merge_sentiment_with_stock_data(
                stock_df_with_tech_market_features,
                news_df_with_sentiment,
                fe_config
            )
            if final_df is None or final_df.empty:
                raise ValueError("Final data merging resulted in empty data.")

            # --- Clean final DataFrame using DataProcessor ---
            # This step is crucial as FinRL environments expect data processed by DataProcessor.df_to_array
            dp_for_final_clean = DataProcessor(data_source="yahoofinance") # Or actual source
            final_df_cleaned = dp_for_final_clean.clean_data(final_df.copy())
            # Note: dp.df_to_array() is usually called just before passing to the environment.
            # Here we save the cleaned DataFrame.
            
            logger.info(f"Final processed DataFrame shape: {final_df_cleaned.shape}")
            mlflow_utils.log_metrics({"final_df_rows": final_df_cleaned.shape[0], "final_df_cols": final_df_cleaned.shape[1]})
            if 'tic' in final_df_cleaned.columns:
                 mlflow_utils.log_metrics({"final_df_unique_tickers": final_df_cleaned['tic'].nunique()})


            # --- Save Processed Data & Log DVC artifact ---
            processed_dvc_file_rel_path = save_processed_data(final_df_cleaned, global_vars_config, fe_config)
            if processed_dvc_file_rel_path:
                # Construct full path for MLflow artifact logging if needed, or log relative path
                full_dvc_file_path = os.path.join(project_root, processed_dvc_file_rel_path)
                if os.path.exists(full_dvc_file_path):
                    mlflow_utils.log_artifact(full_dvc_file_path, artifact_path="dvc_metadata")
                    logger.info(f"Logged processed data DVC metadata: {full_dvc_file_path}")
                else:
                    logger.warning(f"DVC file {full_dvc_file_path} not found for MLflow logging, but DVC add might have succeeded for the directory.")
            
            total_pipeline_duration = time.time() - pipeline_start_time
            mlflow_utils.log_metrics({"total_pipeline_duration_sec": round(total_pipeline_duration, 2)})
            # mlflow.set_tag("status", "COMPLETED") # Handled by placeholder
            logger.info("MLOps Feature Engineering Pipeline completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred in the feature engineering pipeline: {e}", exc_info=True)
            # mlflow.set_tag("status", "FAILED") # Handled by placeholder
            # mlflow.set_tag("error_type", e.__class__.__name__)
            # mlflow.set_tag("error_message", str(e))
            sys.exit(1)
        # finally: # Ensure MLflow run ends
            # current_run = mlflow.active_run() # Handled by placeholder
            # if current_run: mlflow.end_run()


if __name__ == "__main__":
    main()