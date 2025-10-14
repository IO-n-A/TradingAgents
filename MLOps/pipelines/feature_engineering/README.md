# MLOps Feature Engineering Pipeline

This directory contains the scripts and configurations for the feature engineering pipeline of the FinAI_algo project. This pipeline is responsible for transforming raw data (financial and news) into a feature-rich dataset suitable for training sentiment models and Reinforcement Learning (RL) agents.

## Files

*   **`__init__.py`**: (If this directory becomes a sub-package, though not explicitly defined as such in the initial plan, it's good practice if other modules might import from it directly).
*   **`build_features.py`**:
    *   The main script that orchestrates the feature engineering process.
    *   **Functionality:**
        1.  **Loads Configurations:** Reads settings from:
            *   `MLOps/config/common/global_vars.yaml` (for data paths).
            *   `MLOps/config/environments/stock_trading_env_config.yaml` (for the list of technical indicators to generate).
            *   `MLOps/config/sentiment_models/llama3_8b_lora_params.yaml` (for paths to sentiment model components if needed by `SentimentAnalyzerService`).
        2.  **Loads Raw Data:**
            *   Reads raw financial data (e.g., from `data/raw/raw_financial_data.csv`).
            *   Reads raw news data (e.g., from `data/raw/raw_news_data.csv`).
            *   These files are assumed to be tracked by DVC and produced by the data ingestion pipeline.
        3.  **Preprocesses Text Data:**
            *   Utilizes `SentimentTextPreprocessor` (from FinRL or a custom implementation) to clean and prepare news text for sentiment analysis.
        4.  **Generates Sentiment Scores:**
            *   Uses `SentimentAnalyzerService` (to be developed, as per Phase 3.1 of `analysis/coding-strategy.md`) to predict sentiment scores for the preprocessed news text. This service will load a trained sentiment model (e.g., a fine-tuned LLM with LoRA).
            *   Aggregates sentiment scores (e.g., daily average per ticker).
        5.  **Merges Data:**
            *   Combines the aggregated sentiment scores with the financial data, aligning by date and ticker.
        6.  **Generates Financial Features:**
            *   Applies FinRL's `FeatureEngineer` or custom functions to calculate technical indicators (e.g., MACD, RSI, Bollinger Bands) and potentially other financial features (e.g., turbulence index, lagged returns).
        7.  **Saves Processed Data:**
            *   Outputs the final feature-engineered DataFrame to a CSV file (e.g., `data/processed/processed_data_with_features.csv`).
        8.  **Data Versioning (Conceptual):**
            *   Conceptually invokes `dvc add` to track the processed data file with DVC.
    *   **Execution:**
        ```bash
        python MLOps/pipelines/feature_engineering/build_features.py \
          --global_config_path MLOps/config/common/global_vars.yaml \
          --env_config_path MLOps/config/environments/stock_trading_env_config.yaml \
          --sentiment_model_params_path MLOps/config/sentiment_models/llama3_8b_lora_params.yaml \
          --raw_financial_data_filename raw_financial_data.csv \
          --raw_news_data_filename raw_news_data.csv \
          --processed_data_filename processed_data_with_features.csv
        ```

## Key Features Generated

*   **Sentiment Scores:** Numerical representation of sentiment derived from news data.
*   **Technical Indicators:** Standard financial market indicators (MACD, RSI, CCI, SMA, Bollinger Bands, etc.).
*   **Lagged Features:** Past values of prices, volumes, or returns (if configured).
*   **Turbulence Index:** (If implemented and configured) A measure of market volatility.
*   Other custom features relevant to the trading strategy.

## Output

*   A single CSV file (e.g., `data/processed/processed_data_with_features.csv`) containing:
    *   Original financial data (OHLCV).
    *   Aligned sentiment scores.
    *   Calculated technical indicators and other engineered features.
    *   Columns typically include: `date`, `tic`, `open`, `high`, `low`, `close`, `volume`, `sentiment_score`, `macd`, `rsi_30`, etc.

## Dependencies

*   Pandas, NumPy
*   PyYAML
*   FinRL (for `FeatureEngineer`, `SentimentTextPreprocessor`, and potentially `SentimentAnalyzerService` components)
*   TA-Lib (often a dependency for FinRL's `FeatureEngineer` or for custom technical indicator calculation)
*   Hugging Face Transformers, PEFT, bitsandbytes (dependencies for the `SentimentAnalyzerService` when using LLMs)
*   DVC (command-line tool, for versioning)

## Future Enhancements

*   More sophisticated text preprocessing techniques.
*   Advanced sentiment analysis models and aspect-based sentiment.
*   A wider range of configurable technical indicators and financial features.
*   Feature selection and dimensionality reduction steps.
*   Robust handling of missing data during feature calculation.
*   Metrics exposure for Prometheus monitoring (e.g., processing time, number of features generated).