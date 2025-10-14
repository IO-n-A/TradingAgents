# MLOps Data Ingestion Pipeline

This directory contains the scripts and configurations for the data ingestion pipeline of the FinAI_algo project. The primary goal of this pipeline is to fetch raw financial data and news data from various sources and store it in a structured format for further processing.

## Files

*   **`__init__.py`**: Makes this directory a Python sub-package.
*   **`data_ingestion_pipeline.py`**:
    *   The main script that orchestrates the data ingestion process.
    *   **Functionality:**
        1.  **Loads Configurations:** Reads settings from:
            *   `MLOps/config/common/global_vars.yaml` (for date ranges, output paths, default tickers).
            *   `MLOps/config/data_sources.yaml` (to determine which data sources are enabled and their specific parameters, e.g., API keys if not handled elsewhere).
        2.  **Fetches Financial Data:**
            *   Utilizes FinRL's `DataProcessor` and underlying downloaders (e.g., `YahooDownloader`) or other custom downloaders to fetch historical stock data (OHLCV - Open, High, Low, Close, Volume) for a configured list of tickers and date range.
            *   Saves the fetched financial data to a CSV file (e.g., `data/raw/raw_financial_data.csv`).
        3.  **Fetches News Data:**
            *   (Placeholder for development as per Phase 2.1 of `analysis/coding-strategy.md`)
            *   When implemented, this will fetch news articles, headlines, or other textual data relevant to the specified tickers and date range from configured news APIs or web sources.
            *   Saves the fetched news data to a CSV file (e.g., `data/raw/raw_news_data.csv`).
        4.  **Data Versioning (Conceptual):**
            *   After saving the raw data files, the script conceptually invokes `dvc add` to track these files with DVC (Data Version Control). In a fully automated setup, this DVC command would be executed by the orchestration layer.
    *   **Execution:**
        ```bash
        python MLOps/pipelines/data_ingestion/data_ingestion_pipeline.py \
          --global_config_path MLOps/config/common/global_vars.yaml \
          --data_sources_config_path MLOps/config/data_sources.yaml \
          --financial_data_filename raw_financial_data_v2.csv \
          --news_data_filename raw_news_data_v2.csv
        ```
        (Arguments allow overriding default config paths and output filenames.)

## Data Sources

*   **Financial Data:**
    *   Primarily uses Yahoo Finance via FinRL's utilities as a default.
    *   Can be extended to support other sources like Alpaca, IEX Cloud, etc., by adding new downloader components and configurations in `MLOps/config/data_sources.yaml`.
*   **News Data:**
    *   The specific news sources and fetching mechanism are to be developed.
    *   Configuration for news APIs (keys, endpoints, parameters) will be managed in `MLOps/config/data_sources.yaml` or `config/api_keys.yaml`.

## Output

*   Raw financial data (e.g., `data/raw/raw_financial_data.csv`).
    *   Schema typically includes: `date`, `tic` (ticker), `open`, `high`, `low`, `close`, `volume`, `adj_close` (if available).
*   Raw news data (e.g., `data/raw/raw_news_data.csv`).
    *   Schema might include: `date`, `tic` (if news is ticker-specific), `source`, `headline`, `summary`, `full_text_url`.

## Dependencies

*   Pandas
*   PyYAML
*   FinRL (for financial data downloaders)
*   Requests (or similar, for news API interaction when implemented)
*   DVC (command-line tool, for versioning)

## Future Enhancements

*   More robust error handling and retry mechanisms for data fetching.
*   Support for incremental data fetching to update datasets efficiently.
*   Integration with a wider range of financial data providers and news APIs.
*   Schema validation for fetched data.
*   Direct integration with DVC Python API for programmatic versioning.
*   Metrics exposure for Prometheus monitoring (e.g., number of records fetched, success/failure status).