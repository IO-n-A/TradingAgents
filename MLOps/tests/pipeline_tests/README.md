# MLOps Pipeline Tests (Unit Tests)

This directory is dedicated to unit tests for individual components and functions within the MLOps pipelines of the FinAI_algo project. Unit tests are focused on testing the smallest, isolated parts of the codebase to ensure their correctness.

## Objective

The primary goal of these pipeline unit tests is to:

*   Verify the logic of individual functions and classes within each pipeline script (e.g., `data_ingestion_pipeline.py`, `build_features.py`, `train_rl_agent.py`, `train_sentiment_model.py`).
*   Ensure that data transformations, calculations, and utility functions behave as expected given specific inputs.
*   Catch bugs and regressions early in the development cycle.
*   Provide documentation for how individual components are intended to work.

## Scope of Pipeline Unit Tests

*   **Testing individual functions:**
    *   Example: A function in `build_features.py` that calculates a specific technical indicator. The unit test would provide sample input data (e.g., a small Pandas DataFrame with price series) and assert that the calculated indicator values are correct.
    *   Example: A text preprocessing function in `SentimentTextPreprocessor`.
*   **Testing methods of classes:**
    *   Example: Methods of `SentimentAnalyzerService` (e.g., `predict_sentiment` with a mock model).
    *   Example: Methods of `YahooDownloader` (e.g., `fetch_data` with a mocked API response from `yfinance` or `requests`).
*   **Testing argument parsing and configuration loading:**
    *   Ensuring that pipeline scripts correctly parse command-line arguments and load configurations.
*   **Boundary conditions and error handling:**
    *   Testing how functions handle empty inputs, invalid data types, or other edge cases.
    *   Verifying that custom exceptions are raised appropriately.

## What Not to Test Here (Typically)

*   **Interactions between multiple pipeline scripts:** These are covered by integration tests in `MLOps/tests/integration_tests/`.
*   **Full end-to-end pipeline runs with large datasets:** Unit tests should use small, controlled inputs.
*   **Actual machine learning model training or performance:** Unit tests focus on the correctness of the code surrounding the ML, not the statistical performance of the model itself (which is covered by evaluation scripts, backtesting, and MLflow tracking).
*   **Direct interactions with live external services (databases, APIs, DVC remote storage):** These should be mocked.

## Example Test Scenarios

*   **`test_data_ingestion.py`**:
    *   `test_load_config_valid()`: Test loading a valid YAML config.
    *   `test_load_config_invalid_path()`: Test behavior when config path is wrong.
    *   `test_yahoo_downloader_fetch_data_mocked()`: Mock the `yfinance.download()` call and verify that the downloader processes the mocked response correctly.
    *   `test_news_downloader_fetch_news_mocked()`: (When implemented) Mock news API calls.
*   **`test_feature_engineering.py`**: (Will be created)
    *   `test_calculate_sma()`: Test a simple moving average calculation.
    *   `test_sentiment_text_preprocessor_clean_text()`: Test a specific text cleaning step.
    *   `test_sentiment_analyzer_service_predict_mocked()`: Mock the underlying sentiment model and test the service's input/output handling.
    *   `test_merge_financial_and_sentiment_data()`: Test the logic for merging financial data with aggregated sentiment scores, covering cases with matching and missing dates/tickers.
*   **`test_train_rl_agent.py`**: (Will be created)
    *   `test_environment_initialization()`: Test that the `StockTradingEnv` can be initialized with sample processed data and configuration.
    *   `test_agent_hyperparameter_loading()`: Test that agent hyperparameters are correctly loaded from config.
*   **`test_train_sentiment_model.py`**: (Will be created)
    *   `test_tokenize_function()`: Test the tokenization logic for text data.
    *   `test_lora_config_creation()`: Test that LoRA configuration objects are created correctly based on input parameters.

## Test Structure and Tools

*   **Framework:** `pytest` is recommended.
*   **File Naming:** Test files should typically start with `test_` (e.g., `test_data_ingestion.py`). Test functions within these files should also start with `test_`.
*   **Fixtures:** Use `pytest` fixtures (`@pytest.fixture`) to provide reusable setup code for tests (e.g., creating sample DataFrames, mock objects, temporary configuration files).
*   **Mocking:** Use `unittest.mock` or `pytest-mock` extensively to isolate the unit under test from its dependencies (e.g., file system operations, API calls, complex objects).
*   **Assertions:** Use `pytest`'s `assert` statement for checking expected outcomes. For DataFrame comparisons, `pandas.testing.assert_frame_equal` is very useful. For NumPy arrays, `numpy.testing.assert_array_equal`.

## Running Pipeline Unit Tests

```bash
pytest MLOps/tests/pipeline_tests/
```

These unit tests form the foundation of the testing pyramid for the MLOps system, ensuring that individual building blocks are sound.