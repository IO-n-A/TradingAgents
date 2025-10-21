# MLOps/tests/pipeline_tests/test_data_ingestion.py
"""
Unit tests for the data_ingestion_pipeline.py script.
"""
import pytest
import os
import pandas as pd
import yaml
from unittest.mock import patch, mock_open, MagicMock
import argparse

import sys
# Add project root to sys.path to ensure MLOps module can be found
# __file__ is MLOps/tests/pipeline_tests/test_data_ingestion.py
# We need to go up three levels to reach the project root c:/Users/Jonas/code/FinAI_algo
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import the script to be tested (adjust path if necessary)
# This assumes that the MLOps directory is in PYTHONPATH or tests are run from project root.
try:
    from MLOps.pipelines.data_ingestion.data_ingestion_pipeline import (
        load_config,
        run_data_ingestion_pipeline,
        PlaceholderYahooDownloader, # For testing its mock
        PlaceholderNewsDownloader   # For testing its mock
    )
except ImportError as e:
    print(f"DEBUG: ImportError encountered while trying to import pipeline components: {e}")
    import traceback
    print("Full Traceback:")
    print(traceback.format_exc())
    pytest.skip(f"Skipping data ingestion tests due to ImportError: {e}. Check traceback above. Ensure MLOps is in PYTHONPATH and all dependencies are installed.", allow_module_level=True)


# --- Fixtures ---

@pytest.fixture
# Pytest Fixture: temp_config_dir
# Description: Creates a temporary directory structure for configuration files (`config/common/`)
#              within the pytest temporary path (`tmp_path`).
# Input: tmp_path (pytest fixture).
# Output: Path object to the created 'config' directory.
# Dependencies: pytest.
def temp_config_dir(tmp_path):
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Fixture: temp_config_dir, Purpose: Creates a temporary directory for config files, Output: Path to temp config dir.")
    """Creates a temporary directory for config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    common_dir = config_dir / "common"
    common_dir.mkdir()
    return config_dir

# Pytest Fixture: sample_global_vars_config_content
# Description: Provides a sample dictionary representing the content of a `global_vars.yaml` file.
# Input: None.
# Output: Dictionary with sample global configuration.
# Dependencies: None.
@pytest.fixture
def sample_global_vars_config_content():
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Fixture: sample_global_vars_config_content, Purpose: Provides sample global_vars.yaml content, Output: Dictionary.")
    return {
        "date_ranges": {
            "training_start_date": "2023-01-01",
            "training_end_date": "2023-01-10"
        },
        "paths": {
            "raw_data_dir": "test_raw_data/", # Relative to where test runs, or use tmp_path
            "processed_data_dir": "test_processed_data/",
            "model_registry_dir": "test_model_registry/",
            "results_dir": "test_results/",
            "log_dir": "test_logs/"
        },
        "default_tickers": ["TEST1", "TEST2"]
    }

# Pytest Fixture: sample_data_sources_config_content
# Description: Provides a sample dictionary representing the content of a `data_sources.yaml` file.
# Input: None.
# Output: Dictionary with sample data sources configuration.
# Dependencies: None.
@pytest.fixture
def sample_data_sources_config_content():
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Fixture: sample_data_sources_config_content, Purpose: Provides sample data_sources.yaml content, Output: Dictionary.")
    return {
        "yahoofinance": {"enabled": True},
        "news_api": {
            "enabled": True,
            "parameters": {"language": "en"}
        }
    }

# Pytest Fixture: create_sample_config_files
# Description: Creates sample `global_vars.yaml` and `data_sources.yaml` files in a temporary
#              directory structure, along with a temporary raw data output directory.
#              Used to set up a realistic file environment for testing the pipeline.
# Input: tmp_path (pytest fixture), sample_global_vars_config_content (fixture), sample_data_sources_config_content (fixture).
# Output: Dictionary containing paths to the created config files and directories.
# Dependencies: pytest, yaml.
@pytest.fixture
def create_sample_config_files(tmp_path, sample_global_vars_config_content, sample_data_sources_config_content):
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Fixture: create_sample_config_files, Purpose: Creates sample config files in a temp directory, Output: Dictionary of paths.")
    """Creates sample config files in a temporary directory."""
    config_root = tmp_path / "MLOps_test_configs"
    config_root.mkdir()
    
    common_dir = config_root / "config" / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    
    global_vars_path = common_dir / "global_vars.yaml"
    with open(global_vars_path, 'w') as f:
        # Make a copy to modify for this test run, ensuring raw_data_dir is absolute
        current_global_vars_content = sample_global_vars_config_content.copy()
        # Ensure paths sub-dictionary exists
        if "paths" not in current_global_vars_content:
            current_global_vars_content["paths"] = {}
        
        # Construct the absolute path for raw_data_dir for this test run
        # The original sample_global_vars_config_content["paths"]["raw_data_dir"] is like "test_raw_data/"
        relative_raw_data_dir_name = current_global_vars_content["paths"].get("raw_data_dir", "test_raw_data/")
        absolute_raw_data_dir = tmp_path / relative_raw_data_dir_name
        absolute_raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Update the content to be written to YAML to use this absolute path
        current_global_vars_content["paths"]["raw_data_dir"] = str(absolute_raw_data_dir)
        
        yaml.dump(current_global_vars_content, f) # Write modified content

    data_sources_path = config_root / "config" / "data_sources.yaml"
    (config_root / "config").mkdir(exist_ok=True) # Ensure parent 'config' dir exists
    with open(data_sources_path, 'w') as f:
        yaml.dump(sample_data_sources_config_content, f)
        
    return {
        "global_config_path": str(global_vars_path),
        "data_sources_config_path": str(data_sources_path),
        "temp_root": tmp_path,
        "raw_data_output_dir": str(absolute_raw_data_dir) # This is what the test asserts against
    }


# --- Tests for load_config ---

# Test Function: test_load_config_valid
# Description: Tests the `load_config` function with a valid YAML file.
# Input: tmp_path (pytest fixture).
# Output: Asserts that the loaded configuration matches the expected content.
# Dependencies: pytest, yaml, load_config (from pipeline script).
def test_load_config_valid(tmp_path):
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_load_config_valid, Purpose: Tests loading a valid YAML config file, Output: None (assertions).")
    """Test loading a valid YAML config file."""
    config_content = {"key": "value", "number": 123}
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    
    loaded_config = load_config(str(config_file))
    assert loaded_config == config_content

# Test Function: test_load_config_file_not_found
# Description: Tests the `load_config` function with a non-existent file path.
# Input: None.
# Output: Asserts that a FileNotFoundError is raised.
# Dependencies: pytest, load_config (from pipeline script).
def test_load_config_file_not_found():
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_load_config_file_not_found, Purpose: Tests loading a non-existent config file, Output: None (assertions).")
    """Test loading a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

# Test Function: test_load_config_invalid_yaml
# Description: Tests the `load_config` function with an invalid YAML file.
# Input: tmp_path (pytest fixture).
# Output: Asserts that a yaml.YAMLError is raised.
# Dependencies: pytest, yaml, load_config (from pipeline script).
def test_load_config_invalid_yaml(tmp_path):
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_load_config_invalid_yaml, Purpose: Tests loading an invalid YAML file, Output: None (assertions).")
    """Test loading an invalid YAML file."""
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, 'w') as f:
        f.write("key: value\n  bad_indent: oops") # Invalid YAML
    
    with pytest.raises(yaml.YAMLError): # Or a more specific error depending on PyYAML version
        load_config(str(config_file))


# --- Tests for run_data_ingestion_pipeline (with mocks) ---

# Test Function: test_run_data_ingestion_pipeline_success
# Description: Tests the `run_data_ingestion_pipeline` function for a successful run
#              where both financial and news data are fetched (using mocks).
#              It checks if the mock downloaders are called and if output files are created with expected content.
# Input: create_sample_config_files (fixture), tmp_path (pytest fixture), sample_global_vars_config_content (fixture).
# Output: Asserts mock calls and file creation/content.
# Dependencies: pytest, unittest.mock.patch, os, pandas (pd), argparse,
#               run_data_ingestion_pipeline (from pipeline script).
def test_run_data_ingestion_pipeline_success(
    create_sample_config_files, tmp_path, sample_global_vars_config_content
):
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_run_data_ingestion_pipeline_success, Purpose: Tests successful run of data ingestion pipeline with mocks, Output: None (assertions).")
    """Test the main pipeline function with mocked downloaders."""

    # Prepare arguments for the pipeline function using actual fixture values
    raw_data_output_dir = create_sample_config_files["raw_data_output_dir"]
    args_dict = {
        "global_config_path": create_sample_config_files["global_config_path"],
        "data_sources_config_path": create_sample_config_files["data_sources_config_path"],
        "financial_data_filename": "test_financial_data.csv",
        "news_data_filename": "test_news_data.csv"
    }
    args_namespace = argparse.Namespace(**args_dict)

    dummy_financial_df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'tic': ['TEST1', 'TEST1'], 'close': [100, 101]
    })
    dummy_news_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01']), # Changed 'date' to 'timestamp'
        'ticker': ['TEST1'],
        'headline': ['Test news headline'],
        'summary': ['Test summary'],      # Added to match pipeline's expected columns
        'full_text': ['Test full text'],  # Added
        'source_id': ['TestSouce'],       # Changed 'source' to 'source_id'
        'url': ['http://test.url'],       # Added
        'raw_data': ['{}']                # Added
    })

    with patch('MLOps.pipelines.data_ingestion.data_ingestion_pipeline.YahooDownloader') as MockYahooDownloader, \
         patch('MLOps.pipelines.data_ingestion.data_ingestion_pipeline.NewsDataDownloader') as MockNewsDownloader:

        # Setup mock instances and their return values
        mock_yahoo_downloader_instance = MockYahooDownloader.return_value
        mock_yahoo_downloader_instance.fetch_data.return_value = dummy_financial_df

        mock_news_downloader_instance = MockNewsDownloader.return_value
        mock_news_downloader_instance.fetch_news.return_value = dummy_news_df

        # Run the pipeline
        run_data_ingestion_pipeline(args_namespace)

        # Assertions
        mock_yahoo_downloader_instance.fetch_data.assert_called_once()
        mock_news_downloader_instance.fetch_news.assert_called_once()

        # Check if output files were created in the correct temporary directory
        expected_financial_file = os.path.join(raw_data_output_dir, "test_financial_data.csv")
        expected_news_file = os.path.join(raw_data_output_dir, "test_news_data.csv")
        
        assert os.path.exists(expected_financial_file)
        assert os.path.exists(expected_news_file)

        # Optionally, check content of the CSVs
        df_fin_out = pd.read_csv(expected_financial_file, parse_dates=['date'])
        pd.testing.assert_frame_equal(df_fin_out, dummy_financial_df, check_dtype=False)

        df_news_out = pd.read_csv(expected_news_file, parse_dates=['timestamp'])
        # Ensure the dummy_news_df 'timestamp' column is also just date part for comparison if read_csv truncates time
        dummy_news_df_for_comparison = dummy_news_df.copy()
        dummy_news_df_for_comparison['timestamp'] = pd.to_datetime(dummy_news_df_for_comparison['timestamp']).dt.normalize()
        df_news_out['timestamp'] = pd.to_datetime(df_news_out['timestamp']).dt.normalize()
        pd.testing.assert_frame_equal(df_news_out, dummy_news_df_for_comparison, check_dtype=False)


# Test Function: test_run_data_ingestion_pipeline_one_source_disabled
# Description: Tests the `run_data_ingestion_pipeline` function when one data source (news)
#              is disabled in the configuration. It verifies that the disabled source's
#              downloader is not called and its output file is not created.
# Input: create_sample_config_files (fixture), tmp_path (pytest fixture).
# Output: Asserts mock calls and file creation status.
# Dependencies: pytest, unittest.mock.patch, os, pandas (pd), yaml, argparse,
#               run_data_ingestion_pipeline (from pipeline script).
def test_run_data_ingestion_pipeline_one_source_disabled(
    create_sample_config_files, tmp_path
):
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_run_data_ingestion_pipeline_one_source_disabled, Purpose: Tests pipeline with one data source disabled, Output: None (assertions).")
    """Test pipeline when one data source is disabled in config."""
    
    # Modify data_sources_config to disable news
    disabled_news_config_content = {
        "yahoofinance": {"enabled": True},
        "news_api": {"enabled": False} # News disabled
    }
    disabled_data_sources_path = tmp_path / "disabled_news_sources.yaml"
    with open(disabled_data_sources_path, 'w') as f:
        yaml.dump(disabled_news_config_content, f)

    args_dict = {
        "global_config_path": create_sample_config_files["global_config_path"],
        "data_sources_config_path": str(disabled_data_sources_path), # Use modified config
        "financial_data_filename": "test_financial_data_only.csv",
        "news_data_filename": "test_news_data_disabled.csv"
    }
    args_namespace = argparse.Namespace(**args_dict)
    
    raw_data_output_dir = create_sample_config_files["raw_data_output_dir"]

    with patch('MLOps.pipelines.data_ingestion.data_ingestion_pipeline.YahooDownloader') as MockYahooDownloader, \
         patch('MLOps.pipelines.data_ingestion.data_ingestion_pipeline.NewsDataDownloader') as MockNewsDownloader:

        mock_yahoo_downloader_instance = MockYahooDownloader.return_value
        mock_yahoo_downloader_instance.fetch_data.return_value = pd.DataFrame({'tic': ['T1']})

        run_data_ingestion_pipeline(args_namespace)

        mock_yahoo_downloader_instance.fetch_data.assert_called_once()
        MockNewsDownloader.return_value.fetch_news.assert_not_called()

    assert os.path.exists(os.path.join(raw_data_output_dir, "test_financial_data_only.csv"))
    # When news is disabled, the pipeline's fetch_news_data creates an empty placeholder file.
    # So, we expect this file to exist.
    assert os.path.exists(os.path.join(raw_data_output_dir, "test_news_data_disabled.csv"))


# --- Tests for Placeholder Downloaders (Optional, if they had more logic) ---

# Test Function: test_placeholder_yahoo_downloader
# Description: Tests the `fetch_data` method of the `PlaceholderYahooDownloader`.
#              Verifies that it returns a non-empty DataFrame with expected columns and data characteristics.
# Input: None.
# Output: Asserts DataFrame properties.
# Dependencies: pandas (pd), PlaceholderYahooDownloader (from pipeline script).
def test_placeholder_yahoo_downloader():
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_placeholder_yahoo_downloader, Purpose: Tests the PlaceholderYahooDownloader's fetch_data method, Output: None (assertions).")
    """Test the placeholder Yahoo downloader's fetch_data method."""
    downloader = PlaceholderYahooDownloader()
    tickers = ["AAPL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    df = downloader.fetch_data(start_date, end_date, tickers)
    
    assert not df.empty
    assert "tic" in df.columns
    assert set(df["tic"].unique()) == set(tickers)
    assert pd.to_datetime(df["date"].min()) >= pd.to_datetime(start_date)
    assert pd.to_datetime(df["date"].max()) <= pd.to_datetime(end_date)

# Test Function: test_placeholder_news_downloader
# Description: Tests the `fetch_news` method of the `PlaceholderNewsDownloader`.
#              Verifies that it returns a non-empty DataFrame with expected columns.
# Input: None.
# Output: Asserts DataFrame properties.
# Dependencies: pandas (pd), PlaceholderNewsDownloader (from pipeline script).
def test_placeholder_news_downloader():
    print(f"File: MLOps/tests/pipeline_tests/test_data_ingestion.py, Test: test_placeholder_news_downloader, Purpose: Tests the PlaceholderNewsDownloader's fetch_news method, Output: None (assertions).")
    """Test the placeholder News downloader's fetch_news method."""
    downloader = PlaceholderNewsDownloader(api_key="dummy_key")
    tickers = ["GOOG"]
    start_date = "2023-02-01"
    end_date = "2023-02-03"
    df = downloader.fetch_news(tickers, start_date, end_date)

    assert not df.empty
    assert "ticker" in df.columns
    assert "headline" in df.columns
    # Add more specific checks if the dummy data generation was more complex


# It's good practice to also test for failure cases, e.g.,
# - What happens if a downloader raises an exception?
# - What happens if config files are malformed in ways not caught by basic load_config tests?
# These would require more intricate mocking or setup.

if __name__ == '__main__':
    # This allows running pytest directly on this file:
    # python MLOps/tests/pipeline_tests/test_data_ingestion.py
    # However, it's more common to run pytest from the project root:
    # pytest MLOps/tests/pipeline_tests/test_data_ingestion.py
    # or just `pytest` to discover all tests.
    pytest.main()