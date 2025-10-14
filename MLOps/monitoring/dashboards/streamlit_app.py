# MLOps/monitoring/dashboards/streamlit_app.py

import streamlit as st
import pandas as pd
import os
import sys
import yaml
import logging
import logging.config
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the main project directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging (streamlit apps might need their own specific setup or can use root)
# For simplicity, using basic config here if main config file is not found or causes issues in streamlit context
try:
    logging_config_path = os.path.join(project_root, 'config', 'logging_config.py')
    if os.path.exists(logging_config_path):
        # Assuming logging_config.py uses fileConfig or similar that can be called
        # For fileConfig, it expects a filename. If it's a .py, it might need to be imported and a setup function called.
        # This part might need adjustment based on how logging_config.py is structured.
        # If logging_config.py directly configures logging on import, this might be tricky.
        # A common pattern is to have a setup_logging() function in logging_config.py.
        # For now, let's assume it's a fileConfig compatible file or handle it.
        # If it's a .py file that configures upon import, this won't work as expected.
        # A more robust way for .py config is:
        # import importlib.util
        # spec = importlib.util.spec_from_file_location("logging_config", logging_config_path)
        # logging_config_module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(logging_config_module)
        # if hasattr(logging_config_module, 'setup_logging'):
        # logging_config_module.setup_logging()
        # else: # Fallback if no setup function
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Used basicConfig as logging_config.py structure is not directly fileConfig compatible or setup function not found.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
except Exception: # Catch potential errors if logging config is not streamlit-friendly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---
# Function: load_config
# Description: Loads a YAML configuration file from the given path.
#              Handles errors by displaying them in the Streamlit app and returning an empty dictionary.
# Input:
#   config_path: String, the path to the YAML configuration file.
# Output: A dictionary containing the loaded configuration, or an empty dictionary on error.
# Dependencies: yaml, streamlit (st).
def load_config(config_path: str) -> dict:
    print(f"File: MLOps/monitoring/dashboards/streamlit_app.py, Function: load_config, Purpose: Loads YAML configuration for Streamlit app, Output: Config dictionary or empty dict on error.")
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading configuration from {config_path}: {e}")
        return {}

# Function: load_csv_data
# Description: Loads data from a CSV file into a pandas DataFrame.
#              It handles file not found errors and other loading exceptions by displaying
#              warnings/errors in the Streamlit app and returning an empty DataFrame.
#              Attempts to convert common date-like columns to datetime objects.
# Input:
#   data_path: String, the path to the CSV file.
#   file_description: String, a description of the data being loaded (for error messages).
# Output: A pandas DataFrame with the loaded data, or an empty DataFrame on error.
# Dependencies: pandas (pd), os, streamlit (st), logging.
def load_csv_data(data_path: str, file_description: str) -> pd.DataFrame:
    print(f"File: MLOps/monitoring/dashboards/streamlit_app.py, Function: load_csv_data, Purpose: Loads CSV data for Streamlit app, handles errors, Output: Pandas DataFrame or empty DataFrame on error.")
    """Loads CSV data, handling potential errors for Streamlit."""
    if not data_path or not os.path.exists(data_path):
        st.warning(f"{file_description} data file not found at {data_path}. Some dashboard sections may be unavailable.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(data_path)
        # Attempt to parse date columns if they exist
        for col in ['date', 'timestamp', 'Date']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass # Ignore if conversion fails, keep as is
        logger.info(f"Successfully loaded {file_description} data from {data_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading {file_description} data from {data_path}: {e}")
        return pd.DataFrame()

# --- Paths ---
# These paths should point to where your pipeline outputs are stored.
# They might be managed by DVC in a real scenario.
RAW_DATA_DIR = os.path.join(project_root, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(project_root, "data", "processed")
BACKTEST_RESULTS_DIR = os.path.join(project_root, "MLOps", "results", "backtesting")
SENTIMENT_PREDICTIONS_DIR = os.path.join(project_root, "data", "sentiment_predictions_output") # Assuming sentiment service outputs here

# --- Main Dashboard Logic ---
# Main Streamlit Application Logic
# Description: This section defines the Streamlit application's UI and logic.
#              It includes page configuration, title, sidebar navigation, data loading,
#              and different dashboard sections for Data Quality, Sentiment Model,
#              RL Agent Performance, and Correlations. Each section visualizes
#              relevant data and metrics.
# Input: User interaction with the Streamlit interface.
# Output: Renders a web-based dashboard.
# Dependencies: streamlit (st), pandas (pd), matplotlib.pyplot, seaborn (sns),
#               load_all_data (local function), and other helper functions/data paths.
print(f"File: MLOps/monitoring/dashboards/streamlit_app.py, Section: Main Streamlit App Logic, Purpose: Defines the UI and logic for the MLOps monitoring dashboard, Output: Renders web dashboard.")
st.set_page_config(layout="wide", page_title="FinAI MLOps Dashboard")
st.title("📈 FinAI MLOps Monitoring Dashboard")

# --- Sidebar for Navigation/Filters ---
st.sidebar.header("Dashboard Sections")
dashboard_section = st.sidebar.radio(
    "Select a section:",
    ("Data Quality", "Sentiment Model", "RL Agent Performance", "Correlations")
)

# --- Load Data (cached for performance) ---
@st.cache_data
# Function: load_all_data (cached)
# Description: Loads all necessary data for the dashboard from various CSV files.
#              This includes raw financial and news data, processed features,
#              RL agent backtesting results (latest account value), and sentiment predictions.
#              The function is cached using `@st.cache_data` for performance.
# Input: None (uses predefined global path variables like RAW_DATA_DIR, PROCESSED_DATA_DIR, etc.).
# Output: A dictionary where keys are data descriptions (e.g., "raw_financial") and
#         values are the corresponding pandas DataFrames.
# Dependencies: streamlit (st), os, load_csv_data (local function).
# Globals: RAW_DATA_DIR, PROCESSED_DATA_DIR, BACKTEST_RESULTS_DIR, SENTIMENT_PREDICTIONS_DIR.
@st.cache_data
def load_all_data():
    print(f"File: MLOps/monitoring/dashboards/streamlit_app.py, Function: load_all_data, Purpose: Loads all data sources for the Streamlit dashboard, cached for performance, Output: Dictionary of DataFrames.")
    data = {}
    data["raw_financial"] = load_csv_data(os.path.join(RAW_DATA_DIR, "raw_financial_data.csv"), "Raw Financial")
    data["raw_news"] = load_csv_data(os.path.join(RAW_DATA_DIR, "raw_news_data_placeholder.csv"), "Raw News (Placeholder)") # Or actual news file
    data["processed_features"] = load_csv_data(os.path.join(PROCESSED_DATA_DIR, "processed_feature_engineered_data.csv"), "Processed Features")
    
    # For RL Agent Performance - find the latest backtest results
    # This is a simple way; a more robust method would query MLflow or a results database.
    latest_account_value_file = None
    if os.path.exists(BACKTEST_RESULTS_DIR):
        backtest_files = [f for f in os.listdir(BACKTEST_RESULTS_DIR) if f.startswith("account_value_") and f.endswith(".csv")]
        if backtest_files:
            latest_account_value_file = max([os.path.join(BACKTEST_RESULTS_DIR, f) for f in backtest_files], key=os.path.getctime, default=None)
    
    data["rl_account_value"] = load_csv_data(latest_account_value_file, "RL Agent Account Value")
    
    # Placeholder for sentiment predictions (assuming a file structure)
    # This would be output by the SentimentAnalyzerService if it saves its batch predictions
    latest_sentiment_output_file = None
    if os.path.exists(SENTIMENT_PREDICTIONS_DIR):
        sentiment_files = [f for f in os.listdir(SENTIMENT_PREDICTIONS_DIR) if f.endswith(".csv")] # Adjust naming
        if sentiment_files:
            latest_sentiment_output_file = max([os.path.join(SENTIMENT_PREDICTIONS_DIR, f) for f in sentiment_files], key=os.path.getctime, default=None)
    data["sentiment_predictions"] = load_csv_data(latest_sentiment_output_file, "Sentiment Predictions")

    return data

app_data = load_all_data()


# ==========================
# === Data Quality Section ===
# ==========================
if dashboard_section == "Data Quality":
    st.header("📊 Data Quality Monitoring")

    st.subheader("Raw Financial Data Overview")
    if not app_data["raw_financial"].empty:
        st.dataframe(app_data["raw_financial"].head())
        st.write(f"Shape: {app_data['raw_financial'].shape}")
        st.write("Summary Statistics (Numeric Columns):")
        st.dataframe(app_data["raw_financial"].describe(include=float).transpose()) # Describe only numeric
        
        # Example: Plot volume over time for a selected ticker
        if 'tic' in app_data["raw_financial"].columns and 'volume' in app_data["raw_financial"].columns and 'date' in app_data["raw_financial"].columns:
            tickers = app_data["raw_financial"]['tic'].unique()
            selected_ticker_dq = st.selectbox("Select Ticker for Volume Plot:", tickers, key="dq_ticker_vol")
            if selected_ticker_dq:
                ticker_data = app_data["raw_financial"][app_data["raw_financial"]['tic'] == selected_ticker_dq]
                fig, ax = plt.subplots()
                ax.plot(ticker_data['date'], ticker_data['volume'])
                ax.set_title(f"Volume for {selected_ticker_dq}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volume")
                plt.xticks(rotation=45)
                st.pyplot(fig)
    else:
        st.info("Raw financial data not available.")

    st.subheader("Raw News Data Overview (Placeholder)")
    if not app_data["raw_news"].empty:
        st.dataframe(app_data["raw_news"].head())
        st.write(f"Shape: {app_data['raw_news'].shape}")
        # Add more news-specific visualizations when actual news data is available
        # e.g., number of news items over time, distribution of sources
    else:
        st.info("Raw news data (placeholder or actual) not available or empty.")

    st.subheader("Processed Feature-Engineered Data Overview")
    if not app_data["processed_features"].empty:
        st.dataframe(app_data["processed_features"].head())
        st.write(f"Shape: {app_data['processed_features'].shape}")
        st.write("Summary Statistics (Numeric Columns):")
        st.dataframe(app_data["processed_features"].describe(include=float).transpose())
    else:
        st.info("Processed feature-engineered data not available.")

# ==============================
# === Sentiment Model Section ===
# ==============================
elif dashboard_section == "Sentiment Model":
    st.header("💬 Sentiment Model Monitoring")
    
    sentiment_df = app_data["sentiment_predictions"] # This is placeholder data for now
    
    if not sentiment_df.empty and all(col in sentiment_df.columns for col in ['positive', 'negative', 'neutral', 'date']):
        st.subheader("Distribution of Sentiment Scores")
        st.dataframe(sentiment_df.head())
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(sentiment_df['positive'], kde=True, ax=ax[0], color='green').set_title('Positive Scores')
        sns.histplot(sentiment_df['negative'], kde=True, ax=ax[1], color='red').set_title('Negative Scores')
        sns.histplot(sentiment_df['neutral'], kde=True, ax=ax[2], color='blue').set_title('Neutral Scores')
        st.pyplot(fig)

        st.subheader("Sentiment Scores Over Time")
        # Requires 'date' column in sentiment_df
        if 'date' in sentiment_df.columns:
            sentiment_over_time = sentiment_df.set_index('date')[['positive', 'negative', 'neutral']].resample('M').mean() # Monthly average
            st.line_chart(sentiment_over_time)
        else:
            st.warning("Sentiment data does not have a 'date' column for time series plotting.")
            
    elif not sentiment_df.empty:
        st.warning("Sentiment data loaded, but missing one or more required columns: 'positive', 'negative', 'neutral', 'date'.")
        st.dataframe(sentiment_df.head())
    else:
        st.info("Sentiment prediction data not available. This section requires output from the SentimentAnalyzerService.")


# ====================================
# === RL Agent Performance Section ===
# ====================================
elif dashboard_section == "RL Agent Performance":
    st.header("🤖 RL Agent Performance")
    
    account_value_df = app_data["rl_account_value"]
    if not account_value_df.empty and 'account_value' in account_value_df.columns and 'date' in account_value_df.columns:
        st.subheader("Agent Account Value Over Time")
        st.line_chart(account_value_df.set_index('date')['account_value'])
        
        st.subheader("Performance Statistics")
        # This would ideally load pre-calculated stats from a backtest run or calculate them here.
        # For now, just showing the head of the account value df.
        st.dataframe(account_value_df.tail())
        st.markdown("""
            *Note: Detailed performance statistics (Sharpe Ratio, Max Drawdown, etc.) 
            would be displayed here, typically generated by `FinRL.finrl.plot.backtest_stats` 
            during the backtesting pipeline and saved as an artifact.*
            """)
        # Example: if backtest_stats were saved as a CSV/JSON, load and display it.
        # stats_path = os.path.join(BACKTEST_RESULTS_DIR, "latest_perf_stats.csv")
        # if os.path.exists(stats_path):
        #     perf_stats = pd.read_csv(stats_path, index_col=0)
        #     st.dataframe(perf_stats)
        # else:
        #     st.info("Performance statistics file not found.")
            
        # Placeholder for plot image
        # In a real scenario, the backtest plot image would be saved by the pipeline
        # and loaded here.
        # plot_image_path = os.path.join(BACKTEST_RESULTS_DIR, "latest_backtest_plot.png")
        # if os.path.exists(plot_image_path):
        #     st.image(plot_image_path, caption="Backtest Performance Plot")
        # else:
        #     st.info("Backtest plot image not found.")
            
    else:
        st.info("RL agent account value data not available. Run the backtesting pipeline.")

# ==========================
# === Correlations Section ===
# ==========================
elif dashboard_section == "Correlations":
    st.header("🔗 Correlations Analysis")
    
    processed_df = app_data["processed_features"]
    if not processed_df.empty:
        st.subheader("Correlation Heatmap of Features")
        
        # Select numeric columns for correlation
        numeric_cols = processed_df.select_dtypes(include=float).columns.tolist()
        
        # Limit number of columns for readability or let user select
        if len(numeric_cols) > 15:
            st.warning(f"Found {len(numeric_cols)} numeric features. Displaying correlation for a subset.")
            # Example: select a subset including sentiment features if they exist
            potential_sentiment_features = [
                'sentiment_positive_prob', 'sentiment_negative_prob', 'sentiment_neutral_prob',
                'sentiment_composite_score', 'sentiment_label',
                'sentiment_composite_score_lag1', 'sentiment_label_lag1', 'sentiment_composite_score_ma3'
            ]
            sentiment_features_present = [f for f in potential_sentiment_features if f in numeric_cols]
            
            # Select some price/volume and technical indicators
            other_features = [col for col in ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi_30'] if col in numeric_cols]
            
            cols_for_corr = list(set(sentiment_features_present + other_features[:15-len(sentiment_features_present)]))
            
            if not cols_for_corr: # Fallback if no specific features found
                 cols_for_corr = numeric_cols[:15]
        else:
            cols_for_corr = numeric_cols

        if cols_for_corr:
            corr_matrix = processed_df[cols_for_corr].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")

        # Time series plot of sentiment vs. price (example for one ticker)
        if 'tic' in processed_df.columns and \
           'close' in processed_df.columns and \
           'sentiment_composite_score' in processed_df.columns and \
           'date' in processed_df.columns:
            
            tickers_corr = processed_df['tic'].unique()
            selected_ticker_corr = st.selectbox("Select Ticker for Sentiment vs. Price Plot:", tickers_corr, key="corr_ticker_select")
            
            if selected_ticker_corr:
                ticker_corr_data = processed_df[processed_df['tic'] == selected_ticker_corr].set_index('date')
                
                fig, ax1 = plt.subplots(figsize=(12,6))
                
                color = 'tab:red'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Close Price', color=color)
                ax1.plot(ticker_corr_data.index, ticker_corr_data['close'], color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
                color = 'tab:blue'
                ax2.set_ylabel('Sentiment Composite Score', color=color)
                ax2.plot(ticker_corr_data.index, ticker_corr_data['sentiment_composite_score'], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                fig.tight_layout() # otherwise the right y-label is slightly clipped
                plt.title(f"Close Price vs. Sentiment Score for {selected_ticker_corr}")
                st.pyplot(fig)
        else:
            st.info("Required columns ('tic', 'close', 'sentiment_composite_score', 'date') not all present for Sentiment vs. Price plot.")
            
    else:
        st.info("Processed feature data not available for correlation analysis.")

# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides monitoring for the FinAI_algo project.")