# core/news_fetcher.py
# This script defines a class to fetch general financial news from NewsAPI.org and Finnhub,
# saves the data, and logs its execution.

import logging
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import yaml
import subprocess
from typing import List, Dict, Optional, Tuple

try:
    from newsapi import NewsApiClient
    from newsapi.newsapi_exception import NewsAPIException
except ImportError:
    logging.getLogger(__name__).critical(
        "The 'newsapi-python' library is not installed. This is a required dependency for NewsAPI.org. "
        "Please install it by running 'pip install newsapi-python'."
    )
    sys.exit(1)

try:
    import finnhub
    # It's good practice to also import specific exceptions if the library provides them
    # e.g., from finnhub.exceptions import FinnhubAPIException, FinnhubAuthenticationException
except ImportError:
    logging.getLogger(__name__).critical(
        "The 'finnhub-python' library is not installed. This is a required dependency for Finnhub. "
        "Please install it by running 'pip install finnhub-python'."
    )
    sys.exit(1)

# Path to the project root assuming the script is in core/
# This ensures that paths are relative to the project root when the script is executed.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_KEYS_PATH = os.path.join(PROJECT_ROOT, 'config', 'api_keys.yaml')
GET_TIME_ID_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'helpers', 'get_time_id.py')
BACKLOG_FILE_PATH = os.path.join(PROJECT_ROOT, 'log', 'backlog.md')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'market_news_us.csv')

# Configure logging
# Basic configuration for console logging; more advanced configuration can be centralized.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__)

class MarketNewsFetcher:
    """
    Fetches general financial news from NewsAPI.org (primary) and Finnhub (fallback).

    This class handles API key loading, client initialization, data fetching,
    and basic API limit awareness. It aims to collect news headlines,
    summaries, sources, and publication dates.

    Attributes:
        newsapi_client (Optional[NewsApiClient]): Client for NewsAPI.org.
        finnhub_client (Optional[finnhub.Client]): Client for Finnhub.
        newsapi_key (Optional[str]): API key for NewsAPI.org.
        finnhub_key (Optional[str]): API key for Finnhub.
    """

    # This method initializes the MarketNewsFetcher class.
    # It loads API keys and initializes clients for NewsAPI and Finnhub.
    # No specific inputs are required beyond self, and it doesn't return anything directly but sets instance attributes.
    def __init__(self) -> None:
        """
        Initializes the MarketNewsFetcher.

        Loads API keys from `config/api_keys.yaml` and initializes the
        API clients for NewsAPI.org and Finnhub.
        """
        self.newsapi_client: Optional[NewsApiClient] = None
        self.finnhub_client: Optional[finnhub.Client] = None
        self.newsapi_key: Optional[str] = None
        self.finnhub_key: Optional[str] = None

        self._load_api_keys()
        self._initialize_clients()
        # print("MarketNewsFetcher initialized. API keys loaded and clients potentially set up.")
        # This print statement confirms that the __init__ method has completed its setup process.
        return

    # This method loads API keys from the configuration file.
    # It reads 'NEWS_API_ORG_KEY' and 'FINNHUB_API_KEY' from the YAML file.
    # It doesn't take inputs and doesn't return anything, but sets instance attributes for the keys.
    def _load_api_keys(self) -> None:
        """
        Loads API keys from the `config/api_keys.yaml` file.

        Sets `self.newsapi_key` and `self.finnhub_key` attributes.
        Logs warnings if the configuration file or keys are not found.
        """
        try:
            if not os.path.exists(API_KEYS_PATH):
                logger.warning(f"API keys file not found at {API_KEYS_PATH}. News fetching will likely fail.")
                # print(f"API keys file not found at {API_KEYS_PATH}. Keys will be None.")
                # This print statement indicates the outcome of the API key file search.
                return

            with open(API_KEYS_PATH, 'r') as f:
                api_keys_config = yaml.safe_load(f)

            self.newsapi_key = api_keys_config.get("NEWS_API_ORG_KEY")
            self.finnhub_key = api_keys_config.get("FINNHUB_API_KEY")

            if not self.newsapi_key:
                logger.warning("NEWS_API_ORG_KEY not found in api_keys.yaml.")
            if not self.finnhub_key:
                logger.warning("FINNHUB_API_KEY not found in api_keys.yaml.")
            # print(f"API keys loaded. NewsAPI key found: {bool(self.newsapi_key)}, Finnhub key found: {bool(self.finnhub_key)}.")
            # This print statement summarizes whether the necessary API keys were successfully loaded.

        except Exception as e:
            logger.error(f"Error loading API keys from {API_KEYS_PATH}: {e}", exc_info=True)
            # print(f"An error occurred while loading API keys: {e}. Keys will remain None.")
            # This print statement informs about an error during the key loading process.
        return

    # This method initializes the API clients for NewsAPI and Finnhub.
    # It uses the loaded API keys to create client instances.
    # It doesn't take inputs and doesn't return anything, but sets client instance attributes.
    def _initialize_clients(self) -> None:
        """
        Initializes the NewsAPI and Finnhub clients using the loaded API keys.

        Sets `self.newsapi_client` and `self.finnhub_client` attributes.
        Logs information about client initialization status.
        """
        if NewsApiClient and self.newsapi_key:
            try:
                self.newsapi_client = NewsApiClient(api_key=self.newsapi_key)
                logger.info("NewsAPI client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize NewsAPI client: {e}", exc_info=True)
        # This check is now largely handled by the initial ImportError checks,
        # but kept for logical completeness if the script structure changes.
        # elif not NewsApiClient: # This case should ideally not be reached if script exits on ImportError
        #     logger.critical("NewsAPI client library (newsapi-python) not installed. Exiting.") # Should have exited earlier
        #     sys.exit(1)
        elif not self.newsapi_key:
            logger.warning("NewsAPI key not available. NewsAPI client not initialized. NewsAPI fetching will be skipped.")

        if self.finnhub_key: # finnhub library presence is checked at the start
            try:
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
                logger.info("Finnhub client initialized successfully.")
            except Exception as e: # General exception for unforeseen client init issues
                logger.error(f"Failed to initialize Finnhub client: {e}", exc_info=True)
        # elif not finnhub: # This case should ideally not be reached
        #     logger.critical("Finnhub client library (finnhub-python) not installed. Exiting.") # Should have exited earlier
        #     sys.exit(1)
        elif not self.finnhub_key:
            logger.warning("Finnhub API key not available. Finnhub client not initialized. Finnhub fetching will be skipped.")
        return

    # This method fetches news articles from NewsAPI.org based on keywords and a date range.
    # It queries the API, processes the articles, and standardizes them into a list of dictionaries.
    # Returns a list of news articles or an empty list if fetching fails or no articles are found.
    def _fetch_news_newsapi(self, keywords: List[str], from_date: str, to_date: str) -> List[Dict]:
        """
        Fetches news from NewsAPI.org.

        Args:
            keywords (List[str]): Keywords to search for (e.g., ["US economy", "NASDAQ"]).
            from_date (str): Start date for news fetching (YYYY-MM-DD).
            to_date (str): End date for news fetching (YYYY-MM-DD).

        Returns:
            List[Dict]: A list of news articles, each a dictionary with standardized keys.
                        Returns an empty list on failure or if no news is found.
        """
        if not self.newsapi_client:
            logger.warning("NewsAPI client is not initialized. Cannot fetch from NewsAPI.")
            # print("NewsAPI client not available; fetching skipped. An empty list will be returned.")
            # This print statement explains that fetching cannot proceed and what the outcome will be.
            return []

        query_string = " OR ".join(f'"{k}"' for k in keywords) # Use quotes for exact phrases
        logger.info(f"Fetching news from NewsAPI with query: '{query_string}', from: {from_date}, to: {to_date}")
        processed_articles: List[Dict] = []
        try:
            # NewsAPI free tier allows fetching articles up to a month old.
            # Page size up to 100. get_everything handles pagination internally for the first 100 results on free tier.
            all_articles_response = self.newsapi_client.get_everything(
                q=query_string,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt', # 'relevancy', 'popularity', 'publishedAt'
                page_size=100 # Max for free tier for 'everything' endpoint
            )
            
            articles = all_articles_response.get('articles', [])
            logger.info(f"NewsAPI returned {len(articles)} articles for query '{query_string}'.")

            for article in articles:
                processed_articles.append({
                    'published_at': article.get('publishedAt'),
                    'source_name': article.get('source', {}).get('name'),
                    'headline': article.get('title'),
                    'description': article.get('description'),
                    'api_source': 'NewsAPI.org',
                    'url': article.get('url'), # Optional, but good to have
                    'raw_data': str(article) # Optional, for debugging
                })
            # print(f"Fetched {len(processed_articles)} articles from NewsAPI. Data is ready for further processing.")
            # This print statement summarizes the fetching result from NewsAPI.
            return processed_articles
        except NewsAPIException as e: # More specific exception handling
            msg = str(e).lower()
            if "apikeyinvalid" in msg or "apikeyrequired" in msg or "apikeymissing" in msg:
                logger.error(f"NewsAPI request failed due to API key issue: {e}. Please check your NEWS_API_ORG_KEY.", exc_info=True)
            elif "ratelimited" in msg:
                logger.warning(f"NewsAPI rate limit hit: {e}. Consider reducing request frequency or upgrading your plan.", exc_info=True)
            elif "maximumresultsreached" in msg:
                 logger.warning(f"NewsAPI maximum results reached for the query/plan: {e}", exc_info=True)
            else:
                logger.error(f"NewsAPI request failed with NewsAPIException: {e}", exc_info=True)
            return []
        except requests.exceptions.RequestException as e: # Catch potential network errors
            logger.error(f"Network error during NewsAPI request: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during NewsAPI fetch: {e}", exc_info=True)
            return []

    # This method fetches general news from Finnhub for a specific category and date range.
    # It queries the API, filters articles by date, and standardizes them.
    # Returns a list of news articles or an empty list if fetching fails or no relevant articles are found.
    def _fetch_news_finnhub(self, category: str, from_date_dt: datetime, to_date_dt: datetime) -> List[Dict]:
        """
        Fetches general news from Finnhub.

        Args:
            category (str): News category (e.g., 'general', 'forex', 'crypto', 'merger').
            from_date_dt (datetime): Start datetime object for filtering.
            to_date_dt (datetime): End datetime object for filtering.

        Returns:
            List[Dict]: A list of news articles, each a dictionary.
                        Returns an empty list on failure or if no news is found.
        """
        if not self.finnhub_client:
            logger.warning("Finnhub client is not initialized. Cannot fetch from Finnhub.")
            # print("Finnhub client not available; fetching skipped. An empty list will be returned.")
            # This print statement explains that fetching cannot proceed and what the outcome will be.
            return []

        logger.info(f"Fetching general news from Finnhub for category: '{category}'")
        processed_articles: List[Dict] = []
        try:
            # Finnhub's general_news does not take date ranges directly.
            # We fetch recent news and filter manually. min_id=0 fetches latest.
            # Free tier limit: 60 calls/minute.
            news_items = self.finnhub_client.general_news(category, min_id=0)
            logger.info(f"Finnhub returned {len(news_items)} general news items for category '{category}'.")

            for item in news_items:
                published_timestamp = item.get('datetime')
                if published_timestamp:
                    # Finnhub 'datetime' is a Unix timestamp (seconds)
                    article_dt = datetime.fromtimestamp(published_timestamp, tz=timezone.utc)
                    # Filter by date range
                    if from_date_dt <= article_dt <= to_date_dt:
                        processed_articles.append({
                            'published_at': article_dt.isoformat(),
                            'source_name': item.get('source'),
                            'headline': item.get('headline'),
                            'description': item.get('summary'), # Finnhub 'summary' is used as description
                            'api_source': 'Finnhub',
                            'url': item.get('url'), # Optional
                            'raw_data': str(item) # Optional
                        })
            logger.info(f"Filtered to {len(processed_articles)} articles from Finnhub within the date range.")
            # print(f"Fetched and filtered {len(processed_articles)} articles from Finnhub. Data is ready for further processing.")
            # This print statement summarizes the fetching and filtering result from Finnhub.
            return processed_articles
        # Attempt to catch more specific Finnhub exceptions if available and documented
        # For example, if finnhub.FinnhubAPIException exists:
        # except finnhub.FinnhubAPIException as e:
        #     if e.status_code == 401: # Unauthorized
        #         logger.error(f"Finnhub API authentication failed (key issue?): {e}", exc_info=True)
        #     elif e.status_code == 429: # Rate limit
        #         logger.warning(f"Finnhub rate limit hit: {e}", exc_info=True)
        #     else:
        #         logger.error(f"Finnhub API request failed: {e}", exc_info=True)
        #     return []
        except requests.exceptions.RequestException as e: # Catch potential network errors
            logger.error(f"Network error during Finnhub request: {e}", exc_info=True)
            return []
        except Exception as e: # General fallback
            logger.error(f"Finnhub request or processing failed: {e}", exc_info=True)
            return []

    # This method orchestrates the news fetching process, trying NewsAPI first and then Finnhub as a fallback.
    # It combines data from the successful source into a Pandas DataFrame.
    # Returns a DataFrame containing the fetched news articles and counts of articles from each source.
    def fetch_market_news(self, newsapi_keywords: List[str], finnhub_category: str,
                            start_date_str: str, end_date_str: str) -> Tuple[pd.DataFrame, int, int]:
        """
        Fetches market news, trying NewsAPI first, then Finnhub as a fallback.

        Args:
            newsapi_keywords (List[str]): Keywords for NewsAPI.
            finnhub_category (str): Category for Finnhub.
            start_date_str (str): Start date (YYYY-MM-DD).
            end_date_str (str): End date (YYYY-MM-DD).

        Returns:
            Tuple[pd.DataFrame, int, int]: A DataFrame with columns
            ['published_at', 'source_name', 'headline', 'description', 'api_source', 'url', 'raw_data'],
            and the counts of articles fetched from NewsAPI and Finnhub respectively.
            Returns an empty DataFrame and zero counts if all sources fail.
        """
        all_fetched_articles: List[Dict] = []
        newsapi_article_count = 0
        finnhub_article_count = 0

        # Try NewsAPI (Primary)
        if self.newsapi_client:
            logger.info("Attempting to fetch news from NewsAPI (Primary)...")
            # Basic rate limit delay before NewsAPI call
            time.sleep(self.get_api_delay("newsapi"))
            newsapi_articles = self._fetch_news_newsapi(newsapi_keywords, start_date_str, end_date_str)
            if newsapi_articles:
                all_fetched_articles.extend(newsapi_articles)
                newsapi_article_count = len(newsapi_articles)
                logger.info(f"Successfully fetched {newsapi_article_count} articles from NewsAPI.")
            else:
                logger.warning("NewsAPI fetch attempt yielded no articles or failed. Will attempt fallback.")
        else:
            logger.warning("NewsAPI client not available. Skipping NewsAPI fetch.")

        # Try Finnhub (Fallback if NewsAPI yielded nothing or client not available)
        if not all_fetched_articles and self.finnhub_client:
            logger.info("Attempting to fetch news from Finnhub (Fallback)...")
            from_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            to_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=999999).replace(tzinfo=timezone.utc)
            
            # Basic rate limit delay before Finnhub call
            time.sleep(self.get_api_delay("finnhub"))
            finnhub_articles = self._fetch_news_finnhub(finnhub_category, from_date_dt, to_date_dt)
            if finnhub_articles:
                all_fetched_articles.extend(finnhub_articles)
                finnhub_article_count = len(finnhub_articles)
                logger.info(f"Successfully fetched {finnhub_article_count} articles from Finnhub as fallback.")
            else:
                logger.warning("Finnhub fetch attempt (fallback) also yielded no articles or failed.")
        elif not all_fetched_articles: # If NewsAPI already got data, or Finnhub client not available
             logger.info("Skipping Finnhub fallback as NewsAPI provided data or Finnhub client is unavailable.")


        # df will be created even if all_fetched_articles is empty, to handle header creation later
        df = pd.DataFrame(all_fetched_articles)

        if not all_fetched_articles:
            logger.warning("No news articles fetched from any source. An empty DataFrame will be used.")
        else:
            # Standardize and ensure required columns only if there's data
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
            # Drop rows where critical fields like 'published_at' or 'headline' are NaT/None AFTER conversion
            df.dropna(subset=['published_at', 'headline'], inplace=True)

            # Ensure all desired columns exist, fill with None if not.
            # 'url' and 'raw_data' are optional for the final CSV but good to have in the DataFrame if present
            final_df_columns = ['published_at', 'source_name', 'headline', 'description', 'api_source', 'url', 'raw_data']
            for col in final_df_columns:
                if col not in df.columns:
                    df[col] = None
            
            df = df[final_df_columns] # Reorder/select columns
            df.sort_values(by='published_at', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info(f"Total news articles processed: {len(df)}. From NewsAPI: {newsapi_article_count}, From Finnhub: {finnhub_article_count}.")

        return df, newsapi_article_count, finnhub_article_count

    # This method provides a delay duration based on the API source to respect rate limits.
    # It helps in managing API call frequencies.
    # Returns a float representing the delay in seconds.
    def get_api_delay(self, source: str) -> float:
        """
        Returns a delay time in seconds for API rate limiting.

        Args:
            source (str): The API source ("newsapi" or "finnhub").

        Returns:
            float: Delay in seconds.
        """
        if source == "finnhub":
            # Finnhub: 60 requests/minute for free tier. Delay slightly more than 1s.
            # print("Calculated API delay for Finnhub. A delay of 1.1 seconds will be applied.")
            # This print statement confirms the delay calculation for Finnhub.
            return 1.1
        elif source == "newsapi":
            # NewsAPI: 100 requests/day for free tier.
            # Individual calls are less frequent, but a small delay can be polite.
            # print("Calculated API delay for NewsAPI. A delay of 0.5 seconds will be applied.")
            # This print statement confirms the delay calculation for NewsAPI.
            return 0.5 # Shorter delay as calls are less frequent overall
        # print("Calculated API delay for an unknown source. A default delay of 0.1 seconds will be applied.")
        # This print statement indicates a default delay for unrecognized sources.
        return 0.1 # Default small delay


# This function defines the main pipeline for fetching, saving, and logging news data.
# It sets up parameters, calls the MarketNewsFetcher, saves results, and logs to backlog.md.
# It does not return anything but orchestrates the entire process.
def run_news_fetching_pipeline() -> None:
    """
    Main function to run the news fetching, saving, and logging pipeline.

    This function orchestrates the entire process:
    1. Defines news sources, keywords, and date ranges.
    2. Instantiates and uses `MarketNewsFetcher`.
    3. Saves the fetched data to a CSV file.
    4. Logs the operation's summary to `log/backlog.md`.
    """
    logger.info("Starting news fetching pipeline...")

    # --- Configuration ---
    NEWSAPI_KEYWORDS = ["US economy", "NASDAQ", "S&P 500", "market sentiment", "stock market", "Federal Reserve", "interest rates"]
    FINNHUB_CATEGORY = 'general' # Other options: 'forex', 'crypto', 'merger'
    DAYS_TO_FETCH = 7 # Fetch news for the last 7 days

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    logger.info(f"Fetching news from {start_date_str} to {end_date_str}.")

    # --- Fetching ---
    fetcher = MarketNewsFetcher()
    news_df, newsapi_count, finnhub_count = fetcher.fetch_market_news(
        newsapi_keywords=NEWSAPI_KEYWORDS,
        finnhub_category=FINNHUB_CATEGORY,
        start_date_str=start_date_str,
        end_date_str=end_date_str
    )

    # --- Saving Data ---
    # Ensure the output directory exists
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created data directory: {OUTPUT_DIR}")
    except OSError as e:
        logger.error(f"Could not create data directory {OUTPUT_DIR}: {e}", exc_info=True)
        # If directory creation fails, we probably can't save, so log and prepare for backlog.
        # The pipeline will still attempt to log to backlog.

    # Define headers for the CSV file, even if it's empty
    output_columns = ['published_at', 'source_name', 'headline', 'description', 'api_source']
    
    # Attempt to save the data or an empty file with headers
    try:
        if not news_df.empty:
            columns_to_save = [col for col in output_columns if col in news_df.columns]
            news_df[columns_to_save].to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
            logger.info(f"Successfully saved {len(news_df)} news articles to {OUTPUT_CSV_PATH}")
        else:
            # Create empty CSV with headers if no data was fetched or processed
            empty_df_with_headers = pd.DataFrame(columns=output_columns)
            empty_df_with_headers.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
            logger.warning(f"No news data to save. Created an empty CSV file with headers at {OUTPUT_CSV_PATH}.")
    except Exception as e:
        logger.error(f"Failed to save news data to CSV {OUTPUT_CSV_PATH}: {e}", exc_info=True)
        # Even if saving fails, proceed to backlog logging to record the attempt and failure.

    # --- Backlog Logging ---
    # This section will now always execute to log the outcome.
    try:
        process_result = subprocess.run(
            [sys.executable, GET_TIME_ID_SCRIPT_PATH],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        time_id_str = process_result.stdout.strip()
        logger.info(f"Successfully executed get_time_id.py: {time_id_str}")

        # Construct summary message based on outcome
        total_articles_saved = len(news_df) if not news_df.empty else 0
        
        if total_articles_saved > 0:
            summary_message = (
                f"Successfully fetched news data for US markets covering the last {DAYS_TO_FETCH} days. "
                f"Retrieved {newsapi_count} articles from NewsAPI and {finnhub_count} articles from Finnhub. "
                f"Saved a total of {total_articles_saved} combined news articles to {os.path.basename(OUTPUT_CSV_PATH)}."
            )
        elif newsapi_count == 0 and finnhub_count == 0 and (fetcher.newsapi_client or fetcher.finnhub_client):
            # This condition implies APIs were attempted but returned no news, or clients were initialized but failed to fetch
            # We need to distinguish if it was "no news found" vs "API failure"
            # For simplicity here, if counts are zero but clients existed, assume "no news found" or "APIs failed but script ran"
            # A more sophisticated check would involve inspecting specific API error flags if set by fetch methods
            summary_message = (
                f"News fetching completed for US markets for the last {DAYS_TO_FETCH} days, but no articles were found or retrieved from NewsAPI or Finnhub. "
                f"An empty {os.path.basename(OUTPUT_CSV_PATH)} with headers was created. "
                f"This allows downstream processes to continue with an empty dataset."
            )
        else: # Covers cases like API keys missing, or other critical init failures before fetching
             summary_message = (
                f"News fetching process for US markets (last {DAYS_TO_FETCH} days) encountered issues preventing data retrieval (e.g., API key problems, client init failures). "
                f"An empty {os.path.basename(OUTPUT_CSV_PATH)} with headers was created to prevent downstream pipeline failures. "
                f"Please check logs for specific errors regarding API connectivity or configuration."
            )

        logger.info(f"Backlog summary: {summary_message}")

        full_log_entry = f"{time_id_str} {summary_message}\n"
        
        existing_content = ""
        if os.path.exists(BACKLOG_FILE_PATH):
            with open(BACKLOG_FILE_PATH, 'r', encoding='utf-8') as f_read:
                existing_content = f_read.read()
        
        with open(BACKLOG_FILE_PATH, 'w', encoding='utf-8') as f_write:
            f_write.write(full_log_entry)
            # Ensure there's a newline separating the new entry from old content,
            # but avoid double newlines if existing_content was empty or already ended with one.
            if existing_content:
                if not existing_content.startswith('\n') and full_log_entry.endswith('\n'):
                     f_write.write(existing_content) # existing_content might not start with \n if file was manually edited
                elif existing_content.startswith('\n'):
                     f_write.write(existing_content)
                else: # existing_content does not start with \n and full_log_entry does not end with \n (unlikely with current format)
                     f_write.write("\n" + existing_content)


        logger.info(f"Successfully prepended summary to {BACKLOG_FILE_PATH}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute {GET_TIME_ID_SCRIPT_PATH} for backlog logging: {e.stderr}", exc_info=True)
    except FileNotFoundError:
        logger.error(f"Script {GET_TIME_ID_SCRIPT_PATH} not found for backlog logging.", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to write backlog log to {BACKLOG_FILE_PATH}: {e}", exc_info=True)

    logger.info("News fetching pipeline finished.")
    return

if __name__ == "__main__":
    # This block allows the script to be run directly from the command line.
    # It calls the main pipeline function to perform all defined tasks.
    run_news_fetching_pipeline()
    # print("Script execution finished. The news fetching pipeline has been run.")
    # This print statement confirms that the main execution block has completed.