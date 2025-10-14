# MLOps/pipelines/ingestion/market_news_scheduled_ingest.py
# This script fetches general financial news headlines and summaries relevant to the US market.
# It supports NewsAPI.org as the primary source and Finnhub as a fallback,
# configurable via command-line arguments. The script manages API keys, respects API limits,
# logs its operations, and saves data in a DVC-compatible format.

import argparse
import datetime
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Union

import requests
import yaml
import finnhub
from newsapi import NewsApiClient

# Determine project root and add to sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sys
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import custom logging setup
try:
    from config.logging_config import setup_logging, SUCCESS_LEVEL_NUM
    setup_logging() # Initialize logging for the script
    logger = logging.getLogger(__name__)
    logger.info("Custom logging initialized successfully for market_news_scheduled_ingest.py.")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    SUCCESS_LEVEL_NUM = 25
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
    logging.Logger.success = success
    logger.warning(
        "Could not import custom logging_config. Using basic logging for market_news_scheduled_ingest.py."
    )

# --- Constants and Configuration Loading ---
DEFAULT_API_KEYS_PATH: str = os.path.join(project_root, "config/api_keys.yaml")
DEFAULT_DATA_SOURCES_CONFIG_PATH: str = os.path.join(project_root, "MLOps/config/data_sources.yaml")
DATA_OUTPUT_BASE_DIR: str = "data/raw/news_articles" # Default, can be overridden by data_sources.yaml
DATE_FORMAT: str = "%Y-%m-%d"

# API Limits (can be fine-tuned in data_sources.yaml if needed, but script defaults are sensible)
NEWSAPI_MAX_REQUESTS_PER_DAY: int = 100
NEWSAPI_PAGE_SIZE: int = 100
FINNHUB_MAX_REQUESTS_PER_MINUTE: int = 60
FINNHUB_DELAY_BETWEEN_CALLS: float = 60.0 / FINNHUB_MAX_REQUESTS_PER_MINUTE

# Default queries/categories (will be overridden by data_sources.yaml)
DEFAULT_NEWSAPI_QUERIES: List[str] = ["US stock market"]
DEFAULT_FINNHUB_CATEGORY: str = "general"

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                logger.warning(f"Configuration file '{config_path}' is empty.")
                return {}
            logger.info(f"Configuration loaded successfully from '{config_path}'.")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        return {}

def load_api_keys(keys_path: str = DEFAULT_API_KEYS_PATH) -> Dict[str, Any]:
    """
    Loads API keys from a YAML configuration file.
    This function securely retrieves API keys needed for accessing news services.
    It handles file errors and returns a dictionary of API keys.

    Args:
        keys_path (str): Path to the API keys YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing API keys.
                        Returns an empty dictionary if the file is not found, is empty, or an error occurs.
    """
    # This function loads API keys from the specified YAML file.
    # It handles potential file errors and returns a dictionary of keys.
    # These keys are essential for authenticating with news providers.
    try:
        with open(keys_path, "r") as f:
            api_keys = yaml.safe_load(f)
            if api_keys is None:
                logger.warning(f"API keys file '{keys_path}' is empty.")
                return {}
            logger.success(f"API keys loaded successfully from '{keys_path}'.", extra={'filename_summary': __name__})
            return api_keys
    except FileNotFoundError:
        logger.error(
            f"CRITICAL: API keys file '{keys_path}' not found. Cannot proceed without API keys.",
            extra={'filename_summary': __name__}
        )
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing API keys file '{keys_path}': {e}", extra={'filename_summary': __name__})
        return {}
    # print(f"API keys loading attempted from '{keys_path}'.")
    # print("If successful, keys for NewsAPI and Finnhub should be available for the script.")


def fetch_news_from_newsapi(
    api_key: str,
    queries: List[str],
    page_size: int, # Now passed from config
    language: str,  # Now passed from config
    sort_by: str,   # Now passed from config
) -> List[Dict[str, Any]]:
    """
    Fetches news articles from NewsAPI.org based on a list of queries.
    This function iterates through the provided queries, making one API request per query
    to collect relevant news articles. It respects the API's page size and focuses on recent news.

    Args:
        api_key (str): The API key for NewsAPI.org.
        queries (List[str]): A list of search terms (e.g., "NASDAQ", "US economy").
        page_size (int): Number of articles to fetch per query (max 100).
        language (str): Language of the articles (e.g., "en").
        sort_by (str): How to sort articles (e.g., "publishedAt", "relevancy", "popularity").

    Returns:
        List[Dict[str, Any]]: A list of fetched news articles. Each article is a dictionary.
    """
    # This function connects to NewsAPI to retrieve articles for specified queries.
    # It manages API authentication and parameters for fetching news.
    # The result is a list of article data, or an empty list if issues occur.
    all_articles: List[Dict[str, Any]] = []
    if not api_key:
        logger.error("NewsAPI key is missing. Cannot fetch news from NewsAPI.org.", extra={'filename_summary': __name__})
        return all_articles

    newsapi = NewsApiClient(api_key=api_key)
    # NewsAPI free tier has a limit of 100 requests per day.
    # Each query below is one request.
    # We should not exceed NEWSAPI_MAX_REQUESTS_PER_DAY in a single run if script runs multiple times.
    # For a daily script, a few broad queries are typical.

    # Calculate date range for "today" or "recent" news. NewsAPI `everything` endpoint
    # is better for historical, but `top-headlines` or `everything` with a from_param can work.
    # For general market news, `everything` with recent dates is often best.
    # Let's fetch from yesterday up to today to ensure recent articles are captured.
    today_dt = datetime.date.today()
    yesterday_dt = today_dt - datetime.timedelta(days=1)
    today_str = today_dt.strftime(DATE_FORMAT)
    yesterday_str = yesterday_dt.strftime(DATE_FORMAT)

    logger.info(f"Fetching news from NewsAPI.org for queries: {queries} for date range: {yesterday_str} to {today_str}", extra={'filename_summary': __name__})

    for query_idx, q in enumerate(queries):
        if query_idx >= NEWSAPI_MAX_REQUESTS_PER_DAY: # Simple check, assumes 1 query = 1 request
            logger.warning(f"Approaching NewsAPI daily request limit. Stopping at query: {q}", extra={'filename_summary': __name__})
            break
        try:
            logger.info(f"Fetching NewsAPI query ({query_idx+1}/{len(queries)}): '{q}'", extra={'filename_summary': __name__})
            response = newsapi.get_everything(
                q=q,
                language=language,
                sort_by=sort_by,
                page_size=page_size, # Get up to 100 articles for this query
                from_param=yesterday_str, # Articles from yesterday
                to=today_str             # Articles up to today
            )

            if response.get("status") == "ok":
                articles = response.get("articles", [])
                logger.success(
                    f"Successfully fetched {len(articles)} articles for query '{q}'. Total so far: {len(all_articles) + len(articles)}",
                    extra={'filename_summary': __name__, 'data_payload': {'query': q, 'count': len(articles)}}
                )
                all_articles.extend(articles)
            else:
                error_message = response.get('message', 'Unknown error')
                logger.error(f"Error fetching news for query '{q}' from NewsAPI: {error_message}", extra={'filename_summary': __name__})
            
            # Optional: Add a small delay if making many queries, though NewsAPI's main limit is daily.
            # time.sleep(0.5) # Not strictly needed for daily limit but good for server politeness

        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException fetching news for query '{q}': {e}", exc_info=True, extra={'filename_summary': __name__})
        except Exception as e:
            logger.error(f"Unexpected error fetching news for query '{q}': {e}", exc_info=True, extra={'filename_summary': __name__})
    
    # print(f"NewsAPI fetching complete. Total articles retrieved: {len(all_articles)}.")
    # print("This list contains dictionaries, each representing a news article with its details.")
    return all_articles


def fetch_news_from_finnhub(
    api_key: str, category: str = DEFAULT_FINNHUB_CATEGORY, min_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches general market news from Finnhub.
    This function retrieves news articles for a specified category. Finnhub's free tier
    has a rate limit (60 calls/minute), so delays are incorporated if multiple calls were needed
    (though for general news, one call might be sufficient for recent items).

    Args:
        api_key (str): The API key for Finnhub.
        category (str): News category (e.g., 'general', 'forex', 'crypto').
        min_id (Optional[str]): For pagination, the ID of the last news item to get older news.

    Returns:
        List[Dict[str, Any]]: A list of fetched news articles. Each article is a dictionary.
    """
    # This function connects to Finnhub to retrieve news for a given category.
    # It handles API authentication and respects rate limits.
    # The result is a list of article data, or an empty list on failure.
    all_articles: List[Dict[str, Any]] = []
    if not api_key:
        logger.error("Finnhub API key is missing. Cannot fetch news from Finnhub.", extra={'filename_summary': __name__})
        return all_articles

    finnhub_client = finnhub.Client(api_key=api_key)
    logger.info(f"Fetching news from Finnhub for category: {category}", extra={'filename_summary': __name__})

    try:
        # Finnhub's market_news returns recent news.
        # For free tier, it's usually limited in how far back it goes.
        # Pagination is via min_id, but for a daily script, one call for recent news is typical.
        if min_id:
             # This part is more for fetching historical data in chunks, less for daily "latest"
            logger.info(f"Fetching Finnhub news with min_id: {min_id}", extra={'filename_summary': __name__})
            # Finnhub API does not directly support 'count' or 'page_size' for market_news in the same way as NewsAPI.
            # It returns a batch (typically 50-200 items for free tier) and min_id for next batch.
            # For a daily script, we usually want the latest, so min_id is less relevant unless fetching history.
            news_items = finnhub_client.general_news(category, min_id=min_id)
        else:
            news_items = finnhub_client.general_news(category) # Fetches most recent

        if news_items: # news_items is a list of dictionaries
            logger.success(
                f"Successfully fetched {len(news_items)} articles from Finnhub for category '{category}'.",
                extra={'filename_summary': __name__, 'data_payload': {'category': category, 'count': len(news_items)}}
            )
            all_articles.extend(news_items)
        else:
            logger.info(f"No news articles returned from Finnhub for category '{category}'.", extra={'filename_summary': __name__})

        # Respect Finnhub's 60 req/min limit if making multiple calls in a loop (not typical for this script's daily run)
        # time.sleep(FINNHUB_DELAY_BETWEEN_CALLS) # Only if looping for more categories or pagination

    except finnhub.FinnhubAPIException as e:
        logger.error(f"FinnhubAPIException fetching news for category '{category}': {e}", exc_info=True, extra={'filename_summary': __name__})
    except Exception as e:
        logger.error(f"Unexpected error fetching news from Finnhub for category '{category}': {e}", exc_info=True, extra={'filename_summary': __name__})
    
    # print(f"Finnhub fetching complete. Total articles retrieved: {len(all_articles)}.")
    # print("This list contains dictionaries, each representing a news article from Finnhub.")
    return all_articles


def save_news_data(
    articles: List[Dict[str, Any]], base_path: str, current_date_str: str, source_api_name: str
) -> None:
    """
    Saves the collected news articles to JSON files in a structured directory.
    Each article is saved as an individual JSON file. The directory structure is
    `base_path/YYYY-MM-DD/article_SOURCE_TIMESTAMP_ID.json` or similar to ensure uniqueness.

    Args:
        articles (List[Dict[str, Any]]): List of article dictionaries.
        base_path (str): The base directory for saving data (e.g., 'data/raw/news_articles').
        current_date_str (str): The current date as a string (YYYY-MM-DD) for the subdirectory.
        source_api_name (str): Name of the API source (e.g., "newsapi", "finnhub") for filename prefix.
    """
    # This function writes the fetched news articles to individual JSON files.
    # It organizes files by date and includes source information in filenames for clarity.
    # Proper error handling for file I/O is implemented.
    output_dir = os.path.join(base_path, current_date_str)
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}", extra={'filename_summary': __name__})
    except OSError as e:
        logger.error(f"Error creating directory {output_dir}: {e}", exc_info=True, extra={'filename_summary': __name__})
        return

    if not articles:
        logger.info("No articles to save.", extra={'filename_summary': __name__})
        return

    for idx, article_data in enumerate(articles):
        try:
            # Create a somewhat unique filename. Using a hash of title or URL could be more robust.
            # For simplicity, using index and a timestamp from the article if available.
            timestamp_str = ""
            if 'publishedAt' in article_data and article_data['publishedAt']: # NewsAPI
                dt_obj = datetime.datetime.fromisoformat(article_data['publishedAt'].replace('Z', '+00:00'))
                timestamp_str = dt_obj.strftime("%Y%m%dT%H%M%S")
            elif 'datetime' in article_data and article_data['datetime']: # Finnhub (Unix timestamp)
                dt_obj = datetime.datetime.fromtimestamp(article_data['datetime'], tz=datetime.timezone.utc)
                timestamp_str = dt_obj.strftime("%Y%m%dT%H%M%S")
            
            article_id_part = article_data.get('id', str(idx)) # Finnhub has 'id', NewsAPI doesn't have a simple one.
            if not article_id_part: article_id_part = str(idx)


            # Sanitize title for filename or use a generic ID
            title_part = article_data.get('title', 'untitled')
            safe_title = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in title_part)[:50].strip().replace(' ', '_')
            if not safe_title: safe_title = "article"

            filename = f"{source_api_name}_{timestamp_str}_{safe_title}_{article_id_part}.json"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(article_data, f, ensure_ascii=False, indent=4)
            logger.debug(f"Successfully saved article to {file_path}", extra={'filename_summary': __name__})
        except Exception as e:
            logger.error(
                f"Error saving article {idx} to JSON: {e}",
                exc_info=True,
                extra={'filename_summary': __name__, 'data_payload': {'article_index': idx}}
            )
    logger.success(f"Completed saving {len(articles)} articles to {output_dir}", extra={'filename_summary': __name__})
    # print(f"News data saving process completed. JSON files were written to '{output_dir}'.")
    # print(f"Each article is in its own JSON file, prefixed by source and timestamp.")


def main():
    """
    Main function to orchestrate the news data ingestion process.
    It parses arguments for selecting the news source, loads configurations and API keys,
    fetches news according to the selected source and its config, and saves it.
    """
    parser = argparse.ArgumentParser(
        description="Fetch financial news from NewsAPI.org or Finnhub based on MLOps/config/data_sources.yaml."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["newsapi", "finnhub"],
        required=True,
        help="The news API source to use ('newsapi' or 'finnhub'). Must be enabled in data_sources.yaml.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_DATA_SOURCES_CONFIG_PATH,
        help="Path to the data sources YAML configuration file."
    )
    parser.add_argument(
        "--api_keys_path",
        type=str,
        default=DEFAULT_API_KEYS_PATH,
        help="Path to the API keys YAML file."
    )
    args = parser.parse_args()

    logger.info(f"Starting market news ingestion process with source: {args.source} using config: {args.config_path}")

    # Load configurations
    data_config = load_yaml_config(args.config_path)
    if not data_config:
        logger.critical("Failed to load data_sources.yaml. Exiting.")
        return

    api_keys = load_api_keys(args.api_keys_path)
    if not api_keys:
        logger.critical("No API keys loaded. Exiting.")
        return

    # Determine output directory from config or use default
    output_base_dir = data_config.get("data_paths", {}).get("raw_news_data_dir", DATA_OUTPUT_BASE_DIR)

    current_date_str = datetime.date.today().strftime(DATE_FORMAT)
    all_articles: List[Dict[str, Any]] = []

    news_sources_config = data_config.get("news_data", {}).get("sources", [])
    selected_source_config = None
    for src_cfg in news_sources_config:
        if src_cfg.get("source_name") == args.source:
            selected_source_config = src_cfg
            break
    
    if not selected_source_config:
        logger.error(f"Configuration for source '{args.source}' not found in {args.config_path}.")
        return
    
    if not selected_source_config.get("enabled", False):
        logger.warning(f"News source '{args.source}' is not enabled in the configuration. Exiting.")
        return

    if args.source == "newsapi":
        newsapi_key = api_keys.get("NEWS_API_ORG_KEY")
        if not newsapi_key:
            logger.critical("NEWS_API_ORG_KEY not found in api_keys.yaml. Cannot use NewsAPI.")
            return
        
        queries_to_use = selected_source_config.get("queries", DEFAULT_NEWSAPI_QUERIES)
        language = selected_source_config.get("language", "en")
        sort_by = selected_source_config.get("sort_by", "publishedAt")
        page_size = selected_source_config.get("page_size", NEWSAPI_PAGE_SIZE)

        logger.info(f"Using NewsAPI with queries: {queries_to_use}, lang: {language}, sort: {sort_by}, page_size: {page_size}")
        all_articles = fetch_news_from_newsapi(
            api_key=newsapi_key,
            queries=queries_to_use,
            page_size=page_size,
            language=language,
            sort_by=sort_by
        )

    elif args.source == "finnhub":
        finnhub_key = api_keys.get("FINNHUB_API_KEY")
        if not finnhub_key:
            logger.critical("FINNHUB_API_KEY not found in api_keys.yaml. Cannot use Finnhub.")
            return
        
        category_to_use = selected_source_config.get("category", DEFAULT_FINNHUB_CATEGORY)
        # Finnhub delay is handled by a constant FINNHUB_DELAY_BETWEEN_CALLS if multiple calls were made,
        # but current fetch_news_from_finnhub makes one call.
        logger.info(f"Using Finnhub with category: {category_to_use}")
        all_articles = fetch_news_from_finnhub(api_key=finnhub_key, category=category_to_use)

    else:
        logger.error(f"Invalid source specified: {args.source}") # Should be caught by argparse
        return

    if all_articles:
        save_news_data(all_articles, output_base_dir, current_date_str, args.source)
        logger.success(f"Successfully fetched and saved {len(all_articles)} articles from {args.source}.")
    else:
        logger.warning(f"No articles were fetched from {args.source}. Nothing to save.")

    logger.info("Market news ingestion process completed.")


if __name__ == "__main__":
    main()