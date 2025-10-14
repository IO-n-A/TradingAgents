# core/feature_engineering/technical_indicators_calculator.py
"""
Calculates standard technical indicators (SMA, EMA, RSI, MACD) on daily price data.
It also incorporates VIX and Turbulence metrics if available in the input OHLCV data.
"""
import pandas as pd
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import logging
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# This function calculates various technical indicators from OHLCV data.
# It takes a pandas DataFrame with 'open', 'high', 'low', 'close', and 'volume' columns as input.
# It returns a DataFrame enriched with SMA, EMA, RSI, MACD, and potentially VIX and turbulence data.
def calculate_technical_indicators(
    ohlcv_df: pd.DataFrame,
    sma_windows: Optional[List[int]] = None,
    ema_windows: Optional[List[int]] = None,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9
) -> pd.DataFrame:
    """
    Calculates standard technical indicators and incorporates VIX and Turbulence if present.

    The function computes Simple Moving Averages (SMA), Exponential Moving Averages (EMA),
    Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD)
    for the given OHLCV data. It also appends 'vix' and 'turbulence' columns
    if they exist in the input DataFrame.

    Args:
        ohlcv_df (pd.DataFrame): DataFrame with at least 'open', 'high', 'low', 'close', 'volume' columns.
                                 May also contain 'vix' and 'turbulence' columns.
        sma_windows (Optional[List[int]]): List of window sizes for SMA calculation. Defaults to [20, 50].
        ema_windows (Optional[List[int]]): List of window sizes for EMA calculation. Defaults to [20, 50].
        rsi_window (int): Window size for RSI calculation. Defaults to 14.
        macd_fast (int): Fast period for MACD. Defaults to 12.
        macd_slow (int): Slow period for MACD. Defaults to 26.
        macd_signal (int): Signal period for MACD. Defaults to 9.

    Returns:
        pd.DataFrame: DataFrame with original OHLCV data and added technical indicators.
                      Returns an empty DataFrame if input is empty or 'close' column is missing.
    """
    if sma_windows is None:
        sma_windows = [20, 50]
    if ema_windows is None:
        ema_windows = [20, 50]

    if ohlcv_df.empty:
        logger.warning("Input DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()
    if 'close' not in ohlcv_df.columns:
        logger.error("'close' column is missing from the input DataFrame. Cannot calculate indicators.")
        return pd.DataFrame()

    df = ohlcv_df.copy()

    # Calculate SMAs
    for window in sma_windows:
        if len(df) >= window:
            sma_indicator = SMAIndicator(close=df['close'], window=window, fillna=False)
            df[f'sma_{window}'] = sma_indicator.sma_indicator()
            logger.debug(f"Calculated SMA with window {window}.")
        else:
            logger.warning(f"Not enough data points to calculate SMA with window {window}. Skipping.")
            df[f'sma_{window}'] = pd.NA

    # Calculate EMAs
    for window in ema_windows:
        if len(df) >= window:
            ema_indicator = EMAIndicator(close=df['close'], window=window, fillna=False)
            df[f'ema_{window}'] = ema_indicator.ema_indicator()
            logger.debug(f"Calculated EMA with window {window}.")
        else:
            logger.warning(f"Not enough data points to calculate EMA with window {window}. Skipping.")
            df[f'ema_{window}'] = pd.NA

    # Calculate RSI
    if len(df) >= rsi_window:
        rsi_indicator = RSIIndicator(close=df['close'], window=rsi_window, fillna=False)
        df['rsi'] = rsi_indicator.rsi()
        logger.debug(f"Calculated RSI with window {rsi_window}.")
    else:
        logger.warning(f"Not enough data points to calculate RSI with window {rsi_window}. Skipping.")
        df['rsi'] = pd.NA

    # Calculate MACD
    if len(df) >= macd_slow: # MACD calculation depends on the longest window (slow period)
        macd_indicator = MACD(close=df['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal, fillna=False)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_diff'] = macd_indicator.macd_diff()
        logger.debug(f"Calculated MACD with fast={macd_fast}, slow={macd_slow}, signal={macd_signal}.")
    else:
        logger.warning(f"Not enough data points to calculate MACD. Skipping.")
        df['macd'] = pd.NA
        df['macd_signal'] = pd.NA
        df['macd_diff'] = pd.NA

    # Incorporate VIX and Turbulence if available
    if 'vix' in df.columns:
        logger.info("VIX column found and included.")
    else:
        logger.info("VIX column not found in input data.")
        df['vix'] = pd.NA # Add VIX column with NAs if not present, for schema consistency

    if 'turbulence' in df.columns:
        logger.info("Turbulence column found and included.")
    else:
        logger.info("Turbulence column not found in input data.")
        df['turbulence'] = pd.NA # Add turbulence column with NAs if not present

    # This function has calculated technical indicators such as SMA, EMA, RSI, and MACD.
    # The resulting DataFrame includes these new features alongside the original OHLCV data, plus VIX and turbulence if they were provided.
    print(f"Technical indicators calculated. DataFrame shape: {df.shape}")
    return df

if __name__ == '__main__':
    # Example Usage:
    # Create a sample OHLCV DataFrame
    data = {
        'open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        'high': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5],
        'low': [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5],
        'close': [10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2, 17.2, 18.2, 19.2, 20.2, 21.2, 22.2, 23.2, 24.2, 25.2, 26.2, 27.2, 28.2, 29.2, 30.2, 31.2, 32.2, 33.2, 34.2, 35.2, 36.2, 37.2, 38.2, 39.2, 40.2, 41.2, 42.2, 43.2, 44.2, 45.2, 46.2, 47.2, 48.2, 49.2, 50.2, 51.2, 52.2, 53.2, 54.2, 55.2, 56.2, 57.2, 58.2, 59.2],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900],
        'vix': [15, 16, 15.5, 16.5, 17, 16, 15, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 17.5, 17, 16.5, 16, 15.5, 15, 14.5, 14, 13.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 18.5, 18, 17.5, 17, 16.5, 16, 15.5, 15, 14.5, 14, 13.5, 13, 12.5],
        'turbulence': [0.1, 0.12, 0.11, 0.13, 0.14, 0.12, 0.1, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
    }
    sample_ohlcv_df = pd.DataFrame(data)
    sample_ohlcv_df['date'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(sample_ohlcv_df)))
    sample_ohlcv_df.set_index('date', inplace=True)

    logger.info("Starting technical indicator calculation for sample data...")
    enriched_df = calculate_technical_indicators(sample_ohlcv_df)

    if not enriched_df.empty:
        logger.info("Successfully calculated technical indicators for sample data.")
        print("\nEnriched DataFrame with Technical Indicators (first 5 rows):")
        print(enriched_df.head())
        print("\nEnriched DataFrame with Technical Indicators (last 5 rows):")
        print(enriched_df.tail())
        print(f"\nColumns in enriched DataFrame: {enriched_df.columns.tolist()}")
    else:
        logger.error("Failed to calculate technical indicators for sample data.")

    # Example with missing VIX and Turbulence
    sample_ohlcv_no_extra_df = sample_ohlcv_df[['open', 'high', 'low', 'close', 'volume']].copy()
    logger.info("\nStarting technical indicator calculation for sample data without VIX/Turbulence...")
    enriched_no_extra_df = calculate_technical_indicators(sample_ohlcv_no_extra_df)
    if not enriched_no_extra_df.empty:
        logger.info("Successfully calculated technical indicators for sample data without VIX/Turbulence.")
        print("\nEnriched DataFrame (no VIX/Turbulence input, first 5 rows):")
        print(enriched_no_extra_df.head())
        print(f"\nColumns in enriched DataFrame: {enriched_no_extra_df.columns.tolist()}")
    else:
        logger.error("Failed to calculate technical indicators for sample data without VIX/Turbulence.")

    # Example with insufficient data for some indicators
    short_ohlcv_df = sample_ohlcv_df.head(10).copy()
    logger.info("\nStarting technical indicator calculation for short sample data...")
    enriched_short_df = calculate_technical_indicators(short_ohlcv_df)
    if not enriched_short_df.empty:
        logger.info("Successfully calculated technical indicators for short sample data.")
        print("\nEnriched DataFrame (short data, first 5 rows):")
        print(enriched_short_df.head())
        print(f"\nColumns in enriched DataFrame: {enriched_short_df.columns.tolist()}")
    else:
        logger.error("Failed to calculate technical indicators for short sample data.")