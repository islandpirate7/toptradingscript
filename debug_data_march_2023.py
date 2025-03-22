#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Data for March 2023
--------------------------------------
This script checks the quality of data for March 2023 and
prints detailed information about the data to help debug issues.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_alpaca_credentials():
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials.get('paper', {})
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {e}")
        return {}

def get_alpaca_client():
    """Get Alpaca API client"""
    try:
        import alpaca_trade_api as tradeapi
        
        credentials = load_alpaca_credentials()
        api_key = credentials.get('api_key', '')
        api_secret = credentials.get('api_secret', '')
        base_url = credentials.get('base_url', 'https://paper-api.alpaca.markets')
        
        if not api_key or not api_secret:
            logger.error("Missing Alpaca API credentials")
            return None
        
        return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    except Exception as e:
        logger.error(f"Error creating Alpaca client: {e}")
        return None

def get_historical_data(symbol, start_date, end_date, timeframe='1D'):
    """Get historical price data from Alpaca
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date
        end_date (datetime): End date
        timeframe (str): Timeframe (e.g. '1D', '1H')
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    try:
        # Get Alpaca client
        api = get_alpaca_client()
        if api is None:
            return None
        
        # Format dates as strings in YYYY-MM-DD format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get bars
        bars = api.get_bars(
            symbol,
            timeframe,
            start=start_str,
            end=end_str,
            adjustment='raw'
        ).df
        
        # Reset index to make timestamp a column
        if not bars.empty:
            bars = bars.reset_index()
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in bars.columns and not pd.api.types.is_datetime64_any_dtype(bars['timestamp']):
                bars['timestamp'] = pd.to_datetime(bars['timestamp'])
                
            logger.info(f"Fetched {len(bars)} bars for {symbol}")
        else:
            logger.warning(f"No data returned for {symbol}")
            
        return bars
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def check_data_quality(df, symbol):
    """Check data quality and print detailed information
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        symbol (str): Stock symbol
        
    Returns:
        dict: Dictionary with data quality metrics
    """
    if df is None or df.empty:
        logger.warning(f"No data for {symbol}")
        return {
            'symbol': symbol,
            'data_available': False,
            'trading_days': 0,
            'has_gaps': False,
            'has_zeros': False,
            'has_nulls': False,
            'price_range': (0, 0),
            'avg_volume': 0,
            'potential_buy_signals': 0,
            'potential_sell_signals': 0,
            'ultra_buy_signals': 0,
            'ultra_sell_signals': 0
        }
    
    # Check for missing days
    if 'timestamp' in df.columns:
        df['date'] = df['timestamp'].dt.date
        unique_dates = df['date'].nunique()
        date_range = (df['date'].min(), df['date'].max())
        expected_days = np.busday_count(date_range[0], date_range[1]) + 1
        has_gaps = unique_dates < expected_days
    else:
        unique_dates = 0
        has_gaps = True
    
    # Check for zeros in price data
    has_zeros = (df['close'] == 0).any() or (df['open'] == 0).any()
    
    # Check for nulls
    has_nulls = df.isnull().any().any()
    
    # Get price range
    price_range = (df['low'].min(), df['high'].max())
    
    # Get average volume
    avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
    
    # Calculate basic indicators for signal generation testing
    if len(df) > 20:
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Bollinger Bands
        df['std_20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Count potential signals
        potential_buy_signals = ((df['close'] < df['bb_lower']) & (df['rsi_14'] < 30)).sum()
        potential_sell_signals = ((df['close'] > df['bb_upper']) & (df['rsi_14'] > 70)).sum()
        
        # Ultra aggressive signals (using much more relaxed conditions)
        ultra_buy_signals = ((df['close'] < df['sma_20']) | (df['rsi_14'] < 45)).sum()
        ultra_sell_signals = ((df['close'] > df['sma_20']) | (df['rsi_14'] > 55)).sum()
    else:
        potential_buy_signals = 0
        potential_sell_signals = 0
        ultra_buy_signals = 0
        ultra_sell_signals = 0
    
    return {
        'symbol': symbol,
        'data_available': True,
        'trading_days': unique_dates,
        'expected_days': expected_days if 'expected_days' in locals() else 0,
        'has_gaps': has_gaps,
        'has_zeros': has_zeros,
        'has_nulls': has_nulls,
        'price_range': price_range,
        'avg_volume': avg_volume,
        'potential_buy_signals': potential_buy_signals,
        'potential_sell_signals': potential_sell_signals,
        'ultra_buy_signals': ultra_buy_signals,
        'ultra_sell_signals': ultra_sell_signals
    }

def debug_data_march_2023(symbols):
    """Debug data for March 2023
    
    Args:
        symbols (list): List of stock symbols
        
    Returns:
        pd.DataFrame: DataFrame with data quality metrics
    """
    # Define date range
    start_date = dt.datetime(2023, 3, 1)
    end_date = dt.datetime(2023, 3, 31)
    
    logger.info(f"Checking data quality for {len(symbols)} symbols from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get data for each symbol
    results = []
    data_frames = {}
    
    for symbol in symbols:
        df = get_historical_data(symbol, start_date, end_date)
        data_frames[symbol] = df
        results.append(check_data_quality(df, symbol))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nData Quality Summary:")
    print(tabulate(results_df, headers='keys', tablefmt='grid'))
    
    # Print detailed signal potential
    print("\nSignal Potential Summary:")
    signal_df = results_df[['symbol', 'potential_buy_signals', 'potential_sell_signals', 'ultra_buy_signals', 'ultra_sell_signals']]
    print(tabulate(signal_df, headers='keys', tablefmt='grid'))
    
    # Calculate and print overall statistics
    print("\nOverall Statistics:")
    print(f"Total symbols checked: {len(symbols)}")
    print(f"Symbols with data: {results_df['data_available'].sum()}")
    print(f"Symbols with gaps: {results_df['has_gaps'].sum()}")
    print(f"Symbols with zeros: {results_df['has_zeros'].sum()}")
    print(f"Symbols with nulls: {results_df['has_nulls'].sum()}")
    print(f"Average trading days: {results_df['trading_days'].mean():.1f}")
    print(f"Total potential buy signals: {results_df['potential_buy_signals'].sum()}")
    print(f"Total potential sell signals: {results_df['potential_sell_signals'].sum()}")
    print(f"Total ultra aggressive buy signals: {results_df['ultra_buy_signals'].sum()}")
    print(f"Total ultra aggressive sell signals: {results_df['ultra_sell_signals'].sum()}")
    
    return results_df, data_frames

if __name__ == "__main__":
    # Define symbols to check
    symbols = [
        'LIN', 'NVDA', 'NEE', 'VLO', 'MRK', 'PSX', 'DE', 'EOG', 'CVX', 'XLU',
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'PG'
    ]
    
    # Debug data
    results_df, data_frames = debug_data_march_2023(symbols)
    
    # Save results to CSV for further analysis
    results_df.to_csv('data_quality_march_2023.csv', index=False)
    logger.info("Saved data quality results to data_quality_march_2023.csv")
