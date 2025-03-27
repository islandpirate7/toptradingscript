#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock data generator for backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_mock_price_data(symbol, start_date, end_date, seed=None):
    """
    Generate mock price data for a symbol.
    
    Args:
        symbol (str): Symbol to generate data for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Convert dates to datetime
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Add 60 days of history to ensure we have enough data for indicators
    extended_start = start - pd.Timedelta(days=60)
    
    # Generate date range
    date_range = pd.date_range(start=extended_start, end=end, freq='D')
    
    # Filter out weekends
    date_range = date_range[date_range.dayofweek < 5]
    
    # Generate random price data
    base_price = random.uniform(10, 500)  # Random starting price
    volatility = random.uniform(0.01, 0.05)  # Random volatility
    
    # Generate daily returns with slight upward bias
    daily_returns = np.random.normal(0.0005, volatility, size=len(date_range))
    
    # Calculate price series
    prices = [base_price]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]  # Remove the initial base price
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': prices,
        'high': [p * (1 + random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - random.uniform(0, 0.02)) for p in prices],
        'close': [p * (1 + random.uniform(-0.01, 0.01)) for p in prices],
        'volume': [int(random.uniform(100000, 10000000)) for _ in range(len(prices))]
    })
    
    return df

def get_mock_bars(symbols, timeframe, start, end, seed=42):
    """
    Get mock bars for multiple symbols.
    
    Args:
        symbols (list): List of symbols to get data for
        timeframe (str): Timeframe for the bars (e.g., '1D', '1H')
        start (pd.Timestamp): Start date/time
        end (pd.Timestamp): End date/time
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data for all symbols
    """
    all_dfs = []
    
    for symbol in symbols:
        # Generate mock data for this symbol
        df = generate_mock_price_data(
            symbol, 
            start.strftime('%Y-%m-%d'), 
            end.strftime('%Y-%m-%d'),
            seed=seed + hash(symbol) % 10000  # Different seed for each symbol
        )
        
        # Add symbol as a column
        df['symbol'] = symbol
        
        all_dfs.append(df)
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs)
        
        # Set multi-index
        combined_df = combined_df.set_index(['symbol', 'timestamp'])
        
        return combined_df
    else:
        return pd.DataFrame()
