#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check Alpaca Data
----------------
This script checks the data available from Alpaca for March 2024
to ensure we have enough historical data for our backtest.
"""

import json
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from tabulate import tabulate

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_alpaca_api():
    """Initialize Alpaca API
    
    Returns:
        StockHistoricalDataClient: Alpaca Historical Data client
    """
    # Load credentials from file
    with open('alpaca_credentials.json', 'r') as f:
        credentials = json.load(f)
    
    # Use paper trading credentials
    paper_credentials = credentials['paper']
    api_key = paper_credentials['api_key']
    api_secret = paper_credentials['api_secret']
    
    # Initialize Historical Data client
    client = StockHistoricalDataClient(api_key, api_secret)
    
    return client

def get_historical_data(symbols, start_date, end_date, timeframe=TimeFrame.Day):
    """Get historical data for a list of symbols
    
    Args:
        symbols (list): List of symbols to get data for
        start_date (datetime): Start date
        end_date (datetime): End date
        timeframe (TimeFrame): Timeframe for the data
        
    Returns:
        dict: Dictionary of DataFrames with historical data
    """
    client = initialize_alpaca_api()
    
    # Get historical data
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date
    )
    
    try:
        bars = client.get_stock_bars(request_params)
        
        # Convert to dictionary of DataFrames
        data = {}
        for symbol in symbols:
            if symbol in bars.data:
                # Convert bars to DataFrame
                symbol_bars = bars.data[symbol]
                df = pd.DataFrame([bar.dict() for bar in symbol_bars])
                
                # Convert timestamp to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                data[symbol] = df
                logger.info(f"Got {len(df)} bars for {symbol} from {df.index[0]} to {df.index[-1]}")
            else:
                logger.warning(f"No data for {symbol}")
        
        return data
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return {}

def check_data_for_march_2024():
    """Check data available for March 2024"""
    # Define date range
    start_date = dt.datetime(2024, 2, 1)  # Start a month earlier to have enough data for indicators
    end_date = dt.datetime(2024, 3, 31)
    
    # Define symbols from our seasonal configuration
    symbols = [
        'LIN', 'NVDA', 'NEE', 'VLO', 'MRK', 
        'PSX', 'DE', 'EOG', 'CVX', 'XLU'
    ]
    
    logger.info(f"Checking data for {len(symbols)} symbols from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get data
    data = get_historical_data(symbols, start_date, end_date)
    
    # Print summary
    if data:
        print("\nData Summary:")
        summary = []
        
        for symbol, df in data.items():
            start = df.index[0].strftime('%Y-%m-%d')
            end = df.index[-1].strftime('%Y-%m-%d')
            days = len(df)
            
            # Calculate basic statistics
            avg_close = df['close'].mean()
            avg_volume = df['volume'].mean()
            
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Convert to percentage
            
            summary.append([
                symbol, 
                start, 
                end, 
                days, 
                f"${avg_close:.2f}", 
                f"{avg_volume:.0f}",
                f"{volatility:.2f}%"
            ])
        
        # Print table
        headers = ["Symbol", "Start Date", "End Date", "Trading Days", "Avg Close", "Avg Volume", "Volatility"]
        print(tabulate(summary, headers=headers, tablefmt='grid'))
        
        # Check for any gaps in the data
        print("\nChecking for gaps in the data...")
        
        for symbol, df in data.items():
            # Get all business days in the range
            all_days = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Find missing days
            missing_days = all_days.difference(df.index)
            
            if len(missing_days) > 0:
                print(f"\n{symbol} is missing {len(missing_days)} trading days:")
                for day in missing_days:
                    print(f"  - {day.strftime('%Y-%m-%d')}")
            else:
                print(f"{symbol}: No missing trading days")
    else:
        logger.error("No data returned")

if __name__ == "__main__":
    check_data_for_march_2024()
