#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Alpaca API Connection
--------------------------
This script tests the connection to Alpaca API and fetches historical data
for a single symbol to verify that the API credentials are working correctly.
"""

import os
import logging
import datetime
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_alpaca_connection():
    """Test connection to Alpaca API and fetch historical data"""
    # Get API credentials
    api_key = "PK3MIMOSIMVY8A9IYXE5"
    api_secret = "GMXfCCDGQYSPyGrJZPwrIUUgmMO5XIOhXKWJnL3f"
    
    logger.info(f"Testing Alpaca API connection with key: {api_key[:5]}...")
    
    # Initialize Alpaca API client - try both endpoints
    endpoints = [
        'https://paper-api.alpaca.markets',  # Paper trading API
        'https://api.alpaca.markets'         # Live trading API
    ]
    
    api = None
    account = None
    
    for endpoint in endpoints:
        try:
            logger.info(f"Trying endpoint: {endpoint}")
            api = REST(
                api_key,
                api_secret,
                base_url=endpoint
            )
            
            # Test account info
            account = api.get_account()
            logger.info(f"Successfully connected to Alpaca API at {endpoint}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Account equity: ${float(account.equity):.2f}")
            break
        except Exception as e:
            logger.warning(f"Failed to connect to {endpoint}: {e}")
    
    if not account:
        logger.error("Could not connect to any Alpaca API endpoint")
        return False
    
    # Test fetching historical data - use 2023 data as per free tier limitations
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    try:
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to datetime objects
        start = pd.Timestamp(start_date, tz='America/New_York')
        end = pd.Timestamp(end_date, tz='America/New_York')
        
        # Fetch data
        bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat()
        ).df
        
        if bars.empty:
            logger.warning(f"No data found for {symbol} in the specified date range")
            return False
        
        logger.info(f"Successfully fetched {len(bars)} bars for {symbol}")
        logger.info(f"First bar: {bars.iloc[0]}")
        logger.info(f"Last bar: {bars.iloc[-1]}")
        
        return True
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        
        # Try with data API endpoint
        try:
            logger.info("Trying with data API endpoint...")
            api = REST(
                api_key,
                api_secret,
                base_url='https://data.alpaca.markets'
            )
            
            bars = api.get_bars(
                symbol,
                TimeFrame.Day,
                start=start.isoformat(),
                end=end.isoformat()
            ).df
            
            if bars.empty:
                logger.warning(f"No data found for {symbol} in the specified date range")
                return False
            
            logger.info(f"Successfully fetched {len(bars)} bars for {symbol}")
            logger.info(f"First bar: {bars.iloc[0]}")
            logger.info(f"Last bar: {bars.iloc[-1]}")
            
            return True
        except Exception as e2:
            logger.error(f"Error fetching historical data with data API: {e2}")
            return False

if __name__ == "__main__":
    success = test_alpaca_connection()
    if success:
        logger.info("Alpaca API connection test successful!")
    else:
        logger.error("Alpaca API connection test failed!")
