#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Alpaca Historical Data Access
---------------------------------
This script tests access to Alpaca's historical data API specifically,
focusing on 2023 data which should be accessible with a free tier account.
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

def test_alpaca_historical():
    """Test access to Alpaca historical data API"""
    # Get API credentials
    api_key = "PK3MIMOSIMVY8A9IYXE5"
    api_secret = "GMXfCCDGQYSPyGrJZPwrIUUgmMO5XIOhXKWJnL3f"
    
    logger.info(f"Testing Alpaca historical data API with key: {api_key[:5]}...")
    
    # Initialize Alpaca API client specifically for data
    api = REST(
        api_key,
        api_secret,
        base_url='https://data.alpaca.markets'
    )
    
    # Test fetching historical data from 2022 (should be available with free tier)
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2022-01-31"
    
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
            end=end.isoformat(),
            adjustment='raw'
        ).df
        
        if bars.empty:
            logger.warning(f"No data found for {symbol} in the specified date range")
            return False
        
        logger.info(f"Successfully fetched {len(bars)} bars for {symbol}")
        logger.info(f"First bar: {bars.iloc[0]}")
        logger.info(f"Last bar: {bars.iloc[-1]}")
        
        # Try a different date range
        logger.info("Trying a different date range from 2023...")
        start_date2 = "2023-06-01"
        end_date2 = "2023-06-30"
        
        start2 = pd.Timestamp(start_date2, tz='America/New_York')
        end2 = pd.Timestamp(end_date2, tz='America/New_York')
        
        bars2 = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start2.isoformat(),
            end=end2.isoformat(),
            adjustment='raw'
        ).df
        
        if not bars2.empty:
            logger.info(f"Successfully fetched {len(bars2)} bars for {symbol} in second date range")
            logger.info(f"First bar: {bars2.iloc[0]}")
            logger.info(f"Last bar: {bars2.iloc[-1]}")
        
        return True
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        
        # Try with a different API method
        try:
            logger.info("Trying with get_bars_v2 method...")
            request_params = api.get_bars_v2(
                symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                timeframe=TimeFrame.Day,
                adjustment='raw'
            )
            
            bars_v2 = []
            for bar in request_params:
                bars_v2.append(bar)
            
            if not bars_v2:
                logger.warning(f"No data found for {symbol} in the specified date range using v2 API")
                return False
            
            logger.info(f"Successfully fetched {len(bars_v2)} bars for {symbol} using v2 API")
            logger.info(f"First bar: {bars_v2[0]}")
            logger.info(f"Last bar: {bars_v2[-1]}")
            
            return True
        except Exception as e2:
            logger.error(f"Error fetching historical data with v2 API: {e2}")
            return False

if __name__ == "__main__":
    success = test_alpaca_historical()
    if success:
        logger.info("Alpaca historical data API test successful!")
    else:
        logger.error("Alpaca historical data API test failed!")
