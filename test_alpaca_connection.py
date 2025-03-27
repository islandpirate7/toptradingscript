#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Alpaca API Connection
--------------------------
This script tests the connection to the Alpaca API and verifies that the API keys are working.
"""

import os
import yaml
import logging
from datetime import datetime, timedelta
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = 'sp500_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        logger.error(f"Configuration file {config_path} not found")
        return None

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    # Load configuration
    config = load_config()
    if not config or 'alpaca' not in config:
        logger.error("Alpaca configuration not found")
        return False
    
    # Get API credentials
    api_key = config['alpaca']['api_key']
    api_secret = config['alpaca']['api_secret']
    base_url = config['alpaca']['base_url']
    
    logger.info(f"Testing Alpaca API connection with key: {api_key[:5]}...")
    logger.info(f"Using base URL: {base_url}")
    
    try:
        # Initialize Alpaca API client for trading
        api = REST(
            api_key,
            api_secret,
            base_url=base_url
        )
        
        # Test account access
        account = api.get_account()
        logger.info(f"Successfully connected to Alpaca API")
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Account equity: {account.equity}")
        logger.info(f"Account cash: {account.cash}")
        
        # Test getting current positions
        positions = api.list_positions()
        logger.info(f"Current positions: {len(positions)}")
        for position in positions:
            logger.info(f"  {position.symbol}: {position.qty} shares at {position.avg_entry_price}")
        
        # Test getting market data (last 5 trading days)
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            # Convert to string format
            end_str = end_date.strftime('%Y-%m-%d')
            start_str = start_date.strftime('%Y-%m-%d')
            
            logger.info(f"Testing market data access from {start_str} to {end_str}")
            
            # Try to get data for a common stock
            symbol = "AAPL"
            
            # Try using the get_bars method
            try:
                logger.info(f"Attempting to get bars for {symbol}...")
                bars = api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_str,
                    end=end_str,
                    limit=5
                ).df
                
                if bars.empty:
                    logger.warning(f"No data found for {symbol}")
                else:
                    logger.info(f"Successfully retrieved {len(bars)} bars for {symbol}")
                    logger.info(f"First bar: {bars.iloc[0]}")
            except Exception as e:
                logger.error(f"Error getting bars: {e}")
            
            # Try using the get_barset method (older API)
            try:
                logger.info(f"Attempting to get barset for {symbol}...")
                barset = api.get_barset(symbol, 'day', limit=5)
                
                if symbol not in barset or not barset[symbol]:
                    logger.warning(f"No data found for {symbol} in barset")
                else:
                    logger.info(f"Successfully retrieved {len(barset[symbol])} bars for {symbol} using barset")
                    logger.info(f"First bar: {barset[symbol][0]}")
            except Exception as e:
                logger.error(f"Error getting barset: {e}")
                
            # Try using the get_bars method with the data API
            try:
                logger.info(f"Attempting to get data from data API...")
                data_api = REST(
                    api_key,
                    api_secret,
                    base_url='https://data.alpaca.markets'
                )
                
                data_bars = data_api.get_bars(
                    symbol,
                    TimeFrame.Day,
                    start=start_str,
                    end=end_str,
                    limit=5
                ).df
                
                if data_bars.empty:
                    logger.warning(f"No data found for {symbol} from data API")
                else:
                    logger.info(f"Successfully retrieved {len(data_bars)} bars for {symbol} from data API")
                    logger.info(f"First bar: {data_bars.iloc[0]}")
            except Exception as e:
                logger.error(f"Error getting data from data API: {e}")
            
        except Exception as e:
            logger.error(f"Error testing market data access: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
        return False

if __name__ == "__main__":
    success = test_alpaca_connection()
    if success:
        logger.info("Alpaca API connection test completed")
    else:
        logger.error("Alpaca API connection test failed")
