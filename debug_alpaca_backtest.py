#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Alpaca Backtest
--------------------
This script is a simplified version to debug the Alpaca API connection and data fetching.
"""

import os
import sys
import json
import logging
import datetime
import yaml
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_alpaca.log"),
        logging.StreamHandler()  # This will print to console
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def initialize_alpaca_api():
    """Initialize the Alpaca API client with credentials from alpaca_credentials.json"""
    try:
        # Load credentials from JSON file
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca_credentials.json')
        
        logger.info(f"Looking for credentials at: {credentials_path}")
        
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials by default
            paper_creds = credentials.get('paper', {})
            api_key = paper_creds.get('api_key')
            api_secret = paper_creds.get('api_secret')
            base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets')
            
            # Remove /v2 suffix if it's already included to prevent duplication
            if base_url.endswith('/v2'):
                base_url = base_url[:-3]
            
            logger.info(f"Using paper trading credentials from file with base URL: {base_url}")
        else:
            # Fallback to environment variables
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            base_url = 'https://paper-api.alpaca.markets/v2'
            
            if not api_key or not api_secret:
                logger.error("Alpaca API credentials not found")
                return None
        
        # Initialize API
        api = REST(api_key, api_secret, base_url)
        logger.info("Alpaca API initialized successfully")
        
        # Test the API connection
        account = api.get_account()
        logger.info(f"Connected to Alpaca account: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Account equity: {account.equity}")
        
        return api
            
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {str(e)}")
        return None

def fetch_historical_data(api, symbol, start_date, end_date):
    """Fetch historical price data from Alpaca"""
    if not api:
        logger.error("Alpaca API not initialized")
        return None
    
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Format dates as YYYY-MM-DD (without time component)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Using date strings: start={start_str}, end={end_str}")
        
        # Fetch data
        bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start_str,
            end=end_str,
            adjustment='raw'
        ).df
        
        if bars.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        logger.info(f"First bar: {bars.iloc[0]}")
        logger.info(f"Last bar: {bars.iloc[-1]}")
        
        return bars
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    """Main function to run the debug script"""
    import argparse
    
    logger.info("Starting debug script for Alpaca API connection")
    
    parser = argparse.ArgumentParser(description='Debug Alpaca API connection')
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date for data fetch (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-03-31',
                        help='End date for data fetch (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'AMZN'],
                        help='List of symbols to fetch data for')
    
    args = parser.parse_args()
    
    logger.info(f"Arguments: config={args.config}, start={args.start}, end={args.end}, symbols={args.symbols}")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return
    
    logger.info(f"Configuration loaded successfully: {config.keys()}")
    
    # Initialize Alpaca API
    api = initialize_alpaca_api()
    if not api:
        logger.error("Failed to initialize Alpaca API")
        return
    
    # Fetch data for each symbol
    for symbol in args.symbols:
        logger.info(f"Processing symbol: {symbol}")
        bars = fetch_historical_data(api, symbol, args.start, args.end)
        if bars is not None:
            logger.info(f"Successfully fetched data for {symbol}")
        else:
            logger.error(f"Failed to fetch data for {symbol}")

if __name__ == "__main__":
    main()
