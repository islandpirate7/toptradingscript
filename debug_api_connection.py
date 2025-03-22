#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Alpaca API Connection Issues
This script tests connection to Alpaca API and diagnoses common issues
"""

import os
import json
import logging
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"api_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        raise

def test_api_connection(credentials, max_retries=3):
    """Test connection to Alpaca API"""
    logger.info(f"Testing API connection using {credentials['base_url']}")
    
    for attempt in range(max_retries):
        try:
            api = tradeapi.REST(
                key_id=credentials['api_key'],
                secret_key=credentials['api_secret'],
                base_url=credentials['base_url']
            )
            
            # Test account info
            logger.info("Testing account info...")
            account = api.get_account()
            logger.info(f"Account status: {account.status}")
            logger.info(f"Account ID: {account.id}")
            logger.info(f"Account equity: {account.equity}")
            
            # Test clock
            logger.info("Testing market clock...")
            clock = api.get_clock()
            logger.info(f"Market is {'open' if clock.is_open else 'closed'}")
            logger.info(f"Next market open: {clock.next_open}")
            logger.info(f"Next market close: {clock.next_close}")
            
            return api
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return None
    
    logger.error("Failed to connect to Alpaca API after maximum retries")
    return None

def test_data_access(api, symbols=None, days=30):
    """Test access to market data"""
    if symbols is None:
        symbols = ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL"]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Testing data access for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching data for {symbol}...")
            bars = api.get_bars(
                symbol,
                '1D',
                start=pd.Timestamp(start_date, tz='UTC').isoformat(),
                end=pd.Timestamp(end_date, tz='UTC').isoformat(),
                adjustment='raw'
            ).df
            
            logger.info(f"Successfully retrieved {len(bars)} bars for {symbol}")
            if len(bars) > 0:
                logger.info(f"First date: {bars.index[0].date()}")
                logger.info(f"Last date: {bars.index[-1].date()}")
                logger.info(f"Latest close: {bars['close'].iloc[-1]}")
            else:
                logger.warning(f"No data available for {symbol}")
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")

def test_api_endpoints(base_url, api_key, api_secret):
    """Test direct access to API endpoints using requests"""
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }
    
    # Test endpoints
    endpoints = [
        '/v2/account',
        '/v2/clock',
        '/v2/calendar?start=2023-01-01&end=2023-01-10'
    ]
    
    for endpoint in endpoints:
        url = f"{base_url.rstrip('/')}{endpoint}"
        try:
            logger.info(f"Testing direct access to {url}")
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Endpoint {endpoint} accessible: HTTP {response.status_code}")
                logger.debug(f"Response: {response.json()}")
            else:
                logger.error(f"Endpoint {endpoint} error: HTTP {response.status_code}")
                logger.error(f"Error message: {response.text}")
        except Exception as e:
            logger.error(f"Request error for {endpoint}: {str(e)}")

def main():
    """Main function"""
    logger.info("Starting API connection debug")
    
    # Test both paper and live modes
    for mode in ['paper', 'live']:
        try:
            logger.info(f"\n{'='*50}\nTesting {mode.upper()} trading API\n{'='*50}")
            credentials = load_alpaca_credentials(mode=mode)
            
            # Test API connection
            api = test_api_connection(credentials)
            
            if api:
                # Test data access
                test_data_access(api)
                
                # Test direct endpoint access
                test_api_endpoints(
                    credentials['base_url'],
                    credentials['api_key'],
                    credentials['api_secret']
                )
            else:
                logger.error(f"Could not establish connection to {mode} API")
        except Exception as e:
            logger.error(f"Error in {mode} API testing: {str(e)}")
    
    logger.info("API connection debug completed")

if __name__ == "__main__":
    main()
