#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Alpaca Test
-----------------
A simplified test script to verify Alpaca integration with just a few stocks.
"""

import os
import sys
import logging
import datetime as dt
import json
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("SimpleAlpacaTest")

def load_alpaca_credentials():
    """Load Alpaca API credentials from file or environment variables"""
    # Try environment variables first
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    
    # If not found, try config file
    if not api_key or not api_secret:
        try:
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
                api_key = credentials.get('api_key')
                api_secret = credentials.get('api_secret')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load Alpaca credentials: {e}")
            return None, None
    
    return api_key, api_secret

def test_alpaca_connection():
    """Test basic connection to Alpaca API"""
    api_key, api_secret = load_alpaca_credentials()
    
    if not api_key or not api_secret:
        logger.error("Alpaca credentials not found")
        return False
    
    try:
        # Import Alpaca API
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import REST
        
        # Initialize API
        api = REST(api_key, api_secret, base_url="https://paper-api.alpaca.markets")
        
        # Get account info
        account = api.get_account()
        logger.info(f"Connected to Alpaca account: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Account equity: ${float(account.equity):.2f}")
        logger.info(f"Account buying power: ${float(account.buying_power):.2f}")
        
        return True
    except ImportError:
        logger.error("alpaca-trade-api package is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return False

def get_limited_assets(max_stocks=5):
    """Get a limited set of assets from Alpaca"""
    api_key, api_secret = load_alpaca_credentials()
    
    if not api_key or not api_secret:
        logger.error("Alpaca credentials not found")
        return []
    
    try:
        # Import Alpaca API
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import REST
        
        # Initialize API
        api = REST(api_key, api_secret, base_url="https://paper-api.alpaca.markets")
        
        # Get assets
        assets = api.list_assets(status='active', asset_class='us_equity')
        
        # Filter to just a few well-known stocks
        target_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        filtered_assets = [a for a in assets if a.symbol in target_symbols]
        
        logger.info(f"Found {len(filtered_assets)} assets")
        for asset in filtered_assets:
            logger.info(f"Asset: {asset.symbol} - {asset.name} - Tradable: {asset.tradable}")
        
        return filtered_assets
    except Exception as e:
        logger.error(f"Failed to get assets: {e}")
        return []

def test_market_data(symbol='SPY', days=5):
    """Test fetching market data for a symbol"""
    api_key, api_secret = load_alpaca_credentials()
    
    if not api_key or not api_secret:
        logger.error("Alpaca credentials not found")
        return False
    
    try:
        # Import Alpaca API
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import REST, TimeFrame
        
        # Initialize API
        api = REST(api_key, api_secret, base_url="https://paper-api.alpaca.markets")
        
        # Set date range - use older data that should be available with free tier
        # Use data from 2023 instead of recent data
        end_date = dt.date(2023, 12, 31)
        start_date = end_date - dt.timedelta(days=days)
        
        logger.info(f"Fetching {days} days of data for {symbol} from {start_date} to {end_date}")
        
        # Get bars
        bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            adjustment='raw'
        ).df
        
        if bars.empty:
            logger.warning(f"No data found for {symbol}")
            return False
        
        logger.info(f"Retrieved {len(bars)} bars for {symbol}")
        logger.info(f"First bar: {bars.iloc[0]}")
        logger.info(f"Last bar: {bars.iloc[-1]}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to get market data: {e}")
        return False

def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description="Simple Alpaca Test")
    parser.add_argument("--test", choices=["connection", "assets", "data", "all"], 
                        default="all", help="Test to run")
    parser.add_argument("--symbol", default="SPY", help="Symbol to test for market data")
    parser.add_argument("--days", type=int, default=5, help="Days of data to fetch")
    
    args = parser.parse_args()
    
    if args.test == "connection" or args.test == "all":
        logger.info("=== Testing Alpaca Connection ===")
        if test_alpaca_connection():
            logger.info("✅ Connection test passed")
        else:
            logger.error("❌ Connection test failed")
    
    if args.test == "assets" or args.test == "all":
        logger.info("=== Testing Asset Retrieval ===")
        assets = get_limited_assets()
        if assets:
            logger.info(f"✅ Retrieved {len(assets)} assets")
        else:
            logger.error("❌ Asset retrieval failed")
    
    if args.test == "data" or args.test == "all":
        logger.info(f"=== Testing Market Data for {args.symbol} ===")
        if test_market_data(args.symbol, args.days):
            logger.info(f"✅ Market data test passed for {args.symbol}")
        else:
            logger.error(f"❌ Market data test failed for {args.symbol}")
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main()
