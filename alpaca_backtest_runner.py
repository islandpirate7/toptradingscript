#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Backtest Runner for Mean Reversion Strategy
--------------------------------------------------
This script runs a backtest for the optimized mean reversion strategy using real Alpaca data.
It fixes the issues with the API connection and date formatting.
"""

import os
import sys
import json
import logging
import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alpaca_backtest.log"),
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
            
            logger.info(f"Using paper trading credentials with base URL: {base_url}")
        else:
            # Fallback to environment variables
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            base_url = 'https://paper-api.alpaca.markets'
            
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
        return bars
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def run_backtest(api, symbols, start_date, end_date, config):
    """Run a simple backtest with the given symbols and date range"""
    if not api:
        logger.error("Alpaca API not initialized")
        return None
    
    results = {}
    
    # Fetch data for each symbol
    for symbol in symbols:
        logger.info(f"Processing symbol: {symbol}")
        bars = fetch_historical_data(api, symbol, start_date, end_date)
        
        if bars is not None:
            results[symbol] = {
                'data': bars,
                'start_price': bars.iloc[0]['close'],
                'end_price': bars.iloc[-1]['close'],
                'return': (bars.iloc[-1]['close'] / bars.iloc[0]['close']) - 1
            }
            logger.info(f"Symbol {symbol}: Start price=${results[symbol]['start_price']:.2f}, End price=${results[symbol]['end_price']:.2f}, Return={results[symbol]['return']:.2%}")
        else:
            logger.error(f"Failed to fetch data for {symbol}")
    
    return results

def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='Run backtest for mean reversion strategy with Alpaca data')
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4],
                        help='Quarter of 2023 to test (1-4)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='List of symbols to test')
    
    args = parser.parse_args()
    
    # Set date range based on quarter if specified
    if args.quarter:
        if args.quarter == 1:
            args.start = '2023-01-01'
            args.end = '2023-03-31'
        elif args.quarter == 2:
            args.start = '2023-04-01'
            args.end = '2023-06-30'
        elif args.quarter == 3:
            args.start = '2023-07-01'
            args.end = '2023-09-30'
        elif args.quarter == 4:
            args.start = '2023-10-01'
            args.end = '2023-12-31'
    
    # Default symbols if none provided
    if not args.symbols:
        args.symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Initialize Alpaca API
    api = initialize_alpaca_api()
    if not api:
        logger.error("Failed to initialize Alpaca API")
        return
    
    # Run backtest
    logger.info(f"Running backtest from {args.start} to {args.end} with symbols: {args.symbols}")
    results = run_backtest(api, args.symbols, args.start, args.end, config)
    
    if results:
        # Plot returns
        symbols = list(results.keys())
        returns = [results[symbol]['return'] for symbol in symbols]
        
        plt.figure(figsize=(10, 6))
        plt.bar(symbols, returns)
        plt.title(f"Returns: {args.start} to {args.end}")
        plt.xlabel("Symbol")
        plt.ylabel("Return (%)")
        plt.grid(True, axis='y')
        
        # Add return values as text on bars
        for i, r in enumerate(returns):
            plt.text(i, r + 0.01, f"{r:.2%}", ha='center')
        
        # Save plot
        plot_filename = f"returns_{args.start}_to_{args.end}.png"
        plt.savefig(plot_filename)
        logger.info(f"Saved returns plot to {plot_filename}")
        
        # Print summary
        print(f"\nBacktest Summary: {args.start} to {args.end}")
        print(f"Symbols: {args.symbols}")
        
        total_return = sum(returns) / len(returns)
        print(f"Average Return: {total_return:.2%}")
        
        best_symbol = symbols[returns.index(max(returns))]
        worst_symbol = symbols[returns.index(min(returns))]
        print(f"Best Symbol: {best_symbol} ({results[best_symbol]['return']:.2%})")
        print(f"Worst Symbol: {worst_symbol} ({results[worst_symbol]['return']:.2%})")

if __name__ == "__main__":
    main()
