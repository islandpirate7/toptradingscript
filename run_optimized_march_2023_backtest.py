#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Optimized March 2023 Backtest
--------------------------------------
This script runs a backtest for March 2023 using the optimized
ultra aggressive strategy configuration based on our analysis.
"""

import os
import yaml
import json
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from tabulate import tabulate
import alpaca_trade_api as tradeapi

from mean_reversion_strategy_ultra_aggressive import MeanReversionStrategyUltraAggressive
from backtest_march_2023_ultra_aggressive import UltraAggressiveBacktester

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

def run_optimized_march_2023_backtest(config_file):
    """Run optimized backtest for March 2023
    
    Args:
        config_file (str): Path to configuration file
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define date range
    start_date = dt.datetime.strptime(config['general']['backtest_start_date'], '%Y-%m-%d')
    end_date = dt.datetime.strptime(config['general']['backtest_end_date'], '%Y-%m-%d')
    
    logger.info(f"Running optimized backtest for March 2023 from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get symbols from config, prioritizing top performers
    all_symbols = config['general']['symbols']
    
    # Get top performers based on our analysis
    top_performers = ['META', 'JPM', 'MRK', 'VLO', 'CVX', 'TSLA', 'V']
    
    # Reorder symbols to prioritize top performers
    symbols = []
    for symbol in top_performers:
        if symbol in all_symbols:
            symbols.append(symbol)
    
    # Add remaining symbols
    for symbol in all_symbols:
        if symbol not in symbols:
            symbols.append(symbol)
    
    logger.info(f"Running backtest for {len(symbols)} symbols, prioritizing top performers")
    
    # Initialize backtester
    backtester = UltraAggressiveBacktester(config_file)
    
    # Apply ultra aggressive settings
    backtester.apply_ultra_aggressive_settings()
    
    # Apply custom weights for symbols
    mean_reversion_weights = config['strategy_configs']['MeanReversion']['symbol_weights']
    for symbol, weight in mean_reversion_weights.items():
        if symbol in symbols:
            logger.info(f"Setting weight {weight} for {symbol}")
            backtester.strategy.mean_reversion.set_symbol_weight(symbol, weight)
    
    # Modify the generate_signals method to include the symbol
    original_generate_signals = backtester.strategy.mean_reversion.generate_signals
    
    def generate_signals_with_symbol(df, symbol=None):
        """Wrapper to ensure symbol is passed to generate_signals"""
        logger.info(f"Generating signals for {symbol if symbol else 'unknown symbol'}")
        return original_generate_signals(df, symbol)
    
    # Replace the method with our wrapper
    backtester.strategy.mean_reversion.generate_signals = generate_signals_with_symbol
    
    # Modify the CombinedStrategy's generate_signals method to pass the symbol
    original_strategy_generate_signals = backtester.strategy.generate_signals
    
    def strategy_generate_signals_with_symbol(df, symbol=None):
        """Wrapper to ensure symbol is passed to the strategy's generate_signals method"""
        logger.info(f"Strategy generating signals for {symbol if symbol else 'unknown symbol'}")
        return original_strategy_generate_signals(df, symbol)
    
    # Replace the method with our wrapper
    backtester.strategy.generate_signals = strategy_generate_signals_with_symbol
    
    # Run backtest
    results = backtester.run_backtest()
    
    if results:
        # Display results
        print("\nBacktest Summary:")
        print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"End Date: {end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
        
        # Check if results has metrics attribute
        if hasattr(results, 'metrics') and results.metrics:
            metrics = results.metrics
            print(f"Final Capital: ${metrics.get('final_capital', backtester.initial_capital):,.2f}")
            print(f"Total Return: {metrics.get('total_return_pct', 0.0):.2f}%")
            print(f"Annualized Return: {metrics.get('annualized_return_pct', 0.0):.2f}%")
            print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0.0):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
            print(f"Win Rate: {metrics.get('win_rate', 0.0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
        else:
            print("No metrics available in the results")
        
        # Check if we have trades
        if hasattr(results, 'trades') and results.trades:
            # Save trades to CSV
            trades_df = pd.DataFrame(results.trades)
            trades_file = f"output/march_2023_trades_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"\nTrades saved to {trades_file}")
            
            # Display trade summary
            print(f"\nTrade Summary:")
            print(f"Total Trades: {len(results.trades)}")
            
            # Count trades by symbol
            symbol_counts = {}
            for trade in results.trades:
                symbol = trade.get('symbol', 'Unknown')
                if symbol not in symbol_counts:
                    symbol_counts[symbol] = 0
                symbol_counts[symbol] += 1
            
            print("\nTrades by Symbol:")
            for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"{symbol}: {count}")
        else:
            print("\nNo trades were generated during the backtest period")
    else:
        print("No results returned from backtest")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimized backtest for March 2023')
    parser.add_argument('--config', type=str, default='configuration_ultra_aggressive_march_2023.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run backtest
    run_optimized_march_2023_backtest(args.config)
