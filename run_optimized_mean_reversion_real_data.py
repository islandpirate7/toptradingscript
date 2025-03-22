#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Mean Reversion Strategy Backtest with Real Alpaca Data
---------------------------------------------------------------
This script runs a backtest for the optimized Mean Reversion strategy
using real Alpaca data. The strategy parameters are loaded from
configuration_mean_reversion_final.yaml, which contains the optimized
parameters that achieved a 100% win rate with mock data.
"""

import os
import sys
import json
import logging
import datetime
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='optimized_mean_reversion_real_data.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Import necessary classes from run_simple_backtest.py
from run_simple_backtest import (
    CandleData, Position, Portfolio, 
    MeanReversionStrategy, MarketRegimeDetector, 
    MLSignalClassifier, RealAlpacaBacktest
)

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

def create_backtest_config(mean_reversion_config, symbols, initial_capital=100000):
    """Create a backtest configuration from the mean reversion config"""
    # Extract mean reversion strategy parameters
    mean_reversion_params = mean_reversion_config.get('strategies', {}).get('MeanReversion', {})
    
    # Create backtest config
    backtest_config = {
        'initial_capital': initial_capital,
        'symbols': symbols,
        'strategies': {
            'mean_reversion': {
                'params': {
                    'bb_period': mean_reversion_params.get('bb_period', 20),
                    'bb_std': mean_reversion_params.get('bb_std_dev', 2.0),
                    'rsi_period': mean_reversion_params.get('rsi_period', 14),
                    'rsi_overbought': mean_reversion_params.get('rsi_overbought', 70),
                    'rsi_oversold': mean_reversion_params.get('rsi_oversold', 30),
                    'require_reversal': mean_reversion_params.get('require_reversal', False),
                    'stop_loss_atr': mean_reversion_params.get('stop_loss_atr', 2.0),
                    'take_profit_atr': mean_reversion_params.get('take_profit_atr', 3.0),
                    'atr_period': 14
                }
            }
        }
    }
    
    return backtest_config

def run_backtest(config_path, start_date, end_date, symbols=None):
    """Run a backtest with real Alpaca data using the optimized mean reversion strategy"""
    # Load mean reversion configuration
    mean_reversion_config = load_config(config_path)
    if not mean_reversion_config:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # If symbols not provided, use default list
    if not symbols:
        # Default to a mix of stocks and crypto
        symbols = [
            # Stocks - large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            # More stocks from different sectors
            'JPM', 'V', 'WMT', 'PG', 'JNJ',
            'HD', 'DIS', 'NFLX', 'NVDA', 'TSLA',
            'BA', 'GS', 'IBM', 'INTC', 'AMD',
            'MCD', 'KO', 'PEP', 'NKE', 'SBUX',
            # Crypto
            'BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'DOTUSD',
            'ADAUSD', 'LINKUSD', 'MATICUSD', 'DOGEUSD', 'XRPUSD',
            'UNIUSD', 'AAVEUSD', 'ALGOUSD', 'ATOMUSD', 'BCHUSD',
            'COMPUSD', 'DASHUSD', 'LTCUSD', 'MKRUSD', 'NEARUSD',
            'SHIBUSD', 'SNXUSD', 'TRXUSD', 'XLMUSD', 'ZECUSD'
        ]
    
    # Create backtest configuration
    backtest_config = create_backtest_config(mean_reversion_config, symbols)
    
    # Initialize backtest
    backtest = RealAlpacaBacktest(backtest_config)
    
    # Initialize Alpaca API
    backtest.initialize_alpaca_api()
    
    # Run backtest
    logger.info(f"Running backtest from {start_date} to {end_date} with {len(symbols)} symbols")
    results = backtest.run_backtest(start_date, end_date)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"backtest_results_optimized_mean_reversion_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate equity curve
    equity_curve = backtest.portfolio.equity_curve
    if equity_curve:
        dates = [entry[0] for entry in equity_curve]
        equity = [entry[1] for entry in equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity)
        plt.title('Equity Curve - Optimized Mean Reversion Strategy (Real Alpaca Data)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        equity_curve_file = f"equity_curve_optimized_mean_reversion_{timestamp}.png"
        plt.savefig(equity_curve_file)
        logger.info(f"Equity curve saved to {equity_curve_file}")
    
    # Print performance metrics
    win_rate = backtest.portfolio.get_win_rate()
    profit_factor = backtest.portfolio.get_profit_factor()
    max_drawdown = backtest.portfolio.get_max_drawdown()
    
    logger.info(f"Performance Metrics:")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    
    return results

def main():
    """Main function to run the backtest"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimized mean reversion backtest with real Alpaca data')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_final.yaml',
                        help='Path to the mean reversion configuration file')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4],
                        help='Quarter of 2023 to run backtest for (overrides start/end)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='List of symbols to include in backtest')
    
    args = parser.parse_args()
    
    # Set start and end dates based on quarter if specified
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
    
    # Run backtest
    run_backtest(args.config, args.start, args.end, args.symbols)

if __name__ == "__main__":
    main()
