#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest with Seasonality
-------------------------
This script runs a backtest with seasonality enabled
to see how the strategy performs.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

from backtest_combined_strategy import Backtester
from combined_strategy import CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from YAML file
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_seasonality_backtest(config_file, start_date, end_date, period_name="Custom Period"):
    """Run a backtest with seasonality enabled
    
    Args:
        config_file (str): Path to configuration file
        start_date (datetime): Start date for backtest
        end_date (datetime): End date for backtest
        period_name (str): Name of the period for reporting
    """
    # Load configuration
    config = load_config(config_file)
    
    # Ensure seasonality is enabled
    if 'seasonality' not in config:
        config['seasonality'] = {
            'enabled': True,
            'data_file': 'output/seasonal_opportunities_converted.yaml'
        }
    else:
        config['seasonality']['enabled'] = True
    
    # Save the modified config to a temporary file
    temp_config_file = 'temp_seasonality_config.yaml'
    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run backtest
    logger.info(f"Running backtest for {period_name} with seasonality enabled")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    backtester = Backtester(temp_config_file)
    results = backtester.run_backtest(start_date, end_date)
    
    # Print results
    if results and hasattr(results, 'metrics') and results.metrics:
        logger.info("Backtest Results:")
        logger.info(f"Total Return: {results.metrics.get('total_return', 0.0):.2f}%")
        logger.info(f"Win Rate: {results.metrics.get('win_rate', 0.0):.2f}%")
        logger.info(f"Profit Factor: {results.metrics.get('profit_factor', 0.0):.2f}")
        logger.info(f"Max Drawdown: {results.metrics.get('max_drawdown', 0.0):.2f}%")
        logger.info(f"Total Trades: {results.metrics.get('total_trades', 0)}")
        
        # Analyze trades by symbol
        if hasattr(results, 'trades') and results.trades:
            trades_by_symbol = {}
            for trade in results.trades:
                symbol = trade['symbol']
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)
            
            # Print selected stocks
            logger.info("Stocks selected during the backtest:")
            for symbol, trades in trades_by_symbol.items():
                if trades:
                    total_return = sum(trade['return_pct'] for trade in trades)
                    logger.info(f"{symbol}: {len(trades)} trades, Return: {total_return:.2f}%")
        else:
            logger.info("No trades were executed during the backtest period.")
    else:
        logger.info("No results available from the backtest.")
    
    # Clean up temporary file
    if os.path.exists(temp_config_file):
        os.remove(temp_config_file)
    
    return results

def run_q1_2024_backtest(config_file):
    """Run a backtest for Q1 2024 with seasonality enabled
    
    Args:
        config_file (str): Path to configuration file
    """
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 3, 31)
    return run_seasonality_backtest(config_file, start_date, end_date, "Q1 2024")

def run_march_2024_backtest(config_file):
    """Run a backtest for March 2024 with seasonality enabled
    
    Args:
        config_file (str): Path to configuration file
    """
    start_date = dt.datetime(2024, 3, 1)
    end_date = dt.datetime(2024, 3, 31)
    return run_seasonality_backtest(config_file, start_date, end_date, "March 2024")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtest with seasonality')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy_new_stocks.yaml',
                        help='Path to configuration file')
    parser.add_argument('--period', type=str, choices=['march', 'q1', 'all'], default='q1',
                        help='Period to backtest (march, q1, or all)')
    
    args = parser.parse_args()
    
    # Run backtest
    if args.period == 'march':
        run_march_2024_backtest(args.config)
    elif args.period == 'q1':
        run_q1_2024_backtest(args.config)
    elif args.period == 'all':
        run_march_2024_backtest(args.config)
        run_q1_2024_backtest(args.config)
