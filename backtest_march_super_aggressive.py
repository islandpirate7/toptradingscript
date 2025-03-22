#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest with Super Aggressive Settings
--------------------------------------
This script runs a backtest with super aggressive settings
to generate more trading signals for March 2024.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import json
from tabulate import tabulate

from backtest_combined_strategy import Backtester
from combined_strategy import CombinedStrategy
from mean_reversion_strategy_super_aggressive import MeanReversionStrategySuperAggressive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SuperAggressiveBacktester(Backtester):
    """Backtester with super aggressive settings to generate more signals"""
    
    def __init__(self, config_file):
        """Initialize with super aggressive settings"""
        super().__init__(config_file)
        
        # Override strategy with super aggressive settings
        self.apply_super_aggressive_settings()
    
    def apply_super_aggressive_settings(self):
        """Apply super aggressive settings to generate more signals"""
        logger.info("Applying super aggressive settings to generate more signals")
        
        # Replace the mean reversion strategy with our super aggressive version
        self.strategy.mean_reversion = MeanReversionStrategySuperAggressive()
        
        # Modify trend following strategy parameters if available
        if hasattr(self.strategy, 'trend_following'):
            # Lower thresholds for more signals
            if hasattr(self.strategy.trend_following, 'adx_threshold'):
                self.strategy.trend_following.adx_threshold = 10
                logger.info(f"Modified trend following parameters: ADX threshold={self.strategy.trend_following.adx_threshold}")
        
        # Lower the minimum signal score threshold
        self.strategy.min_signal_score = 0.3
        logger.info(f"Lowered minimum signal score threshold to {self.strategy.min_signal_score}")
        
        # Increase seasonality boost and penalty factors
        if hasattr(self.strategy, 'seasonal_boost'):
            self.strategy.seasonal_boost = 0.5
            self.strategy.seasonal_penalty = 0.2  # Lower penalty to allow more signals
            logger.info(f"Modified seasonality factors: boost={self.strategy.seasonal_boost}, penalty={self.strategy.seasonal_penalty}")

def run_march_2024_backtest(config_file):
    """Run a backtest for March 2024 with super aggressive settings
    
    Args:
        config_file (str): Path to configuration file
    """
    start_date = dt.datetime(2024, 3, 1)
    end_date = dt.datetime(2024, 3, 31)
    
    # Use super aggressive backtester
    logger.info(f"Running super aggressive backtest for March 2024")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    backtester = SuperAggressiveBacktester(config_file)
    results = backtester.run_backtest(start_date, end_date)
    
    # Display trade details if there are any trades
    if results and hasattr(results, 'trades') and results.trades:
        logger.info(f"Found {len(results.trades)} trades in the backtest results")
        
        # Create a DataFrame from the trades
        trades_df = pd.DataFrame(results.trades)
        
        # Format the DataFrame for display
        if not trades_df.empty:
            # Convert timestamps to readable format
            if 'entry_time' in trades_df.columns:
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d')
            if 'exit_time' in trades_df.columns:
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d')
            
            # Format numeric columns
            if 'profit_pct' in trades_df.columns:
                trades_df['profit_pct'] = trades_df['profit_pct'].map('{:.2f}%'.format)
            if 'profit_usd' in trades_df.columns:
                trades_df['profit_usd'] = trades_df['profit_usd'].map('${:.2f}'.format)
            
            # Select relevant columns for display
            display_columns = ['symbol', 'direction', 'entry_time', 'exit_time', 'profit_pct', 'profit_usd', 'exit_reason']
            display_columns = [col for col in display_columns if col in trades_df.columns]
            
            # Display the trades
            print("\nTrade Details:")
            print(tabulate(trades_df[display_columns], headers='keys', tablefmt='grid'))
            
            # Calculate and display summary statistics
            print("\nTrade Summary:")
            win_rate = (trades_df['profit_usd'].str.replace('$', '').astype(float) > 0).mean() * 100
            total_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).sum()
            avg_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).mean()
            
            print(f"Total Trades: {len(trades_df)}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"Average Profit per Trade: ${avg_profit:.2f}")
    else:
        logger.warning("No trades found in the backtest results")
    
    return results

def run_q1_2024_backtest(config_file):
    """Run a backtest for Q1 2024 with super aggressive settings
    
    Args:
        config_file (str): Path to configuration file
    """
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 3, 31)
    
    # Use super aggressive backtester
    logger.info(f"Running super aggressive backtest for Q1 2024")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    backtester = SuperAggressiveBacktester(config_file)
    results = backtester.run_backtest(start_date, end_date)
    
    # Display trade details if there are any trades
    if results and hasattr(results, 'trades') and results.trades:
        logger.info(f"Found {len(results.trades)} trades in the backtest results")
        
        # Create a DataFrame from the trades
        trades_df = pd.DataFrame(results.trades)
        
        # Format the DataFrame for display
        if not trades_df.empty:
            # Convert timestamps to readable format
            if 'entry_time' in trades_df.columns:
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d')
            if 'exit_time' in trades_df.columns:
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d')
            
            # Format numeric columns
            if 'profit_pct' in trades_df.columns:
                trades_df['profit_pct'] = trades_df['profit_pct'].map('{:.2f}%'.format)
            if 'profit_usd' in trades_df.columns:
                trades_df['profit_usd'] = trades_df['profit_usd'].map('${:.2f}'.format)
            
            # Select relevant columns for display
            display_columns = ['symbol', 'direction', 'entry_time', 'exit_time', 'profit_pct', 'profit_usd', 'exit_reason']
            display_columns = [col for col in display_columns if col in trades_df.columns]
            
            # Display the trades
            print("\nTrade Details:")
            print(tabulate(trades_df[display_columns], headers='keys', tablefmt='grid'))
            
            # Calculate and display summary statistics
            print("\nTrade Summary:")
            win_rate = (trades_df['profit_usd'].str.replace('$', '').astype(float) > 0).mean() * 100
            total_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).sum()
            avg_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).mean()
            
            print(f"Total Trades: {len(trades_df)}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"Average Profit per Trade: ${avg_profit:.2f}")
    else:
        logger.warning("No trades found in the backtest results")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run super aggressive backtest')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy_march_seasonal.yaml',
                        help='Path to configuration file')
    parser.add_argument('--period', type=str, choices=['march', 'q1', 'all'], default='march',
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
