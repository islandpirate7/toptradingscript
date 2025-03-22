#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Combined Mean Reversion with Seasonality Strategy
----------------------------------------------------
This script runs the combined mean reversion with seasonality strategy
for a specified time period.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from combined_mean_reversion_with_seasonality import CombinedMeanReversionWithSeasonality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_strategy(config_file, start_date, end_date, top_n, max_positions):
    """
    Run the combined strategy for a specified time period.
    
    Args:
        config_file (str): Path to configuration file
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        top_n (int): Number of top stocks to select based on seasonality
        max_positions (int): Maximum number of positions to hold
        
    Returns:
        BacktestResults: Results of the backtest
    """
    # Initialize the strategy
    strategy = CombinedMeanReversionWithSeasonality(config_file)
    
    # Run backtest
    results = strategy.run_backtest(start_date, end_date, top_n, max_positions)
    
    # Plot additional performance metrics
    plot_performance_metrics(results)
    
    return results

def plot_performance_metrics(results):
    """
    Plot additional performance metrics.
    
    Args:
        results: Backtest results object
    """
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Plot monthly returns
    if hasattr(results, 'monthly_returns') and results.monthly_returns is not None and not results.monthly_returns.empty:
        plt.figure(figsize=(12, 6))
        results.monthly_returns.plot(kind='bar', color=results.monthly_returns.map(lambda x: 'g' if x > 0 else 'r'))
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.grid(True, axis='y')
        plt.savefig('output/combined_mean_reversion_seasonality_monthly_returns.png')
        plt.close()
    
    # Plot trade outcomes
    if hasattr(results, 'trade_outcomes') and results.trade_outcomes is not None and not results.trade_outcomes.empty:
        # Plot average P&L by symbol
        if 'symbol' in results.trade_outcomes.columns and 'pnl_pct' in results.trade_outcomes.columns:
            plt.figure(figsize=(10, 6))
            outcomes = results.trade_outcomes.groupby('symbol')['pnl_pct'].mean().sort_values(ascending=False)
            if not outcomes.empty:
                outcomes.plot(kind='bar', color=outcomes.map(lambda x: 'g' if x > 0 else 'r'))
                plt.title('Average P&L by Symbol')
                plt.xlabel('Symbol')
                plt.ylabel('Average P&L (%)')
                plt.grid(True, axis='y')
                plt.savefig('output/combined_mean_reversion_seasonality_pnl_by_symbol.png')
                plt.close()
            
            # Plot win rate by symbol
            plt.figure(figsize=(10, 6))
            win_rates = results.trade_outcomes.groupby('symbol').apply(
                lambda x: (x['pnl_pct'] > 0).mean() * 100
            ).sort_values(ascending=False)
            
            if not win_rates.empty:
                win_rates.plot(kind='bar', color='blue')
                plt.title('Win Rate by Symbol')
                plt.xlabel('Symbol')
                plt.ylabel('Win Rate (%)')
                plt.grid(True, axis='y')
                plt.savefig('output/combined_mean_reversion_seasonality_win_rate_by_symbol.png')
                plt.close()
                
    return results

def main():
    """Main function to run the strategy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run combined mean reversion with seasonality strategy')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of top stocks to select based on seasonality')
    parser.add_argument('--max_positions', type=int, default=5,
                        help='Maximum number of positions to hold')
    parser.add_argument('--quarter', type=str, choices=['1', '2', '3', '4', 'all'], default=None,
                        help='Run for a specific quarter of 2023 (1-4) or all quarters')
    args = parser.parse_args()
    
    # If quarter is specified, override start_date and end_date
    if args.quarter is not None:
        year = 2023
        if args.quarter == 'all':
            # Run all quarters sequentially
            quarters = [('1', f'{year}-01-01', f'{year}-03-31'),
                       ('2', f'{year}-04-01', f'{year}-06-30'),
                       ('3', f'{year}-07-01', f'{year}-09-30'),
                       ('4', f'{year}-10-01', f'{year}-12-31')]
            
            all_results = []
            for q, start, end in quarters:
                logger.info(f"Running Quarter {q}: {start} to {end}")
                args.start_date = start
                args.end_date = end
                results = run_strategy(args.config, args.start_date, args.end_date, args.top_n, args.max_positions)
                all_results.append((q, results))
            
            # Compare quarterly results
            logger.info("=== Quarterly Performance Comparison ===")
            for q, results in all_results:
                logger.info(f"Quarter {q}: Total Return: {results.total_return_pct:.2f}%, " +
                           f"Win Rate: {results.win_rate:.2f}%, " +
                           f"Sharpe: {results.sharpe_ratio:.2f}, " +
                           f"Max DD: {results.max_drawdown_pct:.2f}%")
            
            # Create a comparison chart
            plt.figure(figsize=(12, 8))
            
            # Plot total returns
            plt.subplot(2, 2, 1)
            quarters = [q for q, _ in all_results]
            returns = [r.total_return_pct for _, r in all_results]
            plt.bar(quarters, returns)
            plt.title('Total Return by Quarter (%)')
            plt.ylabel('Return (%)')
            
            # Plot win rates
            plt.subplot(2, 2, 2)
            win_rates = [r.win_rate for _, r in all_results]
            plt.bar(quarters, win_rates)
            plt.title('Win Rate by Quarter (%)')
            plt.ylabel('Win Rate (%)')
            
            # Plot Sharpe ratios
            plt.subplot(2, 2, 3)
            sharpes = [r.sharpe_ratio for _, r in all_results]
            plt.bar(quarters, sharpes)
            plt.title('Sharpe Ratio by Quarter')
            plt.ylabel('Sharpe Ratio')
            
            # Plot max drawdowns
            plt.subplot(2, 2, 4)
            drawdowns = [r.max_drawdown_pct for _, r in all_results]
            plt.bar(quarters, drawdowns)
            plt.title('Max Drawdown by Quarter (%)')
            plt.ylabel('Max Drawdown (%)')
            
            plt.tight_layout()
            plt.savefig('output/quarterly_performance_comparison.png')
            plt.close()
            
            return
        elif args.quarter == '1':
            args.start_date = f'{year}-01-01'
            args.end_date = f'{year}-03-31'
        elif args.quarter == '2':
            args.start_date = f'{year}-04-01'
            args.end_date = f'{year}-06-30'
        elif args.quarter == '3':
            args.start_date = f'{year}-07-01'
            args.end_date = f'{year}-09-30'
        elif args.quarter == '4':
            args.start_date = f'{year}-10-01'
            args.end_date = f'{year}-12-31'
    
    logger.info(f"Running strategy from {args.start_date} to {args.end_date}")
    
    # Run the strategy
    run_strategy(args.config, args.start_date, args.end_date, args.top_n, args.max_positions)

if __name__ == "__main__":
    main()
