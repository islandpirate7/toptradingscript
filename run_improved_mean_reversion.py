#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Improved Mean Reversion Strategy
-----------------------------------
This script runs the improved mean reversion strategy for all quarters of 2023
and compares the results to evaluate performance.
"""

import os
import sys
import yaml
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from improved_mean_reversion import ImprovedMeanReversionBacktest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_quarters():
    """Run backtest for all quarters of 2023"""
    # Define quarters
    quarters = [
        {'name': 'Q1 2023', 'start': '2023-01-01', 'end': '2023-03-31'},
        {'name': 'Q2 2023', 'start': '2023-04-01', 'end': '2023-06-30'},
        {'name': 'Q3 2023', 'start': '2023-07-01', 'end': '2023-09-30'},
        {'name': 'Q4 2023', 'start': '2023-10-01', 'end': '2023-12-31'}
    ]
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Initialize backtest
    config_path = 'configuration_enhanced_mean_reversion.yaml'
    backtest = ImprovedMeanReversionBacktest(config_path)
    
    # Store results for each quarter
    all_results = []
    
    # Run backtest for each quarter
    for quarter in quarters:
        logger.info(f"\n=== Running backtest for {quarter['name']} ===")
        
        # Run backtest
        results = backtest.run_backtest(quarter['start'], quarter['end'])
        
        # Store results
        quarter_results = {
            'quarter': quarter['name'],
            'start_date': quarter['start'],
            'end_date': quarter['end'],
            'initial_capital': results['initial_capital'],
            'final_capital': results['final_capital'],
            'return': results['return'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades'],
            'trades': results['trades']
        }
        
        all_results.append(quarter_results)
        
        # Save quarter results to CSV
        quarter_name = quarter['name'].replace(' ', '_')
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0:
            trades_df.to_csv(f'output/trades_{quarter_name}.csv', index=False)
        
        # Plot equity curve for this quarter
        if 'equity_curve' in results and results['equity_curve']:
            dates = [dt.datetime.fromisoformat(date) for date, _ in results['equity_curve']]
            equity = [value for _, value in results['equity_curve']]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, equity)
            plt.title(f'Equity Curve - {quarter["name"]}')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.savefig(f'output/equity_curve_{quarter_name}.png')
            plt.close()
    
    # Create summary report
    create_summary_report(all_results)

def create_summary_report(all_results):
    """Create a summary report of all quarters"""
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Quarter': result['quarter'],
            'Initial Capital': result['initial_capital'],
            'Final Capital': result['final_capital'],
            'Return (%)': result['return'] * 100,
            'Win Rate (%)': result['win_rate'] * 100,
            'Profit Factor': result['profit_factor'],
            'Max Drawdown (%)': result['max_drawdown'] * 100,
            'Total Trades': result['total_trades']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_df.to_csv('output/summary_results.csv', index=False)
    
    # Print summary
    logger.info("\n=== Summary of Results ===")
    logger.info(summary_df.to_string(index=False))
    
    # Calculate overall performance
    initial_capital = all_results[0]['initial_capital']
    final_capital = all_results[-1]['final_capital']
    overall_return = (final_capital - initial_capital) / initial_capital
    
    # Calculate compound return
    compound_return = 1.0
    for result in all_results:
        compound_return *= (1 + result['return'])
    compound_return -= 1.0
    
    logger.info(f"\nOverall Return: {overall_return:.2%}")
    logger.info(f"Compound Return: {compound_return:.2%}")
    
    # Plot returns by quarter
    plt.figure(figsize=(12, 6))
    quarters = [result['quarter'] for result in all_results]
    returns = [result['return'] * 100 for result in all_results]
    
    plt.bar(quarters, returns)
    plt.title('Returns by Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Return (%)')
    plt.grid(True, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(returns):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    plt.savefig('output/returns_by_quarter.png')
    plt.close()
    
    # Plot win rates by quarter
    plt.figure(figsize=(12, 6))
    win_rates = [result['win_rate'] * 100 for result in all_results]
    
    plt.bar(quarters, win_rates)
    plt.title('Win Rates by Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(win_rates):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    plt.savefig('output/win_rates_by_quarter.png')
    plt.close()
    
    # Plot total trades by quarter
    plt.figure(figsize=(12, 6))
    total_trades = [result['total_trades'] for result in all_results]
    
    plt.bar(quarters, total_trades)
    plt.title('Total Trades by Quarter')
    plt.xlabel('Quarter')
    plt.ylabel('Number of Trades')
    plt.grid(True, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(total_trades):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.savefig('output/trades_by_quarter.png')
    plt.close()

def main():
    """Main function"""
    run_all_quarters()

if __name__ == "__main__":
    main()
