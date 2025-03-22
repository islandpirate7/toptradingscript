#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Comparison Script
Compares the performance of different Mean Reversion strategy implementations
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_backtest_results(results_file: str) -> Dict:
    """Load backtest results from a JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def compare_strategies(strategy1_results: Dict, strategy2_results: Dict, strategy1_name: str, strategy2_name: str):
    """Compare the performance of two strategies"""
    # Create a comparison table
    comparison = {
        'Metric': [
            'Period',
            'Initial Capital',
            'Final Equity',
            'Total Return (%)',
            'Annualized Return (%)',
            'Maximum Drawdown (%)',
            'Sharpe Ratio',
            'Win Rate (%)',
            'Profit Factor',
            'Total Trades',
            'Winning Trades',
            'Losing Trades',
            'Average Trade Duration (days)'
        ],
        strategy1_name: [
            f"{strategy1_results.get('start_date', 'N/A')} to {strategy1_results.get('end_date', 'N/A')}",
            f"${strategy1_results.get('initial_capital', 0):.2f}",
            f"${strategy1_results.get('final_equity', 0):.2f}",
            f"{strategy1_results.get('total_return', 0):.2f}%",
            f"{strategy1_results.get('annualized_return', 0):.2f}%",
            f"{strategy1_results.get('max_drawdown', 0):.2f}%",
            f"{strategy1_results.get('sharpe_ratio', 0):.2f}",
            f"{strategy1_results.get('win_rate', 0):.2f}%",
            f"{strategy1_results.get('profit_factor', 0):.2f}",
            f"{strategy1_results.get('total_trades', 0)}",
            f"{strategy1_results.get('winning_trades', 0)}",
            f"{strategy1_results.get('losing_trades', 0)}",
            calculate_avg_trade_duration(strategy1_results.get('trades', []))
        ],
        strategy2_name: [
            f"{strategy2_results.get('start_date', 'N/A')} to {strategy2_results.get('end_date', 'N/A')}",
            f"${strategy2_results.get('initial_capital', 0):.2f}",
            f"${strategy2_results.get('final_equity', 0):.2f}",
            f"{strategy2_results.get('total_return', 0):.2f}%",
            f"{strategy2_results.get('annualized_return', 0):.2f}%",
            f"{strategy2_results.get('max_drawdown', 0):.2f}%",
            f"{strategy2_results.get('sharpe_ratio', 0):.2f}",
            f"{strategy2_results.get('win_rate', 0):.2f}%",
            f"{strategy2_results.get('profit_factor', 0):.2f}",
            f"{strategy2_results.get('total_trades', 0)}",
            f"{strategy2_results.get('winning_trades', 0)}",
            f"{strategy2_results.get('losing_trades', 0)}",
            calculate_avg_trade_duration(strategy2_results.get('trades', []))
        ]
    }
    
    # Convert to DataFrame for better display
    comparison_df = pd.DataFrame(comparison)
    
    # Calculate differences
    diff_column = []
    for i, metric in enumerate(comparison['Metric']):
        if metric in ['Period', 'Initial Capital']:
            diff_column.append('N/A')
            continue
        
        # Extract numeric values for comparison
        try:
            val1 = float(comparison[strategy1_name][i].replace('$', '').replace('%', ''))
            val2 = float(comparison[strategy2_name][i].replace('$', '').replace('%', ''))
            
            diff = val2 - val1
            
            # Format the difference
            if metric in ['Final Equity']:
                diff_column.append(f"${diff:.2f}")
            elif metric in ['Total Return (%)', 'Annualized Return (%)', 'Maximum Drawdown (%)', 'Win Rate (%)']:
                diff_column.append(f"{diff:.2f}%")
            elif metric in ['Sharpe Ratio', 'Profit Factor']:
                diff_column.append(f"{diff:.2f}")
            else:
                diff_column.append(f"{diff:.0f}")
        except:
            diff_column.append('N/A')
    
    # Add difference column
    comparison_df['Difference'] = diff_column
    
    return comparison_df

def calculate_avg_trade_duration(trades: List[Dict]) -> str:
    """Calculate the average trade duration in days"""
    if not trades:
        return "N/A"
    
    durations = []
    for trade in trades:
        if trade.get('entry_date') and trade.get('exit_date'):
            try:
                entry_date = datetime.datetime.strptime(trade['entry_date'], '%Y-%m-%d')
                exit_date = datetime.datetime.strptime(trade['exit_date'], '%Y-%m-%d')
                duration = (exit_date - entry_date).days
                durations.append(duration)
            except:
                pass
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        return f"{avg_duration:.1f} days"
    else:
        return "N/A"

def analyze_exit_reasons(results: Dict) -> pd.DataFrame:
    """Analyze exit reasons for a strategy"""
    exit_reasons = results.get('exit_reasons', {})
    
    if not exit_reasons:
        return pd.DataFrame()
    
    # Create DataFrame
    exit_data = {
        'Exit Reason': [],
        'Count': [],
        'Win Count': [],
        'Loss Count': [],
        'Win Rate (%)': []
    }
    
    for reason, stats in exit_reasons.items():
        exit_data['Exit Reason'].append(reason)
        exit_data['Count'].append(stats['count'])
        exit_data['Win Count'].append(stats['profit'])
        exit_data['Loss Count'].append(stats['loss'])
        win_rate = stats['profit'] / stats['count'] * 100 if stats['count'] > 0 else 0
        exit_data['Win Rate (%)'].append(f"{win_rate:.2f}%")
    
    return pd.DataFrame(exit_data)

def compare_exit_reasons(strategy1_results: Dict, strategy2_results: Dict, strategy1_name: str, strategy2_name: str):
    """Compare exit reasons between two strategies"""
    exit_df1 = analyze_exit_reasons(strategy1_results)
    exit_df2 = analyze_exit_reasons(strategy2_results)
    
    if exit_df1.empty and exit_df2.empty:
        return None
    
    # Rename columns to include strategy names
    if not exit_df1.empty:
        exit_df1.columns = ['Exit Reason'] + [f"{strategy1_name} {col}" if col != 'Exit Reason' else col for col in exit_df1.columns[1:]]
    
    if not exit_df2.empty:
        exit_df2.columns = ['Exit Reason'] + [f"{strategy2_name} {col}" if col != 'Exit Reason' else col for col in exit_df2.columns[1:]]
    
    # Merge DataFrames
    if exit_df1.empty:
        return exit_df2
    elif exit_df2.empty:
        return exit_df1
    else:
        return pd.merge(exit_df1, exit_df2, on='Exit Reason', how='outer').fillna(0)

def plot_equity_curves(strategy1_results: Dict, strategy2_results: Dict, strategy1_name: str, strategy2_name: str, save_path: str = None):
    """Plot equity curves for both strategies"""
    # Extract equity data
    equity1 = extract_equity_data(strategy1_results)
    equity2 = extract_equity_data(strategy2_results)
    
    if equity1.empty and equity2.empty:
        logger.warning("No equity data to plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot equity curves
    if not equity1.empty:
        plt.plot(equity1.index, equity1['equity'], label=strategy1_name)
    
    if not equity2.empty:
        plt.plot(equity2.index, equity2['equity'], label=strategy2_name)
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def extract_equity_data(results: Dict) -> pd.DataFrame:
    """Extract equity data from results"""
    # This is a placeholder since we don't have direct access to equity curve data in the JSON
    # In a real implementation, you would need to extract this from the results
    return pd.DataFrame()

def find_latest_results(pattern: str) -> str:
    """Find the latest results file matching the pattern"""
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare Mean Reversion strategy implementations')
    parser.add_argument('--original', type=str, help='Path to original strategy results JSON')
    parser.add_argument('--enhanced', type=str, help='Path to enhanced strategy results JSON')
    parser.add_argument('--original_name', type=str, default='Original Strategy', help='Name for original strategy')
    parser.add_argument('--enhanced_name', type=str, default='Enhanced Strategy', help='Name for enhanced strategy')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Find latest results if not specified
    if not args.original:
        args.original = find_latest_results('backtest_results/mean_reversion_results_*.json')
        if not args.original:
            logger.error("No original strategy results found")
            return
    
    if not args.enhanced:
        args.enhanced = find_latest_results('backtest_results/enhanced_mean_reversion_results_*.json')
        if not args.enhanced:
            logger.error("No enhanced strategy results found")
            return
    
    # Load results
    original_results = load_backtest_results(args.original)
    enhanced_results = load_backtest_results(args.enhanced)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Compare strategies
    comparison_df = compare_strategies(original_results, enhanced_results, args.original_name, args.enhanced_name)
    
    # Compare exit reasons
    exit_comparison = compare_exit_reasons(original_results, enhanced_results, args.original_name, args.enhanced_name)
    
    # Print results
    print("\n=== Strategy Comparison ===")
    print(comparison_df.to_string(index=False))
    
    if exit_comparison is not None:
        print("\n=== Exit Reason Comparison ===")
        print(exit_comparison.to_string(index=False))
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"{args.output}/strategy_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    if exit_comparison is not None:
        exit_file = f"{args.output}/exit_reason_comparison_{timestamp}.csv"
        exit_comparison.to_csv(exit_file, index=False)
    
    # Plot equity curves
    equity_file = f"{args.output}/equity_comparison_{timestamp}.png"
    plot_equity_curves(original_results, enhanced_results, args.original_name, args.enhanced_name, equity_file)
    
    print(f"\nResults saved to {args.output} directory")


if __name__ == "__main__":
    main()
