#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compare backtest results across different time periods
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse

def compare_backtest_periods(results_files, output_dir='backtest_comparison'):
    """
    Compare backtest results from multiple CSV files
    
    Args:
        results_files: List of paths to CSV files with backtest results
        output_dir: Directory to save comparison plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    period_results = []
    for file in results_files:
        # Extract period from filename
        period = os.path.basename(file).replace('backtest_results_', '').replace('.csv', '')
        
        # Load data
        df = pd.read_csv(file)
        df['period'] = period
        period_results.append({
            'period': period,
            'data': df,
            'total_trades': len(df),
            'trades_with_returns': df['return'].notna().sum(),
            'win_rate': df['win'].mean(),
            'avg_return': df['return'].mean(),
            'total_return': df['return'].sum(),
            'best_trade': df['return'].max(),
            'worst_trade': df['return'].min(),
            'sharpe': df['return'].mean() / df['return'].std() if df['return'].std() > 0 else 0
        })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(period_results)
    summary_df = summary_df[['period', 'total_trades', 'trades_with_returns', 'win_rate', 
                             'avg_return', 'total_return', 'best_trade', 'worst_trade', 'sharpe']]
    
    # Print summary
    print("=== Backtest Period Comparison ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2%}" if 0 <= x <= 1 else f"{x:.2f}"))
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'period_comparison_summary.csv'), index=False)
    
    # Combine all data for further analysis
    all_data = pd.concat([r['data'] for r in period_results])
    
    # Compare win rates across periods
    plt.figure(figsize=(10, 6))
    win_rates = [r['win_rate'] for r in period_results]
    periods = [r['period'] for r in period_results]
    plt.bar(periods, win_rates)
    plt.title('Win Rate by Period')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(win_rates):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
    plt.savefig(os.path.join(output_dir, 'win_rate_by_period.png'))
    
    # Compare average returns across periods
    plt.figure(figsize=(10, 6))
    avg_returns = [r['avg_return'] for r in period_results]
    plt.bar(periods, avg_returns)
    plt.title('Average Return by Period')
    plt.ylabel('Average Return')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(avg_returns):
        plt.text(i, v + 0.001, f"{v:.2%}", ha='center')
    plt.savefig(os.path.join(output_dir, 'avg_return_by_period.png'))
    
    # Compare total returns across periods
    plt.figure(figsize=(10, 6))
    total_returns = [r['total_return'] for r in period_results]
    plt.bar(periods, total_returns)
    plt.title('Total Return by Period')
    plt.ylabel('Total Return')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(total_returns):
        plt.text(i, v + 1, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(output_dir, 'total_return_by_period.png'))
    
    # Compare performance by direction across periods
    direction_performance = []
    for r in period_results:
        period = r['period']
        data = r['data']
        for direction in data['direction'].unique():
            direction_data = data[data['direction'] == direction]
            if len(direction_data) > 0:
                direction_performance.append({
                    'period': period,
                    'direction': direction,
                    'win_rate': direction_data['win'].mean(),
                    'avg_return': direction_data['return'].mean(),
                    'count': len(direction_data)
                })
    
    direction_df = pd.DataFrame(direction_performance)
    
    # Plot direction performance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='period', y='avg_return', hue='direction', data=direction_df)
    plt.title('Average Return by Direction and Period')
    plt.ylabel('Average Return')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'return_by_direction_period.png'))
    
    # Plot win rate by direction
    plt.figure(figsize=(12, 8))
    sns.barplot(x='period', y='win_rate', hue='direction', data=direction_df)
    plt.title('Win Rate by Direction and Period')
    plt.ylabel('Win Rate')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'win_rate_by_direction_period.png'))
    
    # Find common top performing stocks across periods
    top_stocks = {}
    for r in period_results:
        period = r['period']
        data = r['data']
        stock_perf = data.groupby('symbol')[['return', 'win']].agg({
            'return': 'mean',
            'win': 'mean',
            'return': 'count'
        }).rename(columns={'return': 'count'})
        stock_perf = stock_perf[stock_perf['count'] >= 5]  # At least 5 trades
        top_stocks[period] = stock_perf.sort_values('win', ascending=False).head(10).index.tolist()
    
    # Find intersection of top stocks
    all_periods = list(top_stocks.keys())
    common_stocks = set(top_stocks[all_periods[0]])
    for period in all_periods[1:]:
        common_stocks = common_stocks.intersection(set(top_stocks[period]))
    
    print("\n=== Common Top Performing Stocks Across All Periods ===")
    print(", ".join(common_stocks))
    
    # Save list of common stocks
    with open(os.path.join(output_dir, 'common_top_stocks.txt'), 'w') as f:
        f.write("Common Top Performing Stocks Across All Periods:\n")
        f.write(", ".join(common_stocks))
    
    print(f"\nComparison results saved to {output_dir} directory")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare backtest results across periods')
    parser.add_argument('--results_files', nargs='+', default=glob.glob('backtest_results_*.csv'),
                        help='List of paths to CSV files with backtest results')
    parser.add_argument('--output_dir', type=str, default='backtest_comparison',
                        help='Directory to save comparison plots')
    args = parser.parse_args()
    
    # Compare the results
    compare_backtest_periods(args.results_files, args.output_dir)

if __name__ == "__main__":
    main()
