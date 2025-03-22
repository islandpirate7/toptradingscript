#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze backtest results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse

def analyze_backtest_results(results_file):
    """
    Analyze backtest results from a CSV file
    
    Args:
        results_file: Path to the CSV file with backtest results
    """
    # Load the results
    df = pd.read_csv(results_file)
    
    # Print basic statistics
    print(f"Total trades: {len(df)}")
    print(f"Trades with returns: {df['return'].notna().sum()}")
    print(f"Win rate: {df['win'].mean():.2%}")
    print(f"Average return: {df['return'].mean():.2%}")
    print(f"Total return: {df['return'].sum():.2%}")
    print(f"Best trade: {df['return'].max():.2%}")
    print(f"Worst trade: {df['return'].min():.2%}")
    
    # Analyze by symbol
    print("\nBest performing stocks:")
    symbol_perf = df.groupby('symbol')[['return', 'win']].agg({
        'return': 'mean',
        'win': 'mean'
    }).sort_values('return', ascending=False)
    print(symbol_perf.head(10))
    
    # Analyze by market regime
    print("\nPerformance by market regime:")
    regime_perf = df.groupby('market_regime')[['return', 'win']].agg({
        'return': 'mean',
        'win': 'mean'
    })
    print(regime_perf)
    
    # Analyze by direction
    print("\nPerformance by direction:")
    direction_perf = df.groupby('direction')[['return', 'win']].agg({
        'return': 'mean',
        'win': 'mean'
    })
    print(direction_perf)
    
    # Analyze by score range
    df['score_range'] = pd.cut(df['score'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                              labels=['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    
    print("\nPerformance by score range:")
    score_perf = df.groupby('score_range')[['return', 'win']].agg({
        'return': 'mean',
        'win': 'mean'
    })
    print(score_perf)
    
    # Create plots directory if it doesn't exist
    plots_dir = 'backtest_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot return distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['return'], kde=True)
    plt.title('Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, 'return_distribution.png'))
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    df.sort_values('date', inplace=True)
    df['cumulative_return'] = (1 + df['return']).cumprod() - 1
    plt.plot(range(len(df)), df['cumulative_return'])
    plt.title('Cumulative Returns')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'cumulative_returns.png'))
    
    # Plot win rate by score
    plt.figure(figsize=(10, 6))
    sns.barplot(x='score_range', y='win', data=df)
    plt.title('Win Rate by Score Range')
    plt.xlabel('Score Range')
    plt.ylabel('Win Rate')
    plt.savefig(os.path.join(plots_dir, 'win_rate_by_score.png'))
    
    # Plot average return by score
    plt.figure(figsize=(10, 6))
    sns.barplot(x='score_range', y='return', data=df)
    plt.title('Average Return by Score Range')
    plt.xlabel('Score Range')
    plt.ylabel('Average Return')
    plt.savefig(os.path.join(plots_dir, 'avg_return_by_score.png'))
    
    print(f"\nPlots saved to {plots_dir} directory")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze backtest results')
    parser.add_argument('--results_file', type=str, default='backtest_results_20230101_20230331.csv',
                        help='Path to the CSV file with backtest results')
    args = parser.parse_args()
    
    # Analyze the results
    analyze_backtest_results(args.results_file)

if __name__ == "__main__":
    main()
