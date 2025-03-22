#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Combined Strategy Backtest with Improved Configuration
---------------------------------------------------------
This script runs a backtest for the improved combined strategy
that integrates both mean reversion and trend following approaches.
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from backtest_combined_strategy import Backtester, BacktestResults

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='combined_strategy_improved_backtest.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

def run_backtest(config_file='configuration_combined_strategy.yaml', 
                 start_date='2023-01-01', 
                 end_date='2024-04-30',
                 symbols=None):
    """Run the combined strategy backtest
    
    Args:
        config_file (str): Path to configuration file
        start_date (str): Start date for backtest (YYYY-MM-DD)
        end_date (str): End date for backtest (YYYY-MM-DD)
        symbols (list): List of symbols to backtest, or None to use config
    """
    logger.info(f"Starting combined strategy backtest from {start_date} to {end_date}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override symbols if provided
    if symbols:
        config['general']['symbols'] = symbols
        
    # Override dates
    config['general']['backtest_start_date'] = start_date
    config['general']['backtest_end_date'] = end_date
    
    # Initialize backtester
    backtester = Backtester(config_file)
    
    # Run backtest
    results = backtester.run_backtest(start_date, end_date)
    
    # Generate report
    report = results.generate_report()
    print(report)
    
    # Save report to file
    report_file = f"combined_strategy_report_{start_date}_to_{end_date}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Saved report to {report_file}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    results.equity_curve['equity'].plot()
    plt.title(f"Combined Strategy Equity Curve ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    equity_curve_file = f"combined_strategy_equity_{start_date}_to_{end_date}.png"
    plt.savefig(equity_curve_file)
    plt.close()
    logger.info(f"Saved equity curve to {equity_curve_file}")
    
    # Save trades to CSV
    trades_df = pd.DataFrame(results.trades)
    if not trades_df.empty:
        trades_file = f"combined_strategy_trades_{start_date}_to_{end_date}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved trades to {trades_file}")
    
    # Save metrics to JSON
    metrics_file = f"combined_strategy_metrics_{start_date}_to_{end_date}.json"
    with open(metrics_file, 'w') as f:
        json.dump(results.metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Analyze trades by symbol, strategy, and regime
    analyze_trades(results.trades)
    
    return results

def analyze_trades(trades):
    """Analyze trades by symbol, strategy, and market regime
    
    Args:
        trades (list): List of trade dictionaries
    """
    if not trades:
        logger.info("No trades to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    
    # Analysis by symbol
    print("\n=== ANALYSIS BY SYMBOL ===")
    symbol_analysis = {}
    for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        wins = symbol_trades[symbol_trades['pnl'] > 0]
        win_rate = len(wins) / len(symbol_trades) if len(symbol_trades) > 0 else 0
        total_pnl = symbol_trades['pnl'].sum()
        avg_return = symbol_trades['return_pct'].mean()
        
        symbol_analysis[symbol] = {
            'trades': len(symbol_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return_pct': avg_return
        }
        
        print(f"{symbol}:")
        print(f"  Trades: {len(symbol_trades)}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Avg Return: {avg_return:.2%}")
    
    # Analysis by strategy
    print("\n=== ANALYSIS BY STRATEGY ===")
    if 'strategy' in trades_df.columns:
        for strategy in trades_df['strategy'].unique():
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            wins = strategy_trades[strategy_trades['pnl'] > 0]
            win_rate = len(wins) / len(strategy_trades) if len(strategy_trades) > 0 else 0
            total_pnl = strategy_trades['pnl'].sum()
            avg_return = strategy_trades['return_pct'].mean()
            
            print(f"{strategy.replace('_', ' ').title()}:")
            print(f"  Trades: {len(strategy_trades)}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Avg Return: {avg_return:.2%}")
    
    # Analysis by market regime
    print("\n=== ANALYSIS BY MARKET REGIME ===")
    if 'regime' in trades_df.columns:
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            wins = regime_trades[regime_trades['pnl'] > 0]
            win_rate = len(wins) / len(regime_trades) if len(regime_trades) > 0 else 0
            total_pnl = regime_trades['pnl'].sum()
            avg_return = regime_trades['return_pct'].mean()
            
            print(f"{regime.capitalize()}:")
            print(f"  Trades: {len(regime_trades)}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Avg Return: {avg_return:.2%}")
    
    # Analysis by exit reason
    print("\n=== ANALYSIS BY EXIT REASON ===")
    if 'exit_reason' in trades_df.columns:
        for reason in trades_df['exit_reason'].unique():
            reason_trades = trades_df[trades_df['exit_reason'] == reason]
            wins = reason_trades[reason_trades['pnl'] > 0]
            win_rate = len(wins) / len(reason_trades) if len(reason_trades) > 0 else 0
            total_pnl = reason_trades['pnl'].sum()
            avg_return = reason_trades['return_pct'].mean()
            
            print(f"{reason.replace('_', ' ').title()}:")
            print(f"  Trades: {len(reason_trades)}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Avg Return: {avg_return:.2%}")

def run_quarterly_backtests():
    """Run backtests for each quarter to analyze performance across different time periods"""
    # Define quarters
    quarters = [
        ('2023-01-01', '2023-03-31', 'Q1 2023'),
        ('2023-04-01', '2023-06-30', 'Q2 2023'),
        ('2023-07-01', '2023-09-30', 'Q3 2023'),
        ('2023-10-01', '2023-12-31', 'Q4 2023'),
        ('2024-01-01', '2024-03-31', 'Q1 2024'),
        ('2024-04-01', '2024-04-30', 'April 2024')
    ]
    
    # Run backtest for each quarter
    results = {}
    for start_date, end_date, label in quarters:
        print(f"\n{'='*50}")
        print(f"Running backtest for {label} ({start_date} to {end_date})")
        print(f"{'='*50}\n")
        
        result = run_backtest(start_date=start_date, end_date=end_date)
        results[label] = result.metrics
    
    # Compare quarterly results
    compare_quarterly_results(results)

def compare_quarterly_results(results):
    """Compare quarterly backtest results
    
    Args:
        results (dict): Dictionary of quarterly results
    """
    # Extract key metrics for comparison
    comparison = {}
    for period, metrics in results.items():
        comparison[period] = {
            'total_return_pct': metrics['total_return_pct'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'total_trades': metrics['total_trades']
        }
    
    # Convert to DataFrame for easier display
    comparison_df = pd.DataFrame(comparison).T
    
    # Print comparison
    print("\n=== QUARTERLY PERFORMANCE COMPARISON ===")
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv("combined_strategy_quarterly_comparison.csv")
    
    # Plot key metrics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    comparison_df['total_return_pct'].plot(kind='bar')
    plt.title("Total Return (%)")
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 2)
    comparison_df['win_rate'].plot(kind='bar')
    plt.title("Win Rate")
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 3)
    comparison_df['profit_factor'].plot(kind='bar')
    plt.title("Profit Factor")
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 4)
    comparison_df['sharpe_ratio'].plot(kind='bar')
    plt.title("Sharpe Ratio")
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("combined_strategy_quarterly_comparison.png")
    plt.close()

def run_expanded_symbol_test():
    """Run backtest with an expanded set of symbols"""
    # Define expanded symbol list
    expanded_symbols = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD',
        'INTC', 'IBM', 'CSCO', 'NFLX', 'DIS', 'JPM', 'BAC', 'GS', 'V', 'MA'
    ]
    
    print(f"\n{'='*50}")
    print(f"Running backtest with expanded symbol list ({len(expanded_symbols)} symbols)")
    print(f"{'='*50}\n")
    
    # Run backtest with expanded symbols
    result = run_backtest(symbols=expanded_symbols)
    
    return result

if __name__ == "__main__":
    # Run full period backtest
    print("\n=== RUNNING FULL PERIOD BACKTEST ===\n")
    run_backtest()
    
    # Run quarterly backtests
    print("\n=== RUNNING QUARTERLY BACKTESTS ===\n")
    run_quarterly_backtests()
    
    # Run expanded symbol test
    print("\n=== RUNNING EXPANDED SYMBOL TEST ===\n")
    run_expanded_symbol_test()
