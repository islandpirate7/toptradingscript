#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the improved seasonality integration in the combined strategy.
This script runs backtests with the combined strategy using seasonality data
and analyzes the performance based on seasonality alignment.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the strategy classes
from combined_strategy import CombinedStrategy
from backtest_engine import BacktestEngine
from utils import setup_logging, load_config, calculate_performance_metrics

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def load_alpaca_credentials(credentials_file='alpaca_credentials.json'):
    """
    Load Alpaca API credentials from a JSON file.
    
    Args:
        credentials_file (str): Path to the credentials file
        
    Returns:
        dict: Dictionary containing API key, secret key, and base URL
    """
    try:
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
        
        # Default to paper trading for testing
        paper_creds = credentials.get('paper', {})
        
        # Log the credentials being used (without secrets)
        logger.info(f"Using Alpaca paper trading API with base URL: {paper_creds.get('base_url')}")
        
        return {
            'api_key': paper_creds.get('api_key', ''),
            'api_secret': paper_creds.get('api_secret', ''),
            'base_url': paper_creds.get('base_url', 'https://paper-api.alpaca.markets/v2')
        }
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {e}")
        return {}

def run_backtest(config_file, output_dir='output/seasonality_test'):
    """
    Run a backtest using the combined strategy with seasonality data.
    
    Args:
        config_file (str): Path to the configuration file
        output_dir (str): Directory to save the backtest results
        
    Returns:
        tuple: Backtest results and performance metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(config_file)
    if not config:
        logger.error(f"Failed to load configuration from {config_file}")
        return None, None
    
    # Load Alpaca credentials
    alpaca_credentials = load_alpaca_credentials()
    if not alpaca_credentials:
        logger.error("Failed to load Alpaca credentials")
        return None, None
    
    # Extract backtest parameters
    symbols = config['general']['symbols']
    timeframe = config['general']['timeframe']
    initial_capital = config['general']['initial_capital']
    start_date = config['general']['backtest_start_date']
    end_date = config['general']['backtest_end_date']
    
    # Initialize the strategy
    strategy = CombinedStrategy(config=config)
    
    # Initialize the backtest engine
    backtest = BacktestEngine(
        strategy=strategy,
        symbols=symbols,
        timeframe=timeframe,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        api_key=alpaca_credentials['api_key'],
        api_secret=alpaca_credentials['api_secret'],
        base_url=alpaca_credentials['base_url']
    )
    
    # Run the backtest
    logger.info(f"Running backtest from {start_date} to {end_date}")
    results = backtest.run()
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(results)
    
    # Save results
    results.to_csv(f"{output_dir}/backtest_results.csv")
    
    # Save metrics
    with open(f"{output_dir}/performance_metrics.yaml", 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    return results, metrics

def analyze_seasonality_impact(results, seasonality_file, output_dir='output/seasonality_test'):
    """
    Analyze the impact of seasonality on trading performance.
    
    Args:
        results (pd.DataFrame): Backtest results
        seasonality_file (str): Path to the seasonality data file
        output_dir (str): Directory to save the analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the seasonality analyzer
    try:
        from seasonality_enhanced import SeasonalityEnhanced
        
        # Load the configuration used for the backtest
        config_file = f"{output_dir}/../config_{results['timestamp'].min().strftime('%Y-%m-%d')}_{results['timestamp'].max().strftime('%Y-%m-%d')}.yaml"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = None
        
        # Initialize the seasonality analyzer with the config
        seasonality_analyzer = SeasonalityEnhanced(seasonality_file, config)
        logger.info(f"Initialized seasonality analyzer with data file: {seasonality_file}")
    except Exception as e:
        logger.error(f"Error initializing seasonality analyzer: {e}")
        return
    
    # Filter trades in results
    trades = results[results['action'].isin(['BUY', 'SELL'])]
    if len(trades) == 0:
        logger.warning("No trades found in results")
        return
    
    # Add seasonality score to trades
    trades['date'] = pd.to_datetime(trades['timestamp'])
    trades['month'] = trades['date'].dt.month
    trades['day'] = trades['date'].dt.day
    
    # Calculate seasonality score for each trade
    trades['seasonality_score'] = trades.apply(
        lambda row: seasonality_analyzer.get_seasonal_score(row['symbol'], row['date']), 
        axis=1
    )
    
    # Categorize trades by seasonality score
    score_threshold = 0.6  # Threshold for high seasonality score
    trades['seasonality_category'] = pd.cut(
        trades['seasonality_score'], 
        bins=[0, 0.4, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Group trades by seasonality category
    grouped_trades = trades.groupby('seasonality_category')
    
    # Calculate metrics for each group
    metrics_by_category = {}
    for category, group in grouped_trades:
        metrics_by_category[category] = calculate_trade_metrics(group)
    
    # Add overall metrics
    metrics_by_category['Overall'] = calculate_trade_metrics(trades)
    
    # Print comparison
    logger.info("\n=== Seasonality Impact Analysis ===")
    logger.info(f"Total trades: {len(trades)}")
    
    for category, metrics in metrics_by_category.items():
        if category != 'Overall':
            category_trades = grouped_trades.get_group(category) if category in grouped_trades.groups else pd.DataFrame()
            trade_count = len(category_trades)
            percentage = trade_count / len(trades) * 100 if len(trades) > 0 else 0
            logger.info(f"{category} seasonality trades: {trade_count} ({percentage:.1f}%)")
    
    logger.info("\nPerformance Comparison:")
    for category, metrics in metrics_by_category.items():
        logger.info(f"  {category} seasonality - Win rate: {metrics['win_rate']:.2f}, Avg return: {metrics['avg_return']:.4f}, Profit factor: {metrics['profit_factor']:.2f}")
    
    # Plot comparison
    plot_seasonality_comparison(metrics_by_category, output_dir)
    
    # Save metrics by category
    with open(f"{output_dir}/seasonality_metrics.yaml", 'w') as f:
        yaml.dump(metrics_by_category, f, default_flow_style=False)
    
    # Save trades with seasonality data
    trades.to_csv(f"{output_dir}/trades_with_seasonality.csv")
    
    # Analyze monthly performance
    monthly_performance = analyze_monthly_seasonality(trades, output_dir)
    
    return metrics_by_category

def analyze_monthly_seasonality(trades, output_dir):
    """
    Analyze trading performance by month to identify seasonal patterns.
    
    Args:
        trades (pd.DataFrame): Trades data with seasonality information
        output_dir (str): Directory to save the analysis results
        
    Returns:
        dict: Monthly performance metrics
    """
    # Group trades by month
    trades['month_name'] = trades['date'].dt.strftime('%B')
    trades['month_num'] = trades['date'].dt.month
    
    # Sort by month number
    monthly_groups = trades.groupby('month_num')
    
    # Calculate metrics for each month
    monthly_metrics = {}
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    for month_num, group in monthly_groups:
        month_name = month_names.get(month_num, f"Month {month_num}")
        monthly_metrics[month_name] = calculate_trade_metrics(group)
        monthly_metrics[month_name]['trade_count'] = len(group)
        monthly_metrics[month_name]['avg_seasonality_score'] = group['seasonality_score'].mean()
    
    # Save monthly metrics
    with open(f"{output_dir}/monthly_performance.yaml", 'w') as f:
        yaml.dump(monthly_metrics, f, default_flow_style=False)
    
    # Plot monthly performance
    plot_monthly_performance(monthly_metrics, output_dir)
    
    # Log monthly performance
    logger.info("\n=== Monthly Performance Analysis ===")
    for month, metrics in sorted(monthly_metrics.items(), key=lambda x: list(month_names.values()).index(x[0]) if x[0] in month_names.values() else 99):
        logger.info(f"{month}: Win Rate: {metrics['win_rate']:.2f}, Avg Return: {metrics['avg_return']:.4f}, Trades: {metrics['trade_count']}, Avg Seasonality Score: {metrics['avg_seasonality_score']:.2f}")
    
    return monthly_metrics

def plot_monthly_performance(monthly_metrics, output_dir):
    """
    Plot monthly performance metrics.
    
    Args:
        monthly_metrics (dict): Dictionary of monthly performance metrics
        output_dir (str): Directory to save the plot
    """
    # Sort months in calendar order
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    # Filter to only include months that are in the data
    months = [m for m in month_order if m in monthly_metrics]
    
    if not months:
        logger.warning("No monthly data to plot")
        return
    
    # Extract metrics
    win_rates = [monthly_metrics[m]['win_rate'] for m in months]
    avg_returns = [monthly_metrics[m]['avg_return'] for m in months]
    trade_counts = [monthly_metrics[m]['trade_count'] for m in months]
    seasonality_scores = [monthly_metrics[m]['avg_seasonality_score'] for m in months]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot win rates
    axes[0, 0].bar(months, win_rates, color='green')
    axes[0, 0].set_title('Win Rate by Month')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticklabels(months, rotation=45)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average returns
    axes[0, 1].bar(months, avg_returns, color='blue')
    axes[0, 1].set_title('Average Return by Month')
    axes[0, 1].set_xticklabels(months, rotation=45)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot trade counts
    axes[1, 0].bar(months, trade_counts, color='orange')
    axes[1, 0].set_title('Number of Trades by Month')
    axes[1, 0].set_xticklabels(months, rotation=45)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average seasonality scores
    axes[1, 1].bar(months, seasonality_scores, color='purple')
    axes[1, 1].set_title('Average Seasonality Score by Month')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticklabels(months, rotation=45)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle('Monthly Performance Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/monthly_performance.png")
    plt.close()

def check_seasonal_alignment(trade, seasonal_map):
    """
    Check if a trade is aligned with seasonal opportunities.
    
    Args:
        trade (pd.Series): Trade data
        seasonal_map (dict): Dictionary of seasonal opportunities
        
    Returns:
        bool: True if the trade is aligned with seasonal opportunities, False otherwise
    """
    symbol = trade['symbol']
    month = trade['month']
    action = trade['action']
    
    # Check if the symbol has seasonal data
    if symbol not in seasonal_map:
        return False
    
    # Check if the month has seasonal data for this symbol
    if month not in seasonal_map[symbol]:
        return False
    
    # Get seasonal direction
    direction = seasonal_map[symbol][month]['direction']
    
    # Check alignment
    if (direction == 'LONG' and action == 'BUY') or (direction == 'SHORT' and action == 'SELL'):
        return True
    
    return False

def calculate_trade_metrics(trades):
    """
    Calculate performance metrics for a set of trades.
    
    Args:
        trades (pd.DataFrame): Trades data
        
    Returns:
        dict: Dictionary of performance metrics
    """
    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'avg_return': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    # Calculate returns
    returns = trades['pnl'] / trades['position_value']
    
    # Calculate win rate
    wins = sum(trades['pnl'] > 0)
    win_rate = wins / len(trades) if len(trades) > 0 else 0
    
    # Calculate average return
    avg_return = returns.mean() if len(returns) > 0 else 0
    
    # Calculate profit factor
    gross_profit = sum(trades[trades['pnl'] > 0]['pnl'])
    gross_loss = abs(sum(trades[trades['pnl'] < 0]['pnl']))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calculate total return
    total_return = sum(trades['pnl'])
    
    return {
        'win_rate': win_rate,
        'avg_return': avg_return,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'trade_count': len(trades)
    }

def plot_seasonality_comparison(metrics_by_category, output_dir):
    """
    Plot a comparison of performance across different seasonality categories.
    
    Args:
        metrics_by_category (dict): Dictionary of metrics by seasonality category
        output_dir (str): Directory to save the plot
    """
    # Extract categories and metrics
    categories = [cat for cat in metrics_by_category.keys() if cat != 'Overall']
    categories.sort()  # Sort categories (Low, Medium, High)
    
    # Add Overall category at the end
    if 'Overall' in metrics_by_category:
        categories.append('Overall')
    
    # Extract metrics for each category
    win_rates = [metrics_by_category[cat]['win_rate'] for cat in categories]
    avg_returns = [metrics_by_category[cat]['avg_return'] for cat in categories]
    profit_factors = [metrics_by_category[cat]['profit_factor'] for cat in categories]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define colors for each category
    colors = ['red', 'yellow', 'green', 'blue']
    
    # Plot win rate comparison
    axes[0].bar(categories, win_rates, color=colors[:len(categories)])
    axes[0].set_title('Win Rate by Seasonality Category')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average return comparison
    axes[1].bar(categories, avg_returns, color=colors[:len(categories)])
    axes[1].set_title('Average Return by Seasonality Category')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot profit factor comparison
    axes[2].bar(categories, profit_factors, color=colors[:len(categories)])
    axes[2].set_title('Profit Factor by Seasonality Category')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle('Performance Comparison by Seasonality Category', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/seasonality_comparison.png")
    plt.close()

def analyze_market_regimes(results, config, output_dir='output/seasonality_test'):
    """
    Analyze the performance of the strategy in different market regimes.
    
    Args:
        results (pd.DataFrame): Backtest results
        config (dict): Strategy configuration
        output_dir (str): Directory to save the analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert market regime to string if it's an enum
    if 'market_regime' in results.columns:
        results['market_regime_str'] = results['market_regime'].astype(str)
    else:
        # If no market regime column, we can't do the analysis
        logging.warning("No market regime column found in results")
        return
    
    # Extract regime data
    regime_data = results[~results['market_regime_str'].isna()].copy()
    
    if len(regime_data) == 0:
        logging.warning("No market regime data available for analysis")
        return
    
    # Count days in each regime
    regime_counts = regime_data['market_regime_str'].value_counts().to_dict()
    
    # Group trades by regime
    trades = results[results['action'].isin(['BUY', 'SELL'])].copy()
    
    # Check if market_regime column exists in trades
    if 'market_regime_str' not in trades.columns:
        trades['date'] = pd.to_datetime(trades['timestamp']).dt.date
        
        # Create a date-to-regime mapping
        regime_data['date'] = pd.to_datetime(regime_data['timestamp']).dt.date
        date_regime_map = regime_data[['date', 'market_regime_str']].drop_duplicates().set_index('date')['market_regime_str'].to_dict()
        
        # Add market_regime to trades based on date
        trades['market_regime_str'] = trades['date'].map(date_regime_map)
    
    # Fill any missing regimes with 'unknown'
    trades['market_regime_str'] = trades['market_regime_str'].fillna('unknown')
    
    # Group by regime
    regime_groups = trades.groupby('market_regime_str')
    
    # Calculate metrics by regime
    regime_metrics = {}
    for regime, group in regime_groups:
        regime_metrics[regime] = calculate_trade_metrics(group)
    
    # Print comparison
    print("\n=== Market Regime Analysis ===")
    print("Regime Distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(regime_data)*100:.1f}%)")
    
    print("\nPerformance by Regime:")
    for regime, metrics in regime_metrics.items():
        print(f"  {regime} Regime:")
        print(f"    Win Rate: {metrics['win_rate']:.2f}")
        print(f"    Avg Return: {metrics['avg_return']:.4f}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"    Trade Count: {metrics['trade_count']}")
    
    # Plot comparison
    plot_regime_comparison(regime_metrics, output_dir)
    
    # Save comparison
    with open(f"{output_dir}/regime_analysis.yaml", 'w') as f:
        yaml.dump({
            'regime_counts': regime_counts,
            'regime_metrics': regime_metrics
        }, f, default_flow_style=False)

def plot_regime_comparison(regime_metrics, output_dir):
    """
    Plot a comparison of performance across different market regimes.
    
    Args:
        regime_metrics (dict): Dictionary of metrics by regime
        output_dir (str): Directory to save the plot
    """
    # Extract regimes and metrics
    regimes = list(regime_metrics.keys())
    win_rates = [regime_metrics[r]['win_rate'] for r in regimes]
    avg_returns = [regime_metrics[r]['avg_return'] for r in regimes]
    profit_factors = [regime_metrics[r]['profit_factor'] for r in regimes]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot win rate comparison
    axes[0].bar(regimes, win_rates, color=['green', 'red', 'blue'][:len(regimes)])
    axes[0].set_title('Win Rate by Market Regime')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average return comparison
    axes[1].bar(regimes, avg_returns, color=['green', 'red', 'blue'][:len(regimes)])
    axes[1].set_title('Average Return by Market Regime')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot profit factor comparison
    axes[2].bar(regimes, profit_factors, color=['green', 'red', 'blue'][:len(regimes)])
    axes[2].set_title('Profit Factor by Market Regime')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle('Performance Comparison Across Market Regimes', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/regime_comparison.png")
    plt.close()

def plot_period_comparison(comparison, output_dir):
    """
    Plot a comparison of performance across different periods.
    
    Args:
        comparison (dict): Dictionary of metrics by period
        output_dir (str): Directory to save the plot
    """
    # Extract periods and metrics
    periods = list(comparison.keys())
    win_rates = [comparison[p]['win_rate'] for p in periods]
    avg_returns = [comparison[p]['avg_return'] for p in periods]
    profit_factors = [comparison[p]['profit_factor'] for p in periods]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot win rate comparison
    axes[0].bar(periods, win_rates, color=['green', 'red', 'blue'][:len(periods)])
    axes[0].set_title('Win Rate by Period')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average return comparison
    axes[1].bar(periods, avg_returns, color=['green', 'red', 'blue'][:len(periods)])
    axes[1].set_title('Average Return by Period')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot profit factor comparison
    axes[2].bar(periods, profit_factors, color=['green', 'red', 'blue'][:len(periods)])
    axes[2].set_title('Profit Factor by Period')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle('Performance Comparison Across Periods', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/period_comparison.png")
    plt.close()

def main():
    """Main function to run the seasonality integration test."""
    # Create output directory
    output_dir = 'output/seasonality_test_enhanced'
    os.makedirs(output_dir, exist_ok=True)
    
    # Config file - use the enhanced multi-factor configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    
    # Load config to check settings
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Log key configuration settings
    logger.info(f"=== Testing Enhanced Seasonality Integration ===")
    logger.info(f"Multi-factor stock selection enabled: {config['stock_selection']['enable_multi_factor']}")
    logger.info(f"Technical weight: {config['stock_selection']['technical_weight']}, Seasonality weight: {config['stock_selection']['seasonality_weight']}")
    logger.info(f"Seasonality enabled: {config['seasonality']['enabled']}")
    logger.info(f"Seasonality data file: {config['seasonality']['data_file']}")
    logger.info(f"Seasonality min score threshold: {config['seasonality']['min_score_threshold']}")
    logger.info(f"Number of stocks in universe: {len(config['general']['symbols'])}")
    
    # Set up test periods
    test_periods = [
        ('2023-01-01', '2023-03-31'),  # Q1 2023
        ('2023-04-01', '2023-06-30'),  # Q2 2023
        ('2023-07-01', '2023-09-30'),  # Q3 2023
        ('2023-10-01', '2023-12-31'),  # Q4 2023
    ]
    
    # Run tests for each period
    all_results = {}
    for start_date, end_date in test_periods:
        # Update config with test period
        config['general']['backtest_start_date'] = start_date
        config['general']['backtest_end_date'] = end_date
        
        # Save updated config
        period_config_file = f"{output_dir}/config_{start_date}_{end_date}.yaml"
        with open(period_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run backtest
        logger.info(f"\n=== Running backtest for period: {start_date} to {end_date} ===")
        period_output_dir = f"{output_dir}/{start_date}_{end_date}"
        results, metrics = run_backtest(period_config_file, period_output_dir)
        
        if results is not None:
            # Store results
            all_results[f"{start_date}_{end_date}"] = {
                'results': results,
                'metrics': metrics
            }
            
            # Analyze seasonality impact
            analyze_seasonality_impact(
                results, 
                config['seasonality']['data_file'],
                period_output_dir
            )
            
            # Analyze market regimes
            analyze_market_regimes(
                results,
                config,
                period_output_dir
            )
    
    # Compare performance across periods
    if all_results:
        logger.info("\n=== Performance Comparison Across Periods ===")
        comparison = {}
        for period, data in all_results.items():
            comparison[period] = data['metrics']
        
        # Save comparison
        with open(f"{output_dir}/period_comparison.yaml", 'w') as f:
            yaml.dump(comparison, f, default_flow_style=False)
        
        # Plot comparison
        plot_period_comparison(comparison, output_dir)
        
        logger.info("Seasonality integration test completed successfully")
    else:
        logger.error("No valid results to compare")

if __name__ == "__main__":
    main()
