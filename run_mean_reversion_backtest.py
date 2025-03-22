#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Mean Reversion Strategy Backtest Runner
-----------------------------------------------
This script runs a backtest of the enhanced mean reversion strategy
using historical data from Q1 2024 with relaxed parameters for more signal generation.
"""

import os
import json
import yaml
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from enhanced_mean_reversion_backtest import EnhancedMeanReversionBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mean_reversion_backtest_2024.log"),
        logging.StreamHandler()
    ]
)

# Set specific loggers to DEBUG level for detailed diagnostics
strategy_logger = logging.getLogger("EnhancedMeanReversionStrategy")
strategy_logger.setLevel(logging.DEBUG)

backtest_logger = logging.getLogger("EnhancedMeanReversionBacktest")
backtest_logger.setLevel(logging.DEBUG)

logger = logging.getLogger("mean_reversion_backtest")

def plot_backtest_results(results, output_file="backtest_results.png"):
    """Plot backtest results"""
    # Extract performance metrics
    performance = results.get('performance', {})
    
    # Extract equity curve
    equity_curve = results.get('equity_curve', {})
    
    if not equity_curve:
        logger.warning("No equity curve data to plot")
        return
    
    # Convert equity curve to DataFrame
    dates = []
    equity_values = []
    
    for date_str, equity in equity_curve.items():
        try:
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            dates.append(date)
            equity_values.append(equity)
        except ValueError:
            logger.warning(f"Could not parse date string: {date_str}")
    
    if not dates:
        logger.warning("No valid dates in equity curve")
        return
    
    # Create DataFrame
    equity_df = pd.DataFrame({
        'date': dates,
        'equity': equity_values
    })
    equity_df.set_index('date', inplace=True)
    equity_df.sort_index(inplace=True)
    
    # Calculate drawdown
    equity_df['previous_peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['previous_peak'] - equity_df['equity']) / equity_df['previous_peak']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
    ax1.set_title('Mean Reversion Strategy Backtest Results')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot drawdown
    ax2.fill_between(equity_df.index, equity_df['drawdown'] * 100, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Display key metrics on the plot
    metrics_text = (
        f"Total Return: {performance.get('total_return', 0) * 100:.2f}%\n"
        f"Annualized Return: {performance.get('annualized_return', 0) * 100:.2f}%\n"
        f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}\n"
        f"Max Drawdown: {performance.get('max_drawdown', 0) * 100:.2f}%\n"
        f"Win Rate: {performance.get('win_rate', 0) * 100:.2f}%\n"
        f"Profit Factor: {performance.get('profit_factor', 0):.2f}\n"
        f"Total Trades: {performance.get('total_trades', 0)}\n"
        f"Total Signals: {results.get('total_signals', 0)}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Backtest results plot saved to {output_file}")

def save_results_to_json(results, output_file="mean_reversion_backtest_2024_results.json"):
    """Save backtest results to JSON file"""
    try:
        # Create a deep copy of results to avoid modifying the original
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = value
        
        # Convert Trade objects in trade_history to dictionaries
        if 'trade_history' in serializable_results:
            serializable_results['trade_history'] = [
                trade.to_dict() if hasattr(trade, 'to_dict') else trade 
                for trade in serializable_results['trade_history']
            ]
        
        # Convert Trade objects in current_positions to dictionaries
        if 'current_positions' in serializable_results:
            serializable_positions = {}
            for symbol, position in serializable_results['current_positions'].items():
                if hasattr(position, 'to_dict'):
                    serializable_positions[symbol] = position.to_dict()
                else:
                    serializable_positions[symbol] = position
            serializable_results['current_positions'] = serializable_positions
        
        # Convert datetime objects to strings for JSON serialization
        if 'equity_curve' in serializable_results:
            equity_curve = serializable_results['equity_curve']
            serializable_equity_curve = {}
            for date, value in equity_curve.items():
                if isinstance(date, datetime.datetime):
                    date_str = date.isoformat()
                else:
                    date_str = str(date)
                serializable_equity_curve[date_str] = value
            serializable_results['equity_curve'] = serializable_equity_curve
        
        # Handle any other non-serializable objects
        for key, value in serializable_results.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if hasattr(item, 'to_dict'):
                        value[i] = item.to_dict()
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Backtest results saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving backtest results: {str(e)}")
        return False

def analyze_skipped_signals(backtest):
    """Analyze skipped signals and their reasons"""
    skipped_signals = backtest.skipped_signals_reasons
    
    if not skipped_signals:
        logger.info("No skipped signals to analyze")
        return
    
    logger.info(f"Total skipped signals: {sum(len(reasons) for reasons in skipped_signals.values())}")
    
    # Count reasons by category
    reason_counts = {}
    for symbol, reasons in skipped_signals.items():
        for reason in reasons:
            if reason not in reason_counts:
                reason_counts[reason] = 0
            reason_counts[reason] += 1
    
    logger.info("Skipped signals by reason:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {reason}: {count}")
    
    # Count by symbol
    symbol_counts = {symbol: len(reasons) for symbol, reasons in skipped_signals.items()}
    logger.info("Skipped signals by symbol:")
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {symbol}: {count}")

def analyze_signal_generation(backtest):
    """Analyze signal generation statistics"""
    signals_by_symbol = backtest.signals_by_symbol
    
    if not signals_by_symbol:
        logger.info("No signals were generated")
        return
    
    logger.info(f"Total signals generated: {sum(signals_by_symbol.values())}")
    
    # Count by symbol
    logger.info("Signals generated by symbol:")
    for symbol, count in sorted(signals_by_symbol.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {symbol}: {count}")
    
    # Analyze signal directions if available
    if hasattr(backtest, 'signals_by_direction'):
        signals_by_direction = backtest.signals_by_direction
        logger.info("Signals by direction:")
        for direction, count in signals_by_direction.items():
            logger.info(f"  {direction}: {count}")

def main():
    """Main function"""
    # Load configuration
    config_file = "configuration_mean_reversion_2024_enhanced.yaml"
    
    # Define backtest period (using Q1 2024 data)
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2024, 3, 31)  # Full first quarter of 2024
    
    logger.info(f"Starting backtest with configuration from {config_file}")
    logger.info(f"Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize backtest
    backtest = EnhancedMeanReversionBacktest(config_file)
    
    # Run backtest
    results = backtest.run(start_date, end_date)
    
    # Extract performance metrics from results
    performance = results.get('performance', {})
    
    # Add signal statistics to results
    results['total_signals'] = backtest.total_signals_generated
    results['signals_by_symbol'] = backtest.signals_by_symbol
    results['skipped_signals_count'] = sum(len(reasons) for reasons in backtest.skipped_signals_reasons.values())
    
    # Log results
    logger.info(f"Backtest completed with the following results:")
    logger.info(f"Total Return: {performance.get('total_return', 0) * 100:.2f}%")
    logger.info(f"Annualized Return: {performance.get('annualized_return', 0) * 100:.2f}%")
    logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {performance.get('max_drawdown', 0) * 100:.2f}%")
    logger.info(f"Win Rate: {performance.get('win_rate', 0) * 100:.2f}%")
    logger.info(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
    logger.info(f"Total Trades: {performance.get('total_trades', 0)}")
    logger.info(f"Total Signals Generated: {results.get('total_signals', 0)}")
    logger.info(f"Total Signals Skipped: {results.get('skipped_signals_count', 0)}")
    
    # Analyze signal generation
    analyze_signal_generation(backtest)
    
    # Analyze skipped signals
    analyze_skipped_signals(backtest)
    
    # Save results
    save_results_to_json(results)
    
    # Plot results
    plot_backtest_results(results)

if __name__ == "__main__":
    main()
