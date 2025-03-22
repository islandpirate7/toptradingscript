#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Alpaca Trading System Backtest Runner
---------------------------------------------
This script runs a backtest of the enhanced Alpaca trading system
using historical data from 2023 (compatible with Alpaca free tier).
"""

import os
import json
import yaml
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enhanced_alpaca_trading import EnhancedAlpacaTradingSystem, load_alpaca_credentials, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_backtest")

def plot_backtest_results(results, output_file="enhanced_backtest_results.png"):
    """Plot backtest results"""
    try:
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        equity_curve = results.get('equity_curve', [])
        
        if not equity_curve:
            logger.error("No equity curve data available")
            return False
            
        dates = [dt.datetime.fromisoformat(point[0]) if isinstance(point[0], str) else point[0] 
                for point in equity_curve]
        values = [point[1] for point in equity_curve]
        
        axs[0].plot(dates, values)
        axs[0].set_title('Equity Curve')
        axs[0].set_ylabel('Portfolio Value ($)')
        axs[0].grid(True)
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        axs[1].fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        axs[1].set_title('Drawdown (%)')
        axs[1].set_ylabel('Drawdown (%)')
        axs[1].grid(True)
        
        # Plot daily returns
        daily_returns = np.diff(values) / values[:-1] * 100
        daily_return_dates = dates[1:]
        axs[2].bar(daily_return_dates, daily_returns, color='green', alpha=0.6)
        axs[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[2].set_title('Daily Returns (%)')
        axs[2].set_ylabel('Return (%)')
        axs[2].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Backtest results plot saved to {output_file}")
        
        # Display key metrics on the plot
        metrics_text = (
            f"Total Return: {results.get('total_return_pct', 0):.2f}%\n"
            f"Annualized Return: {results.get('annualized_return_pct', 0):.2f}%\n"
            f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%\n"
            f"Win Rate: {results.get('win_rate', 0):.2f}%\n"
            f"Profit Factor: {results.get('profit_factor', 0):.2f}"
        )
        axs[0].text(0.02, 0.05, metrics_text, transform=axs[0].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        return True
    except Exception as e:
        logger.error(f"Error plotting backtest results: {str(e)}")
        return False

def save_results_to_json(results, output_file="enhanced_backtest_results.json"):
    """Save backtest results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Backtest results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving backtest results: {str(e)}")

def main():
    """Main function"""
    # Load configuration
    config_file = "enhanced_alpaca_config.yaml"
    config = load_config(config_file)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # Load Alpaca credentials
    api_key, api_secret, base_url = load_alpaca_credentials(mode='paper')
    if not all([api_key, api_secret, base_url]):
        logger.error("Failed to load Alpaca credentials. Exiting.")
        return
    
    credentials = {
        'api_key': api_key,
        'api_secret': api_secret,
        'base_url': base_url
    }
    
    # Define backtest period (using 2023 data for Alpaca free tier compatibility)
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Initialize trading system
    trading_system = EnhancedAlpacaTradingSystem(config_file=config_file, mode='paper')
    
    # Run backtest
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    results = trading_system.run_backtest(start_date, end_date)
    
    if results:
        # Log results
        logger.info(f"Backtest completed with the following results:")
        logger.info(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
        logger.info(f"Annualized Return: {results.get('annualized_return_pct', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"Win Rate: {results.get('win_rate', 0):.2f}%")
        logger.info(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        # Save results
        save_results_to_json(results)
        
        # Plot results
        plot_backtest_results(results)
        
        # Compare with original results if available
        try:
            with open("results_2023_yahoo.json", 'r') as f:
                original_results = json.load(f)
            
            logger.info("Comparison with original trading system:")
            logger.info(f"Original Total Return: {original_results.get('total_return_pct', 0):.2f}%")
            logger.info(f"Enhanced Total Return: {results.get('total_return_pct', 0):.2f}%")
            logger.info(f"Improvement: {results.get('total_return_pct', 0) - original_results.get('total_return_pct', 0):.2f}%")
        except Exception as e:
            logger.warning(f"Could not compare with original results: {str(e)}")

if __name__ == "__main__":
    main()
