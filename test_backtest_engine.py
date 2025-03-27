#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the updated backtest engine
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime
from backtest_engine_updated import run_backtest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    """Run a test backtest"""
    print("Testing updated backtest engine...")
    
    # Load configuration
    config_path = 'sp500_config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        print("Please create a configuration file with valid Alpaca API credentials.")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify API credentials exist
    api_key = config.get('alpaca', {}).get('api_key', '')
    api_secret = config.get('alpaca', {}).get('api_secret', '')
    
    if not api_key or not api_secret:
        print("Error: Alpaca API credentials not found in configuration file.")
        print("Please add valid 'api_key' and 'api_secret' under the 'alpaca' section in your sp500_config.yaml file.")
        return
    
    # Run a 3-month backtest
    start_date = "2023-01-01"
    end_date = "2023-03-31"
    initial_capital = 100000
    
    # Ensure required directories exist
    paths = config.get('paths', {})
    if not paths:
        # Try to get paths from backtest section
        paths = {
            'backtest_results': 'backtest_results',
            'plots': 'plots',
            'trades': 'trades',
            'performance': 'performance'
        }
    
    for path_key in ['backtest_results', 'plots', 'trades', 'performance']:
        path = paths.get(path_key, path_key)
        os.makedirs(path, exist_ok=True)
    
    # Run backtest
    print(f"Running backtest from {start_date} to {end_date} with ${initial_capital} initial capital")
    print(f"Using Alpaca API credentials from {config_path}")
    
    metrics, signals = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_signals=40,
        weekly_selection=True
    )
    
    if metrics:
        print("\n=== Backtest Results ===")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annual Return: {metrics.get('annual_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):.2f}")
        
        # Print sector performance
        print("\n=== Sector Performance ===")
        sector_performance = metrics.get('sector_performance', {})
        for sector, perf in sector_performance.items():
            print(f"{sector}: {perf.get('win_rate', 0):.2f}% win rate, {perf.get('return_contribution', 0):.2f}% contribution")
        
        # Print market cap performance
        print("\n=== Market Cap Performance ===")
        large_cap = metrics.get('large_cap_performance', {})
        mid_cap = metrics.get('mid_cap_performance', {})
        print(f"Large Cap: {large_cap.get('num_trades', 0)} trades, {large_cap.get('win_rate', 0):.2f}% win rate, {large_cap.get('return_contribution', 0):.2f}% contribution")
        print(f"Mid Cap: {mid_cap.get('num_trades', 0)} trades, {mid_cap.get('win_rate', 0):.2f}% win rate, {mid_cap.get('return_contribution', 0):.2f}% contribution")
        
        # Save results to file
        results_dir = paths.get('backtest_results', 'backtest_results')
        results_file = os.path.join(results_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    serializable_metrics[key] = {}
                    for k, v in value.items():
                        if hasattr(v, 'item'):  # Check if it's a numpy type
                            serializable_metrics[key][k] = v.item()
                        else:
                            serializable_metrics[key][k] = v
                elif hasattr(value, 'item'):  # Check if it's a numpy type
                    serializable_metrics[key] = value.item()
                else:
                    serializable_metrics[key] = value
            
            json.dump({
                'metrics': serializable_metrics,
                'signals_count': len(signals)
            }, f, indent=4)
        
        print(f"Results saved to {results_file}")
    else:
        print("Backtest failed to generate metrics")

if __name__ == "__main__":
    main()
