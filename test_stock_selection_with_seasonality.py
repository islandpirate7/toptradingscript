#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Stock Selection with Seasonality
-------------------------------------
This script tests the stock selection algorithm with seasonality
to see which stocks are selected for different months.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns

from combined_strategy import CombinedStrategy
from seasonality_enhanced import SeasonalityEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from YAML file
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_stock_selection_for_month(config, month, year=2024, top_n=10):
    """Test stock selection for a specific month
    
    Args:
        config (dict): Configuration dictionary
        month (int): Month to test (1-12)
        year (int): Year to test
        top_n (int): Number of top stocks to select
        
    Returns:
        list: List of selected stocks with scores
    """
    # Initialize strategy
    strategy = CombinedStrategy(config)
    
    # Create a date object for the specified month
    test_date = dt.datetime(year, month, 15)
    
    # Get all symbols from the configuration
    all_symbols = config.get('general', {}).get('symbols', [])
    
    # Check if we have seasonality data for these symbols
    if not hasattr(strategy, 'seasonality_analyzer') or strategy.seasonality_analyzer is None:
        logger.error("Seasonality analyzer not initialized")
        return []
    
    # Get seasonality scores for all symbols
    seasonality_scores = []
    for symbol in all_symbols:
        score, direction = strategy.get_seasonal_score(symbol, test_date)
        seasonality_scores.append({
            'symbol': symbol,
            'score': score,
            'direction': direction
        })
    
    # Sort by score (descending)
    seasonality_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top N
    return seasonality_scores[:top_n]

def test_all_months(config_file, year=2024):
    """Test stock selection for all months
    
    Args:
        config_file (str): Path to configuration file
        year (int): Year to test
    """
    # Load configuration
    config = load_config(config_file)
    
    # Test each month
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    all_results = {}
    
    for month in range(1, 13):
        logger.info(f"Testing stock selection for {month_names[month-1]} {year}")
        selected_stocks = test_stock_selection_for_month(config, month, year)
        
        # Print results
        logger.info(f"Top stocks for {month_names[month-1]} {year}:")
        for i, stock in enumerate(selected_stocks):
            logger.info(f"{i+1}. {stock['symbol']} - Score: {stock['score']:.4f}, Direction: {stock['direction']}")
        
        all_results[month_names[month-1]] = selected_stocks
    
    # Create visualization
    create_seasonality_heatmap(all_results, year)
    
    return all_results

def create_seasonality_heatmap(results, year):
    """Create a heatmap of seasonality scores
    
    Args:
        results (dict): Dictionary of results by month
        year (int): Year of the analysis
    """
    # Extract all unique symbols
    all_symbols = set()
    for month_data in results.values():
        for stock in month_data:
            all_symbols.add(stock['symbol'])
    
    # Create a dataframe for the heatmap
    months = list(results.keys())
    symbols = sorted(list(all_symbols))
    
    # Initialize with NaN values
    data = np.full((len(symbols), len(months)), np.nan)
    
    # Fill in the scores
    for j, month in enumerate(months):
        month_data = results[month]
        for stock in month_data:
            if stock['symbol'] in symbols:
                i = symbols.index(stock['symbol'])
                data[i, j] = stock['score']
    
    # Create dataframe
    df = pd.DataFrame(data, index=symbols, columns=months)
    
    # Create output directory if it doesn't exist
    os.makedirs('output/seasonality_analysis', exist_ok=True)
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(df, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title(f'Seasonality Scores by Month ({year})')
    plt.tight_layout()
    plt.savefig(f'output/seasonality_analysis/seasonality_heatmap_{year}.png')
    plt.close()
    
    # Create bar charts for each month
    for month in months:
        month_data = results[month]
        if not month_data:
            continue
            
        symbols = [stock['symbol'] for stock in month_data]
        scores = [stock['score'] for stock in month_data]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(symbols, scores)
        
        # Color bars based on score
        for i, bar in enumerate(bars):
            if scores[i] > 0.5:
                bar.set_color('green')
            elif scores[i] < -0.5:
                bar.set_color('red')
            else:
                bar.set_color('gray')
                
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Top Stocks for {month} {year}')
        plt.ylabel('Seasonality Score')
        plt.ylim(-1, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'output/seasonality_analysis/top_stocks_{month}_{year}.png')
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test stock selection with seasonality')
    parser.add_argument('--config', type=str, default='configuration_combined_strategy.yaml',
                        help='Path to configuration file')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year to test')
    parser.add_argument('--month', type=int, default=None,
                        help='Month to test (1-12), if not specified, test all months')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top stocks to select')
    args = parser.parse_args()
    
    if args.month:
        # Test a specific month
        config = load_config(args.config)
        results = test_stock_selection_for_month(config, args.month, args.year, args.top_n)
        
        # Print results
        print(f"\nTop {args.top_n} stocks for {dt.datetime(args.year, args.month, 1).strftime('%B %Y')}:")
        for i, stock in enumerate(results):
            print(f"{i+1}. {stock['symbol']} - Score: {stock['score']:.4f}, Direction: {stock['direction']}")
    else:
        # Test all months
        test_all_months(args.config, args.year)
