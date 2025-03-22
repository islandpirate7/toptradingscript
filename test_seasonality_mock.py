#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the improved seasonality integration using mock data.
This script tests the seasonality integration without relying on external API connectivity.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mock_data(symbols, start_date, end_date):
    """
    Generate mock price data for testing.
    
    Args:
        symbols (list): List of symbols to generate data for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary of dataframes with mock data for each symbol
    """
    # Convert dates to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Generate mock data for each symbol
    data = {}
    for symbol in symbols:
        # Generate random starting price between 50 and 500
        base_price = np.random.uniform(50, 500)
        
        # Generate price series with random walk and some seasonality
        prices = []
        for i, date in enumerate(date_range):
            # Add some monthly seasonality effect
            month_effect = 0.01 * np.sin(2 * np.pi * date.month / 12)
            
            # Add some symbol-specific seasonality
            symbol_effect = 0.005 * np.sin(2 * np.pi * (ord(symbol[0]) % 12) / 12 + date.month)
            
            # Random daily change (-1% to +1%)
            daily_change = np.random.uniform(-0.01, 0.01)
            
            # Combine effects
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + daily_change + month_effect + symbol_effect)
            
            prices.append(price)
        
        # Create dataframe
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'close': prices,
            'volume': [int(np.random.uniform(100000, 10000000)) for _ in prices]
        })
        
        data[symbol] = df
    
    logger.info(f"Generated mock data for {len(symbols)} symbols from {start_date} to {end_date}")
    return data

def test_seasonality_calculation():
    """
    Test the seasonality score calculation functionality.
    """
    # Import the SeasonalityEnhanced class
    from seasonality_enhanced import SeasonalityEnhanced
    
    # Load the configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the seasonality analyzer
    seasonality_file = config['seasonality']['data_file']
    seasonality_analyzer = SeasonalityEnhanced(seasonality_file, config)
    
    # Test dates
    test_dates = [
        datetime(2023, 1, 15),  # January
        datetime(2023, 3, 15),  # March
        datetime(2023, 5, 15),  # May
        datetime(2023, 9, 15),  # September
        datetime(2023, 12, 15)  # December
    ]
    
    # Test symbols (use a mix of symbols with and without seasonality data)
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'NEM', 'FDX']
    
    # Calculate and print seasonality scores
    logger.info("\n=== Seasonality Score Test ===")
    
    results = {}
    for symbol in test_symbols:
        symbol_results = {}
        for date in test_dates:
            score = seasonality_analyzer.get_seasonal_score(symbol, date)
            month_name = date.strftime('%B')
            symbol_results[month_name] = score
            logger.info(f"Seasonality score for {symbol} on {date.strftime('%Y-%m-%d')} ({month_name}): {score:.4f}")
        
        results[symbol] = symbol_results
    
    # Plot results
    plot_seasonality_scores(results)
    
    return results

def plot_seasonality_scores(results):
    """
    Plot seasonality scores for different symbols across months.
    
    Args:
        results (dict): Dictionary of seasonality scores by symbol and month
    """
    # Create output directory
    output_dir = 'output/seasonality_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract months and symbols
    months = list(next(iter(results.values())).keys())
    symbols = list(results.keys())
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot scores for each symbol
    for symbol in symbols:
        scores = [results[symbol][month] for month in months]
        plt.plot(months, scores, marker='o', label=symbol)
    
    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Seasonality Score')
    plt.title('Seasonality Scores by Month and Symbol')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonality_scores.png")
    plt.close()

def test_combined_strategy():
    """
    Test the CombinedStrategy class with seasonality integration.
    """
    # Import the CombinedStrategy class
    from combined_strategy import CombinedStrategy
    
    # Load the configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the strategy
    strategy = CombinedStrategy(config)
    
    # Test dates
    test_date = datetime(2023, 3, 15)  # March 15, 2023
    
    # Generate mock data
    symbols = config['general']['symbols'][:20]  # Use first 20 symbols for testing
    start_date = (test_date - timedelta(days=100)).strftime('%Y-%m-%d')
    end_date = test_date.strftime('%Y-%m-%d')
    
    mock_data = generate_mock_data(symbols, start_date, end_date)
    
    # Test stock selection
    logger.info("\n=== Testing Combined Strategy with Seasonality ===")
    
    # Enable multi-factor stock selection
    config['stock_selection']['enable_multi_factor'] = True
    strategy.use_multi_factor = True
    
    # Select stocks using the multi-factor method
    selected_stocks = strategy.select_stocks_multi_factor(mock_data, test_date)
    
    logger.info(f"Selected {len(selected_stocks)} stocks on {test_date.strftime('%Y-%m-%d')}")
    
    # Print details for each selected stock
    for i, stock_info in enumerate(selected_stocks):
        # Handle different return formats
        if isinstance(stock_info, tuple) and len(stock_info) == 2:
            symbol, score = stock_info
        elif isinstance(stock_info, dict) and 'symbol' in stock_info:
            symbol = stock_info['symbol']
            score = stock_info.get('combined_score', 0.0)
        else:
            symbol = stock_info
            score = 0.0  # Default score if not available
            
        # Get seasonal score - this returns a float, not a tuple
        seasonal_score = strategy.seasonality_analyzer.get_seasonal_score(symbol, test_date)
        logger.info(f"{i+1}. {symbol}: Combined Score={score:.4f}, Seasonality Score={seasonal_score:.4f}")
    
    return selected_stocks

def test_seasonality_monthly_weights():
    """
    Test the impact of monthly weights on seasonality scores.
    """
    # Import the SeasonalityEnhanced class
    from seasonality_enhanced import SeasonalityEnhanced
    
    # Load the configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the seasonality analyzer
    seasonality_file = config['seasonality']['data_file']
    seasonality_analyzer = SeasonalityEnhanced(seasonality_file, config)
    
    # Test symbol
    test_symbol = 'NVDA'  # Use a symbol with known seasonality data
    
    # Test all months
    months = range(1, 13)
    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    # Calculate scores with and without monthly weights
    logger.info("\n=== Testing Monthly Weight Impact on Seasonality Scores ===")
    
    # Store results
    results = {
        'with_weights': [],
        'without_weights': []
    }
    
    for month in months:
        # Create test date for this month
        test_date = datetime(2023, month, 15)
        
        # Calculate score with weights (default)
        score_with_weights = seasonality_analyzer.get_seasonal_score(test_symbol, test_date)
        
        # Temporarily disable monthly weights
        original_config = seasonality_analyzer.config
        config_no_weights = original_config.copy() if original_config else {}
        if 'seasonality' in config_no_weights:
            config_no_weights['seasonality'] = config_no_weights['seasonality'].copy()
            if 'monthly_weights' in config_no_weights['seasonality']:
                del config_no_weights['seasonality']['monthly_weights']
        
        seasonality_analyzer.config = config_no_weights
        
        # Calculate score without weights
        score_without_weights = seasonality_analyzer.get_seasonal_score(test_symbol, test_date)
        
        # Restore original config
        seasonality_analyzer.config = original_config
        
        # Store results
        results['with_weights'].append(score_with_weights)
        results['without_weights'].append(score_without_weights)
        
        # Log results
        logger.info(f"{month_names[month-1]}: With Weights={score_with_weights:.4f}, Without Weights={score_without_weights:.4f}, Difference={score_with_weights-score_without_weights:.4f}")
    
    # Plot results
    plot_monthly_weight_impact(month_names, results)
    
    return results

def plot_monthly_weight_impact(months, results):
    """
    Plot the impact of monthly weights on seasonality scores.
    
    Args:
        months (list): List of month names
        results (dict): Dictionary of scores with and without weights
    """
    # Create output directory
    output_dir = 'output/seasonality_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot scores
    plt.plot(months, results['with_weights'], marker='o', label='With Monthly Weights', color='blue')
    plt.plot(months, results['without_weights'], marker='x', label='Without Monthly Weights', color='red')
    
    # Add labels and title
    plt.xlabel('Month')
    plt.ylabel('Seasonality Score')
    plt.title('Impact of Monthly Weights on Seasonality Scores')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_weight_impact.png")
    plt.close()

def main():
    """Main function to run the seasonality integration tests."""
    # Create output directory
    output_dir = 'output/seasonality_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests
    logger.info("=== Starting Seasonality Integration Tests ===")
    
    # Test 1: Seasonality score calculation
    test_seasonality_calculation()
    
    # Test 2: Combined strategy with seasonality
    test_combined_strategy()
    
    # Test 3: Monthly weights impact
    test_seasonality_monthly_weights()
    
    logger.info("=== Seasonality Integration Tests Completed ===")

if __name__ == "__main__":
    main()
