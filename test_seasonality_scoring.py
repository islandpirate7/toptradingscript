#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for evaluating the enhanced seasonality scoring mechanism.
This script tests the seasonality scoring for a set of stocks and displays the distribution of scores.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy modules
from combined_strategy import CombinedStrategy
from seasonality_enhanced import SeasonalityEnhanced

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return None

def test_seasonality_scoring():
    """Test seasonality scoring for a set of stocks"""
    # Load configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    config = load_config(config_file)
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Set logging level to DEBUG for more detailed information
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize strategy
    logger.info("Initializing strategy with configuration...")
    strategy = CombinedStrategy(config)
    
    # Check if seasonality analyzer is initialized
    if hasattr(strategy, 'seasonality_analyzer'):
        logger.info(f"Seasonality analyzer initialized: {strategy.seasonality_analyzer}")
        logger.info(f"Seasonality data file: {strategy.seasonality_analyzer.seasonality_file}")
        
        # Check if seasonality data is loaded
        if hasattr(strategy.seasonality_analyzer, 'seasonality_data'):
            logger.info(f"Seasonality data loaded: {len(strategy.seasonality_analyzer.seasonality_data)} symbols")
            
            # Print a sample of the data
            sample_symbols = list(strategy.seasonality_analyzer.seasonality_data.keys())[:5]
            for symbol in sample_symbols:
                logger.info(f"Sample data for {symbol}: {strategy.seasonality_analyzer.seasonality_data[symbol]}")
        else:
            logger.error("Seasonality data not loaded in the analyzer")
    else:
        logger.error("Seasonality analyzer not initialized in the strategy")
    
    # Get list of symbols from config
    symbols = config['general']['symbols']
    logger.info(f"Testing with {len(symbols)} symbols")
    
    # Test dates
    test_dates = [
        datetime(2023, 1, 15),  # January
        datetime(2023, 3, 15),  # March
        datetime(2023, 6, 15),  # June
        datetime(2023, 9, 15),  # September
        datetime(2023, 12, 15)  # December
    ]
    
    # Store results
    results = []
    
    # Test seasonality scoring for each symbol and date
    for date in test_dates:
        date_scores = []
        for symbol in symbols:
            score, direction = strategy.get_seasonal_score(symbol, date)
            date_scores.append({
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'month': date.strftime('%B'),
                'score': score,
                'direction': direction
            })
        
        # Add to results
        results.extend(date_scores)
        
        # Calculate statistics
        scores = [item['score'] for item in date_scores]
        logger.info(f"Month: {date.strftime('%B')}")
        logger.info(f"  Mean score: {np.mean(scores):.4f}")
        logger.info(f"  Median score: {np.median(scores):.4f}")
        logger.info(f"  Min score: {min(scores):.4f}")
        logger.info(f"  Max score: {max(scores):.4f}")
        logger.info(f"  Std dev: {np.std(scores):.4f}")
        
        # Count directions
        long_count = sum(1 for item in date_scores if item['direction'] == 'LONG')
        short_count = sum(1 for item in date_scores if item['direction'] == 'SHORT')
        logger.info(f"  LONG signals: {long_count}, SHORT signals: {short_count}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    output_file = 'output/seasonality_scores.csv'
    os.makedirs('output', exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved seasonality scores to {output_file}")
    
    # Visualize score distribution
    visualize_score_distribution(df)
    
    return df

def visualize_score_distribution(df):
    """Visualize the distribution of seasonality scores"""
    # Create output directory if it doesn't exist
    os.makedirs('output/seasonality_analysis', exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot overall distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'], bins=20, kde=True)
    plt.title('Distribution of Seasonality Scores')
    plt.xlabel('Score (-1.0 to 1.0)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('output/seasonality_analysis/score_distribution.png')
    
    # Plot distribution by month
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='month', y='score', data=df, order=['January', 'March', 'June', 'September', 'December'])
    plt.title('Seasonality Score Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Score (-1.0 to 1.0)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig('output/seasonality_analysis/score_by_month.png')
    
    # Plot top 10 and bottom 10 stocks for each month
    for month in df['month'].unique():
        month_df = df[df['month'] == month]
        
        # Get average score by symbol
        symbol_scores = month_df.groupby('symbol')['score'].mean().reset_index()
        
        # Sort and get top/bottom 10
        top_symbols = symbol_scores.sort_values('score', ascending=False).head(10)
        bottom_symbols = symbol_scores.sort_values('score').head(10)
        
        # Plot top 10
        plt.figure(figsize=(12, 6))
        sns.barplot(x='symbol', y='score', data=top_symbols)
        plt.title(f'Top 10 Stocks by Seasonality Score - {month}')
        plt.xlabel('Symbol')
        plt.ylabel('Score (-1.0 to 1.0)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'output/seasonality_analysis/top_10_{month}.png')
        
        # Plot bottom 10
        plt.figure(figsize=(12, 6))
        sns.barplot(x='symbol', y='score', data=bottom_symbols)
        plt.title(f'Bottom 10 Stocks by Seasonality Score - {month}')
        plt.xlabel('Symbol')
        plt.ylabel('Score (-1.0 to 1.0)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'output/seasonality_analysis/bottom_10_{month}.png')
    
    logger.info("Saved visualization plots to output/seasonality_analysis/")

if __name__ == "__main__":
    logger.info("Testing seasonality scoring...")
    results_df = test_seasonality_scoring()
    logger.info("Seasonality scoring test completed.")
