#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to analyze stock selection using the combined strategy with seasonality,
technical indicators, and volatility metrics.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import Counter, defaultdict

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

def analyze_stock_selection(selected_stocks, config, test_date):
    """
    Analyze the selected stocks and their scores.
    
    Args:
        selected_stocks (list): List of dictionaries with selected stocks and their scores
        config (dict): Configuration dictionary
        test_date (datetime): Date of the selection
    """
    logger.info(f"\n=== Stock Selection Analysis for {test_date.strftime('%Y-%m-%d')} ===")
    
    # Print number of selected stocks
    logger.info(f"Selected {len(selected_stocks)} stocks")
    
    # Create a dataframe for analysis
    selection_df = pd.DataFrame(selected_stocks)
    
    # Print top stocks with their scores
    logger.info("\nTop Selected Stocks:")
    for i, stock in enumerate(selected_stocks[:10]):
        logger.info(f"{i+1}. {stock['symbol']}: Combined Score={stock['combined_score']:.4f}, "
                   f"Technical={stock['technical_score']:.4f}, Seasonal={stock['seasonal_score']:.4f}, "
                   f"Direction={stock['technical_direction']}")
    
    # Analyze score distributions
    logger.info("\nScore Distributions:")
    for col in ['combined_score', 'technical_score', 'seasonal_score', 'momentum_score', 
                'trend_score', 'volatility_score', 'volume_score']:
        if col in selection_df.columns:
            logger.info(f"{col}: Mean={selection_df[col].mean():.4f}, "
                       f"Median={selection_df[col].median():.4f}, "
                       f"Min={selection_df[col].min():.4f}, "
                       f"Max={selection_df[col].max():.4f}")
    
    # Analyze direction distribution
    if 'technical_direction' in selection_df.columns:
        direction_counts = selection_df['technical_direction'].value_counts()
        logger.info("\nDirection Distribution:")
        for direction, count in direction_counts.items():
            logger.info(f"{direction}: {count} stocks ({count/len(selection_df)*100:.2f}%)")
    
    # Plot score distributions
    plot_score_distributions(selection_df, test_date)
    
    # Plot correlation between scores
    plot_score_correlations(selection_df, test_date)
    
    return selection_df

def plot_score_distributions(df, test_date):
    """
    Plot distributions of various scores.
    
    Args:
        df (pd.DataFrame): DataFrame with score data
        test_date (datetime): Date of the selection
    """
    # Create output directory
    output_dir = 'output/selection_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Score columns to plot
    score_cols = [col for col in ['combined_score', 'technical_score', 'seasonal_score', 
                                  'momentum_score', 'trend_score', 'volatility_score', 'volume_score'] 
                 if col in df.columns]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot histograms for each score
    for i, col in enumerate(score_cols):
        plt.subplot(2, 4, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col.replace('_', ' ').title()} Distribution")
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel('Count')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distributions_{test_date.strftime('%Y%m%d')}.png")
    plt.close()

def plot_score_correlations(df, test_date):
    """
    Plot correlation matrix between different scores.
    
    Args:
        df (pd.DataFrame): DataFrame with score data
        test_date (datetime): Date of the selection
    """
    # Create output directory
    output_dir = 'output/selection_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Score columns to include in correlation
    score_cols = [col for col in ['combined_score', 'technical_score', 'seasonal_score', 
                                 'momentum_score', 'trend_score', 'volatility_score', 'volume_score'] 
                 if col in df.columns]
    
    # Create correlation matrix
    corr = df[score_cols].corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f"Score Correlation Matrix - {test_date.strftime('%Y-%m-%d')}")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_correlations_{test_date.strftime('%Y%m%d')}.png")
    plt.close()

def test_combined_selection():
    """
    Test the combined stock selection strategy with real market data.
    """
    # Import the CombinedStrategy class
    from combined_strategy import CombinedStrategy
    
    # Load the configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the strategy
    strategy = CombinedStrategy(config)
    
    # Test dates - use a specific date for consistency
    test_date = datetime(2023, 3, 15)  # March 15, 2023
    
    # Get symbols from config
    symbols = config['general']['symbols']
    
    # Load historical data
    start_date = (test_date - timedelta(days=100)).strftime('%Y-%m-%d')  # Need enough history for indicators
    end_date = test_date.strftime('%Y-%m-%d')
    
    market_data = generate_mock_data(symbols, start_date, end_date)
    
    if not market_data:
        logger.error("Failed to load market data. Exiting.")
        return
    
    # Enable multi-factor stock selection
    config['stock_selection']['enable_multi_factor'] = True
    strategy.use_multi_factor = True
    
    # Select stocks using the multi-factor method
    logger.info(f"Selecting stocks for {test_date.strftime('%Y-%m-%d')} using multi-factor approach")
    selected_stocks = strategy.select_stocks_multi_factor(market_data, test_date)
    
    # Analyze the selection
    selection_df = analyze_stock_selection(selected_stocks, config, test_date)
    
    # Analyze why specific stocks were selected
    analyze_selection_factors(selected_stocks[:5], market_data, test_date)
    
    return selected_stocks

def analyze_selection_factors(top_stocks, market_data, test_date):
    """
    Analyze why specific stocks were selected by examining their factors in detail.
    
    Args:
        top_stocks (list): List of dictionaries with top selected stocks
        market_data (dict): Dictionary of dataframes with market data
        test_date (datetime): Date of the selection
    """
    logger.info("\n=== Detailed Analysis of Top Stock Selections ===")
    
    for i, stock in enumerate(top_stocks):
        symbol = stock['symbol']
        logger.info(f"\n{i+1}. {symbol} Analysis:")
        
        # Get the stock's data
        if symbol in market_data:
            df = market_data[symbol]
            
            # Get the last 20 days of data for analysis
            recent_data = df.tail(20).copy()
            
            # Calculate some key metrics
            if len(recent_data) > 0:
                last_price = recent_data['close'].iloc[-1]
                price_change_1d = (last_price / recent_data['close'].iloc[-2] - 1) * 100 if len(recent_data) > 1 else 0
                price_change_5d = (last_price / recent_data['close'].iloc[-6] - 1) * 100 if len(recent_data) > 5 else 0
                price_change_20d = (last_price / recent_data['close'].iloc[0] - 1) * 100 if len(recent_data) > 0 else 0
                
                # Calculate volatility (standard deviation of returns)
                returns = recent_data['close'].pct_change().dropna()
                volatility = returns.std() * 100
                
                # Calculate volume ratio (current volume / average volume)
                avg_volume = recent_data['volume'].mean()
                current_volume = recent_data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                logger.info(f"Price: ${last_price:.2f}")
                logger.info(f"1-Day Change: {price_change_1d:.2f}%")
                logger.info(f"5-Day Change: {price_change_5d:.2f}%")
                logger.info(f"20-Day Change: {price_change_20d:.2f}%")
                logger.info(f"20-Day Volatility: {volatility:.2f}%")
                logger.info(f"Volume Ratio: {volume_ratio:.2f}x average")
                
                # Log score components
                logger.info(f"Combined Score: {stock['combined_score']:.4f}")
                logger.info(f"Technical Score: {stock['technical_score']:.4f}")
                logger.info(f"Seasonal Score: {stock['seasonal_score']:.4f}")
                logger.info(f"Momentum Score: {stock.get('momentum_score', 'N/A')}")
                logger.info(f"Trend Score: {stock.get('trend_score', 'N/A')}")
                logger.info(f"Volatility Score: {stock.get('volatility_score', 'N/A')}")
                logger.info(f"Volume Score: {stock.get('volume_score', 'N/A')}")
                logger.info(f"Direction: {stock['technical_direction']}")
                
                # Provide interpretation
                logger.info("\nInterpretation:")
                
                # Momentum interpretation
                if 'momentum_score' in stock and stock['momentum_score'] > 0.7:
                    logger.info("- Strong momentum: The stock shows significant positive price movement")
                elif 'momentum_score' in stock and stock['momentum_score'] < 0.3:
                    logger.info("- Weak momentum: The stock may be experiencing a pullback or consolidation")
                
                # Trend interpretation
                if 'trend_score' in stock and stock['trend_score'] > 0.7:
                    logger.info("- Strong trend: The stock is in a well-defined trend")
                elif 'trend_score' in stock and stock['trend_score'] < 0.3:
                    logger.info("- Weak trend: The stock may be in a choppy or sideways market")
                
                # Volatility interpretation
                if 'volatility_score' in stock and stock['volatility_score'] > 0.7:
                    logger.info("- Favorable volatility: The stock has appropriate volatility for the strategy")
                elif 'volatility_score' in stock and stock['volatility_score'] < 0.3:
                    logger.info("- Unfavorable volatility: The stock may be too volatile or not volatile enough")
                
                # Volume interpretation
                if 'volume_score' in stock and stock['volume_score'] > 0.7:
                    logger.info("- Strong volume: The stock has good liquidity and volume confirmation")
                elif 'volume_score' in stock and stock['volume_score'] < 0.3:
                    logger.info("- Weak volume: The stock may lack liquidity or volume confirmation")
                
                # Seasonal interpretation
                if stock['seasonal_score'] > 0.7:
                    logger.info("- Strong seasonal pattern: The stock historically performs well in this time period")
                elif stock['seasonal_score'] < 0.3:
                    logger.info("- Weak seasonal pattern: The stock historically underperforms in this time period")
                
                # Direction interpretation
                if stock['technical_direction'] == 'LONG':
                    logger.info("- Long signal: Technical indicators suggest a bullish outlook")
                elif stock['technical_direction'] == 'SHORT':
                    logger.info("- Short signal: Technical indicators suggest a bearish outlook")
                else:
                    logger.info("- Neutral signal: Technical indicators are mixed or inconclusive")
            else:
                logger.warning(f"Insufficient data for {symbol}")
        else:
            logger.warning(f"No data available for {symbol}")

def main():
    """
    Main function to run the combined selection analysis.
    """
    logger.info("=== Starting Combined Selection Analysis ===")
    
    # Test combined selection
    selected_stocks = test_combined_selection()
    
    logger.info("=== Combined Selection Analysis Completed ===")
    
    return selected_stocks

if __name__ == "__main__":
    main()
