#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for selecting the top 25 stocks from the S&P 500 index based on multi-factor scoring.
This script loads the enhanced configuration, runs a backtest on S&P 500 stocks, and selects the top 25.
"""

# Set matplotlib to use non-interactive backend before importing it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import alpaca_trade_api as tradeapi
import json
import time
import argparse
import requests
from enum import Enum

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategy modules
from combined_strategy import CombinedStrategy, MarketRegime
from data_loader import AlpacaDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_alpaca_credentials():
    """Load Alpaca API credentials from file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {e}")
        return None

def get_sp500_symbols():
    """
    Get the full list of S&P 500 symbols by scraping Wikipedia.
    
    Returns:
        list: List of S&P 500 stock symbols
    """
    try:
        # Try to fetch the current S&P 500 list from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Use pandas to read the table from the webpage
            tables = pd.read_html(response.text)
            sp500_table = tables[0]  # The first table contains the S&P 500 companies
            
            # Extract the ticker symbols and convert to list
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean up symbols (remove .B, replace dots with hyphens for BRK.B, etc.)
            cleaned_symbols = []
            for symbol in symbols:
                # Replace dots with hyphens (e.g., BRK.B -> BRK-B) for Alpaca compatibility
                symbol = symbol.replace('.', '-')
                cleaned_symbols.append(symbol)
            
            logger.info(f"Successfully fetched {len(cleaned_symbols)} S&P 500 symbols from Wikipedia")
            return cleaned_symbols
        else:
            logger.warning(f"Failed to fetch S&P 500 symbols from Wikipedia, status code: {response.status_code}")
            # Fall back to a static list if web scraping fails
            return get_static_sp500_symbols()
    
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {e}")
        # Fall back to a static list if web scraping fails
        return get_static_sp500_symbols()

def get_static_sp500_symbols():
    """
    Get a static list of S&P 500 symbols (top 100 by market cap).
    This is a fallback if the web scraping fails.
    
    Returns:
        list: List of S&P 500 stock symbols
    """
    logger.info("Using static list of top 100 S&P 500 symbols")
    return [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'BRK.B', 'TSLA', 'UNH',
        'LLY', 'JPM', 'V', 'XOM', 'AVGO', 'PG', 'MA', 'HD', 'COST', 'MRK',
        'CVX', 'ABBV', 'PEP', 'KO', 'ADBE', 'WMT', 'CRM', 'TMO', 'MCD', 'ACN',
        'BAC', 'LIN', 'CSCO', 'ABT', 'NFLX', 'AMD', 'DHR', 'CMCSA', 'PFE', 'NKE',
        'TXN', 'WFC', 'PM', 'INTC', 'ORCL', 'VZ', 'COP', 'IBM', 'AMGN', 'QCOM',
        'UPS', 'HON', 'LOW', 'CAT', 'BA', 'SPGI', 'GE', 'DE', 'INTU', 'AXP',
        'AMAT', 'BKNG', 'GS', 'MS', 'BLK', 'SBUX', 'GILD', 'RTX', 'ADI', 'MDLZ',
        'ISRG', 'TJX', 'MMC', 'SYK', 'PGR', 'ELV', 'DIS', 'EOG', 'VRTX', 'REGN',
        'SO', 'BMY', 'AMT', 'LRCX', 'C', 'ETN', 'TMUS', 'MO', 'SCHW', 'CB',
        'CME', 'PLD', 'ZTS', 'ADP', 'NOW', 'BDX', 'TGT', 'SLB', 'ITW', 'SNPS',
        'SPY'  # Adding SPY for market regime detection
    ]

def load_data_parallel(data_loader, symbols, start_date, end_date, timeframe, max_workers=10):
    """
    Load historical data for multiple symbols in parallel
    
    Args:
        data_loader: Data loader instance
        symbols: List of symbols to load data for
        start_date: Start date for data
        end_date: End date for data
        timeframe: Timeframe for data
        max_workers: Maximum number of parallel workers
        
    Returns:
        dict: Dictionary of symbol -> dataframe
    """
    logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Create a dictionary to store the data
    symbol_data = {}
    
    # Define a function to load data for a single symbol
    def load_symbol_data(symbol):
        try:
            # Format dates as strings in YYYY-MM-DD format without time
            start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, pd.Timestamp) else start_date
            end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, pd.Timestamp) else end_date
            
            # Get data for the symbol
            df = data_loader.load_historical_data(symbol, start_str, end_str, timeframe)
            
            # Check if we have data
            if df is not None and not df.empty:
                # Set the timestamp column as the index if it's not already
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                
                logger.debug(f"Loaded {len(df)} rows for {symbol}")
                return symbol, df
            else:
                logger.warning(f"No data for {symbol}")
                return symbol, None
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return symbol, None
    
    # Use ThreadPoolExecutor to load data in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {executor.submit(load_symbol_data, symbol): symbol for symbol in symbols}
        
        # Process results as they complete
        for future in as_completed(futures):
            symbol, df = future.result()
            if df is not None:
                symbol_data[symbol] = df
    
    logger.info(f"Loaded data for {len(symbol_data)} symbols")
    return symbol_data

def calculate_returns(selections_df, symbol_data, holding_period=5):
    """
    Calculate returns for selected stocks
    
    Args:
        selections_df: DataFrame with stock selections
        symbol_data: Dictionary of symbol -> dataframe with price data
        holding_period: Number of days to hold each position
        
    Returns:
        DataFrame: Updated selections DataFrame with return metrics
    """
    logger.info(f"Calculating returns for {len(selections_df)} selections with holding period of {holding_period} days")
    
    # Create a copy of the selections dataframe
    results_df = selections_df.copy()
    
    # Add columns for returns
    results_df['return'] = np.nan
    results_df['max_return'] = np.nan
    results_df['max_drawdown'] = np.nan
    results_df['win'] = np.nan
    
    # Iterate through each selection
    for idx, row in results_df.iterrows():
        symbol = row['symbol']
        date = row['date']
        direction = row['direction']
        
        # Skip if we don't have data for this symbol
        if symbol not in symbol_data:
            logger.warning(f"No data for {symbol}, skipping return calculation")
            continue
        
        # Get the data for this symbol
        df = symbol_data[symbol]
        
        # Skip if the dataframe is empty
        if df is None or df.empty:
            logger.warning(f"Empty dataframe for {symbol}, skipping return calculation")
            continue
        
        try:
            # Find the index of the selection date
            if isinstance(df.index, pd.DatetimeIndex):
                # If the index is a DatetimeIndex, convert date to datetime if needed
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                
                # Normalize timezone information to prevent comparison issues
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    # If df index has timezone, convert date to match
                    if hasattr(date, 'tz') and date.tz is None:
                        date = date.tz_localize(df.index.tz)
                else:
                    # If df index has no timezone, ensure date has no timezone
                    if hasattr(date, 'tz') and date.tz is not None:
                        date = date.tz_localize(None)
                
                # Find the closest date in the index
                date_idx = df.index.get_indexer([date], method='nearest')[0]
            else:
                # If the index is not a DatetimeIndex, try to find the date directly
                date_mask = df.index == date
                if date_mask.any():
                    date_idx = date_mask.argmax()
                else:
                    logger.warning(f"Date {date} not found in data for {symbol}, skipping return calculation")
                    continue
            
            # Calculate the end index for the holding period
            end_idx = min(date_idx + holding_period, len(df) - 1)
            
            # Skip if we don't have enough data for the holding period
            if end_idx <= date_idx:
                logger.warning(f"Not enough data for holding period for {symbol} on {date}, skipping return calculation")
                continue
            
            # Get the prices
            start_price = df.iloc[date_idx]['close']
            end_price = df.iloc[end_idx]['close']
            
            # Calculate the return based on direction
            if direction == 'LONG':
                ret = (end_price - start_price) / start_price
            else:  # SHORT
                ret = (start_price - end_price) / start_price
            
            # Calculate the maximum return and drawdown during the holding period
            prices = df.iloc[date_idx:end_idx+1]['close'].values
            
            if direction == 'LONG':
                # For long positions
                returns = (prices - start_price) / start_price
                max_return = returns.max()
                max_drawdown = returns.min()
            else:
                # For short positions
                returns = (start_price - prices) / start_price
                max_return = returns.max()
                max_drawdown = returns.min()
            
            # Update the results dataframe
            results_df.loc[idx, 'return'] = ret
            results_df.loc[idx, 'max_return'] = max_return
            results_df.loc[idx, 'max_drawdown'] = max_drawdown
            results_df.loc[idx, 'win'] = 1 if ret > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating return for {symbol} on {date}: {e}")
            continue
    
    logger.info(f"Calculated returns for {results_df['return'].notna().sum()} selections")
    return results_df

def analyze_performance(results_df):
    """
    Analyze performance of stock selections
    
    Args:
        results_df: DataFrame with stock selections and returns
        
    Returns:
        dict: Performance metrics
    """
    logger.info("Analyzing performance")
    
    # Filter out rows with NaN returns
    valid_results = results_df[results_df['return'].notna()].copy()
    
    if len(valid_results) == 0:
        logger.warning("No valid results to analyze")
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0
        }
    
    # Calculate performance metrics
    total_trades = len(valid_results)
    wins = valid_results['win'].sum()
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    avg_return = valid_results['return'].mean()
    total_return = valid_results['return'].sum()
    
    # Calculate profit factor
    winning_trades = valid_results[valid_results['win'] == 1]
    losing_trades = valid_results[valid_results['win'] == 0]
    
    gross_profit = winning_trades['return'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['return'].sum()) if len(losing_trades) > 0 else 0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate Sharpe ratio
    returns_std = valid_results['return'].std()
    sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
    
    # Calculate metrics by direction
    direction_metrics = {}
    for direction in valid_results['direction'].unique():
        direction_df = valid_results[valid_results['direction'] == direction]
        direction_total = len(direction_df)
        direction_wins = direction_df['win'].sum()
        direction_win_rate = direction_wins / direction_total if direction_total > 0 else 0
        direction_avg_return = direction_df['return'].mean()
        
        direction_metrics[direction] = {
            'total_trades': direction_total,
            'win_rate': direction_win_rate,
            'avg_return': direction_avg_return
        }
    
    # Calculate metrics by market regime
    regime_metrics = {}
    for regime in valid_results['market_regime'].unique():
        regime_df = valid_results[valid_results['market_regime'] == regime]
        regime_total = len(regime_df)
        regime_wins = regime_df['win'].sum()
        regime_win_rate = regime_wins / regime_total if regime_total > 0 else 0
        regime_avg_return = regime_df['return'].mean()
        
        regime_metrics[regime] = {
            'total_trades': regime_total,
            'win_rate': regime_win_rate,
            'avg_return': regime_avg_return
        }
    
    # Calculate metrics by score range
    valid_results['score_range'] = pd.cut(valid_results['score'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                         labels=['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    
    score_metrics = {}
    for score_range in valid_results['score_range'].unique():
        score_df = valid_results[valid_results['score_range'] == score_range]
        score_total = len(score_df)
        score_wins = score_df['win'].sum()
        score_win_rate = score_wins / score_total if score_total > 0 else 0
        score_avg_return = score_df['return'].mean()
        
        score_metrics[score_range] = {
            'total_trades': score_total,
            'win_rate': score_win_rate,
            'avg_return': score_avg_return
        }
    
    # Return all metrics
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'direction_metrics': direction_metrics,
        'regime_metrics': regime_metrics,
        'score_metrics': score_metrics
    }

def run_backtest(config_file, start_date, end_date, holding_period=5, top_n=25):
    """
    Run a backtest using the specified configuration.
    
    Args:
        config_file (str): Path to the configuration file
        start_date (str): Start date for the backtest (YYYY-MM-DD)
        end_date (str): End date for the backtest (YYYY-MM-DD)
        holding_period (int): Number of days to hold each position
        top_n (int): Number of top stocks to select
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"Running backtest from {start_date} to {end_date} with holding period {holding_period}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get S&P 500 symbols
    sp500_symbols = get_static_sp500_symbols()
    
    # Update the configuration with S&P 500 symbols
    config['general']['symbols'] = sp500_symbols
    
    # Update top_n_stocks in the configuration
    config['stock_selection']['top_n_stocks'] = top_n
    
    # Load Alpaca credentials
    credentials = load_alpaca_credentials()
    if not credentials:
        logger.error("Failed to load Alpaca credentials")
        return None
    
    # Create Alpaca API client
    api = tradeapi.REST(
        key_id=credentials['paper']['api_key'],
        secret_key=credentials['paper']['api_secret'],
        base_url=credentials['paper']['base_url']
    )
    
    # Create data loader
    data_loader = AlpacaDataLoader(api)
    
    # Convert dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Load historical data
    symbol_data = load_data_parallel(
        data_loader,
        sp500_symbols,
        start_date - timedelta(days=100),  # Load extra data for indicators
        end_date,
        config['general']['timeframe']
    )
    
    # Create strategy instance
    strategy = CombinedStrategy(config)
    
    # Initialize results dataframe
    selections_df = pd.DataFrame(columns=[
        'date', 'symbol', 'score', 'direction', 'position_size', 'market_regime',
        'technical_score', 'seasonal_score', 'technical_weight', 'seasonality_weight',
        'seasonality_confidence'
    ])
    
    # Iterate through each trading day
    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
        
        # Detect market regime - need to get SPY data first
        if 'SPY' in symbol_data:
            spy_data = symbol_data['SPY']
            try:
                market_regime = strategy.detect_market_regime(spy_data)
                logger.info(f"Market regime: {market_regime}")
            except Exception as e:
                logger.error(f"Error detecting market regime: {e}")
                market_regime = MarketRegime.MIXED
        else:
            logger.warning("SPY data not available for market regime detection, using MIXED")
            market_regime = MarketRegime.MIXED
        
        # Select stocks
        try:
            logger.info(f"Calling select_stocks_multi_factor with {len(symbol_data)} symbols, top_n={top_n}, direction='ANY', market_regime={market_regime}")
            
            selections = strategy.select_stocks_multi_factor(
                symbol_data,
                current_date=current_date,
                top_n=top_n,
                direction='ANY',
                market_regime=market_regime
            )
            
            if selections and len(selections) > 0:
                logger.info(f"Selected {len(selections)} stocks for {current_date.strftime('%Y-%m-%d')}")
                
                # Add selections to dataframe
                for selection in selections:
                    selection_data = {
                        'date': current_date,
                        'symbol': selection['symbol'],
                        'score': selection.get('combined_score', 0),  # Use combined_score instead of score
                        'direction': selection['direction'],
                        'position_size': selection['position_size'],
                        'market_regime': market_regime.value if hasattr(market_regime, 'value') else market_regime,
                        'technical_score': selection.get('technical_score', 0),
                        'seasonal_score': selection.get('seasonal_score', 0),
                        'technical_weight': selection.get('technical_weight', 0),
                        'seasonality_weight': selection.get('seasonality_weight', 0),
                        'seasonality_confidence': selection.get('seasonality_confidence', 0)
                    }
                    
                    selections_df = pd.concat([selections_df, pd.DataFrame([selection_data])], ignore_index=True)
            else:
                logger.warning(f"No stocks selected for {current_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Error selecting stocks for {current_date.strftime('%Y-%m-%d')}: {e}")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Calculate returns
    results_df = calculate_returns(selections_df, symbol_data, holding_period)
    
    # Analyze performance
    performance = analyze_performance(results_df)
    
    # Print performance metrics
    logger.info("\n=== Performance Metrics ===")
    logger.info(f"Total Trades: {performance['total_trades']}")
    logger.info(f"Win Rate: {performance['win_rate']:.2%}")
    logger.info(f"Average Return: {performance['avg_return']:.2%}")
    logger.info(f"Total Return: {performance['total_return']:.2%}")
    logger.info(f"Profit Factor: {performance['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    
    logger.info("\n=== Direction Performance ===")
    for direction, metrics in performance.get('direction_metrics', {}).items():
        logger.info(f"{direction}: {metrics['total_trades']} trades, Win Rate: {metrics['win_rate']:.2%}, Avg Return: {metrics['avg_return']:.2%}")
    
    logger.info("\n=== Market Regime Performance ===")
    for regime, metrics in performance.get('regime_metrics', {}).items():
        logger.info(f"{regime}: {metrics['total_trades']} trades, Win Rate: {metrics['win_rate']:.2%}, Avg Return: {metrics['avg_return']:.2%}")
    
    logger.info("\n=== Score Range Performance ===")
    for score_range, metrics in performance.get('score_metrics', {}).items():
        logger.info(f"{score_range}: {metrics['total_trades']} trades, Win Rate: {metrics['win_rate']:.2%}, Avg Return: {metrics['avg_return']:.2%}")
    
    # Save results to CSV
    output_file = f"backtest_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Return results
    return {
        'performance': performance,
        'results_df': results_df
    }

def main():
    """
    Main function to run the backtest
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run S&P 500 stock selection backtest')
    parser.add_argument('--config', type=str, default='configuration_enhanced_multi_factor_500.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, default='2023-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-03-31',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--holding_period', type=int, default=5,
                        help='Holding period in days')
    parser.add_argument('--top_n', type=int, default=25,
                        help='Number of top stocks to select')
    args = parser.parse_args()
    
    # Get configuration file path
    config_file = args.config
    
    # Check if configuration file exists
    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} not found")
        return
    
    # Run backtest
    logger.info(f"\n=== Running backtest for period: {args.start_date} to {args.end_date} ===")
    run_backtest(config_file, args.start_date, args.end_date, args.holding_period, args.top_n)

if __name__ == "__main__":
    main()
