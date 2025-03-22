#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to compare performance with and without seasonality.
This script runs two backtests - one with seasonality enabled and one without,
then compares the results.
"""

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
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import strategy modules
from combined_strategy import CombinedStrategy
from data_loader import AlpacaDataLoader

def load_alpaca_credentials():
    """Load Alpaca API credentials from file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {e}")
        return None

def load_symbol_data(symbol):
    """Load historical data for a single symbol"""
    try:
        # Format dates as strings in YYYY-MM-DD format for Alpaca API
        start_str = '2022-01-01'  # Start a year earlier to have enough data for indicators
        end_str = '2024-01-31'    # End a month after our test period to ensure we have all data
        timeframe = '1D'
        
        df = data_loader.load_historical_data(
            symbol, 
            start_str,
            end_str,
            timeframe
        )
        
        # Debug the DataFrame structure
        if df is not None and not df.empty:
            logger.info(f"Loaded data for {symbol} with shape {df.shape}")
            
            # Set the timestamp column as the index if it exists
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
                logger.info(f"Set timestamp as index for {symbol}")
                
        return symbol, df
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return symbol, None

def load_data_parallel(data_loader, symbols, start_date, end_date, timeframe, max_workers=10):
    """Load historical data for multiple symbols in parallel"""
    symbol_data = {}
    
    def load_symbol_data(symbol):
        try:
            # Format dates as strings in YYYY-MM-DD format for Alpaca API
            start_str = '2022-01-01'  # Start a year earlier to have enough data for indicators
            end_str = '2024-01-31'    # End a month after our test period to ensure we have all data
            timeframe = '1D'
            
            df = data_loader.load_historical_data(
                symbol, 
                start_str,
                end_str,
                timeframe
            )
            
            # Debug the DataFrame structure
            if df is not None and not df.empty:
                logger.info(f"Loaded data for {symbol} with shape {df.shape}")
                
                # Set the timestamp column as the index if it exists
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                    logger.info(f"Set timestamp as index for {symbol}")
                    
            return symbol, df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return symbol, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_symbol_data, symbol) for symbol in symbols]
        for future in as_completed(futures):
            try:
                symbol, df = future.result()
                symbol_data[symbol] = df
            except Exception as e:
                logger.error(f"Error processing symbol data: {e}")
    
    return symbol_data

def calculate_returns(selections_df, symbol_data, holding_period=5):
    """Calculate returns for selected stocks based on a holding period"""
    # Make a copy to avoid modifying the original
    results_df = selections_df.copy()
    
    # Add columns for entry/exit dates and prices
    results_df['entry_date'] = pd.to_datetime(results_df.index)
    results_df['exit_date'] = None
    results_df['entry_price'] = None
    results_df['exit_price'] = None
    results_df['return'] = None
    results_df['adjusted_return'] = None
    
    # Calculate returns for each position
    for idx, row in results_df.iterrows():
        symbol = row['symbol']
        position_size = row['position_size']
        direction = row['direction']
        
        if symbol not in symbol_data or symbol_data[symbol] is None:
            continue
            
        # Get symbol data
        df = symbol_data[symbol]
        
        # Find entry date (current date)
        entry_date = pd.to_datetime(idx)
        entry_idx = df.index.get_indexer([entry_date], method='nearest')[0]
        
        if entry_idx < 0 or entry_idx >= len(df):
            continue
            
        # Get entry price
        entry_price = df.iloc[entry_idx]['close']
        results_df.at[idx, 'entry_price'] = entry_price
        
        # Calculate exit date (entry date + holding period)
        exit_date = entry_date + timedelta(days=holding_period)
        exit_idx = df.index.get_indexer([exit_date], method='nearest')[0]
        
        if exit_idx < 0 or exit_idx >= len(df) or exit_idx <= entry_idx:
            continue
            
        # Get exit price
        exit_price = df.iloc[exit_idx]['close']
        results_df.at[idx, 'exit_price'] = exit_price
        results_df.at[idx, 'exit_date'] = df.index[exit_idx]
        
        # Calculate return
        if direction == 'LONG':
            ret = (exit_price - entry_price) / entry_price
        elif direction == 'SHORT':
            ret = (entry_price - exit_price) / entry_price
        else:  # NEUTRAL - assume long
            ret = (exit_price - entry_price) / entry_price
            
        results_df.at[idx, 'return'] = ret
        
        # Calculate position-adjusted return
        adjusted_ret = ret * position_size
        results_df.at[idx, 'adjusted_return'] = adjusted_ret
    
    # Drop rows with no return data
    results_df = results_df.dropna(subset=['return'])
    
    return results_df

def analyze_performance(results_df, config_file, start_date, end_date, prefix=''):
    """Analyze performance of stock selections"""
    if results_df is None or results_df.empty:
        logger.warning("No results to analyze")
        return None
        
    # Calculate performance metrics
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['return'] > 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    avg_return = results_df['return'].mean() * 100
    avg_win = results_df[results_df['return'] > 0]['return'].mean() * 100 if winning_trades > 0 else 0
    avg_loss = results_df[results_df['return'] < 0]['return'].mean() * 100 if len(results_df[results_df['return'] < 0]) > 0 else 0
    
    # Calculate profit factor
    gross_profit = results_df[results_df['return'] > 0]['return'].sum()
    gross_loss = abs(results_df[results_df['return'] < 0]['return'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate Sharpe ratio (simplified)
    returns = results_df['adjusted_return']
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    
    # Calculate max drawdown
    cumulative_returns = (1 + results_df['adjusted_return']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = abs(drawdown.min())
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Total Trades', 'Win Rate (%)', 'Avg Return (%)', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown (%)'],
        'Value': [total_trades, round(win_rate, 1), round(avg_return, 2), round(profit_factor, 2), round(sharpe_ratio, 1), round(max_drawdown, 2)]
    })
    
    # Save results to CSV
    period_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    config_name = os.path.basename(config_file).replace('.yaml', '')
    
    results_df.to_csv(f"{prefix}{config_name}_{period_str}_detailed_results.csv")
    summary.to_csv(f"{prefix}{config_name}_{period_str}_summary.csv", index=False)
    
    # Print summary
    logger.info(f"=== {prefix} Summary ===")
    for _, row in summary.iterrows():
        logger.info(f"{row['Metric']}: {row['Value']}")
    logger.info("")
    
    return summary

def plot_performance(results_df, output_prefix=''):
    """Create performance visualizations"""
    if results_df is None or results_df.empty:
        logger.error("Error plotting performance: No data available")
        return
    
    try:
        # Set up the plotting style
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cumulative Returns
        plt.subplot(2, 2, 1)
        cumulative_returns = (1 + results_df['adjusted_return']).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns, linewidth=2)
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        
        # Plot 2: Return Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(results_df['return'] * 100, kde=True)
        plt.title('Return Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        
        # Plot 3: Returns by Symbol
        plt.subplot(2, 2, 3)
        symbol_returns = results_df.groupby('symbol')['return'].mean() * 100
        symbol_returns = symbol_returns.sort_values(ascending=False)
        symbol_returns.plot(kind='bar')
        plt.title('Average Return by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Average Return (%)')
        plt.xticks(rotation=45)
        
        # Plot 4: Win Rate by Symbol
        plt.subplot(2, 2, 4)
        win_rates = results_df.groupby('symbol').apply(
            lambda x: (x['return'] > 0).mean() * 100 if len(x) > 0 else 0
        ).sort_values(ascending=False)
        win_rates.plot(kind='bar')
        plt.title('Win Rate by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}performance_analysis.png")
        
        # Additional plots
        plt.figure(figsize=(15, 10))
        
        # Plot 5: Returns by Direction
        plt.subplot(2, 2, 1)
        direction_returns = results_df.groupby('direction')['return'].mean() * 100
        direction_returns.plot(kind='bar')
        plt.title('Average Return by Direction')
        plt.xlabel('Direction')
        plt.ylabel('Average Return (%)')
        
        # Plot 6: Returns by Market Regime
        if 'market_regime' in results_df.columns:
            plt.subplot(2, 2, 2)
            regime_returns = results_df.groupby('market_regime')['return'].mean() * 100
            regime_returns.plot(kind='bar')
            plt.title('Average Return by Market Regime')
            plt.xlabel('Market Regime')
            plt.ylabel('Average Return (%)')
        
        # Plot 7: Scatter of Technical vs Seasonal Score
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['technical_score'], results_df['seasonal_score'], 
                   c=results_df['return'] * 100, cmap='RdYlGn', alpha=0.7)
        plt.colorbar(label='Return (%)')
        plt.title('Technical vs Seasonal Score')
        plt.xlabel('Technical Score')
        plt.ylabel('Seasonal Score')
        
        # Plot 8: Return vs Combined Score
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['combined_score'], results_df['return'] * 100, alpha=0.7)
        plt.title('Return vs Combined Score')
        plt.xlabel('Combined Score')
        plt.ylabel('Return (%)')
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}additional_analysis.png")
        
    except Exception as e:
        logger.error(f"Error plotting performance: {e}")

def run_backtest(config_file, start_date, end_date, use_seasonality=True, period_name=""):
    """
    Run a backtest for the given period with or without seasonality.
    
    Args:
        config_file (str): Path to configuration file
        start_date (datetime): Start date for backtest
        end_date (datetime): End date for backtest
        use_seasonality (bool): Whether to use seasonality in stock selection
        period_name (str): Name of the period for reporting
        
    Returns:
        pd.DataFrame: DataFrame with backtest results
    """
    logger.info(f"Running backtest for {start_date} to {end_date}")
    logger.info(f"Running backtest with seasonality {'ENABLED' if use_seasonality else 'DISABLED'}")
    
    # Load configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get symbols from config
    symbols = config.get('general', {}).get('symbols', [])
    logger.info(f"Running backtest with {len(symbols)} symbols")
    
    # Initialize strategy with the config, not the file path
    strategy = CombinedStrategy(config)
    logger.info(f"Multi-factor stock selection enabled: {strategy.use_multi_factor}")
    
    # If not using seasonality, disable it temporarily
    original_use_seasonality = strategy.use_seasonality
    if not use_seasonality:
        strategy.use_seasonality = False
    
    # Initialize data loader with Alpaca API
    credentials = load_alpaca_credentials()
    if credentials:
        api = tradeapi.REST(
            key_id=credentials['paper']['api_key'],
            secret_key=credentials['paper']['api_secret'],
            base_url=credentials['paper']['base_url']
        )
        data_loader = AlpacaDataLoader(api)
        logger.info("Initialized Alpaca data loader")
    else:
        logger.error("Failed to initialize data loader: No credentials")
        return None
    
    # Load data for each symbol
    symbol_data = {}
    
    # Define a local function to load data for a single symbol
    def _load_symbol_data(symbol):
        """Load historical data for a single symbol"""
        try:
            # Format dates as strings in YYYY-MM-DD format for Alpaca API
            start_str = '2022-01-01'  # Start a year earlier to have enough data for indicators
            end_str = '2024-01-31'    # End a month after our test period to ensure we have all data
            timeframe = '1D'
            
            df = data_loader.load_historical_data(
                symbol, 
                start_str,
                end_str,
                timeframe
            )
            
            # Debug the DataFrame structure
            if df is not None and not df.empty:
                logger.info(f"Loaded data for {symbol} with shape {df.shape}")
                
                # Set the timestamp column as the index if it exists
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                    logger.info(f"Set timestamp as index for {symbol}")
                    
            return symbol, df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return symbol, None
    
    # Load data in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_load_symbol_data, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol, df = future.result()
            if df is not None and not df.empty:
                symbol_data[symbol] = df
    
    logger.info(f"Loaded valid data for {len(symbol_data)} out of {len(symbols)} symbols")
    
    # Get all trading days in the period
    all_dates = []
    for symbol, df in symbol_data.items():
        if df is not None and not df.empty:
            logger.info(f"Processing data for {symbol}, shape: {df.shape}")
            logger.info(f"DataFrame index type: {type(df.index)}")
            logger.info(f"First few index values: {df.index[:5].tolist()}")
            
            # Print start_date and end_date for comparison
            logger.info(f"Target date range: {start_date} to {end_date}")
            
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                logger.info(f"Converted index to DatetimeIndex")
            
            # Create timezone-aware timestamps for comparison
            start_ts = pd.Timestamp(start_date).tz_localize('UTC')
            end_ts = pd.Timestamp(end_date).tz_localize('UTC')
            
            # Filter dates within our target range
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            logger.info(f"Number of dates in range: {mask.sum()}")
            
            dates_in_range = df.index[mask]
            logger.info(f"Dates in range for {symbol}: {len(dates_in_range)}")
            all_dates.extend(dates_in_range)
            
    # Get unique dates and sort
    trading_days = sorted(list(set(all_dates)))
    logger.info(f"Found {len(trading_days)} trading days in the period")
    
    if not trading_days:
        logger.error("No trading days found in the specified period")
        return None
    
    # Initialize results DataFrame
    results = []
    
    # Run backtest for each trading day
    for day_idx, day in enumerate(trading_days):
        # Skip the first day as we need previous data
        if day_idx == 0:
            continue
        
        # Get previous day
        prev_day = trading_days[day_idx - 1]
        
        # Prepare data for this day
        day_data = {}
        for symbol, df in symbol_data.items():
            # Get data up to current day
            mask = df.index <= day
            if mask.any():
                day_data[symbol] = df[mask]
        
        # Skip if not enough data
        if not day_data:
            continue
        
        # Select stocks for this day
        if strategy.use_multi_factor:
            selected_stocks = strategy.select_stocks_multi_factor(day_data, day)
        else:
            selected_stocks = strategy.select_stocks(day_data, day)
        
        if not selected_stocks:
            continue
        
        # Record results
        for symbol, details in selected_stocks.items():
            # Get next day's data for this symbol
            next_day_idx = min(day_idx + 1, len(trading_days) - 1)
            next_day = trading_days[next_day_idx]
            
            # Find the actual return
            if symbol in symbol_data and next_day in symbol_data[symbol].index:
                current_price = symbol_data[symbol].loc[day, 'close']
                next_price = symbol_data[symbol].loc[next_day, 'close']
                
                # Calculate return based on direction
                direction = details.get('direction', 'LONG')
                if direction == 'LONG':
                    pct_return = (next_price - current_price) / current_price * 100
                else:  # SHORT
                    pct_return = (current_price - next_price) / current_price * 100
                
                # Determine if it's a win
                is_win = pct_return > 0
                
                # Record trade
                results.append({
                    'date': day,
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'exit_price': next_price,
                    'return': pct_return,
                    'win': is_win,
                    'score': details.get('total_score', 0),
                    'technical_score': details.get('technical_score', 0),
                    'seasonal_score': details.get('seasonal_score', 0) if use_seasonality else 0,
                    'period': period_name
                })
    
    # Restore original seasonality setting
    strategy.use_seasonality = original_use_seasonality
    
    # Convert results to DataFrame
    if not results:
        logger.warning(f"No results for {period_name} {'with' if use_seasonality else 'without'} seasonality")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    total_trades = len(results_df)
    win_rate = results_df['win'].mean() * 100
    avg_return = results_df['return'].mean()
    profit_factor = calculate_profit_factor(results_df)
    sharpe = calculate_sharpe_ratio(results_df)
    max_drawdown = calculate_max_drawdown(results_df)
    
    logger.info(f"=== {period_name}{'_with_seasonality' if use_seasonality else '_without_seasonality'} Summary ===")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate (%): {win_rate:.1f}")
    logger.info(f"Avg Return (%): {avg_return:.2f}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info(f"Sharpe Ratio: {sharpe:.1f}")
    logger.info(f"Max Drawdown (%): {max_drawdown:.2f}")
    logger.info("")
    
    # Save detailed results to CSV
    filename = f"{period_name.lower().replace(' ', '_')}{'_with' if use_seasonality else '_without'}_seasonality.csv"
    results_df.to_csv(filename, index=False)
    
    return results_df

def calculate_performance_metrics(results_df):
    """Calculate performance metrics for a given results DataFrame"""
    total_trades = len(results_df)
    win_rate = results_df['win'].mean() * 100
    avg_return = results_df['return'].mean()
    profit_factor = calculate_profit_factor(results_df)
    sharpe = calculate_sharpe_ratio(results_df)
    max_drawdown = calculate_max_drawdown(results_df)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

def plot_comparison(period_name, with_seasonality_results, without_seasonality_results):
    """Create comparison visualizations"""
    try:
        # Set up the plotting style
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cumulative Returns
        plt.subplot(2, 2, 1)
        with_seasonality_cumulative_returns = (1 + with_seasonality_results['return']).cumprod()
        without_seasonality_cumulative_returns = (1 + without_seasonality_results['return']).cumprod()
        plt.plot(with_seasonality_cumulative_returns.index, with_seasonality_cumulative_returns, label='With Seasonality')
        plt.plot(without_seasonality_cumulative_returns.index, without_seasonality_cumulative_returns, label='Without Seasonality')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Return Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(with_seasonality_results['return'] * 100, kde=True, label='With Seasonality')
        sns.histplot(without_seasonality_results['return'] * 100, kde=True, label='Without Seasonality')
        plt.title('Return Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 3: Returns by Symbol
        plt.subplot(2, 2, 3)
        with_seasonality_symbol_returns = with_seasonality_results.groupby('symbol')['return'].mean() * 100
        without_seasonality_symbol_returns = without_seasonality_results.groupby('symbol')['return'].mean() * 100
        with_seasonality_symbol_returns.plot(kind='bar', label='With Seasonality')
        without_seasonality_symbol_returns.plot(kind='bar', label='Without Seasonality')
        plt.title('Average Return by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Average Return (%)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 4: Win Rate by Symbol
        plt.subplot(2, 2, 4)
        with_seasonality_win_rates = with_seasonality_results.groupby('symbol').apply(
            lambda x: (x['return'] > 0).mean() * 100 if len(x) > 0 else 0
        ).sort_values(ascending=False)
        without_seasonality_win_rates = without_seasonality_results.groupby('symbol').apply(
            lambda x: (x['return'] > 0).mean() * 100 if len(x) > 0 else 0
        ).sort_values(ascending=False)
        with_seasonality_win_rates.plot(kind='bar', label='With Seasonality')
        without_seasonality_win_rates.plot(kind='bar', label='Without Seasonality')
        plt.title('Win Rate by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{period_name}_comparison.png")
        
    except Exception as e:
        logger.error(f"Error plotting comparison: {e}")

def calculate_profit_factor(results_df):
    """Calculate profit factor (gross profit / gross loss)"""
    if results_df.empty:
        return 0.0
    
    # Calculate gross profit and gross loss
    profits = results_df[results_df['return'] > 0]['return'].sum()
    losses = abs(results_df[results_df['return'] < 0]['return'].sum())
    
    # Avoid division by zero
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    
    return profits / losses

def calculate_sharpe_ratio(results_df, risk_free_rate=0.0):
    """Calculate Sharpe ratio (return / volatility)"""
    if results_df.empty:
        return 0.0
    
    # Calculate mean return and standard deviation
    mean_return = results_df['return'].mean()
    std_return = results_df['return'].std()
    
    # Avoid division by zero
    if std_return == 0:
        return 0.0
    
    # Calculate annualized Sharpe ratio (assuming daily returns)
    sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
    
    return sharpe

def calculate_max_drawdown(results_df):
    """Calculate maximum drawdown as a percentage"""
    if results_df.empty:
        return 0.0
    
    # Calculate cumulative returns
    cumulative_returns = (1 + results_df['return']).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max - 1) * 100
    
    # Get maximum drawdown
    max_drawdown = abs(drawdown.min())
    
    return max_drawdown

def main():
    """Main function to run the seasonality comparison test"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration file
    config_file = "configuration_combined_strategy.yaml"
    logger.info(f"Testing configuration: {config_file}")
    
    # Load symbols from config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config.get('general', {}).get('symbols', [])
    logger.info(f"Loaded {len(symbols)} symbols from configuration")
    
    # Define test periods
    periods = [
        {
            'name': 'Full Year 2023',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 12, 31)
        },
        {
            'name': 'Q1 2023',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 3, 31)
        },
        {
            'name': 'Q2 2023',
            'start_date': datetime(2023, 4, 1),
            'end_date': datetime(2023, 6, 30)
        },
        {
            'name': 'Q3 2023',
            'start_date': datetime(2023, 7, 1),
            'end_date': datetime(2023, 9, 30)
        },
        {
            'name': 'Q4 2023',
            'start_date': datetime(2023, 10, 1),
            'end_date': datetime(2023, 12, 31)
        }
    ]
    
    # Store results for comparison
    all_results = []
    
    # Run tests for each period
    for period in periods:
        logger.info(f"Running backtest for {period['name']}: {period['start_date']} to {period['end_date']}")
        
        # Run with seasonality
        with_seasonality_results = run_backtest(
            config_file,
            period['start_date'],
            period['end_date'],
            use_seasonality=True,
            period_name=period['name']
        )
        
        # Run without seasonality
        without_seasonality_results = run_backtest(
            config_file,
            period['start_date'],
            period['end_date'],
            use_seasonality=False,
            period_name=period['name']
        )
        
        # Compare results
        if with_seasonality_results is not None and without_seasonality_results is not None:
            # Calculate performance metrics for both
            with_metrics = calculate_performance_metrics(with_seasonality_results)
            without_metrics = calculate_performance_metrics(without_seasonality_results)
            
            # Add to results
            all_results.append({
                'period': period['name'],
                'with_seasonality_win_rate': with_metrics['win_rate'],
                'without_seasonality_win_rate': without_metrics['win_rate'],
                'with_seasonality_avg_return': with_metrics['avg_return'],
                'without_seasonality_avg_return': without_metrics['avg_return'],
                'with_seasonality_profit_factor': with_metrics['profit_factor'],
                'without_seasonality_profit_factor': without_metrics['profit_factor'],
                'with_seasonality_sharpe': with_metrics['sharpe'],
                'without_seasonality_sharpe': without_metrics['sharpe'],
                'with_seasonality_max_drawdown': with_metrics['max_drawdown'],
                'without_seasonality_max_drawdown': without_metrics['max_drawdown'],
                'improvement_win_rate': with_metrics['win_rate'] - without_metrics['win_rate'],
                'improvement_avg_return': with_metrics['avg_return'] - without_metrics['avg_return'],
                'improvement_profit_factor': with_metrics['profit_factor'] - without_metrics['profit_factor'],
                'improvement_sharpe': with_metrics['sharpe'] - without_metrics['sharpe'],
                'improvement_max_drawdown': without_metrics['max_drawdown'] - with_metrics['max_drawdown']
            })
            
            # Log comparison
            logger.info(f"=== {period['name']} Comparison ===")
            logger.info(f"Win Rate: With Seasonality={with_metrics['win_rate']:.1f}%, Without Seasonality={without_metrics['win_rate']:.1f}%")
            logger.info(f"Avg Return: With Seasonality={with_metrics['avg_return']:.2f}%, Without Seasonality={without_metrics['avg_return']:.2f}%")
            logger.info(f"Profit Factor: With Seasonality={with_metrics['profit_factor']:.2f}, Without Seasonality={without_metrics['profit_factor']:.2f}")
            logger.info(f"Sharpe Ratio: With Seasonality={with_metrics['sharpe']:.1f}, Without Seasonality={without_metrics['sharpe']:.1f}")
            logger.info(f"Max Drawdown: With Seasonality={with_metrics['max_drawdown']:.2f}%, Without Seasonality={without_metrics['max_drawdown']:.2f}%")
            logger.info("")
            
            # Create comparison visualizations
            plot_comparison(
                period['name'],
                with_seasonality_results,
                without_seasonality_results
            )
    
    # Save all comparison results to CSV
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv("seasonality_comparison_results.csv", index=False)
        logger.info("Saved comparison results to seasonality_comparison_results.csv")
    
    logger.info("Analysis complete. Plots saved to current directory.")

if __name__ == "__main__":
    main()
