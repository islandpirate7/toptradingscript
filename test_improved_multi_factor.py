#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the improved multi-factor stock selection strategy with seasonality.
This script loads the enhanced configuration, runs a backtest, and analyzes the results.
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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

def load_data_parallel(data_loader, symbols, start_date, end_date, timeframe, max_workers=10):
    """Load historical data for multiple symbols in parallel"""
    symbol_data = {}
    
    def load_symbol_data(symbol):
        try:
            df = data_loader.load_historical_data(
                symbol, 
                start=start_date, 
                end=end_date, 
                timeframe=timeframe
            )
            
            if df is not None and not df.empty:
                # Convert timestamp to timezone-naive for consistent comparisons
                if 'timestamp' in df.columns and df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # Set timestamp as index if it's not already
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                
                logger.info(f"Loaded {len(df)} bars for {symbol}")
                return symbol, df
            else:
                logger.warning(f"No data loaded for {symbol}")
                return symbol, None
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return symbol, None
    
    # Use ThreadPoolExecutor to load data in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_symbol = {executor.submit(load_symbol_data, symbol): symbol for symbol in symbols}
        
        # Process results as they complete
        for future in as_completed(future_to_symbol):
            symbol, df = future.result()
            if df is not None:
                symbol_data[symbol] = df
    
    logger.info(f"Successfully loaded data for {len(symbol_data)} out of {len(symbols)} symbols")
    return symbol_data

def calculate_returns(selections_df, symbol_data, holding_period=5):
    """Calculate returns for selected stocks"""
    if selections_df.empty:
        return selections_df
    
    # Add columns for returns
    selections_df['entry_date'] = None
    selections_df['exit_date'] = None
    selections_df['entry_price'] = 0.0
    selections_df['exit_price'] = 0.0
    selections_df['return'] = 0.0
    selections_df['adjusted_return'] = 0.0  # Return adjusted by position size
    
    # Process each selection
    for idx, row in selections_df.iterrows():
        symbol = row['symbol']
        date_dt = row['date']
        position_size = row['position_size']
        
        if symbol not in symbol_data or symbol_data[symbol] is None:
            continue
        
        symbol_df = symbol_data[symbol]
        
        # Find entry date (first trading day on or after selection date)
        entry_mask = symbol_df.index >= date_dt
        if not any(entry_mask):
            continue
            
        entry_idx = symbol_df.index[entry_mask][0]
        entry_price = symbol_df.loc[entry_idx, 'close']
        
        # Find exit date (holding_period trading days later or last available day)
        exit_date = entry_idx
        remaining_days = holding_period
        
        # Get all dates after entry date
        future_dates = symbol_df.index[symbol_df.index > entry_idx]
        
        if len(future_dates) >= holding_period:
            exit_date = future_dates[holding_period-1]
        elif len(future_dates) > 0:
            exit_date = future_dates[-1]
        else:
            # No future dates available, use the last date
            exit_date = symbol_df.index[-1]
            
        exit_price = symbol_df.loc[exit_date, 'close']
        
        # Calculate return
        trade_return = (exit_price - entry_price) / entry_price
        if row['direction'] == 'SHORT':
            trade_return = -trade_return
            
        # Update the dataframe
        selections_df.at[idx, 'entry_date'] = entry_idx
        selections_df.at[idx, 'exit_date'] = exit_date
        selections_df.at[idx, 'entry_price'] = entry_price
        selections_df.at[idx, 'exit_price'] = exit_price
        selections_df.at[idx, 'return'] = trade_return
        selections_df.at[idx, 'adjusted_return'] = trade_return * position_size / 100.0
    
    return selections_df

def analyze_performance(results_df, config_file, start_date, end_date):
    """Analyze performance of stock selections"""
    if results_df.empty or 'return' not in results_df.columns:
        logger.warning("No results to analyze")
        return None, None
    
    # Save detailed results to CSV
    csv_filename = f"{os.path.basename(config_file).split('.')[0]}_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}_detailed_results.csv"
    results_df.to_csv(csv_filename, index=False)
    logger.info(f"Detailed results saved to {csv_filename}")
    
    # Calculate performance metrics
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['return'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    avg_return = results_df['return'].mean() if not results_df.empty else 0
    
    # Calculate profit factor (gross profit / gross loss)
    gross_profit = results_df[results_df['return'] > 0]['return'].sum() if not results_df.empty else 0
    gross_loss = abs(results_df[results_df['return'] < 0]['return'].sum()) if not results_df.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    returns = results_df['return']
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative_returns = (1 + results_df['return'] / 100).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / rolling_max - 1) * 100
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Total Trades', 'Win Rate (%)', 'Avg Return (%)', 'Profit Factor', 'Sharpe Ratio', 'Max Drawdown (%)'],
        'Value': [total_trades, round(win_rate, 2), round(avg_return, 2), round(profit_factor, 2), round(sharpe_ratio, 2), round(max_drawdown, 2)]
    })
    
    # Save summary to CSV
    summary_filename = f"{os.path.basename(config_file).split('.')[0]}_{start_date.replace('-', '')}_to_{end_date.replace('-', '')}_summary.csv"
    summary.to_csv(summary_filename, index=False)
    logger.info(f"Performance summary saved to {summary_filename}")
    
    return summary, results_df

def plot_performance(results_df, output_prefix=''):
    """Create performance visualizations"""
    if results_df.empty or 'return' not in results_df.columns:
        logger.warning("No results to plot")
        return
    
    # Filter out rows with NaN returns
    valid_results = results_df.dropna(subset=['return'])
    
    if valid_results.empty:
        logger.warning("No valid results with returns to plot")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a figure for return distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(valid_results['return'], kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_prefix}return_distribution.png')
    plt.close()
    
    # Create a figure for cumulative returns
    plt.figure(figsize=(12, 8))
    valid_results.sort_values('date', inplace=True)
    valid_results['cumulative_return'] = (1 + valid_results['return'] / 100).cumprod() - 1
    plt.plot(valid_results['date'].unique(), valid_results.groupby('date')['cumulative_return'].last() * 100)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.savefig(f'{output_prefix}cumulative_returns.png')
    plt.close()
    
    # Create a figure for returns by direction
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='technical_direction', y='return', data=valid_results)
    plt.title('Returns by Direction')
    plt.xlabel('Direction')
    plt.ylabel('Return (%)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(f'{output_prefix}returns_by_direction.png')
    plt.close()
    
    # Create a figure for returns by market regime if available
    if 'market_regime' in valid_results.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='market_regime', y='return', data=valid_results)
        plt.title('Returns by Market Regime')
        plt.xlabel('Market Regime')
        plt.ylabel('Return (%)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(f'{output_prefix}returns_by_regime.png')
        plt.close()
    
    logger.info(f"Plots saved with prefix: {output_prefix}")

def run_backtest(config_file, start_date, end_date, symbols=None, holding_period=5):
    """Run backtest for the specified period"""
    logger.info(f"Running backtest for {start_date} to {end_date}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    log_level = config.get('general', {}).get('log_level', 'INFO')
    logger.setLevel(log_level)
    
    # Get symbols from config
    if symbols is None:
        symbols = config.get('general', {}).get('symbols', [])
    if not symbols:
        symbols = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'AMD',
            'NEM', 'DE', 'CAT', 'LMT', 'NEE', 'SO', 'EOG', 'VLO', 'LIN', 'XLE', 'CVX', 'FDX'
        ]
    
    logger.info(f"Running backtest with {len(symbols)} symbols")
    
    # Initialize strategy
    strategy = CombinedStrategy(config)
    
    # Load Alpaca credentials
    credentials = load_alpaca_credentials()
    
    # Use paper trading credentials
    api_key = credentials['paper']['api_key']
    api_secret = credentials['paper']['api_secret']
    base_url = credentials['paper']['base_url']
    
    # Initialize API
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    
    # Initialize data loader
    data_loader = AlpacaDataLoader(api)
    logger.info("Initialized Alpaca data loader")
    
    # Calculate lookback period
    lookback_days = 100  # Default lookback for technical indicators
    
    # Adjust start date to include lookback period
    lookback_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    logger.info(f"Loading historical data from {lookback_start} to {end_date}")
    
    # Load historical data
    symbol_data = load_data_parallel(data_loader, symbols, lookback_start, end_date, '1D')
    
    # Get trading days
    trading_days = []
    for symbol, df in symbol_data.items():
        if df is not None and not df.empty:
            # Get dates after the start_date (excluding lookback period)
            start_timestamp = pd.Timestamp(start_date)
            logger.info(f"Extracting trading days for {symbol} from {start_date} to {end_date}")
            logger.info(f"DataFrame index type: {type(df.index)}, First index: {df.index[0]}, Last index: {df.index[-1]}")
            logger.info(f"DataFrame shape: {df.shape}")
            
            # Convert the index to Timestamp objects if they aren't already
            if not isinstance(df.index[0], pd.Timestamp):
                dates = [pd.Timestamp(date) for date in df.index if pd.Timestamp(date) >= start_timestamp]
            else:
                dates = [date for date in df.index if date >= start_timestamp]
            
            logger.info(f"Found {len(dates)} trading days for {symbol}")
            trading_days.extend(dates)
            break
    
    # Sort and deduplicate trading days
    trading_days = sorted(list(set(trading_days)))
    logger.info(f"Found {len(trading_days)} trading days from {start_date} to {end_date}")
    
    # Initialize results
    selections_df = pd.DataFrame()
    market_regimes = {}
    
    # Process each trading day
    for day in trading_days:
        current_date = pd.Timestamp(day).to_pydatetime()
        
        # Skip if current_date is after end_date
        if current_date > datetime.strptime(end_date, '%Y-%m-%d'):
            continue
        
        logger.info(f"Processing {current_date.strftime('%Y-%m-%d')}")
        
        # Detect market regime using SPY data
        market_regime = None
        if 'SPY' in symbol_data and symbol_data['SPY'] is not None:
            # Get data up to the current date
            spy_df = symbol_data['SPY']
            # Create a mask for dates up to the current day
            mask = spy_df.index.date <= pd.Timestamp(day).date()
            spy_data = spy_df.loc[mask]
            
            if len(spy_data) > 20:  # Ensure enough data for regime detection
                market_regime = strategy.detect_market_regime(spy_data)
                market_regimes[day] = market_regime.name
                logger.info(f"Detected market regime: {market_regime.name}")
        
        # Prepare data for stock selection
        day_data = {}
        for symbol, df in symbol_data.items():
            if df is not None and not df.empty:
                # Get data up to the current day
                mask = df.index <= pd.Timestamp(day)
                symbol_data_to_date = df.loc[mask]
                
                if not symbol_data_to_date.empty:
                    day_data[symbol] = symbol_data_to_date
        
        # Select stocks for the day
        selected_stocks = strategy.select_stocks_multi_factor(day_data, current_date, market_regime)
        
        if selected_stocks:
            # Convert selected stocks to DataFrame
            day_selections = []
            for i, (symbol, data) in enumerate(selected_stocks.items(), 1):
                direction = data.get('direction', 'NEUTRAL')
                position_size = data.get('position_size', 5.0)
                
                day_selections.append({
                    'date': current_date,
                    'symbol': symbol,
                    'rank': i,
                    'technical_score': data.get('technical_score', 0),
                    'seasonal_score': data.get('seasonal_score', 0),
                    'combined_score': data.get('combined_score', 0),
                    'technical_direction': data.get('technical_direction', 'NEUTRAL'),
                    'direction': direction,
                    'position_size': position_size,
                    'market_regime': market_regime.name if market_regime else 'UNKNOWN'
                })
            
            # Append to selections DataFrame
            day_df = pd.DataFrame(day_selections)
            selections_df = pd.concat([selections_df, day_df], ignore_index=True)
            
            # Log selections
            logger.info(f"Selected {len(selected_stocks)} stocks for {current_date.strftime('%Y-%m-%d')}")
            for i, (symbol, data) in enumerate(sorted(selected_stocks.items(), key=lambda x: x[1].get('combined_score', 0), reverse=True), 1):
                if i <= 5:
                    logger.info(f"  {i}. {symbol} - Score: {data.get('combined_score', 0):.4f}, Direction: {data.get('direction', 'NEUTRAL')}, Position Size: {data.get('position_size', 5.0):.1f}%")
                elif i == 6:
                    logger.info(f"  ... and {len(selected_stocks) - 5} more")
                    break
    
    # Calculate returns
    results_df = calculate_returns(selections_df, symbol_data, holding_period)
    
    # Analyze performance
    summary_df, results_df = analyze_performance(results_df, config_file, start_date, end_date)
    
    # Plot performance
    try:
        output_prefix = f"{os.path.basename(config_file).split('.')[0]}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_"
        plot_performance(results_df, output_prefix)
    except Exception as e:
        logger.error(f"Error plotting performance: {e}")
    
    return {
        'selections': selections_df,
        'returns': results_df,
        'performance': summary_df,
        'market_regimes': market_regimes
    }

def main():
    """Main function to run the backtest"""
    # Define configuration file
    config_file = "configuration_combined_strategy.yaml"
    
    logger.info(f"Testing configuration: {config_file}")
    
    # Define test periods
    periods = [
        {
            'name': 'Full Year 2023',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        },
        {
            'name': 'Q1 2023',
            'start_date': '2023-01-01',
            'end_date': '2023-03-31'
        },
        {
            'name': 'Q2 2023',
            'start_date': '2023-04-01',
            'end_date': '2023-06-30'
        },
        {
            'name': 'Q3 2023',
            'start_date': '2023-07-01',
            'end_date': '2023-09-30'
        },
        {
            'name': 'Q4 2023',
            'start_date': '2023-10-01',
            'end_date': '2023-12-31'
        }
    ]
    
    all_results = {}
    
    # Run backtest for each period
    for period in periods:
        logger.info(f"Running backtest for {period['name']}: {period['start_date']} to {period['end_date']}")
        
        results = run_backtest(
            config_file=config_file,
            start_date=period['start_date'],
            end_date=period['end_date'],
            holding_period=5
        )
        
        # Store results
        all_results[period['name']] = results
        
        # Log summary
        if results and 'performance' in results and results['performance'] is not None and not results['performance'].empty:
            summary_df = results['performance']
            logger.info(f"=== {period['name']} Summary ===")
            for _, row in summary_df.iterrows():
                logger.info(f"{row['Metric']}: {row['Value']}")
            logger.info("")
    
    logger.info("Analysis complete. Plots saved to current directory.")
    return all_results

if __name__ == "__main__":
    main()
