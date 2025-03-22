#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze March 2023 Ultra Aggressive Signals
--------------------------------------
This script analyzes the signals generated by the ultra aggressive
strategy for March 2023 and simulates trades based on these signals.
"""

import os
import yaml
import json
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from tabulate import tabulate
import alpaca_trade_api as tradeapi

from mean_reversion_strategy_ultra_aggressive import MeanReversionStrategyUltraAggressive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_alpaca_credentials():
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials.get('paper', {})
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {e}")
        return {}

def get_alpaca_client():
    """Get Alpaca API client"""
    try:
        credentials = load_alpaca_credentials()
        api_key = credentials.get('api_key', '')
        api_secret = credentials.get('api_secret', '')
        base_url = credentials.get('base_url', 'https://paper-api.alpaca.markets')
        
        if not api_key or not api_secret:
            logger.error("Missing Alpaca API credentials")
            return None
        
        return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    except Exception as e:
        logger.error(f"Error creating Alpaca client: {e}")
        return None

def get_historical_data(symbol, start_date, end_date, timeframe='1D'):
    """Get historical price data from Alpaca
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date
        end_date (datetime): End date
        timeframe (str): Timeframe (e.g. '1D', '1H')
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    try:
        # Get Alpaca client
        api = get_alpaca_client()
        if api is None:
            return None
        
        # Format dates as strings in YYYY-MM-DD format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Get bars
        bars = api.get_bars(
            symbol,
            timeframe,
            start=start_str,
            end=end_str,
            adjustment='raw'
        ).df
        
        # Reset index to make timestamp a column
        if not bars.empty:
            bars = bars.reset_index()
            logger.info(f"Fetched {len(bars)} bars for {symbol}")
        else:
            logger.warning(f"No data returned for {symbol}")
            
        return bars
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def simulate_trade(entry_signal, exit_signal, data):
    """Simulate a trade based on entry and exit signals
    
    Args:
        entry_signal (dict): Entry signal
        exit_signal (dict): Exit signal
        data (pd.DataFrame): Price data
        
    Returns:
        dict: Trade details
    """
    entry_date = entry_signal['date']
    entry_price = entry_signal['price']
    entry_direction = entry_signal['direction']
    stop_loss = entry_signal['stop_loss']
    take_profit = entry_signal['take_profit']
    
    exit_date = exit_signal['date'] if exit_signal else None
    exit_price = exit_signal['price'] if exit_signal else None
    exit_reason = 'signal' if exit_signal else None
    
    # If no exit signal, check if stop loss or take profit was hit
    if not exit_signal:
        # Filter data after entry date
        future_data = data[data['timestamp'] > entry_date]
        
        if not future_data.empty:
            for _, row in future_data.iterrows():
                current_price = row['close']
                current_date = row['timestamp']
                
                # Check if stop loss was hit
                if entry_direction == 'LONG' and current_price <= stop_loss:
                    exit_price = stop_loss
                    exit_date = current_date
                    exit_reason = 'stop_loss'
                    break
                elif entry_direction == 'SHORT' and current_price >= stop_loss:
                    exit_price = stop_loss
                    exit_date = current_date
                    exit_reason = 'stop_loss'
                    break
                
                # Check if take profit was hit
                if entry_direction == 'LONG' and current_price >= take_profit:
                    exit_price = take_profit
                    exit_date = current_date
                    exit_reason = 'take_profit'
                    break
                elif entry_direction == 'SHORT' and current_price <= take_profit:
                    exit_price = take_profit
                    exit_date = current_date
                    exit_reason = 'take_profit'
                    break
            
            # If no exit yet, use the last price
            if not exit_date:
                exit_price = future_data.iloc[-1]['close']
                exit_date = future_data.iloc[-1]['timestamp']
                exit_reason = 'end_of_period'
        else:
            # If no future data, use the last available price
            exit_price = entry_price  # Assume no profit/loss
            exit_date = entry_date
            exit_reason = 'no_future_data'
    
    # Ensure exit_price is not None
    if exit_price is None:
        exit_price = entry_price
        exit_reason = 'no_exit_price'
        if exit_date is None:
            exit_date = entry_date
    
    # Calculate profit
    if entry_direction == 'LONG':
        profit_pct = (exit_price - entry_price) / entry_price * 100
    else:  # SHORT
        profit_pct = (entry_price - exit_price) / entry_price * 100
    
    # Assume $10,000 position size
    position_size = 10000
    profit_usd = position_size * profit_pct / 100
    
    return {
        'symbol': entry_signal.get('symbol', 'Unknown'),
        'direction': entry_direction,
        'entry_date': entry_date,
        'entry_price': entry_price,
        'exit_date': exit_date,
        'exit_price': exit_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'exit_reason': exit_reason,
        'profit_pct': profit_pct,
        'profit_usd': profit_usd,
        'duration_days': (exit_date - entry_date).days + 1 if exit_date else 0
    }

def analyze_signals(symbols, start_date, end_date):
    """Analyze signals for a list of symbols
    
    Args:
        symbols (list): List of stock symbols
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        tuple: (signals_df, trades_df)
    """
    all_signals = []
    all_trades = []
    
    # Create ultra aggressive strategy
    strategy = MeanReversionStrategyUltraAggressive()
    
    for symbol in symbols:
        logger.info(f"Analyzing signals for {symbol}")
        
        # Get historical data
        data = get_historical_data(symbol, start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"No data available for {symbol}")
            continue
        
        # Generate signals
        signals = strategy.generate_signals(data, symbol)
        
        # Add symbol to signals
        for signal in signals:
            signal['symbol'] = symbol
        
        if signals:
            logger.info(f"Found {len(signals)} signals for {symbol}")
            all_signals.extend(signals)
            
            # Simulate trades
            active_position = None
            
            for i, signal in enumerate(signals):
                # Skip first signal if it's a SHORT (we want to start with a LONG)
                if i == 0 and signal['direction'] == 'SHORT':
                    continue
                
                if active_position is None:
                    # Enter new position
                    active_position = signal
                else:
                    # Exit existing position if direction is opposite
                    if active_position['direction'] != signal['direction']:
                        trade = simulate_trade(active_position, signal, data)
                        all_trades.append(trade)
                        active_position = signal
            
            # Close any remaining position
            if active_position:
                trade = simulate_trade(active_position, None, data)
                all_trades.append(trade)
        else:
            logger.info(f"No signals found for {symbol}")
    
    # Convert to DataFrames
    signals_df = pd.DataFrame(all_signals)
    trades_df = pd.DataFrame(all_trades)
    
    return signals_df, trades_df

def analyze_march_2023_signals(config_file):
    """Analyze signals for March 2023
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        tuple: (signals_df, trades_df)
    """
    # Define date range
    start_date = dt.datetime(2023, 3, 1)
    end_date = dt.datetime(2023, 3, 31)
    
    logger.info(f"Analyzing signals for March 2023 from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get symbols from config
    symbols = config.get('general', {}).get('symbols', [])
    
    if not symbols:
        logger.error("No symbols found in configuration")
        return None, None
    
    logger.info(f"Analyzing signals for {len(symbols)} symbols")
    
    # Analyze signals
    signals_df, trades_df = analyze_signals(symbols, start_date, end_date)
    
    # Display results
    if not signals_df.empty:
        logger.info(f"Found {len(signals_df)} signals for {len(signals_df['symbol'].unique())} symbols")
        
        # Count signals by symbol
        signal_counts = signals_df['symbol'].value_counts()
        print("\nSignal Counts by Symbol:")
        print(tabulate(signal_counts.reset_index().rename(columns={'index': 'Symbol', 'symbol': 'Signal Count'}), 
                      headers='keys', tablefmt='grid'))
        
        # Count signals by direction
        direction_counts = signals_df['direction'].value_counts()
        print("\nSignal Counts by Direction:")
        print(tabulate(direction_counts.reset_index().rename(columns={'index': 'Direction', 'direction': 'Signal Count'}), 
                      headers='keys', tablefmt='grid'))
    else:
        logger.warning("No signals found")
    
    if not trades_df.empty:
        logger.info(f"Simulated {len(trades_df)} trades")
        
        # Format the DataFrame for display
        display_df = trades_df.copy()
        
        # Convert timestamps to readable format
        if 'entry_date' in display_df.columns:
            display_df['entry_date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')
        if 'exit_date' in display_df.columns:
            display_df['exit_date'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d')
        
        # Format numeric columns
        if 'profit_pct' in display_df.columns:
            display_df['profit_pct'] = display_df['profit_pct'].map('{:.2f}%'.format)
        if 'profit_usd' in display_df.columns:
            display_df['profit_usd'] = display_df['profit_usd'].map('${:.2f}'.format)
        
        # Select relevant columns for display
        display_columns = ['symbol', 'direction', 'entry_date', 'exit_date', 'profit_pct', 'profit_usd', 'exit_reason']
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Display the trades
        print("\nTrade Details:")
        print(tabulate(display_df[display_columns], headers='keys', tablefmt='grid'))
        
        # Calculate and display summary statistics
        print("\nTrade Summary:")
        win_rate = (trades_df['profit_usd'] > 0).mean() * 100
        total_profit = trades_df['profit_usd'].sum()
        avg_profit = trades_df['profit_usd'].mean()
        
        print(f"Total Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Average Profit per Trade: ${avg_profit:.2f}")
        
        # Group by symbol
        symbol_performance = trades_df.groupby('symbol').agg({
            'profit_usd': ['sum', 'mean', 'count'],
            'profit_pct': ['mean']
        })
        
        # Flatten the column names
        symbol_performance.columns = ['_'.join(col).strip() for col in symbol_performance.columns.values]
        
        # Rename columns for clarity
        symbol_performance = symbol_performance.rename(columns={
            'profit_usd_sum': 'Total Profit ($)',
            'profit_usd_mean': 'Avg Profit per Trade ($)',
            'profit_usd_count': 'Trade Count',
            'profit_pct_mean': 'Avg Return (%)'
        })
        
        # Sort by total profit
        symbol_performance = symbol_performance.sort_values('Total Profit ($)', ascending=False)
        
        print("\nPerformance by Symbol:")
        print(tabulate(symbol_performance.reset_index(), headers='keys', tablefmt='grid'))
        
        # Save results to CSV
        trades_df.to_csv('march_2023_trades.csv', index=False)
        logger.info("Saved trade results to march_2023_trades.csv")
        
        # Plot profit distribution
        plt.figure(figsize=(10, 6))
        plt.hist(trades_df['profit_usd'], bins=20, alpha=0.7)
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Profit Distribution')
        plt.xlabel('Profit ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('march_2023_profit_distribution.png')
        logger.info("Saved profit distribution chart to march_2023_profit_distribution.png")
    else:
        logger.warning("No trades simulated")
    
    return signals_df, trades_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze signals for March 2023')
    parser.add_argument('--config', type=str, default='configuration_ultra_aggressive_march_2023.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Analyze signals
    signals_df, trades_df = analyze_march_2023_signals(args.config)
