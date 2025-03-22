#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Mean Reversion Strategy Backtest
-----------------------------------------
This script runs a backtest for the optimized mean reversion strategy using real Alpaca data.
It implements the optimized parameters identified through previous testing.
"""

import os
import sys
import json
import logging
import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame

# Import our custom modules
from mean_reversion_strategy import MeanReversionStrategy
from portfolio import Portfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimized_mean_reversion_backtest.log"),
        logging.StreamHandler()  # This will print to console
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def initialize_alpaca_api():
    """Initialize the Alpaca API client with credentials from alpaca_credentials.json"""
    try:
        # Load credentials from JSON file
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca_credentials.json')
        
        logger.info(f"Looking for credentials at: {credentials_path}")
        
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            
            # Use paper trading credentials by default
            paper_creds = credentials.get('paper', {})
            api_key = paper_creds.get('api_key')
            api_secret = paper_creds.get('api_secret')
            base_url = paper_creds.get('base_url', 'https://paper-api.alpaca.markets')
            
            # Remove /v2 suffix if it's already included to prevent duplication
            if base_url.endswith('/v2'):
                base_url = base_url[:-3]
            
            logger.info(f"Using paper trading credentials with base URL: {base_url}")
        else:
            # Fallback to environment variables
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            base_url = 'https://paper-api.alpaca.markets'
            
            if not api_key or not api_secret:
                logger.error("Alpaca API credentials not found")
                return None
        
        # Initialize API
        api = REST(api_key, api_secret, base_url)
        logger.info("Alpaca API initialized successfully")
        
        # Test the API connection
        account = api.get_account()
        logger.info(f"Connected to Alpaca account: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Account equity: {account.equity}")
        
        return api
            
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {str(e)}")
        return None

def fetch_historical_data(api, symbol, start_date, end_date):
    """Fetch historical price data from Alpaca"""
    if not api:
        logger.error("Alpaca API not initialized")
        return None
    
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Format dates as YYYY-MM-DD (without time component)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Using date strings: start={start_str}, end={end_str}")
        
        # Fetch data
        bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start_str,
            end=end_str,
            adjustment='raw'
        ).df
        
        if bars.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        return bars
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def run_backtest(api, symbols, start_date, end_date, config):
    """Run a backtest with the given symbols and date range"""
    if not api:
        logger.error("Alpaca API not initialized")
        return None
    
    # Initialize strategy
    strategy = MeanReversionStrategy(config)
    
    # Initialize portfolio
    initial_capital = config.get('initial_capital', 100000)
    portfolio = Portfolio(initial_capital)
    
    # Fetch data for each symbol
    symbol_data = {}
    for symbol in symbols:
        bars = fetch_historical_data(api, symbol, start_date, end_date)
        if bars is not None:
            # Generate signals
            df = strategy.generate_signals(bars)
            symbol_data[symbol] = df
            logger.info(f"Generated signals for {symbol}: {len(df[df['signal'] != 0])} signals")
    
    if not symbol_data:
        logger.error("No data fetched for any symbol")
        return None
    
    # Get all unique dates from the data
    all_dates = set()
    for symbol, df in symbol_data.items():
        all_dates.update(df.index.date)
    
    all_dates = sorted(all_dates)
    logger.info(f"Running backtest with {len(all_dates)} trading days")
    
    # Process each date
    for date in all_dates:
        # Process open positions first
        for symbol, position in list(portfolio.open_positions.items()):
            if symbol in symbol_data:
                # Get data for this date
                day_data = symbol_data[symbol][symbol_data[symbol].index.date == date]
                
                if not day_data.empty:
                    # Check for stop loss or take profit
                    if position.direction == 'long':
                        # Check if price hit stop loss
                        if day_data['low'].iloc[0] <= position.stop_loss:
                            portfolio.close_position(symbol, position.stop_loss, day_data.index[0], "stop_loss")
                        # Check if price hit take profit
                        elif day_data['high'].iloc[0] >= position.take_profit:
                            portfolio.close_position(symbol, position.take_profit, day_data.index[0], "take_profit")
                    else:  # short
                        # Check if price hit stop loss
                        if day_data['high'].iloc[0] >= position.stop_loss:
                            portfolio.close_position(symbol, position.stop_loss, day_data.index[0], "stop_loss")
                        # Check if price hit take profit
                        elif day_data['low'].iloc[0] <= position.take_profit:
                            portfolio.close_position(symbol, position.take_profit, day_data.index[0], "take_profit")
        
        # Process new signals
        for symbol, df in symbol_data.items():
            # Skip if we already have a position for this symbol
            if symbol in portfolio.open_positions:
                continue
            
            # Get data for this date
            day_data = df[df.index.date == date]
            
            if not day_data.empty and day_data['signal'].iloc[0] != 0:
                signal = day_data['signal'].iloc[0]
                price = day_data['close'].iloc[0]
                timestamp = day_data.index[0]
                
                # Calculate position size (fixed dollar amount per trade)
                position_size_dollars = initial_capital * 0.05  # 5% of initial capital per trade
                position_size = position_size_dollars / price
                
                if signal == 1:  # Buy signal
                    portfolio.open_position(
                        symbol=symbol,
                        entry_price=price,
                        entry_time=timestamp,
                        position_size=position_size,
                        direction='long',
                        stop_loss=day_data['stop_loss'].iloc[0],
                        take_profit=day_data['take_profit'].iloc[0]
                    )
                elif signal == -1:  # Sell signal
                    portfolio.open_position(
                        symbol=symbol,
                        entry_price=price,
                        entry_time=timestamp,
                        position_size=position_size,
                        direction='short',
                        stop_loss=day_data['stop_loss'].iloc[0],
                        take_profit=day_data['take_profit'].iloc[0]
                    )
        
        # Update equity curve at the end of the day
        portfolio.update_equity_curve(datetime.datetime.combine(date, datetime.time(16, 0)))
    
    # Close any remaining open positions at the end of the backtest
    for symbol, position in list(portfolio.open_positions.items()):
        if symbol in symbol_data:
            last_price = symbol_data[symbol]['close'].iloc[-1]
            last_time = symbol_data[symbol].index[-1]
            portfolio.close_position(symbol, last_price, last_time, "end_of_backtest")
    
    # Get performance metrics
    metrics = portfolio.get_performance_metrics()
    
    logger.info(f"Backtest completed: {len(metrics['trades'])} trades, Return: {metrics['return']:.2%}, Win Rate: {metrics['win_rate']:.2%}")
    
    return metrics

def plot_equity_curve(equity_curve, title, filename):
    """Plot and save the equity curve"""
    timestamps = [ts for ts, _ in equity_curve]
    equity = [eq for _, eq in equity_curve]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, equity)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(filename)
    logger.info(f"Saved equity curve plot to {filename}")

def save_trade_results(trades, filename):
    """Save trade results to CSV file"""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Symbol', 'Direction', 'Entry Price', 'Entry Time', 'Exit Price', 'Exit Time', 'P/L', 'Reason'])
        
        for trade in trades:
            writer.writerow([
                trade['symbol'],
                trade['direction'],
                trade['entry_price'],
                trade['entry_time'],
                trade['exit_price'],
                trade['exit_time'],
                trade['profit_loss'],
                trade['reason']
            ])
    
    logger.info(f"Saved trades to {filename}")

def save_summary(metrics, filename, start_date, end_date, symbols):
    """Save summary to text file"""
    with open(filename, 'w') as f:
        f.write(f"Backtest Summary: {start_date} to {end_date}\n")
        f.write(f"Symbols: {symbols}\n\n")
        f.write(f"Initial Capital: ${metrics['initial_capital']:.2f}\n")
        f.write(f"Final Capital: ${metrics['final_capital']:.2f}\n")
        f.write(f"Return: {metrics['return']:.2%}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
    
    logger.info(f"Saved summary to {filename}")

def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='Run backtest for optimized mean reversion strategy with Alpaca data')
    parser.add_argument('--config', type=str, default='multi_strategy_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4],
                        help='Quarter of 2023 to test (1-4)')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='List of symbols to test')
    
    args = parser.parse_args()
    
    # Set date range based on quarter if specified
    if args.quarter:
        if args.quarter == 1:
            args.start = '2023-01-01'
            args.end = '2023-03-31'
        elif args.quarter == 2:
            args.start = '2023-04-01'
            args.end = '2023-06-30'
        elif args.quarter == 3:
            args.start = '2023-07-01'
            args.end = '2023-09-30'
        elif args.quarter == 4:
            args.start = '2023-10-01'
            args.end = '2023-12-31'
    
    # Default symbols if none provided
    if not args.symbols:
        args.symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Initialize Alpaca API
    api = initialize_alpaca_api()
    if not api:
        logger.error("Failed to initialize Alpaca API")
        return
    
    # Run backtest
    logger.info(f"Running backtest from {args.start} to {args.end} with symbols: {args.symbols}")
    metrics = run_backtest(api, args.symbols, args.start, args.end, config)
    
    if metrics:
        # Plot equity curve
        plot_title = f"Equity Curve: {args.start} to {args.end}"
        plot_filename = f"equity_curve_{args.start}_to_{args.end}.png"
        plot_equity_curve(metrics['equity_curve'], plot_title, plot_filename)
        
        # Save trade results
        trades_filename = f"trades_{args.start}_to_{args.end}.csv"
        save_trade_results(metrics['trades'], trades_filename)
        
        # Save summary
        summary_filename = f"summary_{args.start}_to_{args.end}.txt"
        save_summary(metrics, summary_filename, args.start, args.end, args.symbols)
        
        # Print summary
        print(f"\nBacktest Summary: {args.start} to {args.end}")
        print(f"Symbols: {args.symbols}")
        print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
        print(f"Final Capital: ${metrics['final_capital']:.2f}")
        print(f"Return: {metrics['return']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")

if __name__ == "__main__":
    main()
