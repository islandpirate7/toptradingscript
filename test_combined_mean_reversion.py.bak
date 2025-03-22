#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to run the mean reversion strategy on the top 20 stocks selected
using the combined strategy with seasonality, technical indicators, and volatility metrics.
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
import argparse
from combined_strategy import CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_alpaca_data(symbols, start_date, end_date):
    """
    Load historical data from Alpaca API.
    
    Args:
        symbols (list): List of symbols to load data for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary of dataframes with data for each symbol
    """
    try:
        # Import Alpaca API client
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        # Load credentials
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        # Use paper trading credentials
        paper_credentials = credentials.get('paper', {})
        api_key = paper_credentials.get('api_key')
        api_secret = paper_credentials.get('api_secret')
        
        if not api_key or not api_secret:
            logger.error("Missing Alpaca API credentials")
            return {}
        
        # Initialize client
        client = StockHistoricalDataClient(api_key, api_secret)
        
        # Convert dates to datetime
        start = pd.Timestamp(start_date, tz='America/New_York').date()
        end = pd.Timestamp(end_date, tz='America/New_York').date()
        
        # Request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        # Get bars
        bars = client.get_stock_bars(request_params)
        
        # Convert to dictionary of dataframes
        data = {}
        for symbol in symbols:
            if symbol in bars.data:
                # Convert to dataframe
                symbol_bars = bars.data[symbol]
                df = pd.DataFrame()
                
                # Extract data
                df['timestamp'] = [bar.timestamp for bar in symbol_bars]
                df['open'] = [bar.open for bar in symbol_bars]
                df['high'] = [bar.high for bar in symbol_bars]
                df['low'] = [bar.low for bar in symbol_bars]
                df['close'] = [bar.close for bar in symbol_bars]
                df['volume'] = [bar.volume for bar in symbol_bars]
                
                data[symbol] = df
                logger.debug(f"Loaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"No data available for {symbol}")
        
        logger.info(f"Loaded historical data for {len(data)} symbols from {start_date} to {end_date}")
        return data
    except Exception as e:
        logger.error(f"Error loading Alpaca data: {e}")
        return {}

def select_top_stocks(strategy, test_date, lookback_days=100, top_n=20):
    """
    Select the top stocks using the combined strategy.
    
    Args:
        strategy (CombinedStrategy): Combined strategy instance
        test_date (datetime): Date for stock selection
        lookback_days (int): Number of days of historical data to use
        top_n (int): Number of top stocks to select
        
    Returns:
        list: List of selected stock symbols
    """
    # Get symbols from config
    symbols = strategy.config['general']['symbols']
    
    # Load historical data
    start_date = (test_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    end_date = test_date.strftime('%Y-%m-%d')
    
    market_data = load_alpaca_data(symbols, start_date, end_date)
    
    if not market_data:
        logger.error("Failed to load market data for stock selection. Exiting.")
        return []
    
    # Select stocks using the multi-factor method
    logger.info(f"Selecting top {top_n} stocks for {test_date.strftime('%Y-%m-%d')} using multi-factor approach")
    selected_stocks = strategy.select_stocks_multi_factor(market_data, test_date, top_n=top_n)
    
    # Extract symbols from selected stocks
    selected_symbols = []
    for stock in selected_stocks:
        if isinstance(stock, dict) and 'symbol' in stock:
            selected_symbols.append(stock['symbol'])
        elif isinstance(stock, tuple) and len(stock) >= 1:
            selected_symbols.append(stock[0])
        else:
            selected_symbols.append(stock)
    
    logger.info(f"Selected {len(selected_symbols)} stocks: {', '.join(selected_symbols)}")
    return selected_symbols

def run_mean_reversion_backtest(symbols, start_date, end_date, config):
    """
    Run a backtest of the mean reversion strategy on the selected symbols.
    
    Args:
        symbols (list): List of symbols to test
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary with backtest results
    """
    # Import the AlpacaBacktest class
    from test_optimized_mean_reversion_alpaca import AlpacaBacktest
    import datetime
    from alpaca_trade_api.rest import TimeFrame
    
    # Create a subclass to fix the date format issue
    class FixedAlpacaBacktest(AlpacaBacktest):
        def fetch_historical_data(self, symbol, start_date, end_date):
            """Fetch historical price data from Alpaca with fixed date format"""
            if not self.api:
                self.logger.error("Alpaca API not initialized")
                return None
            
            try:
                # Convert string dates to datetime if needed
                if isinstance(start_date, str):
                    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                if isinstance(end_date, str):
                    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                
                self.logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
                
                # Determine timeframe based on symbol type
                timeframe = TimeFrame.Day
                
                # Fetch data - use date strings in YYYY-MM-DD format instead of isoformat
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    self.logger.warning(f"No data returned for {symbol}")
                    return None
                
                # Convert to CandleData objects
                from test_optimized_mean_reversion_alpaca import CandleData
                candles = []
                for index, row in bars.iterrows():
                    candle = CandleData(
                        timestamp=index.to_pydatetime(),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    candles.append(candle)
                
                self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
                return candles
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                return None
    
    # Save the selected symbols to a temporary config file for the backtest
    temp_config = config.copy()
    temp_config['general']['symbols'] = symbols
    
    # Set optimized mean reversion parameters
    if 'strategy_configs' not in temp_config:
        temp_config['strategy_configs'] = {}
    
    # Update with optimized parameters from previous testing
    temp_config['strategy_configs']['MeanReversion'] = {
        'bb_period': 20,
        'bb_std_dev': 1.9,
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'require_reversal': True,
        'stop_loss_atr': 1.8,
        'take_profit_atr': 3.0,
        'atr_period': 14,
        'volume_filter': True
    }
    
    # Write the temporary config to a file
    temp_config_path = 'temp_backtest_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f)
    
    # Initialize the backtest with our fixed subclass
    backtest = FixedAlpacaBacktest(temp_config_path)
    
    # Initialize Alpaca API
    backtest.initialize_alpaca_api()
    
    # Set the symbols to test
    backtest.set_symbols(symbols)
    
    # Run the backtest
    logger.info(f"Running mean reversion backtest on {len(symbols)} selected stocks from {start_date} to {end_date}")
    results = backtest.run_backtest(start_date, end_date)
    
    # Clean up temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    return results

def analyze_backtest_results(results):
    """
    Analyze the backtest results.
    
    Args:
        results (dict): Dictionary with backtest results
    """
    # Extract trades
    trades = results.get('trades', [])
    
    if not trades:
        logger.warning("No trades were executed in the backtest")
        return
    
    # Convert to DataFrame for analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate overall metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Calculate returns
    trades_df['return'] = trades_df['profit_loss'] / trades_df['entry_price']
    
    avg_return = trades_df['return'].mean() * 100 if total_trades > 0 else 0
    median_return = trades_df['return'].median() * 100 if total_trades > 0 else 0
    max_return = trades_df['return'].max() * 100 if total_trades > 0 else 0
    min_return = trades_df['return'].min() * 100 if total_trades > 0 else 0
    
    # Calculate profit factor
    total_gains = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
    total_losses = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
    
    # Print overall metrics
    logger.info("\n=== Performance Metrics ===")
    logger.info(f"Total trades: {total_trades}")
    logger.info(f"Win rate: {win_rate:.2f}%")
    logger.info(f"Average return: {avg_return:.2f}%")
    logger.info(f"Median return: {median_return:.2f}%")
    logger.info(f"Maximum return: {max_return:.2f}%")
    logger.info(f"Minimum return: {min_return:.2f}%")
    logger.info(f"Profit factor: {profit_factor:.2f}")
    
    # Analyze performance by direction
    if 'direction' in trades_df.columns:
        logger.info("\n=== Performance by Direction ===")
        for direction, group in trades_df.groupby('direction'):
            dir_trades = len(group)
            dir_wins = len(group[group['profit_loss'] > 0])
            dir_win_rate = dir_wins / dir_trades * 100 if dir_trades > 0 else 0
            dir_avg_return = group['return'].mean() * 100 if dir_trades > 0 else 0
            
            logger.info(f"{direction}: {dir_trades} trades, Win rate: {dir_win_rate:.2f}%, Avg return: {dir_avg_return:.2f}%")
    
    # Analyze performance by symbol
    logger.info("\n=== Performance by Symbol ===")
    symbol_stats = {}
    for symbol, group in trades_df.groupby('symbol'):
        sym_trades = len(group)
        sym_wins = len(group[group['profit_loss'] > 0])
        sym_win_rate = sym_wins / sym_trades * 100 if sym_trades > 0 else 0
        sym_avg_return = group['return'].mean() * 100 if sym_trades > 0 else 0
        
        symbol_stats[symbol] = {
            'trades': sym_trades,
            'win_rate': sym_win_rate,
            'avg_return': sym_avg_return
        }
        
        logger.info(f"{symbol}: {sym_trades} trades, Win rate: {sym_win_rate:.2f}%, Avg return: {sym_avg_return:.2f}%")
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Plot return distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(trades_df['return'] * 100, kde=True)
    plt.title('Trade Return Distribution')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('output/return_distribution.png')
    plt.close()
    
    # Plot cumulative returns
    trades_df['cumulative_return'] = (1 + trades_df['return']).cumprod() - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(trades_df.index, trades_df['cumulative_return'] * 100)
    plt.title('Cumulative Returns')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.savefig('output/cumulative_returns.png')
    plt.close()
    
    # Plot performance by symbol
    symbols = list(symbol_stats.keys())
    win_rates = [symbol_stats[s]['win_rate'] for s in symbols]
    avg_returns = [symbol_stats[s]['avg_return'] for s in symbols]
    
    plt.figure(figsize=(14, 7))
    
    # Sort by average return
    sorted_indices = np.argsort(avg_returns)
    sorted_symbols = [symbols[i] for i in sorted_indices]
    sorted_win_rates = [win_rates[i] for i in sorted_indices]
    sorted_avg_returns = [avg_returns[i] for i in sorted_indices]
    
    plt.subplot(1, 2, 1)
    plt.barh(sorted_symbols, sorted_avg_returns)
    plt.title('Average Return by Symbol')
    plt.xlabel('Average Return (%)')
    plt.ylabel('Symbol')
    plt.grid(True, axis='x')
    
    plt.subplot(1, 2, 2)
    plt.barh(sorted_symbols, sorted_win_rates)
    plt.title('Win Rate by Symbol')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Symbol')
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig('output/symbol_performance.png')
    plt.close()
    
    logger.info("\nAnalysis complete. Plots saved to output directory.")
    
    return symbol_stats

def initialize_strategy(use_multi_factor):
    """Initialize the combined strategy with the specified configuration"""
    # Import the CombinedStrategy class
    from combined_strategy import CombinedStrategy
    
    # Load the configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the strategy with the loaded config
    strategy = CombinedStrategy(config)
    
    # Enable multi-factor stock selection if requested
    strategy.use_multi_factor = use_multi_factor
    
    # Log the strategy weights
    logger.info(f"Initialized Combined Strategy with weights: MR={strategy.mr_weight}, TF={strategy.tf_weight}")
    
    return strategy

def plot_backtest_results(results, symbol_stats, filename):
    # Extract trades
    trades = results.get('trades', [])
    
    if not trades:
        logger.warning("No trades were executed in the backtest")
        return
    
    # Convert to DataFrame for analysis
    trades_df = pd.DataFrame(trades)
    
    # Check if we have the necessary columns
    logger.info(f"Trade DataFrame columns: {trades_df.columns.tolist()}")
    
    # Calculate returns if not present
    if 'return' not in trades_df.columns and 'profit_loss' in trades_df.columns and 'entry_price' in trades_df.columns:
        # Assuming standard position size of 100 shares if quantity not available
        if 'quantity' not in trades_df.columns:
            position_size = 100  # Default position size
            trades_df['return'] = trades_df['profit_loss'] / (trades_df['entry_price'] * position_size)
        else:
            trades_df['return'] = trades_df['profit_loss'] / (trades_df['entry_price'] * trades_df['quantity'])
    
    # Plot return distribution if we have return data
    if 'return' in trades_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(trades_df['return'] * 100, kde=True)
        plt.title('Trade Return Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.savefig(f'output/{filename}_return_distribution.png')
        plt.close()
        
        # Plot cumulative returns
        trades_df['cumulative_return'] = (1 + trades_df['return']).cumprod() - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(trades_df.index, trades_df['cumulative_return'] * 100)
        plt.title('Cumulative Returns')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True)
        plt.savefig(f'output/{filename}_cumulative_returns.png')
        plt.close()
    else:
        logger.warning("No return data available for plotting return distribution and cumulative returns")
    
    # Plot performance by symbol
    symbols = list(symbol_stats.keys())
    win_rates = [symbol_stats[s]['win_rate'] for s in symbols]
    avg_returns = [symbol_stats[s]['avg_return'] for s in symbols]
    
    if symbols:
        plt.figure(figsize=(14, 7))
        
        # Sort by average return
        sorted_indices = np.argsort(avg_returns)
        sorted_symbols = [symbols[i] for i in sorted_indices]
        sorted_win_rates = [win_rates[i] for i in sorted_indices]
        sorted_avg_returns = [avg_returns[i] for i in sorted_indices]
        
        plt.subplot(1, 2, 1)
        plt.barh(sorted_symbols, sorted_avg_returns)
        plt.title('Average Return by Symbol')
        plt.xlabel('Average Return (%)')
        plt.ylabel('Symbol')
        plt.grid(True, axis='x')
        
        plt.subplot(1, 2, 2)
        plt.barh(sorted_symbols, sorted_win_rates)
        plt.title('Win Rate by Symbol')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Symbol')
        plt.grid(True, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'output/{filename}_symbol_performance.png')
        plt.close()
    
    logger.info("\nAnalysis complete. Plots saved to output directory.")

def main():
    """Main function to run the combined mean reversion test"""
    logger.info("=== Starting Combined Mean Reversion Test ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test combined mean reversion strategy')
    parser.add_argument('--multi-factor', action='store_true', help='Use multi-factor stock selection')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4], default=2, help='Quarter to test (1-4)')
    args = parser.parse_args()
    
    # Set date range based on quarter
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
    
    # Initialize the strategy
    use_multi_factor = args.multi_factor
    logger.info(f"Multi-factor stock selection enabled: {use_multi_factor}")
    strategy = initialize_strategy(use_multi_factor)
    
    # Load the configuration
    config_file = 'configuration_enhanced_multi_factor_500.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs('output', exist_ok=True)
    
    # Set test parameters
    stock_selection_date = datetime.strptime(args.start, '%Y-%m-%d')  # Use start date for initial selection
    top_n = 20  # Number of stocks to select
    
    # Select top stocks
    selected_symbols = select_top_stocks(strategy, stock_selection_date, top_n)
    
    if not selected_symbols:
        logger.error("No symbols selected. Exiting.")
        return
    
    # Run mean reversion backtest
    results = run_mean_reversion_backtest(selected_symbols, args.start, args.end, config)
    
    # Analyze results
    symbol_stats = analyze_backtest_results(results)
    
    # Generate plots
    plot_backtest_results(results, symbol_stats, f'combined_mean_reversion_q{args.quarter}')
    
    logger.info("=== Combined Mean Reversion Test Completed ===")

if __name__ == "__main__":
    main()
