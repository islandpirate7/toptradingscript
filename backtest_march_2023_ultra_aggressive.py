#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest with Ultra Aggressive Settings for March 2023
--------------------------------------
This script runs a backtest with ultra aggressive settings
to generate many more trading signals for March 2023 (historical data).
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import json
from tabulate import tabulate

from backtest_combined_strategy import Backtester
from combined_strategy import CombinedStrategy
from mean_reversion_strategy_ultra_aggressive import MeanReversionStrategyUltraAggressive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraAggressiveBacktester(Backtester):
    """Backtester for ultra aggressive strategies"""
    
    def __init__(self, config_file=None):
        """Initialize the ultra aggressive backtester
        
        Args:
            config_file (str, optional): Path to configuration file. Defaults to None.
        """
        # Call parent constructor
        super().__init__(config_file)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize ultra aggressive strategy
        self.ultra_aggressive_strategy = MeanReversionStrategyUltraAggressive(self.config)
        
    def apply_ultra_aggressive_settings(self):
        """Apply ultra aggressive settings to the strategy"""
        self.logger.info("Applying ultra aggressive settings to generate many more signals")
        
        # Get the combined strategy
        combined_strategy = self.strategy
        
        # Apply ultra aggressive settings to mean reversion strategy
        mr_strategy = combined_strategy.mean_reversion
        
        # Ultra aggressive settings - slightly more balanced
        mr_strategy.bb_period = 5      # Short but not too short
        mr_strategy.bb_std = 0.5       # Tight but not too tight
        mr_strategy.rsi_period = 3     # Short but not too short
        mr_strategy.rsi_overbought = 60  # More realistic threshold
        mr_strategy.rsi_oversold = 40    # More realistic threshold
        mr_strategy.require_reversal = False  # Don't require price reversal
        mr_strategy.min_bb_penetration = 0.0  # No penetration required
        mr_strategy.use_volume_filter = True  # Enable volume filter for better signals
        
        # Apply ultra aggressive settings to trend following strategy
        tf_strategy = combined_strategy.trend_following
        
        # Ultra aggressive settings - slightly more balanced
        tf_strategy.ema_short = 5      # Short but not too short
        tf_strategy.ema_long = 10      # Short but not too short
        tf_strategy.atr_period = 5     # Short but not too short
        tf_strategy.rsi_period = 3     # Short but not too short
        tf_strategy.rsi_overbought = 60  # More realistic threshold
        tf_strategy.rsi_oversold = 40    # More realistic threshold
        
        # Set low min signal score to allow most signals
        self.min_signal_score = 0.1
        
        # Update combined strategy settings
        combined_strategy.min_signal_score = 0.1
        
        self.logger.info(f"Ultra aggressive settings applied: BB period={mr_strategy.bb_period}, BB std={mr_strategy.bb_std}, "
                        f"RSI period={mr_strategy.rsi_period}, RSI thresholds={mr_strategy.rsi_oversold}/{mr_strategy.rsi_overbought}, "
                        f"Min signal score={self.min_signal_score}")
    
    def is_same_day(self, date1, date2):
        """Check if two dates are on the same day
        
        Args:
            date1 (datetime): First date
            date2 (datetime): Second date
            
        Returns:
            bool: True if dates are on the same day, False otherwise
        """
        if isinstance(date1, pd.Timestamp):
            date1 = date1.to_pydatetime()
        if isinstance(date2, pd.Timestamp):
            date2 = date2.to_pydatetime()
            
        return (date1.year == date2.year and 
                date1.month == date2.month and 
                date1.day == date2.day)
    
    def run_backtest(self, start_date, end_date):
        """Run backtest with ultra aggressive settings
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            BacktestResults: Backtest results
        """
        # Apply ultra aggressive settings
        self.apply_ultra_aggressive_settings()
        
        # Run backtest
        return super().run_backtest(start_date, end_date)
    
    def prioritize_signals_by_volatility(self, signals):
        """Prioritize signals by volatility (ATR)
        
        Args:
            signals (list): List of signals
            
        Returns:
            list: Prioritized signals
        """
        if not signals or len(signals) <= self.max_positions:
            return signals
        
        # Sort by ATR (higher ATR first)
        sorted_signals = sorted(signals, key=lambda x: x.get('atr', 0), reverse=True)
        
        # Take only the top max_positions signals
        prioritized_signals = sorted_signals[:self.max_positions]
        
        self.logger.info(f"Prioritized {len(prioritized_signals)} signals out of {len(signals)} based on volatility")
        
        return prioritized_signals
    
    def generate_signals(self, df, symbol):
        """Generate signals with volatility prioritization
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            symbol (str): Symbol to generate signals for
            
        Returns:
            list: List of signals
        """
        # Generate signals using the strategy
        signals = self.strategy.generate_signals(df, symbol)
        
        # Prioritize signals by volatility
        if len(signals) > self.max_positions:
            signals = self.prioritize_signals_by_volatility(signals)
        
        return signals
    
    def get_historical_data(self, symbol, start_date, end_date):
        """Get historical data for a symbol from pre-fetched data
        
        Args:
            symbol (str): Symbol to get data for
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pd.DataFrame: Historical data
        """
        if symbol in self.data and self.data[symbol] is not None and not self.data[symbol].empty:
            return self.data[symbol]
        return None
    
    def run_march_2023_backtest(self, config_file):
        """Run a backtest for March 2023 with ultra aggressive settings
        
        Args:
            config_file (str): Path to configuration file
        """
        start_date = dt.datetime(2023, 3, 1)
        end_date = dt.datetime(2023, 3, 31)
        
        # Use ultra aggressive backtester
        logger.info(f"Running ultra aggressive backtest for March 2023")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        backtester = UltraAggressiveBacktester(config_file)
        
        # Enable verbose logging for debugging
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Add a handler to log to a file
        file_handler = logging.FileHandler('backtest_debug.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # Add detailed logging to the mean reversion strategy
        original_generate_signals = backtester.strategy.mean_reversion.generate_signals
        
        def generate_signals_with_logging(df):
            logger.debug(f"Generating signals for dataframe with {len(df)} rows")
            logger.debug(f"First few rows of data: {df.head(3)}")
            logger.debug(f"Last few rows of data: {df.tail(3)}")
            
            # Calculate indicators
            df_with_indicators = backtester.strategy.mean_reversion.calculate_indicators(df)
            
            # Log indicator values
            logger.debug(f"Indicator values for last 3 rows:")
            for col in ['close', 'sma', 'bb_upper', 'bb_lower', 'rsi']:
                if col in df_with_indicators.columns:
                    logger.debug(f"{col}: {df_with_indicators[col].tail(3).values}")
            
            # Call original function
            signals = original_generate_signals(df)
            logger.debug(f"Generated {len(signals)} signals")
            
            # Log signal details
            if signals:
                for i, signal in enumerate(signals[:5]):  # Log first 5 signals
                    logger.debug(f"Signal {i+1}: {signal}")
            
            return signals
        
        # Replace the generate_signals method with our logging version
        backtester.strategy.mean_reversion.generate_signals = generate_signals_with_logging
        
        # Run the backtest
        results = backtester.run_backtest(start_date, end_date)
        
        # Reset logging level
        logging.getLogger().setLevel(logging.INFO)
        
        # Display trade details if there are any trades
        if results and hasattr(results, 'trades') and results.trades:
            logger.info(f"Found {len(results.trades)} trades in the backtest results")
            
            # Create a DataFrame from the trades
            trades_df = pd.DataFrame(results.trades)
            
            # Format the DataFrame for display
            if not trades_df.empty:
                # Convert timestamps to readable format
                if 'entry_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d')
                if 'exit_time' in trades_df.columns:
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d')
                
                # Format numeric columns
                if 'profit_pct' in trades_df.columns:
                    trades_df['profit_pct'] = trades_df['profit_pct'].map('{:.2f}%'.format)
                if 'profit_usd' in trades_df.columns:
                    trades_df['profit_usd'] = trades_df['profit_usd'].map('${:.2f}'.format)
                
                # Select relevant columns for display
                display_columns = ['symbol', 'direction', 'entry_time', 'exit_time', 'profit_pct', 'profit_usd', 'exit_reason']
                display_columns = [col for col in display_columns if col in trades_df.columns]
                
                # Display the trades
                print("\nTrade Details:")
                print(tabulate(trades_df[display_columns], headers='keys', tablefmt='grid'))
                
                # Calculate and display summary statistics
                print("\nTrade Summary:")
                win_rate = (trades_df['profit_usd'].str.replace('$', '').astype(float) > 0).mean() * 100
                total_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).sum()
                avg_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).mean()
                
                print(f"Total Trades: {len(trades_df)}")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Total Profit: ${total_profit:.2f}")
                print(f"Average Profit per Trade: ${avg_profit:.2f}")
        else:
            logger.warning("No trades found in the backtest results")
            
            # Check if we have any signals at all
            logger.info("Checking for signals in the raw data...")
            
            # Get the configuration
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get symbols from config
            symbols = config.get('general', {}).get('symbols', [])
            
            # Check for signals in each symbol
            for symbol in symbols:
                # Get historical data
                import alpaca_trade_api as tradeapi
                
                # Load credentials
                with open('alpaca_credentials.json', 'r') as f:
                    credentials = json.load(f)
                
                # Get paper trading credentials
                paper_credentials = credentials.get('paper', {})
                api_key = paper_credentials.get('api_key', '')
                api_secret = paper_credentials.get('api_secret', '')
                base_url = paper_credentials.get('base_url', 'https://paper-api.alpaca.markets')
                
                # Create API client
                api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
                
                # Get historical data
                try:
                    bars = api.get_bars(
                        symbol,
                        '1D',
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        adjustment='raw'
                    ).df
                    
                    if not bars.empty:
                        # Reset index
                        bars = bars.reset_index()
                        
                        # Create ultra aggressive strategy
                        strategy = MeanReversionStrategyUltraAggressive()
                        
                        # Generate signals
                        signals = strategy.generate_signals(bars)
                        
                        if signals:
                            logger.info(f"Found {len(signals)} signals for {symbol}")
                            for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                                logger.info(f"Signal {i+1}: {signal}")
                        else:
                            logger.info(f"No signals found for {symbol}")
                except Exception as e:
                    logger.error(f"Error checking signals for {symbol}: {e}")
    
    def run_q1_2023_backtest(self, config_file):
        """Run a backtest for Q1 2023 with ultra aggressive settings
        
        Args:
            config_file (str): Path to configuration file
        """
        start_date = dt.datetime(2023, 1, 1)
        end_date = dt.datetime(2023, 3, 31)
        
        # Use ultra aggressive backtester
        logger.info(f"Running ultra aggressive backtest for Q1 2023")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        backtester = UltraAggressiveBacktester(config_file)
        results = backtester.run_backtest(start_date, end_date)
        
        # Display trade details if there are any trades
        if results and hasattr(results, 'trades') and results.trades:
            logger.info(f"Found {len(results.trades)} trades in the backtest results")
            
            # Create a DataFrame from the trades
            trades_df = pd.DataFrame(results.trades)
            
            # Format the DataFrame for display
            if not trades_df.empty:
                # Convert timestamps to readable format
                if 'entry_time' in trades_df.columns:
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d')
                if 'exit_time' in trades_df.columns:
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d')
                
                # Format numeric columns
                if 'profit_pct' in trades_df.columns:
                    trades_df['profit_pct'] = trades_df['profit_pct'].map('{:.2f}%'.format)
                if 'profit_usd' in trades_df.columns:
                    trades_df['profit_usd'] = trades_df['profit_usd'].map('${:.2f}'.format)
                
                # Select relevant columns for display
                display_columns = ['symbol', 'direction', 'entry_time', 'exit_time', 'profit_pct', 'profit_usd', 'exit_reason']
                display_columns = [col for col in display_columns if col in trades_df.columns]
                
                # Display the trades
                print("\nTrade Details:")
                print(tabulate(trades_df[display_columns], headers='keys', tablefmt='grid'))
                
                # Calculate and display summary statistics
                print("\nTrade Summary:")
                win_rate = (trades_df['profit_usd'].str.replace('$', '').astype(float) > 0).mean() * 100
                total_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).sum()
                avg_profit = trades_df['profit_usd'].str.replace('$', '').astype(float).mean()
                
                print(f"Total Trades: {len(trades_df)}")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Total Profit: ${total_profit:.2f}")
                print(f"Average Profit per Trade: ${avg_profit:.2f}")
        else:
            logger.warning("No trades found in the backtest results")
    
    if __name__ == "__main__":
        import argparse
        
        parser = argparse.ArgumentParser(description='Run ultra aggressive backtest for 2023')
        parser.add_argument('--config', type=str, default='configuration_combined_strategy_march_seasonal.yaml',
                            help='Path to configuration file')
        parser.add_argument('--period', type=str, choices=['march', 'q1', 'all'], default='march',
                            help='Period to backtest (march, q1, or all)')
        
        args = parser.parse_args()
        
        # Run backtest
        if args.period == 'march':
            UltraAggressiveBacktester.run_march_2023_backtest(UltraAggressiveBacktester(args.config), args.config)
        elif args.period == 'q1':
            UltraAggressiveBacktester.run_q1_2023_backtest(UltraAggressiveBacktester(args.config), args.config)
        elif args.period == 'all':
            UltraAggressiveBacktester.run_march_2023_backtest(UltraAggressiveBacktester(args.config), args.config)
            UltraAggressiveBacktester.run_q1_2023_backtest(UltraAggressiveBacktester(args.config), args.config)
