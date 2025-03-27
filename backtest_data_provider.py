#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Data Provider
---------------------
This module provides historical market data for backtesting purposes.
It can either fetch data from Alpaca API or use local CSV files.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import glob

logger = logging.getLogger(__name__)

class BacktestDataProvider:
    """
    Provides historical market data for backtesting
    """
    
    def __init__(self, data_dir='./data', use_local_data=True):
        """
        Initialize the data provider
        
        Args:
            data_dir (str): Directory for storing/loading data
            use_local_data (bool): Whether to use local CSV files
        """
        self.data_dir = data_dir
        self.use_local_data = use_local_data
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Cache for loaded data
        self.data_cache = {}
    
    def get_bars(self, symbols, timeframe, start, end, alpaca=None):
        """
        Get historical price bars for a list of symbols
        
        Args:
            symbols (list): List of symbols to get data for
            timeframe (str): Timeframe for the bars (e.g., '1D', '1H')
            start (pd.Timestamp): Start date/time
            end (pd.Timestamp): End date/time
            alpaca (AlpacaAPI, optional): Alpaca API instance to use if local data not available
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if self.use_local_data:
            # Try to load data from local CSV files
            all_dfs = []
            missing_symbols = []
            
            for symbol in symbols:
                df = self._load_local_data(symbol, start, end)
                if df is not None:
                    # Add symbol as a column
                    df['symbol'] = symbol
                    all_dfs.append(df)
                else:
                    missing_symbols.append(symbol)
            
            # If we have missing symbols and alpaca is provided, try to fetch them
            if missing_symbols and alpaca:
                try:
                    logger.info(f"Fetching data for {len(missing_symbols)} symbols from Alpaca")
                    # Fetch data from Alpaca
                    alpaca_df = alpaca.get_bars(missing_symbols, timeframe, start, end)
                    if alpaca_df is not None and not alpaca_df.empty:
                        # Save data to local CSV files
                        for symbol in missing_symbols:
                            symbol_df = alpaca_df.loc[symbol]
                            if not symbol_df.empty:
                                self._save_local_data(symbol, symbol_df)
                                all_dfs.append(symbol_df)
                except Exception as e:
                    logger.error(f"Error fetching data from Alpaca: {str(e)}")
                    # Generate synthetic data for missing symbols
                    for symbol in missing_symbols:
                        df = self._generate_synthetic_data(symbol, start, end)
                        df['symbol'] = symbol
                        all_dfs.append(df)
                        self._save_local_data(symbol, df)
            elif missing_symbols:
                # Generate synthetic data for missing symbols
                for symbol in missing_symbols:
                    df = self._generate_synthetic_data(symbol, start, end)
                    df['symbol'] = symbol
                    all_dfs.append(df)
                    self._save_local_data(symbol, df)
            
            # Combine all DataFrames
            if all_dfs:
                combined_df = pd.concat(all_dfs)
                
                # Set multi-index
                combined_df = combined_df.set_index(['symbol', 'timestamp'])
                
                return combined_df
            else:
                return pd.DataFrame()
        elif alpaca:
            # Use Alpaca API directly
            try:
                return alpaca.get_bars(symbols, timeframe, start, end)
            except Exception as e:
                logger.error(f"Error fetching data from Alpaca: {str(e)}")
                # Fall back to synthetic data
                return self._get_synthetic_bars(symbols, start, end)
        else:
            # No Alpaca API and no local data, generate synthetic data
            return self._get_synthetic_bars(symbols, start, end)
    
    def _load_local_data(self, symbol, start, end):
        """
        Load data from a local CSV file
        
        Args:
            symbol (str): Symbol to load data for
            start (pd.Timestamp): Start date/time
            end (pd.Timestamp): End date/time
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data or None if file not found
        """
        # Check if data is in cache
        if symbol in self.data_cache:
            df = self.data_cache[symbol]
            # Filter by date range
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            return df[mask].copy()
        
        # Check if CSV file exists
        csv_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            try:
                # Load data from CSV
                df = pd.read_csv(csv_path)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by date range
                mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
                filtered_df = df[mask].copy()
                
                # Cache the full dataframe
                self.data_cache[symbol] = df
                
                return filtered_df
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                return None
        else:
            return None
    
    def _save_local_data(self, symbol, df):
        """
        Save data to a local CSV file
        
        Args:
            symbol (str): Symbol to save data for
            df (pd.DataFrame): DataFrame with OHLCV data
        """
        try:
            # Ensure df has a timestamp column
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            elif 'timestamp' not in df.columns:
                df = df.reset_index()
            
            # Save to CSV
            csv_path = os.path.join(self.data_dir, f"{symbol}.csv")
            df.to_csv(csv_path, index=False)
            
            # Update cache
            self.data_cache[symbol] = df
            
            logger.info(f"Saved data for {symbol} to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
    
    def _generate_synthetic_data(self, symbol, start, end, seed=None):
        """
        Generate synthetic price data for a symbol
        
        Args:
            symbol (str): Symbol to generate data for
            start (pd.Timestamp): Start date/time
            end (pd.Timestamp): End date/time
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if seed is None:
            # Use symbol hash as seed for consistency
            seed = hash(symbol) % 10000
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Add 60 days of history to ensure we have enough data for indicators
        extended_start = start - pd.Timedelta(days=60)
        
        # Generate date range
        date_range = pd.date_range(start=extended_start, end=end, freq='D')
        
        # Filter out weekends
        date_range = date_range[date_range.dayofweek < 5]
        
        # Generate random price data
        base_price = random.uniform(10, 500)  # Random starting price
        volatility = random.uniform(0.01, 0.05)  # Random volatility
        
        # Generate daily returns with slight upward bias
        daily_returns = np.random.normal(0.0005, volatility, size=len(date_range))
        
        # Calculate price series
        prices = [base_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]  # Remove the initial base price
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': [p * (1 + random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - random.uniform(0, 0.02)) for p in prices],
            'close': [p * (1 + random.uniform(-0.01, 0.01)) for p in prices],
            'volume': [int(random.uniform(100000, 10000000)) for _ in range(len(prices))]
        })
        
        return df
    
    def _get_synthetic_bars(self, symbols, start, end):
        """
        Get synthetic bars for multiple symbols
        
        Args:
            symbols (list): List of symbols to get data for
            start (pd.Timestamp): Start date/time
            end (pd.Timestamp): End date/time
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        all_dfs = []
        
        for symbol in symbols:
            # Generate synthetic data for this symbol
            df = self._generate_synthetic_data(symbol, start, end)
            
            # Add symbol as a column
            df['symbol'] = symbol
            
            all_dfs.append(df)
        
        # Combine all DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs)
            
            # Set multi-index
            combined_df = combined_df.set_index(['symbol', 'timestamp'])
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def list_available_symbols(self):
        """
        List all symbols that have local data available
        
        Returns:
            list: List of symbols
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        symbols = [os.path.basename(f).replace(".csv", "") for f in csv_files]
        return symbols
