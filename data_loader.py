#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader module for fetching historical market data from various sources.
Currently supports Alpaca API for historical data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AlpacaDataLoader:
    """Data loader for Alpaca API"""
    
    def __init__(self, api):
        """
        Initialize the Alpaca data loader.
        
        Args:
            api: Alpaca API instance
        """
        self.api = api
        logger.info("Initialized Alpaca data loader")
    
    def load_historical_data(self, symbol, start, end, timeframe='1D'):
        """
        Load historical data from Alpaca API.
        
        Args:
            symbol (str): Symbol to load data for
            start (str): Start date in YYYY-MM-DD format
            end (str): End date in YYYY-MM-DD format
            timeframe (str): Timeframe for the data (e.g., '1D', '1H')
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        try:
            logger.info(f"Loading {timeframe} data for {symbol} from {start} to {end}")
            
            # Convert string dates to datetime if needed
            if isinstance(start, str):
                start = pd.Timestamp(start).date()
            if isinstance(end, str):
                end = pd.Timestamp(end).date()
            
            # Fetch bars from Alpaca
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                adjustment='raw'
            ).df
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol} from {start} to {end}")
                return None
            
            # Reset index to make timestamp a column
            bars = bars.reset_index()
            
            # Rename columns to match our standard format
            bars = bars.rename(columns={
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Calculate additional indicators
            self._calculate_indicators(bars)
            
            logger.info(f"Loaded {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for the data.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        try:
            # Calculate returns
            df['daily_return'] = df['close'].pct_change()
            
            # Calculate moving averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['sma_20']
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Calculate volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def load_latest_data(self, symbols, lookback_days=100, timeframe='1D'):
        """
        Load the latest data for a list of symbols.
        
        Args:
            symbols (list): List of symbols to load data for
            lookback_days (int): Number of days to look back
            timeframe (str): Timeframe for the data
            
        Returns:
            dict: Dictionary of DataFrames with historical data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        data = {}
        for symbol in symbols:
            df = self.load_historical_data(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                timeframe=timeframe
            )
            
            if df is not None and not df.empty:
                data[symbol] = df
        
        return data
