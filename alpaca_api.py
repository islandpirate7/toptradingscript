import alpaca_trade_api as tradeapi
import logging
import os
import pandas as pd
import numpy as np
import random
from datetime import timedelta

logger = logging.getLogger(__name__)

class AlpacaAPI:
    """
    Wrapper class for Alpaca API to standardize interactions with the Alpaca platform.
    """
    
    def __init__(self, api_key, api_secret, base_url, data_url=None):
        """
        Initialize the Alpaca API wrapper.
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            base_url (str): Alpaca API base URL (paper or live)
            data_url (str, optional): Alpaca Data API URL for market data
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.data_url = data_url or 'https://data.alpaca.markets'
        
        # Initialize the Alpaca API client for trading
        self.api = tradeapi.REST(
            api_key,
            api_secret,
            base_url,
            api_version='v2'
        )
        
        # Initialize a separate API client for market data if data_url is provided
        self.data_api = tradeapi.REST(
            api_key,
            api_secret,
            self.data_url,
            api_version='v2'
        )
        
        logger.info(f"AlpacaAPI initialized with base URL: {base_url}")
        logger.info(f"AlpacaAPI initialized with data URL: {self.data_url}")
    
    def get_account(self):
        """Get account information."""
        try:
            return self.api.get_account()
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            return None
    
    def get_positions(self):
        """Get current positions."""
        try:
            return self.api.list_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self, status=None):
        """Get orders with optional status filter."""
        try:
            if status:
                return self.api.list_orders(status=status)
            else:
                return self.api.list_orders()
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_bars(self, symbols, timeframe, start, end):
        """
        Get historical price bars for a list of symbols.
        
        Args:
            symbols (list): List of symbols to get data for
            timeframe (str): Timeframe for the bars (e.g., '1D', '1H')
            start (pd.Timestamp): Start date/time
            end (pd.Timestamp): End date/time
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data or None if data cannot be retrieved
        """
        try:
            # Convert timeframe to Alpaca format
            if timeframe == '1D':
                timeframe = tradeapi.TimeFrame.Day
            elif timeframe == '1H':
                timeframe = tradeapi.TimeFrame.Hour
            elif timeframe == '15Min':
                timeframe = tradeapi.TimeFrame.Minute
                timeframe = timeframe._replace(value=15)
            elif timeframe == '5Min':
                timeframe = tradeapi.TimeFrame.Minute
                timeframe = timeframe._replace(value=5)
            elif timeframe == '1Min':
                timeframe = tradeapi.TimeFrame.Minute
            
            # Format dates properly for Alpaca API
            if isinstance(start, str):
                start = pd.Timestamp(start)
            if isinstance(end, str):
                end = pd.Timestamp(end)
                
            # Convert to the format expected by Alpaca API (YYYY-MM-DD)
            # The error shows that Alpaca doesn't want the time part
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            logger.info(f"Requesting bars for {len(symbols)} symbols from {start_str} to {end_str}")
            
            # Get bars from Alpaca Data API
            bars = self.data_api.get_bars(
                symbols,
                timeframe,
                start=start_str,
                end=end_str,
                adjustment='raw'
            ).df
            
            logger.info(f"Retrieved {len(bars) if bars is not None else 0} bars for {len(symbols)} symbols")
            return bars
            
        except Exception as e:
            logger.error(f"Error getting bars: {str(e)}")
            return None
    
    def get_asset(self, symbol):
        """Get asset information."""
        try:
            return self.api.get_asset(symbol)
        except Exception as e:
            logger.error(f"Error getting asset information for {symbol}: {str(e)}")
            return None
    
    def get_clock(self):
        """Get market clock information."""
        try:
            return self.api.get_clock()
        except Exception as e:
            logger.error(f"Error getting market clock: {str(e)}")
            return None
    
    def get_calendar(self, start=None, end=None):
        """Get market calendar information."""
        try:
            return self.api.get_calendar(start=start, end=end)
        except Exception as e:
            logger.error(f"Error getting market calendar: {str(e)}")
            return []

    def submit_order(self, symbol, qty, side, type='market', time_in_force='day', limit_price=None, stop_price=None):
        """Submit an order to Alpaca."""
        try:
            return self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {str(e)}")
            return None
