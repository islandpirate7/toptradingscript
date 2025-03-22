#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Data Provider
-------------------
This module provides a wrapper around the Alpaca API for retrieving
historical market data.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Optional, Union
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaDataProvider:
    """Alpaca Data Provider for retrieving historical market data"""
    
    def __init__(self, use_paper: bool = True, credentials_file: str = 'alpaca_credentials.json'):
        """
        Initialize the Alpaca Data Provider.
        
        Args:
            use_paper (bool): Whether to use paper trading API
            credentials_file (str): Path to credentials file
        """
        self.use_paper = use_paper
        self.credentials = self._load_credentials(credentials_file)
        
        # Initialize clients
        self._init_clients()
        
        logger.info(f"Initialized Alpaca Data Provider with {'paper' if use_paper else 'live'} trading API")
    
    def _load_credentials(self, credentials_file: str) -> Dict:
        """
        Load credentials from JSON file.
        
        Args:
            credentials_file (str): Path to credentials file
            
        Returns:
            dict: Credentials dictionary
        """
        try:
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)
            
            # Validate credentials
            if self.use_paper and 'paper' not in credentials:
                raise ValueError("Paper trading credentials not found in credentials file")
            elif not self.use_paper and 'live' not in credentials:
                raise ValueError("Live trading credentials not found in credentials file")
            
            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            raise
    
    def _init_clients(self):
        """Initialize Alpaca clients"""
        try:
            # Get credentials based on environment
            creds = self.credentials['paper'] if self.use_paper else self.credentials['live']
            
            # Initialize historical data client
            self.stock_historical_client = StockHistoricalDataClient(
                api_key=creds['api_key'],
                secret_key=creds['api_secret']
            )
            
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=creds['api_key'],
                secret_key=creds['api_secret'],
                paper=self.use_paper
            )
            
            logger.info("Alpaca clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Alpaca clients: {str(e)}")
            raise
    
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D',
        adjustment: str = 'all'
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol (str): Symbol to get data for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            timeframe (str): Timeframe (1Min, 5Min, 15Min, 30Min, 1H, 1D, 1W)
            adjustment (str): Adjustment type (raw, split, dividend, all)
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        try:
            # Convert dates to datetime objects
            start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
            
            # Add one day to end_date to include it in the results
            end_dt = end_dt + dt.timedelta(days=1)
            
            # Map timeframe string to TimeFrame object
            if timeframe == '1Min':
                tf = TimeFrame.Minute
                multiplier = 1
            elif timeframe == '5Min':
                tf = TimeFrame.Minute
                multiplier = 5
            elif timeframe == '15Min':
                tf = TimeFrame.Minute
                multiplier = 15
            elif timeframe == '30Min':
                tf = TimeFrame.Minute
                multiplier = 30
            elif timeframe == '1H':
                tf = TimeFrame.Hour
                multiplier = 1
            elif timeframe == '1D':
                tf = TimeFrame.Day
                multiplier = 1
            elif timeframe == '1W':
                tf = TimeFrame.Week
                multiplier = 1
            else:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_dt,
                end=end_dt,
                adjustment=adjustment
            )
            
            # Get data
            bars = self.stock_historical_client.get_stock_bars(request_params)
            
            # Check if data is available
            if bars is None or bars.df.empty:
                logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                return None
            
            # Convert to DataFrame
            df = bars.df
            
            # If multi-index with symbol, get only the data for this symbol
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'trade_count': 'trade_count',
                'vwap': 'vwap'
            })
            
            # Add additional columns
            df['symbol'] = symbol
            
            # Create a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Return data
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
    
    def get_tradable_assets(self, asset_class: str = 'us_equity') -> List[str]:
        """
        Get list of tradable assets.
        
        Args:
            asset_class (str): Asset class (us_equity, crypto, etc.)
            
        Returns:
            list: List of tradable asset symbols
        """
        try:
            # Map asset class string to enum
            asset_class_map = {
                'us_equity': AssetClass.US_EQUITY,
                'crypto': AssetClass.CRYPTO
            }
            
            if asset_class not in asset_class_map:
                raise ValueError(f"Invalid asset class: {asset_class}")
            
            # Create request
            request_params = GetAssetsRequest(
                asset_class=asset_class_map[asset_class],
                status=AssetStatus.ACTIVE
            )
            
            # Get assets
            assets = self.trading_client.get_all_assets(request_params)
            
            # Extract symbols
            symbols = [asset.symbol for asset in assets if asset.tradable]
            
            logger.info(f"Retrieved {len(symbols)} tradable {asset_class} assets")
            return symbols
        except Exception as e:
            logger.error(f"Error getting tradable assets: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        try:
            # Get account
            account = self.trading_client.get_account()
            
            # Convert to dictionary
            account_dict = {
                'id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'status': account.status
            }
            
            logger.info(f"Retrieved account information: ID={account.id}, Equity=${account.equity}")
            return account_dict
        except Exception as e:
            logger.error(f"Error getting account information: {e}")
            return {}
