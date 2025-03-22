import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

class YahooFinanceDataSource:
    """Data source that fetches stock data from Yahoo Finance"""
    
    def __init__(self, config: Dict):
        """Initialize the data source with configuration"""
        self.logger = logging.getLogger("YahooFinance")
        self.cache = {}  # Cache for historical data
        self.cache_expiry = {}  # Expiry time for cached data
        self.cache_duration = config.get("cache_duration", 3600)  # Cache duration in seconds (default 1 hour)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)
        self.logger.info("Yahoo Finance data source initialized")
    
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1m") -> pd.DataFrame:
        """
        Get historical data for a symbol
        
        Parameters:
        symbol (str): Stock symbol
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
        pd.DataFrame: DataFrame with historical data
        """
        cache_key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache and current_time < self.cache_expiry.get(cache_key, 0):
            self.logger.debug(f"Using cached data for {cache_key}")
            return self.cache[cache_key]
        
        # Fetch data with retries
        for retry in range(self.max_retries):
            try:
                self.logger.info(f"Fetching historical data for {symbol}, period={period}, interval={interval}")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                # Process data
                if data.empty:
                    self.logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Rename columns to match our expected format
                data = data.rename(columns={
                    'Date': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Convert timestamp to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Sort by timestamp
                data = data.sort_values('timestamp')
                
                # Cache the result
                self.cache[cache_key] = data
                self.cache_expiry[cache_key] = current_time + self.cache_duration
                
                return data
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol} (attempt {retry+1}/{self.max_retries}): {str(e)}")
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # If we reach here, all retries failed
        self.logger.error(f"Failed to fetch data for {symbol} after {self.max_retries} attempts")
        return pd.DataFrame()
    
    def get_latest_data(self, symbol: str, lookback_days: int = 5) -> pd.DataFrame:
        """
        Get the most recent data for a symbol
        
        Parameters:
        symbol (str): Stock symbol
        lookback_days (int): Number of days to look back for intraday data
        
        Returns:
        pd.DataFrame: DataFrame with recent data
        """
        # For recent data, we want to fetch 1-minute data for the lookback period
        today = dt.datetime.now().date()
        
        # Calculate appropriate period parameter based on lookback_days
        if lookback_days <= 5:
            period = f"{lookback_days}d"
        elif lookback_days <= 30:
            period = "1mo"
        elif lookback_days <= 90:
            period = "3mo"
        elif lookback_days <= 180:
            period = "6mo"
        else:
            period = "1y"
        
        # Get data for the lookback period
        df = self.get_historical_data(symbol, period=period, interval="1m")
        
        # Filter to only include data within the lookback period
        if not df.empty:
            cutoff_date = today - dt.timedelta(days=lookback_days)
            df = df[df['timestamp'].dt.date >= cutoff_date]
        
        return df
    
    def get_daily_data(self, symbol: str, years: int = 2) -> pd.DataFrame:
        """
        Get daily data for a symbol
        
        Parameters:
        symbol (str): Stock symbol
        years (int): Number of years of history to fetch
        
        Returns:
        pd.DataFrame: DataFrame with daily data
        """
        period = f"{years}y"
        return self.get_historical_data(symbol, period=period, interval="1d")
    
    def get_live_quote(self, symbol: str) -> Dict:
        """
        Get current price quote for a symbol
        
        Parameters:
        symbol (str): Stock symbol
        
        Returns:
        Dict: Dictionary with quote information
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get information
            info = ticker.info
            
            # Add some real-time quote data if available
            quote = ticker.history(period="1d", interval="1m").iloc[-1].to_dict()
            
            result = {
                'symbol': symbol,
                'price': quote.get('Close', info.get('currentPrice', 0)),
                'change': quote.get('Close', 0) - quote.get('Open', 0),
                'percent_change': ((quote.get('Close', 0) / quote.get('Open', 0)) - 1) * 100 if quote.get('Open', 0) else 0,
                'volume': quote.get('Volume', info.get('volume', 0)),
                'timestamp': dt.datetime.now(),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'day_high': quote.get('High', info.get('dayHigh', 0)),
                'day_low': quote.get('Low', info.get('dayLow', 0))
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'price': 0,
                'change': 0,
                'percent_change': 0,
                'volume': 0,
                'timestamp': dt.datetime.now(),
                'error': str(e)
            }
    
    def get_market_index_data(self, index_symbol: str = "^GSPC", period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get data for a market index (default is S&P 500)
        
        Parameters:
        index_symbol (str): Index symbol (^GSPC for S&P 500, ^VIX for VIX)
        period (str): Data period
        interval (str): Data interval
        
        Returns:
        pd.DataFrame: DataFrame with index data
        """
        return self.get_historical_data(index_symbol, period, interval)
    
    def get_vix_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get VIX data
        
        Parameters:
        period (str): Data period
        interval (str): Data interval
        
        Returns:
        pd.DataFrame: DataFrame with VIX data
        """
        return self.get_historical_data("^VIX", period, interval)
    
    def convert_to_candle_data(self, df: pd.DataFrame) -> List:
        """
        Convert DataFrame to list of CandleData objects
        
        Parameters:
        df (pd.DataFrame): DataFrame with OHLCV data
        
        Returns:
        List: List of CandleData objects
        """
        from multi_strategy_system import CandleData
        
        candles = []
        
        for _, row in df.iterrows():
            candle = CandleData(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume'])
            )
            candles.append(candle)
        
        return candles
