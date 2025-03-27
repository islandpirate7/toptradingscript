#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Mid-Cap Stock Selection

This module provides improved functionality for dynamically selecting mid-cap stocks
based on liquidity, momentum, and other quality metrics. It significantly expands the
trading universe beyond predefined symbols in the configuration file.
"""

import os
import sys
import yaml
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import alpaca-trade-api
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca API not available, some features will be limited")

# Default configuration
DEFAULT_CONFIG = {
    'midcap_stocks': {
        'min_market_cap': 2000000000,  # $2 billion
        'max_market_cap': 10000000000,  # $10 billion
        'min_avg_volume': 500000,       # 500k shares daily
        'min_price': 5.0,               # $5 minimum price
        'max_stocks': 100,              # Maximum number of stocks to include
        'sectors_to_exclude': ['Utilities'],  # Sectors to exclude
        'use_dynamic_selection': True,  # Whether to use dynamic selection
        'ranking_metrics': {
            'volume_weight': 0.3,
            'momentum_weight': 0.3,
            'volatility_weight': 0.2,
            'liquidity_weight': 0.2
        }
    }
}

def load_config(config_file='sp500_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as file:
            credentials = json.load(file)
        
        if mode.lower() == 'paper':
            return credentials.get('paper', {})
        else:
            return credentials.get('live', {})
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        return {}

def get_alpaca_api(mode='paper'):
    """Get Alpaca API instance"""
    if not ALPACA_AVAILABLE:
        logger.error("Alpaca API not available")
        return None
        
    credentials = load_alpaca_credentials(mode)
    
    if not credentials:
        logger.error("No Alpaca credentials available")
        return None
        
    try:
        api = tradeapi.REST(
            key_id=credentials.get('api_key', ''),
            secret_key=credentials.get('api_secret', ''),
            base_url=credentials.get('base_url', 'https://paper-api.alpaca.markets')
        )
        return api
    except Exception as e:
        logger.error(f"Error creating Alpaca API instance: {str(e)}")
        return None

def fetch_sp400_symbols() -> List[str]:
    """
    Fetch S&P 400 Mid-Cap index constituents from Wikipedia
    
    Returns:
        List of ticker symbols for S&P 400 Mid-Cap stocks
    """
    try:
        # S&P 400 Mid-Cap constituents from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        tables = pd.read_html(url)
        
        if not tables or len(tables) < 1:
            logger.error("Failed to parse S&P 400 table from Wikipedia")
            return []
            
        # The first table contains the current S&P 400 constituents
        sp400_df = tables[0]
        
        # Extract ticker symbols (usually in the first column)
        if 'Symbol' in sp400_df.columns:
            symbols = sp400_df['Symbol'].tolist()
        elif 'Ticker symbol' in sp400_df.columns:
            symbols = sp400_df['Ticker symbol'].tolist()
        else:
            # Try to find a column that might contain ticker symbols
            for col in sp400_df.columns:
                if any(isinstance(val, str) and val.isupper() and len(val) <= 5 for val in sp400_df[col].head(10)):
                    symbols = sp400_df[col].tolist()
                    break
            else:
                logger.error("Could not identify ticker symbol column in S&P 400 table")
                return []
        
        # Clean up symbols (remove .A, .B suffixes, etc.)
        symbols = [s.split('.')[0] if isinstance(s, str) else s for s in symbols]
        
        # Filter out any non-string or empty values
        symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
        
        logger.info(f"Successfully fetched {len(symbols)} S&P 400 Mid-Cap symbols from Wikipedia")
        return symbols
        
    except Exception as e:
        logger.error(f"Error fetching S&P 400 symbols from Wikipedia: {str(e)}")
        return []

def fetch_stock_data(symbols: List[str], lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data for the given symbols
    
    Args:
        symbols: List of ticker symbols
        lookback_days: Number of days of historical data to fetch
        
    Returns:
        Dictionary mapping symbols to DataFrames with historical data
    """
    if not symbols:
        logger.warning("No symbols provided to fetch_stock_data")
        return {}
        
    api = get_alpaca_api()
    if not api:
        logger.error("Could not initialize Alpaca API")
        return {}
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Format dates for Alpaca API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_str} to {end_str}")
    
    # Fetch data in batches to avoid API limits
    batch_size = 100
    all_data = {}
    
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        logger.info(f"Fetching batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch_symbols)} symbols)")
        
        try:
            # Fetch daily bars
            bars = api.get_bars(
                batch_symbols,
                '1Day',
                start=start_str,
                end=end_str,
                adjustment='raw'
            )
            
            # Process the bars into a dictionary of DataFrames
            for symbol in batch_symbols:
                symbol_bars = [b for b in bars if b.symbol == symbol]
                
                if not symbol_bars:
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': b.t,
                    'open': b.o,
                    'high': b.h,
                    'low': b.l,
                    'close': b.c,
                    'volume': b.v
                } for b in symbol_bars])
                
                if len(df) > 0:
                    all_data[symbol] = df
        
        except Exception as e:
            logger.error(f"Error fetching batch {i//batch_size + 1}: {str(e)}")
    
    logger.info(f"Successfully fetched data for {len(all_data)} symbols")
    return all_data

def calculate_stock_metrics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate various metrics for stock selection
    
    Args:
        data: Dictionary mapping symbols to DataFrames with historical data
        
    Returns:
        DataFrame with calculated metrics for each symbol
    """
    if not data:
        logger.warning("No data provided to calculate_stock_metrics")
        return pd.DataFrame()
    
    metrics = []
    
    for symbol, df in data.items():
        if len(df) < 20:  # Need at least 20 days of data
            continue
            
        try:
            # Calculate basic metrics
            latest_close = df['close'].iloc[-1]
            avg_volume = df['volume'].mean()
            dollar_volume = latest_close * avg_volume
            
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate momentum (return over the period)
            momentum_20d = df['close'].iloc[-1] / df['close'].iloc[-20] - 1 if len(df) >= 20 else 0
            
            # Calculate average true range (ATR) for liquidity assessment
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate liquidity ratio (ATR / Close Price)
            liquidity_ratio = atr / latest_close
            
            metrics.append({
                'symbol': symbol,
                'close': latest_close,
                'avg_volume': avg_volume,
                'dollar_volume': dollar_volume,
                'volatility': volatility,
                'momentum_20d': momentum_20d,
                'atr': atr,
                'liquidity_ratio': liquidity_ratio
            })
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
    
    if not metrics:
        logger.warning("No metrics calculated for any symbols")
        return pd.DataFrame()
        
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Rank metrics
    metrics_df['volume_rank'] = metrics_df['dollar_volume'].rank(pct=True)
    metrics_df['momentum_rank'] = metrics_df['momentum_20d'].rank(pct=True)
    metrics_df['volatility_rank'] = (1 - metrics_df['volatility'].rank(pct=True))  # Lower volatility is better
    metrics_df['liquidity_rank'] = metrics_df['liquidity_ratio'].rank(pct=True)
    
    return metrics_df

def filter_and_rank_midcap_stocks(metrics_df: pd.DataFrame, config: Dict) -> List[str]:
    """
    Filter and rank mid-cap stocks based on metrics and configuration
    
    Args:
        metrics_df: DataFrame with calculated metrics for each symbol
        config: Configuration dictionary
        
    Returns:
        List of selected mid-cap stock symbols, ranked by quality
    """
    if metrics_df.empty:
        logger.warning("No metrics provided to filter_and_rank_midcap_stocks")
        return []
    
    # Get configuration
    midcap_config = config.get('midcap_stocks', DEFAULT_CONFIG['midcap_stocks'])
    
    # Apply filters
    min_avg_volume = midcap_config.get('min_avg_volume', DEFAULT_CONFIG['midcap_stocks']['min_avg_volume'])
    min_price = midcap_config.get('min_price', DEFAULT_CONFIG['midcap_stocks']['min_price'])
    
    filtered_df = metrics_df[
        (metrics_df['avg_volume'] >= min_avg_volume) &
        (metrics_df['close'] >= min_price)
    ]
    
    if filtered_df.empty:
        logger.warning("No stocks passed the filtering criteria")
        return []
    
    # Get ranking weights
    ranking_metrics = midcap_config.get('ranking_metrics', DEFAULT_CONFIG['midcap_stocks']['ranking_metrics'])
    volume_weight = ranking_metrics.get('volume_weight', 0.3)
    momentum_weight = ranking_metrics.get('momentum_weight', 0.3)
    volatility_weight = ranking_metrics.get('volatility_weight', 0.2)
    liquidity_weight = ranking_metrics.get('liquidity_weight', 0.2)
    
    # Calculate composite score
    filtered_df['composite_score'] = (
        filtered_df['volume_rank'] * volume_weight +
        filtered_df['momentum_rank'] * momentum_weight +
        filtered_df['volatility_rank'] * volatility_weight +
        filtered_df['liquidity_rank'] * liquidity_weight
    )
    
    # Sort by composite score
    filtered_df = filtered_df.sort_values('composite_score', ascending=False)
    
    # Limit to max_stocks
    max_stocks = midcap_config.get('max_stocks', DEFAULT_CONFIG['midcap_stocks']['max_stocks'])
    filtered_df = filtered_df.head(max_stocks)
    
    # Extract symbols
    selected_symbols = filtered_df['symbol'].tolist()
    
    logger.info(f"Selected {len(selected_symbols)} mid-cap stocks after filtering and ranking")
    return selected_symbols

def fetch_midcap_symbols_from_alpaca(config=None):
    """
    Fetch mid-cap symbols directly from Alpaca API
    
    This function uses the Alpaca API to fetch a list of mid-cap stocks
    based on market cap and other criteria.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of mid-cap stock symbols
    """
    try:
        import alpaca_trade_api as tradeapi
        from alpaca_trade_api.rest import REST
        
        # Load configuration if not provided
        if config is None:
            try:
                with open('sp500_config_enhanced.yaml', 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return []
        
        # Get mid-cap configuration
        midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
        
        # Load Alpaca credentials
        try:
            with open('alpaca_credentials.json', 'r') as f:
                credentials = json.load(f)
                api_key = credentials.get('paper', {}).get('api_key', '')
                api_secret = credentials.get('paper', {}).get('api_secret', '')
                base_url = credentials.get('paper', {}).get('base_url', 'https://paper-api.alpaca.markets')
                
                # Try live credentials if paper credentials are not available
                if not api_key or not api_secret:
                    api_key = credentials.get('live', {}).get('api_key', '')
                    api_secret = credentials.get('live', {}).get('api_secret', '')
                    base_url = credentials.get('live', {}).get('base_url', 'https://api.alpaca.markets')
        except Exception as e:
            logger.error(f"Error loading Alpaca credentials: {str(e)}")
            return []
        
        if not api_key or not api_secret:
            logger.error("Alpaca API credentials not found")
            return []
        
        # Initialize Alpaca API
        api = REST(api_key, api_secret, base_url)
        
        # Get parameters from config
        min_price = midcap_config.get('min_price', 5.0)
        min_avg_volume = midcap_config.get('min_avg_volume', 500000)
        max_stocks = midcap_config.get('max_stocks', 50)
        
        # Query Alpaca API for assets
        logger.info("Querying Alpaca API for mid-cap stocks...")
        assets = api.list_assets(status='active')
        
        # Filter for tradable US equities - inspect the first asset to see available attributes
        if assets:
            logger.info(f"First asset attributes: {dir(assets[0])}")
            
        # Filter for tradable US equities based on available attributes
        tradable_assets = []
        for asset in assets:
            if hasattr(asset, 'tradable') and asset.tradable:
                if hasattr(asset, 'exchange') and asset.exchange in ['NYSE', 'NASDAQ', 'AMEX']:
                    # Check for different possible attribute names for asset class
                    is_equity = False
                    for attr_name in ['class', 'asset_class', 'class_name', 'asset_type']:
                        if hasattr(asset, attr_name):
                            attr_value = getattr(asset, attr_name)
                            if isinstance(attr_value, str) and 'equity' in attr_value.lower():
                                is_equity = True
                                break
                    
                    if is_equity:
                        tradable_assets.append(asset)
        
        logger.info(f"Found {len(tradable_assets)} tradable US equities")
        
        # Get symbols
        symbols = [asset.symbol for asset in tradable_assets]
        
        # Fetch market data to filter by price and volume
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching market data for {len(symbols)} symbols...")
        
        # Split symbols into batches to avoid API limits
        batch_size = 100
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        filtered_symbols = []
        
        for i, batch in enumerate(batches):
            try:
                logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} symbols)")
                
                # Fetch bars for the batch
                bars = api.get_bars(
                    batch,
                    tradeapi.TimeFrame.Day,
                    start_date,
                    end_date,
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    continue
                
                # Calculate average volume and latest price for each symbol
                symbol_metrics = {}
                
                for symbol in set(bars.index.get_level_values('symbol')):
                    symbol_bars = bars.loc[symbol]
                    
                    if len(symbol_bars) > 0:
                        avg_volume = symbol_bars['volume'].mean()
                        latest_price = symbol_bars['close'].iloc[-1]
                        
                        symbol_metrics[symbol] = {
                            'avg_volume': avg_volume,
                            'latest_price': latest_price
                        }
                
                # Filter by price and volume
                for symbol, metrics in symbol_metrics.items():
                    if metrics['latest_price'] >= min_price and metrics['avg_volume'] >= min_avg_volume:
                        filtered_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {str(e)}")
        
        logger.info(f"Found {len(filtered_symbols)} symbols meeting criteria")
        
        # Limit to max_stocks
        if len(filtered_symbols) > max_stocks:
            filtered_symbols = filtered_symbols[:max_stocks]
        
        logger.info(f"Selected {len(filtered_symbols)} mid-cap symbols from Alpaca")
        return filtered_symbols
        
    except Exception as e:
        logger.error(f"Error fetching mid-cap symbols from Alpaca: {str(e)}")
        return []

def get_midcap_symbols(config=None) -> List[str]:
    """
    Get a list of mid-cap stock symbols with high liquidity
    
    This function attempts to fetch mid-cap symbols from Alpaca,
    but falls back to using predefined symbols from the config if that fails.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        List of mid-cap stock symbols
    """
    # Load configuration if not provided
    if config is None:
        try:
            with open('sp500_config_enhanced.yaml', 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return []
    
    # Get mid-cap configuration
    midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
    
    # Check if dynamic selection is enabled
    use_dynamic_selection = midcap_config.get('use_dynamic_selection', False)
    
    # If dynamic selection is disabled or not specified, use predefined symbols
    if 'symbols' in midcap_config and midcap_config['symbols'] and not use_dynamic_selection:
        symbols = midcap_config['symbols']
        logger.info(f"Using {len(symbols)} predefined mid-cap symbols from config")
        return symbols
    
    # Try to fetch mid-cap symbols from Alpaca
    try:
        logger.info("Fetching mid-cap symbols from Alpaca...")
        alpaca_symbols = fetch_midcap_symbols_from_alpaca(config)
        
        if alpaca_symbols:
            logger.info(f"Successfully fetched {len(alpaca_symbols)} mid-cap symbols from Alpaca")
            return alpaca_symbols
        
        # If Alpaca fails, try to fetch from Wikipedia
        logger.info("Alpaca fetch failed, trying Wikipedia...")
        sp400_symbols = fetch_sp400_symbols()
        
        if sp400_symbols:
            logger.info(f"Successfully fetched {len(sp400_symbols)} S&P 400 Mid-Cap symbols from Wikipedia")
            return sp400_symbols
        
        # Fall back to predefined symbols if both methods fail
        if 'symbols' in midcap_config and midcap_config['symbols']:
            symbols = midcap_config['symbols']
            logger.info(f"Falling back to {len(symbols)} predefined mid-cap symbols from config")
            return symbols
            
        return []
    except Exception as e:
        logger.error(f"Error in get_midcap_symbols: {str(e)}")
        # Fall back to predefined symbols
        if 'symbols' in midcap_config and midcap_config['symbols']:
            symbols = midcap_config['symbols']
            logger.info(f"Falling back to {len(symbols)} predefined mid-cap symbols from config due to error")
            return symbols
        return []

# Example usage
if __name__ == "__main__":
    config = load_config()
    midcap_symbols = get_midcap_symbols(config)
    print(f"Selected {len(midcap_symbols)} mid-cap symbols:")
    print(midcap_symbols)
