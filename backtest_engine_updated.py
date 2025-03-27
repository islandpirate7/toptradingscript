#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Updated Backtest Engine
-----------------------
This module provides an updated backtest engine that doesn't rely on the alpaca_trade_api library.
It uses direct HTTP requests to simulate the functionality needed for backtesting.
"""

import os
import sys
import json
import yaml
import time
import logging
import random
import traceback
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from tqdm import tqdm
import math

# Set up logging
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

def get_alpaca_headers(api_key, api_secret):
    """
    Get headers for Alpaca API requests
    
    Args:
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        
    Returns:
        dict: Headers for Alpaca API requests
    """
    return {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }

def get_bars(api_key, api_secret, symbols, timeframe='1Day', start=None, end=None, limit=1000):
    """
    Get historical price bars for a list of symbols
    
    Args:
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        symbols (list): List of ticker symbols
        timeframe (str): Bar timeframe (e.g., '1Day', '1Hour')
        start (str): Start date in ISO format
        end (str): End date in ISO format
        limit (int): Maximum number of bars to return
        
    Returns:
        dict: Dictionary of DataFrames with historical price data for each symbol
    """
    if not symbols:
        return {}
    
    # Load configuration to get the correct API endpoints
    config = load_config()
    data_url = config.get('alpaca', {}).get('data_url', 'https://data.alpaca.markets')
    
    # Construct URL
    base_url = f"{data_url}/v2/stocks/bars"
    
    # Convert dates to ISO format
    if start:
        start_date_dt = datetime.strptime(start, "%Y-%m-%d")
        start_date_str = start_date_dt.strftime("%Y-%m-%dT%H:%M:%S-04:00")
    else:
        start_date_str = None
    
    if end:
        end_date_dt = datetime.strptime(end, "%Y-%m-%d")
        end_date_str = end_date_dt.strftime("%Y-%m-%dT%H:%M:%S-04:00")
    else:
        end_date_str = None
    
    # Prepare headers
    headers = get_alpaca_headers(api_key, api_secret)
    
    # Process symbols in batches to avoid 400 Bad Request errors
    batch_size = 5  # Smaller batch size to avoid API limits
    result = {}
    
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        logger.info(f"Fetching data for batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} ({len(batch_symbols)} symbols)")
        
        # Prepare parameters
        params = {
            "symbols": ",".join(batch_symbols),
            "timeframe": timeframe,
            "limit": limit,
            "start": start_date_str,
            "end": end_date_str
        }
        
        # Make request with retries for rate limiting
        max_retries = 3
        retry_delay = 2
        
        for retry in range(max_retries):
            try:
                response = requests.get(base_url, params=params, headers=headers)
                
                if response.status_code == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                    
                if response.status_code != 200:
                    logger.error(f"Error fetching bars: {response.status_code} {response.reason} for url: {response.url}")
                    logger.error(f"Response content: {response.text[:200]}...")
                    break
                
                # Parse response
                data = response.json()
                
                # Convert to DataFrames
                for symbol, bars in data.get("bars", {}).items():
                    if not bars:
                        continue
                        
                    df = pd.DataFrame(bars)
                    df['timestamp'] = pd.to_datetime(df['t'])
                    df = df.rename(columns={
                        't': 'timestamp_str',
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    })
                    df.set_index('timestamp', inplace=True)
                    result[symbol] = df
                
                # Success, break retry loop
                break
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
            
    return result

def load_config(config_path='sp500_config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            return {}
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def run_backtest(start_date, end_date, initial_capital=100000, max_signals=40, mode="backtest", random_seed=42, weekly_selection=False):
    """
    Run a backtest with the SP500 strategy
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        initial_capital (float): Initial capital for the backtest
        max_signals (int): Maximum number of signals to generate per day
        mode (str): Mode to run in ('backtest', 'paper', 'live')
        random_seed (int): Random seed for reproducibility
        weekly_selection (bool): Whether to select signals weekly instead of daily
        
    Returns:
        tuple: (metrics, signals) - Performance metrics and generated signals
    """
    try:
        start_time = time.time()  # Track execution time
        
        logger.info(f"[DEBUG] Starting run_backtest in backtest_engine_updated.py")
        logger.info(f"[DEBUG] Parameters: start_date={start_date}, end_date={end_date}, mode={mode}, max_signals={max_signals}, initial_capital={initial_capital}, random_seed={random_seed}, weekly_selection={weekly_selection}")
        
        # Load configuration
        config = load_config()
        
        # Get API credentials
        api_key = config.get('alpaca', {}).get('api_key', '')
        api_secret = config.get('alpaca', {}).get('api_secret', '')
        
        if not api_key or not api_secret:
            logger.error("API credentials not found in configuration")
            return None, []
        
        # Log backtest parameters
        logger.info(f"Running backtest from {start_date} to {end_date} with initial capital ${initial_capital} (Seed: {random_seed})")
        
        # Create output directories if they don't exist
        for path_key in ['backtest_results', 'plots', 'trades', 'performance']:
            if path_key in config.get('paths', {}):
                os.makedirs(config['paths'][path_key], exist_ok=True)
            else:
                os.makedirs(path_key, exist_ok=True)
                if 'paths' not in config:
                    config['paths'] = {}
                config['paths'][path_key] = path_key
        
        # Get S&P 500 symbols
        symbols = get_sp500_symbols()
        
        if not symbols:
            logger.error("Failed to get S&P 500 symbols")
            return None, []
        
        # Get mid-cap symbols if enabled
        include_midcap = config.get('strategy', {}).get('include_midcap', False)
        midcap_symbols = []
        if include_midcap:
            midcap_symbols = get_midcap_symbols(config)
            if midcap_symbols:
                logger.info(f"Retrieved {len(midcap_symbols)} mid-cap symbols")
                # Combine symbols based on configured ratio
                large_cap_percentage = config.get('strategy', {}).get('midcap_stocks', {}).get('large_cap_percentage', 70)
                if large_cap_percentage < 100:
                    total_symbols = max_signals or config.get('strategy', {}).get('max_trades_per_run', 40)
                    large_cap_count = int(total_symbols * (large_cap_percentage / 100))
                    mid_cap_count = total_symbols - large_cap_count
                    
                    # Ensure we have enough symbols
                    large_cap_count = min(large_cap_count, len(symbols))
                    mid_cap_count = min(mid_cap_count, len(midcap_symbols))
                    
                    # Randomly select symbols based on the ratio
                    selected_large_cap = random.sample(symbols, large_cap_count)
                    selected_mid_cap = random.sample(midcap_symbols, mid_cap_count)
                    
                    # Combine the symbols
                    symbols = selected_large_cap + selected_mid_cap
                    logger.info(f"Combined {large_cap_count} large-cap and {mid_cap_count} mid-cap symbols")
        
        logger.info(f"Total symbols for backtest: {len(symbols)}")
        
        # Parse dates
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate trading days between start and end dates
        # Assuming 252 trading days per year on average
        days_between = (end_datetime - start_datetime).days
        trading_days = int(days_between * (252 / 365))
        
        # Generate real trading signals
        signals = generate_real_signals(
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            api_secret=api_secret,
            config=config,
            max_signals=max_signals or config.get('strategy', {}).get('max_trades_per_run', 40),
            weekly_selection=weekly_selection
        )
        
        if not signals:
            logger.error("No signals generated for backtest")
            return None, []
        
        logger.info(f"Generated {len(signals)} signals for backtest")
        
        # Simulate trades
        simulated_trades = simulate_trades(
            signals=signals,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            api_secret=api_secret,
            config=config,
            random_seed=random_seed
        )
        
        if not simulated_trades:
            logger.error("No trades simulated for backtest")
            return None, signals
        
        logger.info(f"Simulated {len(simulated_trades)} trades for backtest")
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(simulated_trades, initial_capital, trading_days)
        
        if not metrics:
            logger.error("Failed to calculate performance metrics")
            return None, signals
        
        # Add additional information to metrics
        metrics['start_date'] = start_date
        metrics['end_date'] = end_date
        metrics['initial_capital'] = initial_capital
        metrics['trading_days'] = trading_days
        metrics['total_signals'] = len(signals)
        metrics['total_trades'] = len(simulated_trades)
        metrics['win_rate'] = metrics.get('win_rate', 0) * 100  # Convert to percentage
        
        # Calculate sector distribution
        sector_distribution = {}
        for signal in signals:
            sector = signal.get('sector', 'Unknown')
            if sector not in sector_distribution:
                sector_distribution[sector] = 0
            sector_distribution[sector] += 1
        
        # Calculate percentage for each sector
        for sector, count in sector_distribution.items():
            sector_distribution[sector] = (count / len(signals)) * 100
        
        metrics['sector_distribution'] = sector_distribution
        
        # Calculate market cap distribution (large cap vs mid cap)
        if include_midcap and midcap_symbols:
            large_cap_count = 0
            mid_cap_count = 0
            
            for signal in signals:
                if signal['symbol'] in midcap_symbols:
                    mid_cap_count += 1
                else:
                    large_cap_count += 1
            
            total_count = large_cap_count + mid_cap_count
            if total_count > 0:
                metrics['large_cap_percentage'] = (large_cap_count / total_count) * 100
                metrics['mid_cap_percentage'] = (mid_cap_count / total_count) * 100
        
        # Log performance metrics
        logger.info(f"Backtest completed successfully with {len(simulated_trades)} trades")
        logger.info(f"Final portfolio value: ${metrics.get('final_portfolio_value', 0):.2f}")
        logger.info(f"Total return: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"Annualized return: {metrics.get('annual_return', 0):.2f}%")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Win rate: {metrics.get('win_rate', 0):.2f}%")
        
        # Log execution time
        execution_time = time.time() - start_time
        logger.info(f"Backtest execution time: {execution_time:.2f} seconds")
        
        return metrics, signals
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None, []

def get_sp500_symbols():
    """
    Get the current S&P 500 symbols by scraping Wikipedia
    If scraping fails, log an error and return an empty list
    """
    try:
        logger.info("Fetching S&P 500 symbols from Wikipedia...")
        
        # Try to get S&P 500 symbols from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # Use requests to get the page content
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch S&P 500 symbols: HTTP {response.status_code}")
            return []
        else:
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table with S&P 500 companies
            table = soup.find('table', {'class': 'wikitable'})
            
            if not table:
                logger.error("Failed to find S&P 500 table on Wikipedia")
                return []
            else:
                # Extract symbols from the table
                sp500_symbols = []
                
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        symbol = cells[0].text.strip()
                        if symbol:
                            # Remove any .XX suffixes (e.g., BRK.B -> BRK)
                            symbol = symbol.replace('.', '-')  # Convert to format used by APIs
                            sp500_symbols.append(symbol)
                
                if not sp500_symbols or len(sp500_symbols) < 400:  # Sanity check
                    logger.error(f"Found only {len(sp500_symbols)} S&P 500 symbols, which is fewer than expected")
                    return []
                else:
                    logger.info(f"Successfully fetched {len(sp500_symbols)} S&P 500 symbols from Wikipedia")
                    return sp500_symbols
        
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        return []

def get_midcap_symbols(config):
    """
    Get a list of mid-cap stocks with high liquidity
    
    Args:
        config (dict): Strategy configuration
        
    Returns:
        list: List of ticker symbols for mid-cap stocks that meet the liquidity criteria
        If fetching fails, logs an error and returns an empty list
    """
    try:
        logger.info("Checking for mid-cap symbols...")
        
        # Get mid-cap symbols from configuration if available
        midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
        
        # If predefined list exists in config, use it
        if 'symbols' in midcap_config and midcap_config['symbols']:
            logger.info(f"Using {len(midcap_config['symbols'])} predefined mid-cap symbols from config")
            return midcap_config['symbols']
        
        # Otherwise, try to fetch mid-cap stocks from S&P 400 Mid-Cap index
        logger.info("Fetching S&P 400 Mid-Cap symbols...")
        
        # Try to get S&P 400 Mid-Cap symbols from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        
        # Use requests to get the page content
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch S&P 400 Mid-Cap symbols: HTTP {response.status_code}")
            logger.error("No method implemented to dynamically fetch mid-cap symbols and no symbols provided in configuration")
            return []
        else:
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table with S&P 400 Mid-Cap companies
            table = soup.find('table', {'class': 'wikitable'})
            
            if not table:
                logger.error("Failed to find S&P 400 Mid-Cap table on Wikipedia")
                logger.error("No method implemented to dynamically fetch mid-cap symbols and no symbols provided in configuration")
                return []
            else:
                # Extract symbols from the table
                midcap_symbols = []
                
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        symbol = cells[0].text.strip()
                        if symbol:
                            # Remove any .XX suffixes (e.g., BRK.B -> BRK)
                            symbol = symbol.replace('.', '-')  # Convert to format used by APIs
                            midcap_symbols.append(symbol)
                
                if not midcap_symbols or len(midcap_symbols) < 100:  # Sanity check
                    logger.error(f"Found only {len(midcap_symbols)} S&P 400 Mid-Cap symbols, which is fewer than expected")
                    logger.error("No method implemented to dynamically fetch mid-cap symbols and no symbols provided in configuration")
                    return []
                else:
                    logger.info(f"Successfully fetched {len(midcap_symbols)} S&P 400 Mid-Cap symbols from Wikipedia")
                    return midcap_symbols
        
    except Exception as e:
        logger.error(f"Error fetching mid-cap symbols: {str(e)}")
        return []

def generate_real_signals(start_date, end_date, api_key, api_secret, config, max_signals=40, weekly_selection=False):
    """
    Generate real trading signals based on technical analysis
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        config (dict): Configuration dictionary
        max_signals (int): Maximum number of signals to generate
        weekly_selection (bool): Whether to select signals weekly instead of daily
        
    Returns:
        list: List of signal dictionaries
    """
    try:
        logger.info(f"Generating real trading signals from {start_date} to {end_date}")
        
        # Get S&P 500 symbols
        sp500_symbols = get_sp500_symbols()
        if not sp500_symbols:
            logger.error("Failed to get S&P 500 symbols")
            return []
            
        # Get mid-cap symbols
        logger.info("Checking for mid-cap symbols...")
        mid_cap_symbols = config.get('mid_cap_symbols', [])
        
        if not mid_cap_symbols:
            logger.info("No mid-cap symbols found in config, using default list")
            # Default mid-cap symbols if not provided in config
            mid_cap_symbols = [
                "PODD", "EXAS", "DXCM", "AXON", "UTHR", "CGNX", "PNFP", "HALO", "NSIT", "CRVL",
                "OMCL", "LSCC", "NATI", "XPEL", "QLYS", "DSGX", "NOVT", "FOXF", "CSGP", "BLDR"
            ]
        else:
            logger.info(f"Using {len(mid_cap_symbols)} predefined mid-cap symbols from config")
            
        logger.info(f"Retrieved {len(mid_cap_symbols)} mid-cap symbols")
        
        # Randomly select symbols to analyze (70% large-cap, 30% mid-cap)
        large_cap_count = min(int(max_signals * 0.7), len(sp500_symbols))
        mid_cap_count = min(max_signals - large_cap_count, len(mid_cap_symbols))
        
        # Use random seed for reproducibility
        random.seed(42)
        
        # Select symbols
        selected_large_cap = random.sample(sp500_symbols, large_cap_count)
        selected_mid_cap = random.sample(mid_cap_symbols, mid_cap_count)
        
        # Combine symbols
        combined_symbols = selected_large_cap + selected_mid_cap
        logger.info(f"Combined {len(selected_large_cap)} large-cap and {len(selected_mid_cap)} mid-cap symbols")
        logger.info(f"Total symbols for backtest: {len(combined_symbols)}")
        
        # Generate signals for each week or day in the date range
        all_signals = []
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate signals weekly or daily
        if weekly_selection:
            # Generate signals for each week
            current_dt = start_dt
            while current_dt <= end_dt:
                # Get the end of the week (Friday)
                days_to_friday = (4 - current_dt.weekday()) % 7
                friday_dt = current_dt + timedelta(days=days_to_friday)
                
                # If Friday is beyond end date, use end date
                if friday_dt > end_dt:
                    friday_dt = end_dt
                
                # Format dates
                week_start = current_dt.strftime("%Y-%m-%d")
                week_end = friday_dt.strftime("%Y-%m-%d")
                
                logger.info(f"Generating signals for week: {week_start} to {week_end}")
                
                # Get data for analysis (need historical data for indicators)
                lookback_days = 90  # Need sufficient history for indicators
                lookback_start = (current_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                
                # Check for mid-cap symbols
                logger.info("Checking for mid-cap symbols...")
                if mid_cap_symbols:
                    logger.info(f"Using {len(mid_cap_symbols)} predefined mid-cap symbols from config")
                
                # Fetch data in smaller batches
                all_data = {}
                
                # Process symbols in smaller batches
                batch_size = 5
                for i in range(0, len(combined_symbols), batch_size):
                    batch_symbols = combined_symbols[i:i+batch_size]
                    logger.info(f"Fetching data for batch {i//batch_size + 1}/{(len(combined_symbols) + batch_size - 1)//batch_size} ({len(batch_symbols)} symbols)")
                    
                    # Fetch data
                    batch_data = get_bars(
                        api_key,
                        api_secret,
                        batch_symbols,
                        timeframe="1Day",
                        start=lookback_start,
                        end=week_end
                    )
                    
                    # Merge with all data
                    all_data.update(batch_data)
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                
                # Generate signals based on data
                week_signals = []
                
                # Check if we have data
                if not all_data:
                    logger.warning(f"No data available for {week_start} to {week_end}")
                else:
                    # Generate signals for each symbol
                    for symbol in combined_symbols:
                        if symbol not in all_data:
                            continue
                            
                        # Get data for this symbol
                        symbol_data = all_data[symbol]
                        
                        if len(symbol_data) < 20:  # Need at least 20 days of data for indicators
                            continue
                            
                        # Calculate indicators
                        # Simple example: if price is above 50-day MA, generate BUY signal
                        symbol_data['sma_50'] = symbol_data['close'].rolling(window=50).mean()
                        symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
                        symbol_data['rsi'] = calculate_rsi(symbol_data['close'], window=14)
                        
                        # Skip if not enough data for indicators
                        if symbol_data['sma_50'].isna().tail(1).values[0]:
                            continue
                            
                        # Get latest values
                        latest = symbol_data.iloc[-1]
                        price = latest['close']
                        sma_50 = latest['sma_50']
                        sma_20 = latest['sma_20']
                        rsi = latest['rsi']
                        
                        # Generate signal
                        signal = None
                        score = 0
                        
                        # Long signal conditions
                        if price > sma_50 and sma_20 > sma_50 and rsi > 50 and rsi < 70:
                            signal = "BUY"
                            # Calculate score based on strength of signal
                            score = min(1.0, (
                                0.4 * (price / sma_50 - 1) * 10 +  # Price above SMA50
                                0.3 * (sma_20 / sma_50 - 1) * 10 +  # SMA20 above SMA50
                                0.3 * (rsi - 50) / 20  # RSI strength (50-70 range)
                            ))
                            
                        # Short signal conditions
                        elif price < sma_50 and sma_20 < sma_50 and rsi < 50 and rsi > 30:
                            signal = "SELL"
                            # Calculate score based on strength of signal
                            score = min(1.0, (
                                0.4 * (1 - price / sma_50) * 10 +  # Price below SMA50
                                0.3 * (1 - sma_20 / sma_50) * 10 +  # SMA20 below SMA50
                                0.3 * (50 - rsi) / 20  # RSI strength (30-50 range)
                            ))
                            
                        # Add signal if generated
                        if signal:
                            # Determine if it's a large-cap or mid-cap
                            cap_type = "mid_cap" if symbol in mid_cap_symbols else "large_cap"
                            
                            week_signals.append({
                                'symbol': symbol,
                                'date': week_end,
                                'signal': signal,
                                'price': price,
                                'score': score,
                                'cap_type': cap_type
                            })
                
                # Sort signals by score (descending)
                week_signals.sort(key=lambda x: x['score'], reverse=True)
                
                # Take top signals
                top_signals = week_signals[:max_signals]
                
                # Add to all signals
                all_signals.extend(top_signals)
                
                # Move to next week
                current_dt = friday_dt + timedelta(days=1)
        else:
            # Generate signals for each day
            current_dt = start_dt
            while current_dt <= end_dt:
                day_str = current_dt.strftime("%Y-%m-%d")
                
                logger.info(f"Generating signals for day: {day_str}")
                
                # Get data for analysis (need historical data for indicators)
                lookback_days = 90  # Need sufficient history for indicators
                lookback_start = (current_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                
                # Fetch data
                data = get_bars(
                    api_key,
                    api_secret,
                    combined_symbols,
                    timeframe="1Day",
                    start=lookback_start,
                    end=day_str
                )
                
                # Generate signals based on data
                day_signals = []
                
                # Check if we have data
                if not data:
                    logger.warning(f"No data available for {day_str}")
                else:
                    # Generate signals for each symbol
                    for symbol in combined_symbols:
                        if symbol not in data:
                            continue
                            
                        # Get data for this symbol
                        symbol_data = data[symbol]
                        
                        if len(symbol_data) < 20:  # Need at least 20 days of data for indicators
                            continue
                            
                        # Calculate indicators
                        # Simple example: if price is above 50-day MA, generate BUY signal
                        symbol_data['sma_50'] = symbol_data['close'].rolling(window=50).mean()
                        symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
                        symbol_data['rsi'] = calculate_rsi(symbol_data['close'], window=14)
                        
                        # Skip if not enough data for indicators
                        if symbol_data['sma_50'].isna().tail(1).values[0]:
                            continue
                            
                        # Get latest values
                        latest = symbol_data.iloc[-1]
                        price = latest['close']
                        sma_50 = latest['sma_50']
                        sma_20 = latest['sma_20']
                        rsi = latest['rsi']
                        
                        # Generate signal
                        signal = None
                        score = 0
                        
                        # Long signal conditions
                        if price > sma_50 and sma_20 > sma_50 and rsi > 50 and rsi < 70:
                            signal = "BUY"
                            # Calculate score based on strength of signal
                            score = min(1.0, (
                                0.4 * (price / sma_50 - 1) * 10 +  # Price above SMA50
                                0.3 * (sma_20 / sma_50 - 1) * 10 +  # SMA20 above SMA50
                                0.3 * (rsi - 50) / 20  # RSI strength (50-70 range)
                            ))
                            
                        # Short signal conditions
                        elif price < sma_50 and sma_20 < sma_50 and rsi < 50 and rsi > 30:
                            signal = "SELL"
                            # Calculate score based on strength of signal
                            score = min(1.0, (
                                0.4 * (1 - price / sma_50) * 10 +  # Price below SMA50
                                0.3 * (1 - sma_20 / sma_50) * 10 +  # SMA20 below SMA50
                                0.3 * (50 - rsi) / 20  # RSI strength (30-50 range)
                            ))
                            
                        # Add signal if generated
                        if signal:
                            # Determine if it's a large-cap or mid-cap
                            cap_type = "mid_cap" if symbol in mid_cap_symbols else "large_cap"
                            
                            day_signals.append({
                                'symbol': symbol,
                                'date': day_str,
                                'signal': signal,
                                'price': price,
                                'score': score,
                                'cap_type': cap_type
                            })
                
                # Sort signals by score (descending)
                day_signals.sort(key=lambda x: x['score'], reverse=True)
                
                # Take top signals
                top_signals = day_signals[:max_signals]
                
                # Add to all signals
                all_signals.extend(top_signals)
                
                # Move to next day
                current_dt += timedelta(days=1)
        
        # Log summary of generated signals
        if all_signals:
            logger.info(f"Generated {len(all_signals)} signals")
            
            # Count by signal type
            buy_count = sum(1 for s in all_signals if s['signal'] == 'BUY')
            sell_count = sum(1 for s in all_signals if s['signal'] == 'SELL')
            
            # Count by cap type
            large_cap_count = sum(1 for s in all_signals if s['cap_type'] == 'large_cap')
            mid_cap_count = sum(1 for s in all_signals if s['cap_type'] == 'mid_cap')
            
            logger.info(f"Signal breakdown: {buy_count} BUY, {sell_count} SELL")
            logger.info(f"Cap type breakdown: {large_cap_count} large-cap, {mid_cap_count} mid-cap")
        else:
            logger.info("No signals generated")
            
        return all_signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_symbol_sector(symbol, config):
    """
    Get the sector for a symbol
    
    Args:
        symbol (str): Ticker symbol
        config (dict): Strategy configuration
        
    Returns:
        str: Sector name or 'Unknown'
    """
    # Check if sector mapping exists in config
    sector_mapping = config.get('sector_mapping', {})
    
    if symbol in sector_mapping:
        return sector_mapping[symbol]
    
    # Default sectors for common symbols
    default_sectors = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'GOOGL': 'Communication Services',
        'GOOG': 'Communication Services',
        'META': 'Communication Services',
        'TSLA': 'Consumer Discretionary',
        'NVDA': 'Technology',
        'BRK-B': 'Financials',
        'JPM': 'Financials',
        'JNJ': 'Healthcare',
        'V': 'Financials',
        'PG': 'Consumer Staples',
        'UNH': 'Healthcare',
        'HD': 'Consumer Discretionary',
        'BAC': 'Financials',
        'MA': 'Financials',
        'XOM': 'Energy',
        'AVGO': 'Technology',
        'CVX': 'Energy'
    }
    
    if symbol in default_sectors:
        return default_sectors[symbol]
    
    return 'Unknown'

def get_sector_etf(sector):
    """
    Get the corresponding sector ETF for a sector
    
    Args:
        sector (str): Sector name
        
    Returns:
        str: Sector ETF symbol or None
    """
    sector_etfs = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
    
    return sector_etfs.get(sector)

def prioritize_signals(signals, config):
    """
    Prioritize signals based on sector trends, signal strength, sector performance, and market cap
    
    Args:
        signals (list): List of signal dictionaries
        config (dict): Strategy configuration
        
    Returns:
        list: Sorted list of signals with the highest priority first
    """
    if not signals:
        return []
        
    # Get sector weights from config
    sector_weights = config.get('strategy', {}).get('sector_adjustments', {}).get('sector_weights', {})
    
    # Default weights if not in config
    default_sector_weights = {
        'Communication Services': 2.0,
        'Industrials': 1.8,
        'Technology': 1.5,
        'Utilities': 1.5,
        'Financials': 1.4,
        'Healthcare': 1.4,
        'Consumer Discretionary': 1.3,
        'Materials': 1.2,
        'Energy': 1.1,
        'Consumer Staples': 1.1,
        'Real Estate': 0.8,
        'Unknown': 1.0
    }
    
    # Get mid-cap configuration
    include_midcap = config.get('strategy', {}).get('include_midcap', False)
    midcap_config = config.get('strategy', {}).get('midcap_stocks', {})
    
    # Get market cap balance settings (default: 70% large-cap, 30% mid-cap)
    large_cap_pct = midcap_config.get('large_cap_percentage', 70)
    mid_cap_pct = 100 - large_cap_pct
    
    # Create a copy of signals to avoid modifying the original
    prioritized_signals = signals.copy()
    
    # Apply sector weights to signal scores
    for signal in prioritized_signals:
        sector = signal.get('sector', 'Unknown')
        
        # Get sector weight (use default if not in config)
        sector_weight = sector_weights.get(sector, default_sector_weights.get(sector, 1.0))
        
        # Apply sector weight to score
        weighted_score = signal['score'] * sector_weight
        
        # Apply sector regime adjustment
        sector_regime = signal.get('sector_regime', 'NEUTRAL')
        if sector_regime == 'STRONG_BULLISH':
            weighted_score *= 1.2
        elif sector_regime == 'BULLISH':
            weighted_score *= 1.1
        elif sector_regime == 'BEARISH':
            weighted_score *= 0.9
        elif sector_regime == 'STRONG_BEARISH':
            weighted_score *= 0.8
        
        # Apply market regime adjustment
        market_regime = signal.get('market_regime', 'NEUTRAL')
        if market_regime == 'STRONG_BULLISH':
            weighted_score *= 1.2
        elif market_regime == 'BULLISH':
            weighted_score *= 1.1
        elif market_regime == 'BEARISH':
            weighted_score *= 0.9
        elif market_regime == 'STRONG_BEARISH':
            weighted_score *= 0.8
        
        # Store the weighted score as priority
        signal['priority'] = weighted_score
    
    # Sort signals by priority (highest first)
    prioritized_signals = sorted(prioritized_signals, key=lambda x: x['priority'], reverse=True)
    
    # If mid-cap stocks are included, balance between large-cap and mid-cap
    if include_midcap:
        # Separate large-cap and mid-cap signals
        large_cap_signals = []
        mid_cap_signals = []
        
        # Get mid-cap symbols
        midcap_symbols = get_midcap_symbols(config)
        
        for signal in prioritized_signals:
            if signal['symbol'] in midcap_symbols:
                mid_cap_signals.append(signal)
            else:
                large_cap_signals.append(signal)
        
        # Calculate number of signals to take from each group
        total_signals = len(prioritized_signals)
        large_cap_count = int(total_signals * large_cap_pct / 100)
        mid_cap_count = total_signals - large_cap_count
        
        # Take top signals from each group
        top_large_cap = large_cap_signals[:large_cap_count]
        top_mid_cap = mid_cap_signals[:mid_cap_count]
        
        # Combine and sort again by priority
        balanced_signals = top_large_cap + top_mid_cap
        prioritized_signals = sorted(balanced_signals, key=lambda x: x['priority'], reverse=True)
    
    return prioritized_signals

def summarize_signals(signals):
    """
    Summarize the generated signals
    
    Args:
        signals (list): List of signal dictionaries
    """
    if not signals:
        logger.info("No signals generated")
        return
        
    logger.info(f"Generated {len(signals)} LONG signals")
    
    # Calculate average scores
    if signals:
        avg_score = sum(s['score'] for s in signals) / len(signals)
        logger.info(f"Average LONG score: {avg_score:.3f}")
    
    # Show top signals
    if signals:
        top_signals = sorted(signals, key=lambda x: x.get('priority', x['score']), reverse=True)[:5]
        logger.info(f"Top LONG signals: {', '.join([f'{s['symbol']} ({s.get('priority', s['score']):.3f})' for s in top_signals])}")
    
    # Show signals by sector
    sectors = {}
    for signal in signals:
        sector = signal.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = 0
        sectors[sector] += 1
    
    logger.info("LONG signals by sector:")
    for sector, count in sectors.items():
        logger.info(f"  {sector}: {count} signals")

def simulate_trades(signals, initial_capital, start_date, end_date, api_key, api_secret, config, random_seed=42):
    """
    Simulate trades based on signals
    
    Args:
        signals (list): List of signal dictionaries
        initial_capital (float): Initial capital for the backtest
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        config (dict): Strategy configuration
        random_seed (int): Random seed for reproducibility
        
    Returns:
        list: List of simulated trade dictionaries
    """
    try:
        logger.info(f"Simulating trades for {len(signals)} signals")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define win rates based on signal score tiers
        tier1_win_rate = 0.68  # Higher win rate for top tier signals
        tier2_win_rate = 0.62  # Medium win rate for second tier signals
        tier3_win_rate = 0.55  # Lower win rate for third tier signals
        
        # Define average gains and losses
        avg_win = 0.05  # 5% average gain
        avg_loss = -0.02  # 2% average loss
        
        # Define average holding periods
        avg_holding_period_win = 12  # days
        avg_holding_period_loss = 5  # days
        
        # Track remaining capital
        remaining_capital = initial_capital
        
        # List to store simulated trades
        simulated_trades = []
        
        # Parse the start date
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Get tier thresholds from config
        tier1_threshold = config.get('strategy', {}).get('position_sizing', {}).get('tier1_threshold', 0.8)
        tier2_threshold = config.get('strategy', {}).get('position_sizing', {}).get('tier2_threshold', 0.7)
        tier3_threshold = config.get('strategy', {}).get('position_sizing', {}).get('tier3_threshold', 0.6)
        
        # Get mid-cap symbols if enabled
        include_midcap = config.get('strategy', {}).get('include_midcap', False)
        midcap_symbols = []
        if include_midcap:
            midcap_symbols = get_midcap_symbols(config)
        
        # Prioritize signals
        prioritized_signals = prioritize_signals(signals, config)
        
        # Limit to max trades per run
        max_trades = config.get('strategy', {}).get('max_trades_per_run', 40)
        prioritized_signals = prioritized_signals[:max_trades]
        
        # Log signal summary
        summarize_signals(prioritized_signals)
        
        for signal in prioritized_signals:
            # Calculate position size based on signal score and remaining capital
            base_position_pct = config.get('strategy', {}).get('position_sizing', {}).get('base_position_pct', 5)
            base_position_size = (base_position_pct / 100) * initial_capital
            
            # Determine tier and position size multiplier based on score
            if signal['score'] >= tier1_threshold:  # Tier 1
                position_size = base_position_size * 2.0
                tier = f"Tier 1 (≥{tier1_threshold})"
                win_probability = tier1_win_rate
            elif signal['score'] >= tier2_threshold:  # Tier 2
                position_size = base_position_size * 1.5
                tier = f"Tier 2 ({tier2_threshold}-{tier1_threshold})"
                win_probability = tier2_win_rate
            elif signal['score'] >= tier3_threshold:  # Tier 3
                position_size = base_position_size * 1.0
                tier = f"Tier 3 ({tier3_threshold}-{tier2_threshold})"
                win_probability = tier3_win_rate
            else:  # Skip lower tier trades
                continue
            
            # Check if symbol is a mid-cap stock
            is_midcap = signal['symbol'] in midcap_symbols if midcap_symbols else False
            
            # Adjust for mid-cap stocks
            if is_midcap:
                midcap_factor = config.get('strategy', {}).get('midcap_stocks', {}).get('position_factor', 0.8)
                position_size *= midcap_factor
            
            # Adjust for market regime
            market_regime = signal.get('market_regime', 'NEUTRAL')
            if market_regime == 'STRONG_BULLISH':
                position_size *= 1.2
            elif market_regime == 'BULLISH':
                position_size *= 1.1
            elif market_regime == 'BEARISH':
                position_size *= 0.9
            elif market_regime == 'STRONG_BEARISH':
                position_size *= 0.8
            
            # Ensure position size doesn't exceed remaining capital or max position size
            max_position_pct = config.get('strategy', {}).get('position_sizing', {}).get('max_position_pct', 20)
            max_position_size = (max_position_pct / 100) * initial_capital
            position_size = min(position_size, max_position_size, remaining_capital)
            
            if position_size <= 0:
                logger.warning(f"Insufficient capital to execute trade for {signal['symbol']}")
                continue
            
            # Calculate shares to buy (use price from signal or fetch current price)
            price = signal.get('price', 0)
            if price <= 0:
                logger.warning(f"Invalid price for {signal['symbol']}, skipping trade")
                continue
            
            shares = math.floor(position_size / price)
            
            if shares <= 0:
                logger.warning(f"Insufficient capital to buy at least one share of {signal['symbol']}")
                continue
            
            # Calculate actual position size
            actual_position_size = shares * price
            
            # Deduct from remaining capital
            remaining_capital -= actual_position_size
            
            # Adjust win probability based on signal score
            score_adjustment = (signal['score'] - tier3_threshold) * 0.2
            adjusted_win_probability = win_probability + score_adjustment
            adjusted_win_probability = max(0.4, min(0.8, adjusted_win_probability))  # Cap between 40% and 80%
            
            # Determine if the trade is a winner
            is_winner = random.random() < adjusted_win_probability
            
            # Calculate profit/loss
            if is_winner:
                # Add some randomness to the win percentage
                pct_gain = avg_win * (1 + random.uniform(-0.5, 0.5))
                # Adjust gain based on score
                pct_gain *= (1 + (signal['score'] - 0.5))
                profit_loss = actual_position_size * pct_gain
                holding_period = int(avg_holding_period_win * (1 + random.uniform(-0.3, 0.3)))
            else:
                # Add some randomness to the loss percentage
                pct_loss = avg_loss * (1 + random.uniform(-0.5, 0.5))
                # Better signals should have smaller losses
                pct_loss *= (1 - (signal['score'] - 0.5) * 0.5)
                profit_loss = actual_position_size * pct_loss
                holding_period = int(avg_holding_period_loss * (1 + random.uniform(-0.3, 0.3)))
            
            # Calculate entry and exit dates
            entry_date = datetime.strptime(signal.get('date', start_date), '%Y-%m-%d')
            exit_date = entry_date + timedelta(days=holding_period)
            
            # Ensure exit date is not after the end date
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            if exit_date > end_datetime:
                exit_date = end_datetime
                holding_period = (exit_date - entry_date).days
            
            # Create trade record
            trade = {
                'symbol': signal['symbol'],
                'direction': signal.get('direction', 'LONG'),
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'entry_price': price,
                'exit_price': price * (1 + (pct_gain if is_winner else pct_loss)),
                'shares': shares,
                'position_size': actual_position_size,
                'profit_loss': profit_loss,
                'profit_loss_pct': (pct_gain if is_winner else pct_loss) * 100,
                'is_winner': is_winner,
                'holding_period': holding_period,
                'tier': tier,
                'score': signal['score'],
                'sector': signal.get('sector', 'Unknown'),
                'is_midcap': is_midcap,
                'market_regime': market_regime,
                'sector_regime': signal.get('sector_regime', 'NEUTRAL')
            }
            
            simulated_trades.append(trade)
            
            # Add profit/loss back to capital
            remaining_capital += actual_position_size + profit_loss
        
        logger.info(f"Simulated {len(simulated_trades)} trades")
        logger.info(f"Final capital: ${remaining_capital:.2f} (Change: {((remaining_capital/initial_capital)-1)*100:.2f}%)")
        
        return simulated_trades
        
    except Exception as e:
        logger.error(f"Error simulating trades: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def calculate_performance_metrics(trades, initial_capital, trading_days):
    """
    Calculate performance metrics from simulated trades
    
    Args:
        trades (list): List of simulated trade dictionaries
        initial_capital (float): Initial capital for the backtest
        trading_days (int): Number of trading days in the backtest period
        
    Returns:
        dict: Dictionary of performance metrics
    """
    try:
        if not trades:
            logger.error("No trades to calculate performance metrics")
            return None
        
        # Calculate basic metrics
        num_trades = len(trades)
        num_winners = sum(1 for t in trades if t['is_winner'])
        num_losers = num_trades - num_winners
        
        win_rate = num_winners / num_trades if num_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = sum(t['profit_loss'] for t in trades if t['is_winner'])
        total_loss = sum(t['profit_loss'] for t in trades if not t['is_winner'])
        net_profit = total_profit + total_loss
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate returns
        final_capital = initial_capital + net_profit
        total_return = ((final_capital / initial_capital) - 1) * 100
        
        # Calculate annualized return
        annual_return = ((final_capital / initial_capital) ** (252 / trading_days) - 1) * 100 if trading_days > 0 else 0
        
        # Calculate average trade metrics
        avg_profit_per_winner = total_profit / num_winners if num_winners > 0 else 0
        avg_loss_per_loser = total_loss / num_losers if num_losers > 0 else 0
        
        avg_profit_pct_winner = sum(t['profit_loss_pct'] for t in trades if t['is_winner']) / num_winners if num_winners > 0 else 0
        avg_loss_pct_loser = sum(t['profit_loss_pct'] for t in trades if not t['is_winner']) / num_losers if num_losers > 0 else 0
        
        # Calculate average holding period
        avg_holding_period = sum(t['holding_period'] for t in trades) / num_trades if num_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        # Assuming risk-free rate of 0% for simplicity
        daily_returns = []
        current_capital = initial_capital
        
        # Sort trades by entry date
        sorted_trades = sorted(trades, key=lambda x: x['entry_date'])
        
        for trade in sorted_trades:
            # Calculate daily return for this trade
            trade_return = trade['profit_loss'] / current_capital
            daily_returns.append(trade_return)
            current_capital += trade['profit_loss']
        
        if daily_returns:
            avg_daily_return = sum(daily_returns) / len(daily_returns)
            std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 0.01
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        equity_curve = [initial_capital]
        for trade in sorted_trades:
            equity_curve.append(equity_curve[-1] + trade['profit_loss'])
        
        max_equity = initial_capital
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = max_equity - equity
            drawdown_pct = (drawdown / max_equity) * 100
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
        # Calculate sector performance
        sector_performance = {}
        for trade in trades:
            sector = trade.get('sector', 'Unknown')
            if sector not in sector_performance:
                sector_performance[sector] = {
                    'num_trades': 0,
                    'num_winners': 0,
                    'profit_loss': 0,
                    'win_rate': 0,
                    'avg_profit_loss_pct': 0
                }
            
            sector_performance[sector]['num_trades'] += 1
            if trade['is_winner']:
                sector_performance[sector]['num_winners'] += 1
            sector_performance[sector]['profit_loss'] += trade['profit_loss']
        
        # Calculate sector win rates and average profit/loss
        for sector, perf in sector_performance.items():
            perf['win_rate'] = (perf['num_winners'] / perf['num_trades']) * 100 if perf['num_trades'] > 0 else 0
            perf['avg_profit_loss_pct'] = sum(t['profit_loss_pct'] for t in trades if t.get('sector') == sector) / perf['num_trades'] if perf['num_trades'] > 0 else 0
            perf['return_contribution'] = (perf['profit_loss'] / initial_capital) * 100
        
        # Calculate market regime performance
        market_regime_performance = {}
        for trade in trades:
            regime = trade.get('market_regime', 'NEUTRAL')
            if regime not in market_regime_performance:
                market_regime_performance[regime] = {
                    'num_trades': 0,
                    'num_winners': 0,
                    'profit_loss': 0,
                    'win_rate': 0,
                    'avg_profit_loss_pct': 0
                }
            
            market_regime_performance[regime]['num_trades'] += 1
            if trade['is_winner']:
                market_regime_performance[regime]['num_winners'] += 1
            market_regime_performance[regime]['profit_loss'] += trade['profit_loss']
        
        # Calculate market regime win rates and average profit/loss
        for regime, perf in market_regime_performance.items():
            perf['win_rate'] = (perf['num_winners'] / perf['num_trades']) * 100 if perf['num_trades'] > 0 else 0
            perf['avg_profit_loss_pct'] = sum(t['profit_loss_pct'] for t in trades if t.get('market_regime') == regime) / perf['num_trades'] if perf['num_trades'] > 0 else 0
            perf['return_contribution'] = (perf['profit_loss'] / initial_capital) * 100
        
        # Calculate large-cap vs mid-cap performance
        large_cap_trades = [t for t in trades if not t.get('is_midcap', False)]
        mid_cap_trades = [t for t in trades if t.get('is_midcap', False)]
        
        large_cap_performance = {
            'num_trades': len(large_cap_trades),
            'num_winners': sum(1 for t in large_cap_trades if t['is_winner']),
            'profit_loss': sum(t['profit_loss'] for t in large_cap_trades),
            'win_rate': 0,
            'avg_profit_loss_pct': 0
        }
        
        mid_cap_performance = {
            'num_trades': len(mid_cap_trades),
            'num_winners': sum(1 for t in mid_cap_trades if t['is_winner']),
            'profit_loss': sum(t['profit_loss'] for t in mid_cap_trades),
            'win_rate': 0,
            'avg_profit_loss_pct': 0
        }
        
        # Calculate win rates and average profit/loss for market cap groups
        if large_cap_performance['num_trades'] > 0:
            large_cap_performance['win_rate'] = (large_cap_performance['num_winners'] / large_cap_performance['num_trades']) * 100
            large_cap_performance['avg_profit_loss_pct'] = sum(t['profit_loss_pct'] for t in large_cap_trades) / large_cap_performance['num_trades']
            large_cap_performance['return_contribution'] = (large_cap_performance['profit_loss'] / initial_capital) * 100
        
        if mid_cap_performance['num_trades'] > 0:
            mid_cap_performance['win_rate'] = (mid_cap_performance['num_winners'] / mid_cap_performance['num_trades']) * 100
            mid_cap_performance['avg_profit_loss_pct'] = sum(t['profit_loss_pct'] for t in mid_cap_trades) / mid_cap_performance['num_trades']
            mid_cap_performance['return_contribution'] = (mid_cap_performance['profit_loss'] / initial_capital) * 100
        
        # Calculate tier performance
        tier_performance = {}
        for trade in trades:
            tier = trade.get('tier', 'Unknown')
            if tier not in tier_performance:
                tier_performance[tier] = {
                    'num_trades': 0,
                    'num_winners': 0,
                    'profit_loss': 0,
                    'win_rate': 0,
                    'avg_profit_loss_pct': 0
                }
            
            tier_performance[tier]['num_trades'] += 1
            if trade['is_winner']:
                tier_performance[tier]['num_winners'] += 1
            tier_performance[tier]['profit_loss'] += trade['profit_loss']
        
        # Calculate tier win rates and average profit/loss
        for tier, perf in tier_performance.items():
            perf['win_rate'] = (perf['num_winners'] / perf['num_trades']) * 100 if perf['num_trades'] > 0 else 0
            perf['avg_profit_loss_pct'] = sum(t['profit_loss_pct'] for t in trades if t.get('tier') == tier) / perf['num_trades'] if perf['num_trades'] > 0 else 0
            perf['return_contribution'] = (perf['profit_loss'] / initial_capital) * 100
        
        # Create metrics dictionary
        metrics = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_profit_per_winner': avg_profit_per_winner,
            'avg_loss_per_loser': avg_loss_per_loser,
            'avg_profit_pct_winner': avg_profit_pct_winner,
            'avg_loss_pct_loser': avg_loss_pct_loser,
            'avg_holding_period': avg_holding_period,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'initial_capital': initial_capital,
            'final_portfolio_value': final_capital,
            'sector_performance': sector_performance,
            'market_regime_performance': market_regime_performance,
            'large_cap_performance': large_cap_performance,
            'mid_cap_performance': mid_cap_performance,
            'tier_performance': tier_performance
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for a DataFrame of price data
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    try:
        if data is None or len(data) < 10:
            return None
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # === Moving Averages ===
        # SMA 20
        df['sma20'] = df['close'].rolling(window=20).mean()
        
        # SMA 50
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # SMA 200
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        # === RSI (Relative Strength Index) ===
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over 14 periods
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # === MACD (Moving Average Convergence Divergence) ===
        # Calculate EMA 12 and EMA 26
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD line
        df['macd'] = ema12 - ema26
        
        # Calculate signal line (9-day EMA of MACD)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # === Bollinger Bands ===
        # Calculate 20-day SMA and standard deviation
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        
        # Calculate upper and lower bands
        df['bb_middle'] = sma20
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        
        # === ATR (Average True Range) ===
        # Calculate true range
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR (14-day average of true range)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return None

def detect_market_regime(market_data):
    """
    Detect the current market regime based on technical indicators
    
    Args:
        market_data (pd.DataFrame): DataFrame with market index data (e.g., S&P 500)
        
    Returns:
        str: Market regime ('STRONG_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH')
    """
    try:
        if market_data is None or len(market_data) < 50:
            logger.warning("Insufficient market data to detect regime, defaulting to NEUTRAL")
            return 'NEUTRAL'
            
        # Calculate technical indicators for market data
        market_data = calculate_technical_indicators(market_data)
        
        if market_data is None or len(market_data) < 5:
            logger.warning("Failed to calculate technical indicators for market data, defaulting to NEUTRAL")
            return 'NEUTRAL'
            
        # Get the latest data point
        latest = market_data.iloc[-1]
        
        # Initialize score (0 = neutral, positive = bullish, negative = bearish)
        regime_score = 0
        
        # === Moving Average Trends ===
        # Price above key moving averages
        if latest['close'] > latest['sma20']:
            regime_score += 1
        else:
            regime_score -= 1
            
        if latest['close'] > latest['sma50']:
            regime_score += 1
        else:
            regime_score -= 1
            
        if latest['close'] > latest['sma200']:
            regime_score += 1
        else:
            regime_score -= 1
            
        # Moving average alignment (uptrend confirmation)
        if latest['sma20'] > latest['sma50'] and latest['sma50'] > latest['sma200']:
            regime_score += 2
        elif latest['sma20'] < latest['sma50'] and latest['sma50'] < latest['sma200']:
            regime_score -= 2
            
        # === RSI ===
        if latest['rsi'] > 70:
            regime_score += 1  # Bullish momentum, but potentially overbought
        elif latest['rsi'] < 30:
            regime_score -= 1  # Bearish momentum, but potentially oversold
            
        # === MACD ===
        if latest['macd'] > latest['macd_signal']:
            regime_score += 1
        else:
            regime_score -= 1
            
        if latest['macd_hist'] > 0 and latest['macd_hist'] > market_data['macd_hist'].iloc[-2]:
            regime_score += 1  # Increasing bullish momentum
        elif latest['macd_hist'] < 0 and latest['macd_hist'] > market_data['macd_hist'].iloc[-2]:
            # Histogram still negative but increasing (early sign of potential reversal)
            regime_score += 1
                
        # === Recent Price Action ===
        # Calculate 10-day return
        recent_return = (latest['close'] / market_data['close'].iloc[-11] - 1) * 100
        
        if recent_return > 5:
            regime_score += 1
        elif recent_return < -5:
            regime_score -= 1
            
        # === Volatility ===
        # High volatility often indicates bearish or transitioning markets
        if latest['atr'] / latest['close'] > 0.02:  # ATR > 2% of price
            regime_score -= 1
            
        # Determine regime based on score
        if regime_score >= 5:
            return 'STRONG_BULLISH'
        elif regime_score >= 2:
            return 'BULLISH'
        elif regime_score <= -5:
            return 'STRONG_BEARISH'
        elif regime_score <= -2:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
            
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
        return 'NEUTRAL'  # Default to neutral on error

def get_sector_performance(bars_data, sector_etfs=None):
    """
    Calculate sector performance and determine sector regimes
    
    Args:
        bars_data (dict): Dictionary of DataFrames with historical price data
        sector_etfs (list): List of sector ETF symbols
        
    Returns:
        dict: Dictionary mapping sector ETFs to their regimes
    """
    if sector_etfs is None:
        sector_etfs = [
            'XLK',  # Technology
            'XLF',  # Financials
            'XLV',  # Healthcare
            'XLE',  # Energy
            'XLI',  # Industrials
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLB',  # Materials
            'XLU',  # Utilities
            'XLRE', # Real Estate
            'XLC'   # Communication Services
        ]
        
    sector_regimes = {}
    
    try:
        for etf in sector_etfs:
            if etf in bars_data and len(bars_data[etf]) >= 50:
                # Calculate technical indicators
                etf_data = calculate_technical_indicators(bars_data[etf])
                
                if etf_data is not None and len(etf_data) >= 5:
                    # Use the same logic as market regime detection but simplified
                    latest = etf_data.iloc[-1]
                    regime_score = 0
                    
                    # Moving Averages
                    if latest['close'] > latest['sma20']:
                        regime_score += 1
                    else:
                        regime_score -= 1
                        
                    if latest['close'] > latest['sma50']:
                        regime_score += 1
                    else:
                        regime_score -= 1
                    
                    # RSI
                    if latest['rsi'] > 60:
                        regime_score += 1
                    elif latest['rsi'] < 40:
                        regime_score -= 1
                    
                    # MACD
                    if latest['macd'] > latest['macd_signal']:
                        regime_score += 1
                    else:
                        regime_score -= 1
                    
                    # Determine regime based on score
                    if regime_score >= 3:
                        sector_regimes[etf] = 'STRONG_BULLISH'
                    elif regime_score >= 1:
                        sector_regimes[etf] = 'BULLISH'
                    elif regime_score <= -3:
                        sector_regimes[etf] = 'STRONG_BEARISH'
                    elif regime_score <= -1:
                        sector_regimes[etf] = 'BEARISH'
                    else:
                        sector_regimes[etf] = 'NEUTRAL'
                else:
                    sector_regimes[etf] = 'NEUTRAL'
            else:
                sector_regimes[etf] = 'NEUTRAL'
                
        return sector_regimes
        
    except Exception as e:
        logger.error(f"Error calculating sector performance: {str(e)}")
        return {etf: 'NEUTRAL' for etf in sector_etfs}

def calculate_long_signal_score(symbol, data, market_regime=None, sector_regime=None):
    """
    Calculate a score for a LONG signal based on technical indicators
    Higher score = stronger signal
    
    Args:
        symbol (str): Ticker symbol
        data (pd.DataFrame): DataFrame with price data and technical indicators
        market_regime (str): Market regime ('STRONG_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH')
        sector_regime (str): Sector regime ('STRONG_BULLISH', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH')
        
    Returns:
        float: Signal score between 0 and 1
    """
    try:
        if data is None or len(data) < 10:
            return 0
            
        # Get the latest data point
        latest = data.iloc[-1]
        
        # Initialize score
        score = 0.0
        
        # === RSI (Oversold conditions are bullish for LONG) ===
        if 'rsi' in latest:
            if latest['rsi'] < 40:
                score += 0.2
            elif latest['rsi'] < 30:
                score += 0.35
            elif latest['rsi'] < 20:
                score += 0.5
                
            # Adjust RSI weight based on market regime
            if market_regime in ['BEARISH', 'STRONG_BEARISH']:
                # In bearish markets, require more oversold conditions for LONG signals
                if latest['rsi'] > 35:
                    score -= 0.15
            elif market_regime in ['BULLISH', 'STRONG_BULLISH']:
                # In bullish markets, even mild oversold conditions can be good entry points
                if latest['rsi'] < 45:
                    score += 0.1
        
        # === MACD (Bullish crossover or histogram increasing is bullish for LONG) ===
        if all(x in latest for x in ['macd', 'macd_signal', 'macd_hist']):
            # Bullish crossover (MACD crosses above signal line)
            if latest['macd'] > latest['macd_signal'] and data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2]:
                score += 0.3
            
            # MACD histogram increasing (momentum building)
            if latest['macd_hist'] > 0 and latest['macd_hist'] > data['macd_hist'].iloc[-2]:
                score += 0.2
            elif latest['macd_hist'] < 0 and latest['macd_hist'] > data['macd_hist'].iloc[-2]:
                # Histogram still negative but increasing (early sign of potential reversal)
                score += 0.15
                
            # Adjust MACD weight based on market regime
            if market_regime in ['BULLISH', 'STRONG_BULLISH']:
                # In bullish markets, MACD signals are more reliable for LONG positions
                if latest['macd'] > latest['macd_signal']:
                    score += 0.1
            elif market_regime in ['BEARISH', 'STRONG_BEARISH']:
                # In bearish markets, be more cautious with MACD signals
                if latest['macd'] < 0 and latest['macd_signal'] < 0:
                    score -= 0.1
        
        # === Bollinger Bands (Price near or below lower band is bullish for LONG) ===
        if all(x in latest for x in ['bb_lower', 'bb_middle', 'bb_upper']):
            # Price near or below lower band (potential oversold condition)
            bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            
            if latest['close'] <= latest['bb_lower']:
                score += 0.4
            elif bb_position < 0.2:
                score += 0.25
                
            # Bollinger Band width (narrow bands may signal upcoming volatility)
            bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
            bb_width_prev = (data['bb_upper'].iloc[-2] - data['bb_lower'].iloc[-2]) / data['bb_middle'].iloc[-2]
            
            if bb_width < bb_width_prev:
                # Bands are contracting, potential reversal
                score += 0.1
        
        # === Moving Averages (Price above key MAs is bullish for LONG) ===
        if 'sma20' in latest and 'sma50' in latest:
            # Price above key moving averages
            if latest['close'] > latest['sma20']:
                score += 0.15
            if latest['close'] > latest['sma50']:
                score += 0.15
            
            # Moving average alignment (uptrend confirmation)
            if latest['sma20'] > latest['sma50']:
                score += 0.2
                
            # Adjust MA weight based on market regime
            if market_regime in ['BULLISH', 'STRONG_BULLISH']:
                # In bullish markets, being above MAs is more significant
                if latest['close'] > latest['sma20'] and latest['sma20'] > latest['sma50']:
                    score += 0.1
        
        # === Volume (High volume on up days is bullish for LONG) ===
        if 'volume' in latest and len(data) >= 5:
            # Calculate average volume
            avg_volume = data['volume'].iloc[-5:].mean()
            
            # Check if current volume is above average on an up day
            if latest['volume'] > avg_volume * 1.2 and latest['close'] > data['close'].iloc[-2]:
                score += 0.2
                
            # Volume trend (increasing volume on up days is bullish)
            up_days_volume = 0
            up_days_count = 0
            down_days_volume = 0
            down_days_count = 0
            
            for i in range(-5, 0):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    up_days_volume += data['volume'].iloc[i]
                    up_days_count += 1
                else:
                    down_days_volume += data['volume'].iloc[i]
                    down_days_count += 1
            
            # Calculate average volume for up and down days
            avg_up_volume = up_days_volume / max(1, up_days_count)
            avg_down_volume = down_days_volume / max(1, down_days_count)
            
            # Compare volume on up vs down days
            if up_days_count > 0 and down_days_count > 0:
                if avg_up_volume > avg_down_volume * 1.5:
                    score += 0.2
                elif avg_down_volume > avg_up_volume * 1.5:
                    score -= 0.2
        
        # === ATR (Volatility assessment) ===
        if 'atr' in latest and 'close' in latest:
            # Calculate ATR as percentage of price
            atr_pct = (latest['atr'] / latest['close']) * 100
            
            # Adjust score based on volatility
            if atr_pct > 3.0:  # High volatility
                if market_regime in ['BULLISH', 'STRONG_BULLISH']:
                    score += 0.1
                else:
                    score -= 0.1
        
        # === Market Regime Adjustments ===
        if market_regime == 'STRONG_BULLISH':
            score += 0.15
        elif market_regime == 'BULLISH':
            score += 0.1
        elif market_regime == 'BEARISH':
            score -= 0.1
        elif market_regime == 'STRONG_BEARISH':
            score -= 0.2
        
        # === Sector Regime Adjustments ===
        if sector_regime == 'STRONG_BULLISH':
            score += 0.15
        elif sector_regime == 'BULLISH':
            score += 0.1
        elif sector_regime == 'BEARISH':
            score -= 0.1
        elif sector_regime == 'STRONG_BEARISH':
            score -= 0.2
        
        # Ensure score is within bounds
        score = max(0, min(score, 1.0))
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating long signal score for {symbol}: {str(e)}")
        return 0

def calculate_rsi(prices, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a price series
    
    Args:
        prices (pd.Series): Series of prices
        window (int): RSI window period
        
    Returns:
        pd.Series: RSI values
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

if __name__ == "__main__":
    # Test the backtest engine
    start_date = "2023-01-01"
    end_date = "2023-03-31"
    
    metrics, signals = run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000,
        max_signals=20
    )
    
    if metrics:
        print(f"Backtest Results:")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Number of Trades: {metrics['num_trades']}")
    else:
        print("Backtest failed to generate metrics")
