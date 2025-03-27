#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Seasonality Data

This script generates real seasonality data based on historical stock performance.
It analyzes market-wide, sector-specific, and stock-specific seasonal patterns
and saves them to a YAML file that can be used by the trading strategy.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import yaml
import json
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/seasonality_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# S&P 500 sector ETFs
SECTOR_ETFS = {
    'XLF': 'Financial',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

# Market ETF
MARKET_ETF = 'SPY'

def load_api_credentials(credentials_path: str) -> Tuple[str, str]:
    """Load Alpaca API credentials from a JSON file
    
    Args:
        credentials_path (str): Path to the credentials JSON file
        
    Returns:
        Tuple[str, str]: API key and secret
    """
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        
        api_key = credentials.get('APCA_API_KEY_ID')
        api_secret = credentials.get('APCA_API_SECRET_KEY')
        
        if not api_key or not api_secret:
            raise ValueError("API credentials file must contain APCA_API_KEY_ID and APCA_API_SECRET_KEY")
            
        return api_key, api_secret
    except Exception as e:
        logger.error(f"Error loading API credentials: {str(e)}")
        raise

def get_sp500_symbols() -> List[str]:
    """Get a list of S&P 500 symbols
    
    Returns:
        List[str]: List of S&P 500 stock symbols
    """
    try:
        # Try to fetch S&P 500 symbols from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (replace . with - for compatibility with APIs)
            symbols = [s.replace('.', '-') for s in symbols]
            
            logger.info(f"Successfully fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols
        else:
            logger.warning("Failed to fetch S&P 500 symbols from Wikipedia, using backup list")
    except Exception as e:
        logger.warning(f"Error fetching S&P 500 symbols: {str(e)}, using backup list")
    
    # Backup list of some major S&P 500 stocks
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
        'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'ABBV',
        'AVGO', 'LLY', 'COST', 'TMO', 'MCD', 'ABT', 'CSCO', 'ACN', 'WMT', 'CRM',
        'PFE', 'BAC', 'DIS', 'ADBE', 'TXN', 'CMCSA', 'NKE', 'NEE', 'VZ', 'PM',
        'INTC', 'DHR', 'AMD', 'QCOM', 'UPS', 'IBM', 'AMGN', 'SBUX', 'INTU', 'LOW'
    ]

def fetch_historical_data(symbols: List[str], api_key: str, api_secret: str, 
                         start_date: str, end_date: str = None, 
                         timeframe: str = "1Day") -> Dict[str, pd.DataFrame]:
    """Fetch historical data for a list of symbols
    
    Args:
        symbols (List[str]): List of stock symbols
        api_key (str): Alpaca API key
        api_secret (str): Alpaca API secret
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
        timeframe (str, optional): Data timeframe. Defaults to "1Day".
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
    """
    if not symbols:
        logger.warning("Symbol list is empty. Cannot fetch data.")
        return {}
        
    # Set end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    logger.info(f"Fetching historical data from {start_date} to {end_date} for {len(symbols)} symbols")
    
    # Base URL for the Alpaca Data API
    base_url = "https://data.alpaca.markets/v2"
    
    # Headers for authentication
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    
    # Format dates for API
    start = pd.Timestamp(start_date, tz='America/New_York').isoformat()
    end = pd.Timestamp(end_date, tz='America/New_York').isoformat()
    
    # Dictionary to store data for each symbol
    data_dict = {}
    
    # Process symbols with progress bar
    for symbol in tqdm(symbols, desc="Fetching historical data"):
        try:
            # Construct URL for bars endpoint
            url = f"{base_url}/stocks/{symbol}/bars"
            
            # Parameters for the request
            params = {
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "adjustment": "all"  # Adjust for splits, dividends, etc.
            }
            
            # Make the request
            response = requests.get(url, headers=headers, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response
                data = response.json()
                
                # Extract bars data
                bars = data.get('bars', [])
                
                if bars:
                    # Convert to DataFrame
                    df = pd.DataFrame(bars)
                    
                    # Convert timestamp to datetime
                    df['t'] = pd.to_datetime(df['t'])
                    
                    # Set timestamp as index
                    df.set_index('t', inplace=True)
                    
                    # Rename columns
                    df.rename(columns={
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    }, inplace=True)
                    
                    # Store in dictionary
                    data_dict[symbol] = df
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.1)
                else:
                    logger.warning(f"No data returned for {symbol}")
            else:
                logger.warning(f"Failed to fetch data for {symbol}: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
    logger.info(f"Successfully fetched data for {len(data_dict)} out of {len(symbols)} symbols")
    return data_dict

def calculate_daily_returns(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Calculate daily returns for each symbol
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary of historical price data
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of daily returns
    """
    returns_dict = {}
    
    for symbol, df in data_dict.items():
        try:
            # Calculate daily returns
            df_returns = df.copy()
            df_returns['return'] = df['close'].pct_change()
            
            # Add date components for seasonality analysis
            df_returns['date'] = df_returns.index.date
            df_returns['month'] = df_returns.index.month
            df_returns['day_of_month'] = df_returns.index.day
            df_returns['day_of_week'] = df_returns.index.dayofweek
            df_returns['year'] = df_returns.index.year
            
            # Drop NaN values (first row will have NaN return)
            df_returns.dropna(subset=['return'], inplace=True)
            
            returns_dict[symbol] = df_returns
        except Exception as e:
            logger.error(f"Error calculating returns for {symbol}: {str(e)}")
    
    return returns_dict

def analyze_market_seasonality(market_data: pd.DataFrame) -> Dict[str, float]:
    """Analyze market-wide seasonality patterns
    
    Args:
        market_data (pd.DataFrame): Historical data for market ETF
        
    Returns:
        Dict[str, float]: Dictionary mapping date keys to seasonality scores
    """
    logger.info("Analyzing market-wide seasonality patterns")
    
    # Group by month and day of month
    grouped = market_data.groupby(['month', 'day_of_month'])
    
    # Calculate average return for each month-day combination
    avg_returns = grouped['return'].mean()
    
    # Convert to dictionary with date keys in format "MM-DD"
    seasonality_dict = {}
    
    for (month, day), avg_return in avg_returns.items():
        date_key = f"{int(month):02d}-{int(day):02d}"
        
        # Convert average return to a seasonality score between 0 and 1
        # Positive returns -> score > 0.5, Negative returns -> score < 0.5
        # Scale based on the magnitude of the return
        if avg_return > 0:
            # Scale positive returns to 0.5-1.0 range
            score = 0.5 + min(0.5, abs(avg_return) * 10)
        else:
            # Scale negative returns to 0.0-0.5 range
            score = 0.5 - min(0.5, abs(avg_return) * 10)
            
        seasonality_dict[date_key] = round(score, 3)
    
    logger.info(f"Generated {len(seasonality_dict)} market seasonality patterns")
    return seasonality_dict

def analyze_sector_seasonality(sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Analyze sector-specific seasonality patterns
    
    Args:
        sector_data (Dict[str, pd.DataFrame]): Historical data for sector ETFs
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping sectors to their seasonality patterns
    """
    logger.info("Analyzing sector-specific seasonality patterns")
    
    sector_seasonality = {}
    
    for etf, df in sector_data.items():
        sector_name = SECTOR_ETFS[etf]
        logger.info(f"Processing sector: {sector_name}")
        
        # Group by month and day of month
        grouped = df.groupby(['month', 'day_of_month'])
        
        # Calculate average return for each month-day combination
        avg_returns = grouped['return'].mean()
        
        # Convert to dictionary with date keys in format "MM-DD"
        seasonality_dict = {}
        
        for (month, day), avg_return in avg_returns.items():
            date_key = f"{int(month):02d}-{int(day):02d}"
            
            # Convert average return to a seasonality score between 0 and 1
            if avg_return > 0:
                # Scale positive returns to 0.5-1.0 range
                score = 0.5 + min(0.5, abs(avg_return) * 10)
            else:
                # Scale negative returns to 0.0-0.5 range
                score = 0.5 - min(0.5, abs(avg_return) * 10)
                
            seasonality_dict[date_key] = round(score, 3)
        
        sector_seasonality[sector_name] = seasonality_dict
        logger.info(f"Generated {len(seasonality_dict)} seasonality patterns for {sector_name}")
    
    return sector_seasonality

def analyze_stock_seasonality(stock_data: Dict[str, pd.DataFrame], 
                             min_data_points: int = 3) -> Dict[str, Dict[str, float]]:
    """Analyze stock-specific seasonality patterns
    
    Args:
        stock_data (Dict[str, pd.DataFrame]): Historical data for individual stocks
        min_data_points (int, optional): Minimum number of data points required. Defaults to 3.
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping stocks to their seasonality patterns
    """
    logger.info("Analyzing stock-specific seasonality patterns")
    
    stock_seasonality = {}
    
    for symbol, df in tqdm(stock_data.items(), desc="Processing stocks"):
        try:
            # Group by month and day of month
            grouped = df.groupby(['month', 'day_of_month'])
            
            # Calculate average return and count for each month-day combination
            agg_data = grouped['return'].agg(['mean', 'count'])
            
            # Filter out combinations with too few data points
            agg_data = agg_data[agg_data['count'] >= min_data_points]
            
            # Convert to dictionary with date keys in format "MM-DD"
            seasonality_dict = {}
            
            for (month, day), row in agg_data.iterrows():
                date_key = f"{int(month):02d}-{int(day):02d}"
                avg_return = row['mean']
                
                # Convert average return to a seasonality score between 0 and 1
                if avg_return > 0:
                    # Scale positive returns to 0.5-1.0 range
                    score = 0.5 + min(0.5, abs(avg_return) * 10)
                else:
                    # Scale negative returns to 0.0-0.5 range
                    score = 0.5 - min(0.5, abs(avg_return) * 10)
                    
                seasonality_dict[date_key] = round(score, 3)
            
            # Only include stocks with sufficient seasonality patterns
            if len(seasonality_dict) >= 20:  # Arbitrary threshold
                stock_seasonality[symbol] = seasonality_dict
        except Exception as e:
            logger.error(f"Error analyzing seasonality for {symbol}: {str(e)}")
    
    logger.info(f"Generated seasonality patterns for {len(stock_seasonality)} stocks")
    return stock_seasonality

def save_seasonality_data(market_seasonality: Dict[str, float],
                         sector_seasonality: Dict[str, Dict[str, float]],
                         stock_seasonality: Dict[str, Dict[str, float]],
                         output_file: str):
    """Save seasonality data to a YAML file
    
    Args:
        market_seasonality (Dict[str, float]): Market-wide seasonality patterns
        sector_seasonality (Dict[str, Dict[str, float]]): Sector-specific seasonality patterns
        stock_seasonality (Dict[str, Dict[str, float]]): Stock-specific seasonality patterns
        output_file (str): Path to output YAML file
    """
    logger.info(f"Saving seasonality data to {output_file}")
    
    # Combine all seasonality data
    seasonality_data = {
        'market': market_seasonality,
        'sectors': sector_seasonality,
        'stocks': stock_seasonality
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to YAML file
    with open(output_file, 'w') as f:
        yaml.dump(seasonality_data, f, default_flow_style=False)
    
    logger.info(f"Seasonality data saved to {output_file}")
    
    # Log summary statistics
    logger.info(f"Summary statistics:")
    logger.info(f"  Market patterns: {len(market_seasonality)}")
    logger.info(f"  Sectors: {len(sector_seasonality)}")
    logger.info(f"  Stocks: {len(stock_seasonality)}")
    
    total_patterns = len(market_seasonality)
    for sector, patterns in sector_seasonality.items():
        total_patterns += len(patterns)
    for stock, patterns in stock_seasonality.items():
        total_patterns += len(patterns)
    
    logger.info(f"  Total patterns: {total_patterns}")

def generate_seasonality_data(api_credentials_path: str, 
                             lookback_years: int = 5,
                             output_file: str = 'data/seasonality.yaml'):
    """Generate seasonality data based on historical performance
    
    Args:
        api_credentials_path (str): Path to Alpaca API credentials
        lookback_years (int, optional): Number of years to look back. Defaults to 5.
        output_file (str, optional): Path to output file. Defaults to 'data/seasonality.yaml'.
    """
    logger.info(f"Generating seasonality data with {lookback_years} years lookback")
    
    # Load API credentials
    api_key, api_secret = load_api_credentials(api_credentials_path)
    
    # Calculate start date based on lookback years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Get symbols
    sp500_symbols = get_sp500_symbols()
    
    # Combine all symbols to fetch
    all_symbols = sp500_symbols + list(SECTOR_ETFS.keys()) + [MARKET_ETF]
    all_symbols = list(set(all_symbols))  # Remove duplicates
    
    # Fetch historical data
    data_dict = fetch_historical_data(all_symbols, api_key, api_secret, start_date, end_date)
    
    if not data_dict:
        logger.error("Failed to fetch any historical data")
        return
    
    # Calculate daily returns
    returns_dict = calculate_daily_returns(data_dict)
    
    # Separate market, sector, and stock data
    market_data = returns_dict.get(MARKET_ETF)
    
    sector_data = {}
    for etf in SECTOR_ETFS:
        if etf in returns_dict:
            sector_data[etf] = returns_dict[etf]
    
    stock_data = {}
    for symbol in sp500_symbols:
        if symbol in returns_dict:
            stock_data[symbol] = returns_dict[symbol]
    
    # Analyze seasonality patterns
    market_seasonality = analyze_market_seasonality(market_data)
    sector_seasonality = analyze_sector_seasonality(sector_data)
    stock_seasonality = analyze_stock_seasonality(stock_data)
    
    # Save seasonality data
    save_seasonality_data(market_seasonality, sector_seasonality, stock_seasonality, output_file)
    
    logger.info("Seasonality data generation complete")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate seasonality data based on historical performance')
    parser.add_argument('--credentials', type=str, default='alpaca_credentials.json',
                        help='Path to Alpaca API credentials JSON file')
    parser.add_argument('--lookback', type=int, default=5,
                        help='Number of years to look back for historical data')
    parser.add_argument('--output', type=str, default='data/seasonality.yaml',
                        help='Path to output YAML file')
    
    args = parser.parse_args()
    
    try:
        generate_seasonality_data(args.credentials, args.lookback, args.output)
    except Exception as e:
        logger.error(f"Error generating seasonality data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
