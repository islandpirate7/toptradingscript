#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified backtest script for S&P 500 stock selection with 100 stocks
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import alpaca_trade_api as tradeapi
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest_100_stocks.log')
    ]
)
logger = logging.getLogger(__name__)

def load_alpaca_credentials(mode='paper'):
    """Load Alpaca API credentials from JSON file"""
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        raise

def get_top_100_sp500_symbols():
    """Get a static list of top 100 S&P 500 symbols"""
    return [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH",
        "LLY", "JPM", "V", "XOM", "AVGO", "PG", "MA", "HD", "COST", "MRK",
        "CVX", "ABBV", "PEP", "KO", "ADBE", "WMT", "CRM", "BAC", "TMO", "MCD",
        "CSCO", "PFE", "NFLX", "CMCSA", "ABT", "ORCL", "TMUS", "AMD", "DIS", "ACN",
        "DHR", "VZ", "NKE", "TXN", "NEE", "WFC", "PM", "INTC", "INTU", "COP",
        "AMGN", "IBM", "RTX", "HON", "QCOM", "UPS", "CAT", "LOW", "SPGI", "BA",
        "GE", "ELV", "DE", "AMAT", "ISRG", "AXP", "BKNG", "MDLZ", "GILD", "ADI",
        "SBUX", "TJX", "MMC", "SYK", "VRTX", "PLD", "MS", "BLK", "SCHW", "C",
        "ZTS", "CB", "AMT", "ADP", "GS", "ETN", "LRCX", "NOW", "MO", "REGN",
        "EOG", "SO", "BMY", "EQIX", "BSX", "CME", "CI", "PANW", "TGT", "SLB"
    ]

def load_historical_data(api, symbols, start_date, end_date):
    """Load historical data for symbols"""
    logger.info(f"Loading historical data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Load data for each symbol
    data = {}
    for symbol in tqdm(symbols, desc="Loading data"):
        try:
            # Get daily bars
            bars = api.get_bars(
                symbol,
                '1D',
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                adjustment='raw'
            ).df
            
            if len(bars) > 0:
                data[symbol] = bars
            else:
                logger.warning(f"No data available for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    logger.info(f"Loaded data for {len(data)} symbols")
    return data

def calculate_technical_indicators(data):
    """Calculate technical indicators for each symbol"""
    for symbol, df in data.items():
        try:
            # Calculate RSI (14-day)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma20'] + (df['std20'] * 2)
            df['lower_band'] = df['sma20'] - (df['std20'] * 2)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
    
    return data

def run_backtest(start_date, end_date, holding_period=5):
    """Run backtest for the given period"""
    # Initialize Alpaca API
    credentials = load_alpaca_credentials(mode='paper')
    api = tradeapi.REST(
        key_id=credentials['api_key'],
        secret_key=credentials['api_secret'],
        base_url=credentials['base_url']
    )
    
    # Get top 100 S&P 500 symbols
    symbols = get_top_100_sp500_symbols()
    
    # Load historical data
    data = load_historical_data(api, symbols, start_date, end_date)
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Generate trading dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    trading_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    # Initialize results
    all_returns = []
    
    # Run backtest for each date
    for date in tqdm(trading_dates, desc="Running backtest"):
        date_str = date.strftime('%Y-%m-%d')
        
        # For each symbol, calculate returns over holding period
        for symbol, df in data.items():
            try:
                # Find the closest date in the dataframe
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                
                # Calculate end index
                end_idx = min(idx + holding_period, len(df) - 1)
                
                # Skip if we don't have enough data
                if end_idx <= idx:
                    continue
                
                # Get prices
                start_price = df.iloc[idx]['close']
                end_price = df.iloc[end_idx]['close']
                
                # Calculate return for LONG position
                ret = (end_price / start_price - 1) * 100
                
                # Determine direction based on RSI
                rsi = df.iloc[idx]['rsi'] if 'rsi' in df.columns and not pd.isna(df.iloc[idx]['rsi']) else 50
                
                if rsi < 35:  # Oversold - go LONG
                    direction = 'LONG'
                elif rsi > 70:  # Overbought - go SHORT
                    direction = 'SHORT'
                    ret = -ret  # Invert return for SHORT
                else:
                    direction = 'NEUTRAL'
                    ret = 0  # No position for NEUTRAL
                
                # Add to results
                all_returns.append({
                    'symbol': symbol,
                    'date': date_str,
                    'direction': direction,
                    'start_price': start_price,
                    'end_price': end_price,
                    'return': ret,
                    'rsi': rsi
                })
            
            except Exception as e:
                logger.error(f"Error processing {symbol} on {date_str}: {str(e)}")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_returns)
    
    # Calculate overall statistics
    if len(results_df) > 0:
        total_trades = len(results_df)
        trades_with_returns = len(results_df[results_df['direction'] != 'NEUTRAL'])
        win_rate = len(results_df[results_df['return'] > 0]) / trades_with_returns * 100 if trades_with_returns > 0 else 0
        avg_return = results_df['return'].mean()
        total_return = results_df['return'].sum()
        
        logger.info(f"Backtest completed for period {start_date} to {end_date}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Average return: {avg_return:.2f}%")
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Calculate dollar return based on $100,000 starting capital
        position_size = 100000 / 100  # $1,000 per position
        dollar_return = (total_return / 100) * position_size * trades_with_returns
        logger.info(f"Dollar return (based on $100,000 capital): ${dollar_return:.2f}")
        
        # Save results
        output_file = f'backtest_results_{start_date.replace("-", "")}_{end_date.replace("-", "")}_100stocks.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Return the results
        return results_df, dollar_return
    else:
        logger.warning("No valid trades found in the backtest period")
        return pd.DataFrame(), 0

if __name__ == "__main__":
    # Run backtest for Q1 2023
    run_backtest('2023-01-01', '2023-03-31')
