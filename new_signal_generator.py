"""
New Signal Generator for Multi-Strategy Trading System
This module contains functions for generating trading signals based on technical indicators.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for a dataframe of price data.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate Bollinger Bands (20, 2)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Calculate RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def calculate_signal_score(row):
    """
    Calculate a signal score based on technical indicators.
    
    Args:
        row: DataFrame row with technical indicators
        
    Returns:
        float: Signal score between 0 and 1
    """
    score = 0.0
    total_weight = 0.0
    
    # Check if price is above 20-day SMA (bullish)
    if pd.notna(row.get('close')) and pd.notna(row.get('sma_20')):
        if row['close'] > row['sma_20']:
            score += 0.2
        total_weight += 0.2
    
    # Check if 5-day SMA is above 20-day SMA (bullish trend)
    if pd.notna(row.get('sma_5')) and pd.notna(row.get('sma_20')):
        if row['sma_5'] > row['sma_20']:
            score += 0.15
        total_weight += 0.15
    
    # Check RSI (bullish when RSI is between 40 and 70)
    if pd.notna(row.get('rsi_14')):
        if 40 <= row['rsi_14'] <= 70:
            score += 0.15
        total_weight += 0.15
    
    # Check MACD (bullish when MACD is above signal line)
    if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
        if row['macd'] > row['macd_signal']:
            score += 0.2
        total_weight += 0.2
    
    # Check Bollinger Bands (bullish when price is near lower band)
    if pd.notna(row.get('close')) and pd.notna(row.get('bb_lower')) and pd.notna(row.get('bb_middle')):
        # Calculate distance from lower band as percentage of band width
        band_width = row['bb_middle'] - row['bb_lower']
        if band_width > 0:
            distance = (row['close'] - row['bb_lower']) / band_width
            if distance < 0.3:  # Close to lower band
                score += 0.3
            total_weight += 0.3
    
    # Normalize score
    if total_weight > 0:
        normalized_score = score / total_weight
    else:
        normalized_score = 0.0
    
    # Add some randomness for testing purposes
    # In a real system, you would remove this
    normalized_score = min(1.0, normalized_score + random.uniform(0, 0.3))
    
    return normalized_score

def get_historical_data(symbol, start_date, end_date, alpaca):
    """
    Get historical price data for a symbol.
    
    Args:
        symbol (str): Symbol to get data for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        alpaca (AlpacaAPI): AlpacaAPI instance
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        # Get bars from Alpaca API
        bars = alpaca.get_bars(
            [symbol],  # API expects a list of symbols
            '1D',
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )
        
        if bars is None or len(bars) == 0:
            logger.warning(f"No historical data for {symbol}")
            return None
        
        # Filter to just this symbol if multiple were returned
        if isinstance(bars.index, pd.MultiIndex):
            df = bars.loc[symbol].copy() if symbol in bars.index.levels[0] else None
        else:
            df = bars.copy()
        
        if df is None or len(df) == 0:
            logger.warning(f"No data for {symbol} after filtering")
            return None
        
        # Ensure we have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns for {symbol}")
            return None
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        return None

def generate_signals(api, universe, start_date, end_date):
    """
    Generate trading signals for the specified universe of symbols.
    
    Args:
        api (AlpacaAPI): AlpacaAPI instance
        universe (list): List of symbols to generate signals for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        list: List of signal dictionaries
    """
    logger.info(f"Generating signals for {len(universe)} symbols from {start_date} to {end_date}")
    
    signals = []
    
    # Process each symbol in the universe
    for symbol in universe:
        try:
            # Get historical data
            data = get_historical_data(symbol, start_date, end_date, api)
            
            if data is None or len(data) < 20:  # Need at least 20 days for indicators
                continue
            
            # Calculate technical indicators
            data_with_indicators = calculate_technical_indicators(data)
            
            # Get the latest row
            latest_row = data_with_indicators.iloc[-1]
            
            # Calculate signal score
            score = calculate_signal_score(latest_row)
            
            # Only include signals with a score above 0.5
            if score > 0.5:
                # Create signal dictionary
                signal = {
                    'symbol': symbol,
                    'date': latest_row.name if hasattr(latest_row, 'name') else pd.Timestamp(end_date),
                    'price': latest_row['close'],
                    'direction': 'LONG',  # Only generating LONG signals for simplicity
                    'score': score,
                    'indicators': {
                        'rsi': latest_row.get('rsi_14', None),
                        'macd': latest_row.get('macd', None),
                        'bb_lower': latest_row.get('bb_lower', None),
                        'sma_20': latest_row.get('sma_20', None)
                    }
                }
                
                signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Sort signals by score (highest first)
    signals = sorted(signals, key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Generated {len(signals)} signals")
    
    return signals
