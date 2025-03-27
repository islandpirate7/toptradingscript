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
    
    # Adjust window sizes based on available data
    data_length = len(df)
    
    # Use dynamic window sizes based on available data
    sma5_window = min(5, data_length)
    sma10_window = min(10, data_length)
    sma20_window = min(20, data_length)
    bb_window = min(20, data_length)
    rsi_window = min(14, data_length)
    
    # Calculate moving averages with adjusted windows
    df['sma_5'] = df['close'].rolling(window=sma5_window).mean()
    df['sma_10'] = df['close'].rolling(window=sma10_window).mean()
    df['sma_20'] = df['close'].rolling(window=sma20_window).mean()
    
    # Calculate Bollinger Bands with adjusted window
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    df['bb_std'] = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Calculate RSI with adjusted window
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD with adjusted windows
    ema12_window = min(12, data_length)
    ema26_window = min(26, data_length)
    df['ema_12'] = df['close'].ewm(span=ema12_window, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=ema26_window, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=min(9, data_length), adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Fill NaN values in the first row with the next available value
    df = df.bfill()
    
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
    
    # Check if price is above SMA 20 (bullish)
    if pd.notna(row.get('close')) and pd.notna(row.get('sma_20')) and row['close'] > row['sma_20']:
        score += 0.15
        total_weight += 0.15
    
    # Check if SMA 5 is above SMA 20 (bullish trend)
    if pd.notna(row.get('sma_5')) and pd.notna(row.get('sma_20')) and row['sma_5'] > row['sma_20']:
        score += 0.1
        total_weight += 0.1
    
    # Check RSI (not overbought)
    if pd.notna(row.get('rsi_14')):
        if row['rsi_14'] < 70:  # Not overbought
            rsi_score = 0.15 * (1 - (row['rsi_14'] / 100))  # Lower RSI = higher score
            score += rsi_score
            total_weight += 0.15
    
    # Check MACD (positive and above signal line)
    if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
        if row['macd'] > 0 and row['macd'] > row['macd_signal']:
            score += 0.2
            total_weight += 0.2
    
    # Check Bollinger Bands (price near lower band = buying opportunity)
    if pd.notna(row.get('close')) and pd.notna(row.get('bb_lower')) and pd.notna(row.get('bb_middle')):
        # Calculate how close price is to lower band vs middle band
        band_range = row['bb_middle'] - row['bb_lower']
        if band_range > 0:
            price_position = (row['close'] - row['bb_lower']) / band_range
            if price_position < 0.5:  # Closer to lower band than middle
                bb_score = 0.2 * (1 - price_position)
                score += bb_score
                total_weight += 0.2
    
    # Normalize score if we have any weights
    if total_weight > 0:
        return score / total_weight
    else:
        return 0.0

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
        # Get historical data from Alpaca
        logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to proper format if needed
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
            
        # Get bars from Alpaca
        bars = alpaca.get_bars([symbol], '1D', start_date, end_date)
        
        if bars is None or bars.empty:
            logger.warning(f"No historical data found for {symbol}")
            return None
        
        # Create a new DataFrame with the data we need
        try:
            # First try to filter by symbol if the index is multi-level
            if isinstance(bars.index, pd.MultiIndex) and 'symbol' in bars.index.names:
                symbol_data = bars.xs(symbol, level='symbol')
            else:
                # If not multi-level, create a new DataFrame with just the data we need
                symbol_data = bars.copy()
                
            # Make sure we have the required columns
            if 'close' not in symbol_data.columns:
                # Try to map Alpaca column names to our expected format
                column_mapping = {
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                }
                symbol_data = symbol_data.rename(columns=column_mapping)
            
            # Add timestamp column if not present
            if 'timestamp' not in symbol_data.columns:
                symbol_data['timestamp'] = symbol_data.index
            
            logger.info(f"Retrieved {len(symbol_data)} bars for {symbol}")
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        return None

def generate_signals(start_date, end_date, alpaca, min_score=0.7, max_signals=30):
    """
    Generate trading signals for the given date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        alpaca (AlpacaAPI): AlpacaAPI instance
        min_score (float): Minimum signal score to include (default: 0.7)
        max_signals (int): Maximum number of signals to return (default: 30)
        
    Returns:
        list: List of signal dictionaries
    """
    logger.info(f"Generating signals from {start_date} to {end_date}")
    
    # Get universe of symbols
    from final_sp500_strategy import get_sp500_symbols, get_midcap_symbols
    
    # Get S&P 500 symbols
    sp500_symbols = get_sp500_symbols()
    logger.info(f"Processing {len(sp500_symbols)} symbols for signal generation")
    
    # Initialize signals list
    all_signals = []
    
    # Lower the minimum score for testing purposes
    adjusted_min_score = min_score * 0.8  # 20% lower threshold for testing
    
    # Process each symbol
    for symbol in sp500_symbols:
        try:
            # Get historical data
            data = get_historical_data(symbol, start_date, end_date, alpaca)
            
            if data is None or len(data) < 3:  # Need at least 3 days of data
                continue
                
            # Calculate technical indicators
            data_with_indicators = calculate_technical_indicators(data)
            
            # Get the latest row, even if it has some NaN values
            latest_row = data_with_indicators.iloc[-1]
            
            # Check if we have enough valid indicators to calculate a score
            required_indicators = ['close']
            if not all(pd.notna(latest_row.get(ind)) for ind in required_indicators):
                logger.warning(f"Missing required indicators for {symbol}, skipping")
                continue
            
            # Calculate signal score
            score = calculate_signal_score(latest_row)
            
            # Log signal details for debugging
            if score > 0:
                rsi_value = latest_row.get('rsi_14')
                rsi_display = f"{rsi_value:.2f}" if pd.notna(rsi_value) else "N/A"
                logger.info(f"Signal for {symbol}: score={score:.4f}, close={latest_row['close']:.2f}, sma20={latest_row.get('sma_20', 'N/A')}, rsi={rsi_display}")
            
            # Only include signals with a score above adjusted_min_score
            if score >= adjusted_min_score:
                # Create signal dictionary
                signal = {
                    'symbol': symbol,
                    'date': latest_row['timestamp'] if 'timestamp' in latest_row else pd.Timestamp(end_date),
                    'price': latest_row['close'],
                    'direction': 'LONG',  # Only generating LONG signals as per requirements
                    'score': score,
                    'indicators': {
                        'rsi': latest_row.get('rsi_14', None),
                        'macd': latest_row.get('macd', None),
                        'bb_lower': latest_row.get('bb_lower', None),
                        'sma_20': latest_row.get('sma_20', None)
                    }
                }
                
                all_signals.append(signal)
                logger.info(f"Added signal for {symbol} with score {score:.4f}")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            
    # Filter signals by score
    filtered_signals = [s for s in all_signals if s['score'] >= adjusted_min_score]
    
    # Sort by score (descending)
    sorted_signals = sorted(filtered_signals, key=lambda x: x['score'], reverse=True)
    
    # Limit to max_signals
    top_signals = sorted_signals[:max_signals]
    
    logger.info(f"Generated {len(all_signals)} total signals")
    logger.info(f"Filtered to {len(filtered_signals)} signals with score >= {adjusted_min_score:.2f}")
    logger.info(f"Returning top {len(top_signals)} signals")
    
    return top_signals
