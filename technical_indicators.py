#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Indicators
-------------------------------------
This module provides implementations of common technical indicators
without relying on external libraries like TA-Lib.
"""

import numpy as np
import pandas as pd

def calculate_ema(series, period):
    """Calculate Exponential Moving Average
    
    Args:
        series (pd.Series): Price series
        period (int): EMA period
        
    Returns:
        pd.Series: EMA values
    """
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    """Calculate Simple Moving Average
    
    Args:
        series (pd.Series): Price series
        period (int): SMA period
        
    Returns:
        pd.Series: SMA values
    """
    return series.rolling(window=period).mean()

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index
    
    Args:
        series (pd.Series): Price series
        period (int): RSI period
        
    Returns:
        pd.Series: RSI values
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle division by zero
    rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=series.index)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands
    
    Args:
        series (pd.Series): Price series
        period (int): Bollinger Bands period
        std_dev (float): Number of standard deviations
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    middle_band = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        period (int): ADX period
        
    Returns:
        pd.Series: ADX values
    """
    # Calculate True Range
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Plus Directional Movement (+DM)
    plus_dm = high.diff()
    minus_dm = low.diff().multiply(-1)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    
    # Calculate Minus Directional Movement (-DM)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Calculate Smoothed +DM and -DM
    smoothed_plus_dm = plus_dm.rolling(window=period).sum()
    smoothed_minus_dm = minus_dm.rolling(window=period).sum()
    
    # Calculate Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = (smoothed_plus_dm / atr) * 100
    minus_di = (smoothed_minus_dm / atr) * 100
    
    # Calculate Directional Movement Index (DX)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    
    # Calculate Average Directional Index (ADX)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculate Moving Average Convergence Divergence
    
    Args:
        series (pd.Series): Price series
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal EMA period
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        period (int): ATR period
        
    Returns:
        pd.Series: ATR values
    """
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Close prices
        k_period (int): %K period
        d_period (int): %D period
        
    Returns:
        tuple: (stoch_k, stoch_d)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d
