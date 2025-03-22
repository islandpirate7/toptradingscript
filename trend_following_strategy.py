#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trend Following Strategy Implementation
"""

import logging
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define enums and data classes
class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class SignalStrength(Enum):
    WEAK_BUY = "weak_buy"
    MODERATE_BUY = "moderate_buy"
    STRONG_BUY = "strong_buy"
    WEAK_SELL = "weak_sell"
    MODERATE_SELL = "moderate_sell"
    STRONG_SELL = "strong_sell"

# We'll use the CandleData class from mean_reversion_enhanced.py
# This is just a placeholder for documentation
"""
@dataclass
class CandleData:
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
"""

@dataclass
class MarketState:
    """Current state of the market"""
    date: dt.datetime
    regime: str = "neutral"
    volatility: float = 0.0
    trend_strength: float = 0.0
    is_range_bound: bool = False

@dataclass
class Signal:
    """Trading signal"""
    symbol: str
    timestamp: dt.datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    strength: str = "medium"

class TrendFollowingStrategy:
    """Trend following strategy using moving average crossovers"""
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or {}
        self.name = "TrendFollowing"
        self.logger = logging.getLogger(f"Strategy.{self.name}")
        
        # Default parameters
        self.ema_short = 5
        self.ema_long = 20
        self.rsi_period = 14
        self.rsi_threshold = 50
        self.atr_period = 14
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0
        self.min_signal_score = 0.5
        
        # Load parameters from config if provided
        if config:
            self.ema_short = config.get('ema_short', self.ema_short)
            self.ema_long = config.get('ema_long', self.ema_long)
            self.rsi_period = config.get('rsi_period', self.rsi_period)
            self.rsi_threshold = config.get('rsi_threshold', self.rsi_threshold)
            self.atr_period = config.get('atr_period', self.atr_period)
            self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', self.stop_loss_atr_multiplier)
            self.take_profit_atr_multiplier = config.get('take_profit_atr_multiplier', self.take_profit_atr_multiplier)
            self.min_signal_score = config.get('min_signal_score', self.min_signal_score)
    
    def get_param(self, name: str, default=None):
        """Get a parameter value with a default fallback"""
        return self.config.get(name, default)
    
    def _calculate_ema(self, df):
        """Calculate EMAs"""
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
    
    def _calculate_rsi(self, df):
        """Calculate RSI"""
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df):
        """Calculate ATR"""
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
    
    def generate_signals(self, df, symbol=None):
        """Generate trading signals based on trend following strategy
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            symbol (str, optional): Symbol for the data
            
        Returns:
            list: List of signal dictionaries
        """
        # Make a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Calculate indicators
        self._calculate_ema(df)
        self._calculate_rsi(df)
        self._calculate_atr(df)
        
        # Initialize signals list
        signals = []
        
        # Generate signals based on EMA crossover and RSI
        for i in range(max(self.ema_short, self.ema_long, self.rsi_period, self.atr_period) + 1, len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr'] if 'atr' in df.columns and not pd.isna(df.iloc[i]['atr']) else 0
            
            # Skip if ATR is NaN
            if pd.isna(current_atr):
                continue
            
            # Calculate stop loss and take profit levels based on ATR
            stop_loss_long = current_price - (self.stop_loss_atr_multiplier * current_atr)
            take_profit_long = current_price + (self.take_profit_atr_multiplier * current_atr)
            
            stop_loss_short = current_price + (self.stop_loss_atr_multiplier * current_atr)
            take_profit_short = current_price - (self.take_profit_atr_multiplier * current_atr)
            
            # Check for long signal (short EMA crosses above long EMA and RSI > threshold)
            if (df.iloc[i]['ema_short'] > df.iloc[i]['ema_long'] and 
                df.iloc[i-1]['ema_short'] <= df.iloc[i-1]['ema_long'] and 
                df.iloc[i]['rsi'] > self.rsi_threshold):
                
                # Create signal dictionary
                signal = {
                    'date': current_date,
                    'direction': 'LONG',
                    'price': current_price,
                    'stop_loss': stop_loss_long,
                    'take_profit': take_profit_long,
                    'strength': 'strong',
                    'strength_value': 1.0,
                    'strategy': 'trend_following',
                    'atr': current_atr,
                    'weight': 1.5
                }
                
                # Add symbol to signal if provided
                if symbol:
                    signal['symbol'] = symbol
                
                signals.append(signal)
            
            # Check for short signal (short EMA crosses below long EMA and RSI < threshold)
            elif (df.iloc[i]['ema_short'] < df.iloc[i]['ema_long'] and 
                  df.iloc[i-1]['ema_short'] >= df.iloc[i-1]['ema_long'] and 
                  df.iloc[i]['rsi'] < self.rsi_threshold):
                
                # Create signal dictionary
                signal = {
                    'date': current_date,
                    'direction': 'SHORT',
                    'price': current_price,
                    'stop_loss': stop_loss_short,
                    'take_profit': take_profit_short,
                    'strength': 'strong',
                    'strength_value': 1.0,
                    'strategy': 'trend_following',
                    'atr': current_atr,
                    'weight': 1.5
                }
                
                # Add symbol to signal if provided
                if symbol:
                    signal['symbol'] = symbol
                
                signals.append(signal)
        
        # Generate ultra aggressive signals if no signals were generated
        if len(signals) == 0 and len(df) > 0:
            # Generate signals for the last 20 days
            num_days = min(20, len(df))
            for i in range(len(df) - num_days, len(df)):
                current_date = df.index[i]
                current_price = df.iloc[i]['close']
                current_atr = df.iloc[i]['atr'] if 'atr' in df.columns and not pd.isna(df.iloc[i]['atr']) else 0
                
                # Determine direction based on EMA slope
                direction = 'LONG'
                if i > 0 and 'ema_short' in df.columns:
                    if df.iloc[i]['ema_short'] < df.iloc[i-1]['ema_short']:
                        direction = 'SHORT'
                
                # Calculate stop loss and take profit levels based on ATR
                if current_atr > 0:
                    if direction == 'LONG':
                        stop_loss = current_price - (self.stop_loss_atr_multiplier * current_atr)
                        take_profit = current_price + (self.take_profit_atr_multiplier * current_atr)
                    else:  # SHORT
                        stop_loss = current_price + (self.stop_loss_atr_multiplier * current_atr)
                        take_profit = current_price - (self.take_profit_atr_multiplier * current_atr)
                else:
                    # If ATR is 0 or NaN, use a percentage of the price
                    if direction == 'LONG':
                        stop_loss = current_price * 0.95  # 5% below current price
                        take_profit = current_price * 1.05  # 5% above current price
                    else:  # SHORT
                        stop_loss = current_price * 1.05  # 5% above current price
                        take_profit = current_price * 0.95  # 5% below current price
                
                # Create signal dictionary
                signal = {
                    'date': current_date,
                    'direction': direction,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strength': 'strong',
                    'strength_value': 1.0,
                    'strategy': 'trend_following',
                    'atr': current_atr,
                    'weight': 1.5
                }
                
                # Add symbol to signal if provided
                if symbol:
                    signal['symbol'] = symbol
                
                signals.append(signal)
        
        # Log the number of signals generated
        self.logger.info(f"Generated {len(signals)} raw trend following signals for {symbol}")
        
        # Limit the number of signals to avoid overwhelming the system
        if len(signals) > 20:
            signals = signals[-20:]  # Keep only the most recent 20 signals
        
        self.logger.info(f"Returning {len(signals)} processed trend following signals for {symbol}")
        
        return signals
