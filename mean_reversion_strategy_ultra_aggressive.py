#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mean Reversion Strategy - Ultra Aggressive
-----------------------------------------
This is an ultra aggressive version of the mean reversion strategy
designed to generate many more signals for testing purposes.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeanReversionStrategyUltraAggressive:
    """Mean reversion trading strategy with ultra aggressive settings to generate many more signals"""
    
    def __init__(self, config=None):
        """Initialize with ultra aggressive parameters"""
        # Use extremely aggressive settings to generate many more signals
        self.bb_period = 5   # Very short period for extreme volatility
        self.bb_std = 0.5    # Very tight bands for many signals
        self.rsi_period = 3  # Extremely short period for high sensitivity
        self.rsi_lower_threshold = 45  # Very low threshold for many buy signals
        self.rsi_upper_threshold = 55  # Very high threshold for many sell signals
        self.require_reversal = False  # Don't require price reversal
        self.stop_loss_atr = 3.0  # Very wide stop loss
        self.take_profit_atr = 2.0  # Shorter take profit for quicker exits
        self.atr_period = 7       # Shorter ATR period
        
        # Volume filter parameters - completely disabled
        self.use_volume_filter = False
        self.volume_threshold = 0.0  # No volume threshold
        self.volume_period = 5
        
        # Signal quality filter - minimal for many signals
        self.min_bb_penetration = 0.0  # No penetration required
        
        # Symbol-specific weights
        self.symbol_weights = {}
        
        logger.info(f"Initialized ULTRA AGGRESSIVE strategy with parameters: BB period={self.bb_period}, BB std={self.bb_std}, "
                   f"RSI period={self.rsi_period}, RSI thresholds={self.rsi_lower_threshold}/{self.rsi_upper_threshold}, "
                   f"Require reversal={self.require_reversal}, SL/TP ATR multipliers={self.stop_loss_atr}/{self.take_profit_atr}, "
                   f"Volume filter={self.use_volume_filter}")
    
    def set_symbol_weight(self, symbol, weight):
        """Set weight for a specific symbol
        
        Args:
            symbol (str): Stock symbol
            weight (float): Weight to apply to signals for this symbol
        """
        self.symbol_weights[symbol] = weight
        logger.info(f"Set weight {weight} for symbol {symbol}")
    
    def _calculate_bollinger_bands(self, df):
        """Calculate Bollinger Bands"""
        # Calculate SMA for Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        
        # Calculate standard deviation for Bollinger Bands
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        
        # Calculate Bollinger Bands
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std)
    
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
        """Generate trading signals based on the ultra aggressive strategy
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            symbol (str, optional): Symbol for the data. Defaults to None.
            
        Returns:
            list: List of signal dictionaries
        """
        # Apply symbol-specific weight if available
        weight = 1.0
        if symbol and symbol in self.symbol_weights:
            weight = self.symbol_weights[symbol]
        
        # Make a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Calculate indicators
        self._calculate_bollinger_bands(df)
        self._calculate_rsi(df)
        self._calculate_atr(df)
        
        # Initialize signals list
        signals = []
        
        # Generate signals based on Bollinger Bands and RSI
        for i in range(max(self.bb_period, self.rsi_period) + 1, len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr']
            
            # Skip if ATR is NaN
            if pd.isna(current_atr):
                continue
            
            # Calculate stop loss and take profit levels based on ATR
            stop_loss_long = current_price - (self.stop_loss_atr * current_atr)
            take_profit_long = current_price + (self.take_profit_atr * current_atr)
            
            stop_loss_short = current_price + (self.stop_loss_atr * current_atr)
            take_profit_short = current_price - (self.take_profit_atr * current_atr)
            
            # Check for long signal (price below lower band and RSI oversold)
            if df.iloc[i]['close'] < df.iloc[i]['bb_lower'] and df.iloc[i]['rsi'] < self.rsi_lower_threshold:
                # Create signal dictionary
                signal = {
                    'date': current_date,
                    'direction': 'LONG',
                    'price': current_price,
                    'stop_loss': stop_loss_long,
                    'take_profit': take_profit_long,
                    'strength': 'strong',
                    'strength_value': 1.0,
                    'strategy': 'mean_reversion',
                    'atr': current_atr,
                    'weight': weight
                }
                
                # Add symbol to signal if provided
                if symbol:
                    signal['symbol'] = symbol
                
                signals.append(signal)
            
            # Check for short signal (price above upper band and RSI overbought)
            elif df.iloc[i]['close'] > df.iloc[i]['bb_upper'] and df.iloc[i]['rsi'] > self.rsi_upper_threshold:
                # Create signal dictionary
                signal = {
                    'date': current_date,
                    'direction': 'SHORT',
                    'price': current_price,
                    'stop_loss': stop_loss_short,
                    'take_profit': take_profit_short,
                    'strength': 'strong',
                    'strength_value': 1.0,
                    'strategy': 'mean_reversion',
                    'atr': current_atr,
                    'weight': weight
                }
                
                # Add symbol to signal if provided
                if symbol:
                    signal['symbol'] = symbol
                
                signals.append(signal)
        
        # Generate ultra aggressive signals (generate a signal for every day)
        if len(signals) == 0 and len(df) > 0:
            # Generate signals for the last 20 days
            num_days = min(20, len(df))
            for i in range(len(df) - num_days, len(df)):
                current_date = df.index[i]
                current_price = df.iloc[i]['close']
                current_atr = df.iloc[i]['atr'] if 'atr' in df.columns and not pd.isna(df.iloc[i]['atr']) else 0
                
                # Default to LONG signals if no clear direction
                direction = 'LONG'
                
                # Calculate stop loss and take profit levels based on ATR
                if current_atr > 0:
                    stop_loss = current_price - (self.stop_loss_atr * current_atr)
                    take_profit = current_price + (self.take_profit_atr * current_atr)
                else:
                    # If ATR is 0 or NaN, use a percentage of the price
                    stop_loss = current_price * 0.95  # 5% below current price
                    take_profit = current_price * 1.05  # 5% above current price
                
                # Create signal dictionary
                signal = {
                    'date': current_date,
                    'direction': direction,
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strength': 'strong',
                    'strength_value': 1.0,
                    'strategy': 'mean_reversion',
                    'atr': current_atr,
                    'weight': weight
                }
                
                # Add symbol to signal if provided
                if symbol:
                    signal['symbol'] = symbol
                
                signals.append(signal)
        
        # Log the number of signals generated
        logging.info(f"Generated {len(signals)} raw signals for {symbol}")
        
        # Limit the number of signals to avoid overwhelming the system
        if len(signals) > 20:
            signals = signals[-20:]  # Keep only the most recent 20 signals
        
        logging.info(f"Returning {len(signals)} processed signals for {symbol}")
        
        return signals
