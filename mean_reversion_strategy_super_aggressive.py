#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mean Reversion Strategy - Super Aggressive
-----------------------------------------
This is a super aggressive version of the mean reversion strategy
designed to generate more signals for testing purposes.
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

class MeanReversionStrategySuperAggressive:
    """Mean reversion trading strategy with super aggressive settings to generate more signals"""
    
    def __init__(self, config=None):
        """Initialize with super aggressive parameters"""
        # Use very aggressive settings to generate more signals
        self.bb_period = 10  # Shorter period for more volatility
        self.bb_std = 1.0    # Tighter bands for more signals
        self.rsi_period = 7  # Shorter period for more sensitivity
        self.rsi_overbought = 60  # Lower threshold for more sell signals
        self.rsi_oversold = 40    # Higher threshold for more buy signals
        self.require_reversal = False  # Don't require price reversal
        self.stop_loss_atr = 2.0  # Wider stop loss
        self.take_profit_atr = 3.0  # Same take profit
        self.atr_period = 14
        
        # Volume filter parameters - disabled for more signals
        self.use_volume_filter = False
        self.volume_threshold = 0.8  # Lower volume threshold
        self.volume_period = 10
        
        # Signal quality filter - reduced for more signals
        self.min_bb_penetration = 0.1  # Minimal penetration required
        
        logger.info(f"Initialized SUPER AGGRESSIVE strategy with parameters: BB period={self.bb_period}, BB std={self.bb_std}, "
                   f"RSI period={self.rsi_period}, RSI thresholds={self.rsi_oversold}/{self.rsi_overbought}, "
                   f"Require reversal={self.require_reversal}, SL/TP ATR multipliers={self.stop_loss_atr}/{self.take_profit_atr}, "
                   f"Volume filter={self.use_volume_filter}")
    
    def calculate_indicators(self, df):
        """Calculate Bollinger Bands, RSI, ATR and volume indicators for the given dataframe"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate SMA for Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        
        # Calculate standard deviation for Bollinger Bands
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        
        # Calculate Bollinger Bands
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std)
        
        # Calculate BB penetration percentage
        df['upper_penetration'] = (df['close'] - df['bb_upper']) / df['close'] * 100
        df['lower_penetration'] = (df['bb_lower'] - df['close']) / df['close'] * 100
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Calculate price direction (up/down)
        df['price_up'] = df['close'] > df['close'].shift(1)
        df['price_down'] = df['close'] < df['close'].shift(1)
        
        # Calculate volume indicators if volume is available
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=self.volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals based on the super aggressive strategy"""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Generate signals based on Bollinger Bands and RSI
        for i in range(self.bb_period + 1, len(df)):
            # Buy signal conditions - much more relaxed
            buy_condition = (
                (df.iloc[i]['close'] < df.iloc[i]['sma']) and  # Price below SMA (not just BB)
                (df.iloc[i]['rsi'] < self.rsi_oversold)        # RSI condition
            )
            
            # Sell signal conditions - much more relaxed
            sell_condition = (
                (df.iloc[i]['close'] > df.iloc[i]['sma']) and  # Price above SMA (not just BB)
                (df.iloc[i]['rsi'] > self.rsi_overbought)      # RSI condition
            )
            
            # Set signal
            if buy_condition:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif sell_condition:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        # Calculate stop loss and take profit levels for each signal
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for i in range(len(df)):
            if df.iloc[i]['signal'] == 1:  # Buy signal
                df.iloc[i, df.columns.get_loc('stop_loss')] = df.iloc[i]['close'] - (df.iloc[i]['atr'] * self.stop_loss_atr)
                df.iloc[i, df.columns.get_loc('take_profit')] = df.iloc[i]['close'] + (df.iloc[i]['atr'] * self.take_profit_atr)
            elif df.iloc[i]['signal'] == -1:  # Sell signal
                df.iloc[i, df.columns.get_loc('stop_loss')] = df.iloc[i]['close'] + (df.iloc[i]['atr'] * self.stop_loss_atr)
                df.iloc[i, df.columns.get_loc('take_profit')] = df.iloc[i]['close'] - (df.iloc[i]['atr'] * self.take_profit_atr)
        
        # Convert DataFrame signals to a list of signal dictionaries
        signals = []
        
        for i in range(len(df)):
            if df.iloc[i]['signal'] == 1:  # Buy signal
                # Always use medium strength for more consistent signals
                strength = "medium"
                
                signal_dict = {
                    'date': df.index[i],
                    'direction': 'LONG',
                    'price': df.iloc[i]['close'],
                    'stop_loss': df.iloc[i]['stop_loss'],
                    'take_profit': df.iloc[i]['take_profit'],
                    'strength': strength,
                    'strategy': 'mean_reversion',
                    'atr': df.iloc[i]['atr']
                }
                signals.append(signal_dict)
                
            elif df.iloc[i]['signal'] == -1:  # Sell signal
                # Always use medium strength for more consistent signals
                strength = "medium"
                
                signal_dict = {
                    'date': df.index[i],
                    'direction': 'SHORT',
                    'price': df.iloc[i]['close'],
                    'stop_loss': df.iloc[i]['stop_loss'],
                    'take_profit': df.iloc[i]['take_profit'],
                    'strength': strength,
                    'strategy': 'mean_reversion',
                    'atr': df.iloc[i]['atr']
                }
                signals.append(signal_dict)
        
        return signals
