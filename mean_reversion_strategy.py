#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mean Reversion Strategy Implementation
-------------------------------------
This module implements the optimized mean reversion strategy based on Bollinger Bands and RSI.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """Mean reversion trading strategy using Bollinger Bands and RSI"""
    
    def __init__(self, config):
        """Initialize with parameters from config"""
        mr_config = config.get('strategy_configs', {}).get('MeanReversion', {})
        
        # Load optimized parameters from config
        self.bb_period = mr_config.get('bb_period', 20)
        self.bb_std = mr_config.get('bb_std_dev', 1.9)
        self.rsi_period = mr_config.get('rsi_period', 14)
        self.rsi_overbought = mr_config.get('rsi_overbought', 65)
        self.rsi_oversold = mr_config.get('rsi_oversold', 35)
        self.require_reversal = mr_config.get('require_reversal', True)
        self.stop_loss_atr = mr_config.get('stop_loss_atr', 1.8)
        self.take_profit_atr = mr_config.get('take_profit_atr', 3.0)
        self.atr_period = mr_config.get('atr_period', 14)
        
        logger.info(f"Initialized with parameters: BB period={self.bb_period}, BB std={self.bb_std}, "
                   f"RSI period={self.rsi_period}, RSI thresholds={self.rsi_oversold}/{self.rsi_overbought}, "
                   f"Require reversal={self.require_reversal}, SL/TP ATR multipliers={self.stop_loss_atr}/{self.take_profit_atr}")
    
    def calculate_indicators(self, df):
        """Calculate Bollinger Bands, RSI, and ATR for the given dataframe"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate SMA for Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        
        # Calculate standard deviation for Bollinger Bands
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        
        # Calculate Bollinger Bands
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Calculate price reversal
        df['prev_close'] = df['close'].shift(1)
        df['price_up'] = df['close'] > df['prev_close']
        df['price_down'] = df['close'] < df['prev_close']
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals based on the strategy"""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Generate signals based on Bollinger Bands and RSI
        for i in range(self.bb_period + 1, len(df)):
            # Buy signal conditions
            buy_condition = (
                (df.iloc[i]['close'] < df.iloc[i]['bb_lower']) and  # Price below lower BB
                (df.iloc[i]['rsi'] < self.rsi_oversold)             # RSI oversold
            )
            
            # Add price reversal condition if required
            if self.require_reversal:
                buy_condition = buy_condition and df.iloc[i]['price_up']
            
            # Sell signal conditions
            sell_condition = (
                (df.iloc[i]['close'] > df.iloc[i]['bb_upper']) and  # Price above upper BB
                (df.iloc[i]['rsi'] > self.rsi_overbought)           # RSI overbought
            )
            
            # Add price reversal condition if required
            if self.require_reversal:
                sell_condition = sell_condition and df.iloc[i]['price_down']
            
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
        
        return df
