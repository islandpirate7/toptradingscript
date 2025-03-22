#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Mean Reversion Strategy Implementation
-------------------------------------
This module implements an optimized version of the mean reversion strategy
with parameters based on research findings for improved performance.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MeanReversionStrategyOptimized:
    """Mean reversion trading strategy using Bollinger Bands and RSI with optimized parameters"""
    
    def __init__(self, config):
        """Initialize with optimized parameters based on research"""
        mr_config = config.get('strategy_configs', {}).get('MeanReversion', {})
        
        # Load parameters from config but override with optimized values
        self.bb_period = mr_config.get('bb_period', 20)  # Standard setting that works well across markets
        self.bb_std = mr_config.get('bb_std', 1.8)    # Research suggests 1.5-1.8 for more signals with reasonable quality
        self.rsi_period = mr_config.get('rsi_period', 14) # Standard setting
        self.rsi_overbought = mr_config.get('rsi_upper', 70)  # Standard overbought threshold
        self.rsi_oversold = mr_config.get('rsi_lower', 30)    # Standard oversold threshold
        self.require_reversal = mr_config.get('require_reversal', True)  # Require price reversal for better signal quality
        self.stop_loss_atr = mr_config.get('sl_atr_multiplier', 1.5)  # Tighter stop loss for better risk management
        self.take_profit_atr = mr_config.get('tp_atr_multiplier', 3.0)  # Higher take profit for better risk-reward ratio
        self.atr_period = mr_config.get('atr_period', 14)
        
        # Volume filter parameters
        self.use_volume_filter = mr_config.get('use_volume_filter', True)
        self.volume_threshold = mr_config.get('volume_threshold', 1.2)  # Volume should be 1.2x average
        self.volume_period = mr_config.get('volume_period', 20)
        
        # Signal quality filter
        self.min_bb_penetration = mr_config.get('min_bb_penetration', 0.3)  # Minimum penetration % of BB (price must be at least 0.3% beyond band)
        
        logger.info(f"Initialized OPTIMIZED strategy with parameters: BB period={self.bb_period}, BB std={self.bb_std}, "
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
        
        # Calculate volume indicators if volume data is available
        if 'volume' in df.columns:
            df['avg_volume'] = df['volume'].rolling(window=self.volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['avg_volume']
        else:
            df['volume_ratio'] = 1.0  # Default if no volume data
        
        return df
    
    def generate_signals(self, df, symbol=None):
        """Generate trading signals based on the optimized strategy
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            symbol (str, optional): Symbol for the data. Defaults to None.
            
        Returns:
            list: List of signal dictionaries
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Debug info
        logger.info(f"Generating signals for {symbol if symbol else 'unknown'} with {len(df)} rows of data")
        logger.info(f"RSI thresholds: oversold={self.rsi_oversold}, overbought={self.rsi_overbought}")
        logger.info(f"BB std: {self.bb_std}, min penetration: {self.min_bb_penetration}")
        
        # Generate signals based on Bollinger Bands and RSI
        signals_found = 0
        for i in range(self.bb_period + 1, len(df)):
            # Buy signal conditions
            buy_condition = (
                (df.iloc[i]['close'] < df.iloc[i]['bb_lower']) and  # Price below lower BB
                (df.iloc[i]['rsi'] < self.rsi_overbought) and       # RSI below overbought (relaxed condition)
                (df.iloc[i]['lower_penetration'] > self.min_bb_penetration)  # Significant penetration
            )
            
            # Add price reversal condition if required
            if self.require_reversal:
                buy_condition = buy_condition and df.iloc[i]['price_up']
            
            # Add volume filter if enabled
            if self.use_volume_filter and 'volume' in df.columns:
                buy_condition = buy_condition and (df.iloc[i]['volume_ratio'] > self.volume_threshold)
            
            # Sell signal conditions
            sell_condition = (
                (df.iloc[i]['close'] > df.iloc[i]['bb_upper']) and  # Price above upper BB
                (df.iloc[i]['rsi'] > self.rsi_oversold) and         # RSI above oversold (relaxed condition)
                (df.iloc[i]['upper_penetration'] > self.min_bb_penetration)  # Significant penetration
            )
            
            # Add price reversal condition if required
            if self.require_reversal:
                sell_condition = sell_condition and df.iloc[i]['price_down']
            
            # Add volume filter if enabled
            if self.use_volume_filter and 'volume' in df.columns:
                sell_condition = sell_condition and (df.iloc[i]['volume_ratio'] > self.volume_threshold)
            
            # Debug info for the last few rows
            if i >= len(df) - 5:
                date_str = df.index[i].strftime('%Y-%m-%d')
                close = df.iloc[i]['close']
                bb_lower = df.iloc[i]['bb_lower']
                bb_upper = df.iloc[i]['bb_upper']
                rsi = df.iloc[i]['rsi']
                lower_pen = df.iloc[i]['lower_penetration']
                upper_pen = df.iloc[i]['upper_penetration']
                price_up = df.iloc[i]['price_up']
                price_down = df.iloc[i]['price_down']
                
                logger.info(f"Row {i} ({date_str}): Close={close:.2f}, BB Lower={bb_lower:.2f}, "
                           f"BB Upper={bb_upper:.2f}, RSI={rsi:.2f}, "
                           f"Lower Pen={lower_pen:.2f}%, Upper Pen={upper_pen:.2f}%, "
                           f"Price Up={price_up}, Price Down={price_down}")
                
                # Log buy condition components
                logger.info(f"Buy condition components: "
                           f"Close < BB Lower: {df.iloc[i]['close'] < df.iloc[i]['bb_lower']}, "
                           f"RSI < {self.rsi_overbought}: {df.iloc[i]['rsi'] < self.rsi_overbought}, "
                           f"Lower Pen > {self.min_bb_penetration}: {df.iloc[i]['lower_penetration'] > self.min_bb_penetration}, "
                           f"Price Up: {df.iloc[i]['price_up'] if self.require_reversal else 'Not Required'}")
                
                # Log sell condition components
                logger.info(f"Sell condition components: "
                           f"Close > BB Upper: {df.iloc[i]['close'] > df.iloc[i]['bb_upper']}, "
                           f"RSI > {self.rsi_oversold}: {df.iloc[i]['rsi'] > self.rsi_oversold}, "
                           f"Upper Pen > {self.min_bb_penetration}: {df.iloc[i]['upper_penetration'] > self.min_bb_penetration}, "
                           f"Price Down: {df.iloc[i]['price_down'] if self.require_reversal else 'Not Required'}")
                
                logger.info(f"Final conditions: Buy={buy_condition}, Sell={sell_condition}")
            
            # Set signal
            if buy_condition:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                signals_found += 1
            elif sell_condition:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                signals_found += 1
        
        logger.info(f"Found {signals_found} signals for {symbol if symbol else 'unknown'}")
        
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
        
        # Convert DataFrame signals to a list of signal dictionaries to match trend following format
        signals = []
        
        for i in range(len(df)):
            if df.iloc[i]['signal'] == 1:  # Buy signal
                # Determine signal strength based on penetration
                strength = "medium"
                if df.iloc[i]['lower_penetration'] > 1.0:
                    strength = "strong"
                elif df.iloc[i]['lower_penetration'] < 0.5:
                    strength = "weak"
                
                signal_dict = {
                    'date': df.index[i],
                    'direction': 'LONG',
                    'price': df.iloc[i]['close'],
                    'stop_loss': df.iloc[i]['stop_loss'],
                    'take_profit': df.iloc[i]['take_profit'],
                    'strength': strength,
                    'strategy': 'mean_reversion',
                    'atr': df.iloc[i]['atr']  # Include ATR for volatility-based position sizing
                }
                signals.append(signal_dict)
                
            elif df.iloc[i]['signal'] == -1:  # Sell signal
                # Determine signal strength based on penetration
                strength = "medium"
                if df.iloc[i]['upper_penetration'] > 1.0:
                    strength = "strong"
                elif df.iloc[i]['upper_penetration'] < 0.5:
                    strength = "weak"
                
                signal_dict = {
                    'date': df.index[i],
                    'direction': 'SHORT',
                    'price': df.iloc[i]['close'],
                    'stop_loss': df.iloc[i]['stop_loss'],
                    'take_profit': df.iloc[i]['take_profit'],
                    'strength': strength,
                    'strategy': 'mean_reversion',
                    'atr': df.iloc[i]['atr']  # Include ATR for volatility-based position sizing
                }
                signals.append(signal_dict)
        
        return signals
    
    def generate_signal(self, df, symbol=None):
        """
        Generate a single trading signal for the most recent data point.
        This is a wrapper around generate_signals that returns only the most recent signal if available.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            symbol (str, optional): Symbol for the data. Defaults to None.
            
        Returns:
            dict or None: Signal dictionary or None if no signal
        """
        signals = self.generate_signals(df, symbol)
        
        logger.info(f"generate_signal for {symbol}: Found {len(signals)} signals")
        
        # Add symbol to signals if provided
        if symbol and signals:
            for signal in signals:
                signal['symbol'] = symbol
        
        # Return the most recent signal if available
        if signals and len(signals) > 0:
            # Get the most recent signal (last in the list)
            latest_signal = signals[-1]
            logger.info(f"Returning latest signal for {symbol}: {latest_signal}")
            return latest_signal
        
        logger.info(f"No signals found for {symbol}")
        return None
