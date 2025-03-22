#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Strategy Implementation (No TA-Lib)
-------------------------------------
This module implements a combined strategy that integrates both mean reversion
and trend following approaches, adapting to different market conditions.
This version uses custom technical indicators instead of TA-Lib.
"""

import numpy as np
import pandas as pd
import logging
import datetime as dt
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

# Import our custom technical indicators
from technical_indicators import (
    calculate_ema, calculate_sma, calculate_rsi, 
    calculate_bollinger_bands, calculate_adx, 
    calculate_macd, calculate_atr, calculate_stochastic
)

# Import our individual strategies
from mean_reversion_strategy_optimized import MeanReversionStrategyOptimized
from trend_following_strategy import TrendFollowingStrategy, TradeDirection, Signal

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class CombinedStrategy:
    """Combined strategy that integrates mean reversion and trend following approaches"""
    
    def __init__(self, config):
        """Initialize the combined strategy
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Initialize individual strategies
        self.mean_reversion = MeanReversionStrategyOptimized(config)
        self.trend_following = TrendFollowingStrategy(config)
        
        # Set weights for strategies (will be dynamically adjusted based on market regime)
        self.mr_weight = config['strategy_configs']['Combined'].get('mean_reversion_weight', 0.6)
        self.tf_weight = config['strategy_configs']['Combined'].get('trend_following_weight', 0.3)
        
        # Regime detection parameters
        self.adx_threshold = config['strategy_configs']['Combined'].get('adx_threshold', 20)
        self.volatility_period = config['strategy_configs']['Combined'].get('volatility_period', 20)
        self.regime_lookback = config['strategy_configs']['Combined'].get('regime_lookback', 10)
        
        # Signal filtering parameters
        self.min_signal_score = config['general'].get('min_signal_score', 0.7)
        
        # Maximum number of signals per day
        self.max_signals_per_day = config['general'].get('max_signals_per_day', 8)
        
        # Maximum portfolio risk
        self.max_portfolio_risk_pct = config['general'].get('max_portfolio_risk_pct', 0.015)
        
        # Symbol-specific configurations
        self.symbol_configs = config.get('symbol_configs', {})
        
        # Track performance by regime
        self.regime_performance = {
            'trending': {'trades': 0, 'wins': 0, 'total_return': 0},
            'range_bound': {'trades': 0, 'wins': 0, 'total_return': 0},
            'mixed': {'trades': 0, 'wins': 0, 'total_return': 0}
        }
        
        logging.info(f"Initialized Combined Strategy with weights: MR={self.mr_weight}, TF={self.tf_weight}")
    
    def detect_market_regime(self, df):
        """Detect the current market regime
        
        Args:
            df (pd.DataFrame): Price data with indicators
            
        Returns:
            str: Market regime ('trending', 'range_bound', or 'mixed')
        """
        # Calculate ADX for trend strength
        if 'adx' not in df.columns:
            df['adx'] = calculate_adx(df['high'], df['low'], df['close'], period=14)
        
        # Calculate Bollinger Bands width for volatility
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            bb_period = self.config['strategy_configs']['MeanReversion'].get('bb_period', 20)
            bb_std = self.config['strategy_configs']['MeanReversion'].get('bb_std', 1.9)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
                df['close'], period=bb_period, std_dev=bb_std
            )
        
        # Calculate Bollinger Bands width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate Bollinger Bands width change
        df['bb_width_change'] = df['bb_width'].pct_change(self.volatility_period)
        
        # Get current values
        current_adx = df['adx'].iloc[-1]
        current_bb_width_change = df['bb_width_change'].iloc[-1] * 100  # Convert to percentage
        
        # Determine market regime
        if current_adx > self.adx_threshold and current_bb_width_change > 0:
            regime = 'trending'
        elif current_adx < self.adx_threshold and current_bb_width_change < 0:
            regime = 'range_bound'
        else:
            regime = 'mixed'
        
        logging.info(f"Detected market regime: {regime} (ADX: {current_adx:.2f}, BB Width Change: {current_bb_width_change:.2f}%)")
        
        return regime
    
    def adjust_weights_by_regime(self, regime):
        """Adjust strategy weights based on market regime
        
        Args:
            regime (str): Market regime ('trending', 'range_bound', or 'mixed')
            
        Returns:
            tuple: Adjusted weights for mean reversion and trend following
        """
        if regime == 'trending':
            # In trending markets, favor trend following
            mr_weight = 0.2
            tf_weight = 0.8
        elif regime == 'range_bound':
            # In range-bound markets, favor mean reversion
            mr_weight = 0.8
            tf_weight = 0.2
        else:  # mixed
            # In mixed markets, use balanced weights
            mr_weight = 0.5
            tf_weight = 0.5
        
        logging.info(f"Dynamic weight allocation: MR={mr_weight}, TF={tf_weight}")
        
        return mr_weight, tf_weight
    
    def generate_signals(self, df, symbol):
        """Generate trading signals by combining strategies
        
        Args:
            df (pd.DataFrame): Price data with indicators
            symbol (str): Symbol to generate signals for
            
        Returns:
            list: List of signal dictionaries
        """
        # Detect market regime
        regime = self.detect_market_regime(df)
        
        # Adjust weights based on regime
        mr_weight, tf_weight = self.adjust_weights_by_regime(regime)
        
        # Get symbol-specific weight multiplier
        symbol_config = self.symbol_configs.get(symbol, {})
        weight_multiplier = symbol_config.get('weight_multiplier', 1.0)
        risk_multiplier = symbol_config.get('risk_multiplier', 1.0)
        
        # Generate signals from individual strategies
        mr_signals = self.mean_reversion.generate_signals(df, symbol)
        tf_signals = self.trend_following.generate_signals(df, symbol)
        
        # Combine signals with appropriate weights
        combined_signals = []
        
        # Process mean reversion signals
        for signal in mr_signals:
            # Apply weight and symbol-specific multiplier
            signal['weight'] = mr_weight * weight_multiplier
            signal['regime'] = regime
            signal['strategy'] = 'mean_reversion'
            signal['risk_multiplier'] = risk_multiplier
            
            # Convert signal strength to numerical value (0.0-1.0)
            if 'strength' in signal:
                if signal['strength'] == 'strong':
                    signal['strength_value'] = 1.0
                elif signal['strength'] == 'medium':
                    signal['strength_value'] = 0.7
                else:  # weak
                    signal['strength_value'] = 0.4
            else:
                signal['strength_value'] = 0.7  # Default if not specified
            
            # Calculate signal score
            signal['score'] = signal['weight'] * signal['strength_value']
            
            combined_signals.append(signal)
        
        # Process trend following signals
        for signal in tf_signals:
            # Apply weight and symbol-specific multiplier
            signal['weight'] = tf_weight * weight_multiplier
            signal['regime'] = regime
            signal['strategy'] = 'trend_following'
            signal['risk_multiplier'] = risk_multiplier
            
            # Convert signal strength to numerical value (0.0-1.0)
            if 'strength' in signal:
                if signal['strength'] == 'strong':
                    signal['strength_value'] = 1.0
                elif signal['strength'] == 'medium':
                    signal['strength_value'] = 0.7
                else:  # weak
                    signal['strength_value'] = 0.4
            else:
                signal['strength_value'] = 0.7  # Default if not specified
            
            # Calculate signal score
            signal['score'] = signal['weight'] * signal['strength_value']
            
            combined_signals.append(signal)
        
        # Filter signals based on minimum score threshold
        filtered_signals = [s for s in combined_signals if s['score'] >= self.min_signal_score]
        
        # Sort signals by score (highest first) then by timestamp
        filtered_signals.sort(key=lambda x: (x['score'], x['timestamp']), reverse=True)
        
        # Limit the number of signals per day
        if len(filtered_signals) > self.max_signals_per_day:
            filtered_signals = filtered_signals[:self.max_signals_per_day]
        
        return filtered_signals
    
    def calculate_position_size(self, signal, capital, current_positions):
        """Calculate position size based on risk and signal strength
        
        Args:
            signal (dict): Signal dictionary
            capital (float): Available capital
            current_positions (int): Number of current open positions
            
        Returns:
            int: Number of shares to trade
        """
        # Get risk parameters
        max_positions = self.config['general'].get('max_positions', 8)
        position_size_pct = self.config['general'].get('position_size_pct', 0.05)
        
        # Adjust position size based on number of current positions
        if current_positions >= max_positions:
            return 0  # No more positions allowed
        
        # Calculate base position size
        base_position_size = capital * position_size_pct
        
        # Adjust position size based on signal score and risk multiplier
        adjusted_position_size = base_position_size * signal['score'] * signal['risk_multiplier']
        
        # Calculate risk-based position size using stop loss
        price = signal['price']
        stop_loss = signal['stop_loss']
        risk_per_share = abs(price - stop_loss)
        
        # Calculate maximum risk amount (% of capital)
        max_risk_amount = capital * self.max_portfolio_risk_pct * signal['risk_multiplier']
        
        # Calculate position size based on risk
        if risk_per_share > 0:
            risk_based_shares = max_risk_amount / risk_per_share
        else:
            risk_based_shares = 0
        
        # Use the smaller of the two position sizes
        shares = min(adjusted_position_size / price, risk_based_shares)
        
        # Ensure position size is at least 1 share
        return max(1, int(shares))
    
    def update_regime_performance(self, trade):
        """Update performance tracking by market regime
        
        Args:
            trade (dict): Completed trade information
        """
        regime = trade.get('regime', 'unknown')
        if regime in self.regime_performance:
            self.regime_performance[regime]['trades'] += 1
            if trade['pnl'] > 0:
                self.regime_performance[regime]['wins'] += 1
            self.regime_performance[regime]['total_return'] += trade['pnl']
    
    def get_regime_performance(self):
        """Get performance metrics by market regime
        
        Returns:
            dict: Performance metrics by regime
        """
        return self.regime_performance
