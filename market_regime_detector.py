#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Regime Detector
---------------------
This module provides functionality to detect market regimes
and analyze market conditions for trading decisions.
"""

import datetime
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from mean_reversion_enhanced import CandleData, MarketState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Class for detecting market regimes and analyzing market conditions
    """
    
    def __init__(self, config):
        """
        Initialize the market regime detector
        
        Args:
            config: Configuration dictionary or path to config file
        """
        self.logger = logging.getLogger("MarketRegimeDetector")
        
        # Load configuration
        if isinstance(config, str):
            import yaml
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        # Extract market regime parameters
        if 'market_regime_params' in self.config:
            self.params = self.config['market_regime_params']
        else:
            self.params = {
                'short_ma_period': 20,
                'medium_ma_period': 50,
                'long_ma_period': 200,
                'trend_threshold': 0.02,
                'volatility_period': 20,
                'high_volatility_threshold': 0.015,
                'low_volatility_threshold': 0.008,
                'adx_period': 14,
                'adx_threshold': 25,
                'regime_change_lookback': 5
            }
            
        self.logger.info("Market regime detector initialized")
    
    def detect_market_regime(self, candles: List[CandleData], date: datetime.datetime) -> MarketState:
        """
        Detect the current market regime based on price action
        
        Args:
            candles: List of candle data
            date: Current date
            
        Returns:
            MarketState object with regime information
        """
        if len(candles) < self.params['long_ma_period']:
            # Not enough data for analysis
            return MarketState(
                date=date,
                regime="neutral",
                volatility=0.0,
                trend_strength=0.0,
                is_range_bound=False
            )
        
        # Extract price data
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        
        # Calculate moving averages
        short_ma = self._calculate_sma(closes, self.params['short_ma_period'])
        medium_ma = self._calculate_sma(closes, self.params['medium_ma_period'])
        long_ma = self._calculate_sma(closes, self.params['long_ma_period'])
        
        # Calculate volatility (ATR-based)
        volatility = self._calculate_atr(highs, lows, closes, self.params['volatility_period']) / closes[-1]
        
        # Calculate trend strength using custom ADX-like indicator
        trend_strength = self._calculate_trend_strength(highs, lows, closes, self.params['adx_period'])
        
        # Determine if market is range-bound
        is_range_bound = (volatility < self.params['low_volatility_threshold']) and (trend_strength < self.params['adx_threshold'] * 0.5)
        
        # Determine market regime
        if short_ma > medium_ma and medium_ma > long_ma:
            # Strong bullish trend
            if trend_strength > self.params['adx_threshold']:
                regime = "strong_bullish"
            else:
                regime = "bullish"
        elif short_ma < medium_ma and medium_ma < long_ma:
            # Strong bearish trend
            if trend_strength > self.params['adx_threshold']:
                regime = "strong_bearish"
            else:
                regime = "bearish"
        elif short_ma > medium_ma and medium_ma < long_ma:
            # Potential bullish transition
            regime = "transitional"
        elif short_ma < medium_ma and medium_ma > long_ma:
            # Potential bearish transition
            regime = "transitional"
        else:
            # Neutral or unclear
            regime = "neutral"
        
        # Adjust regime if range-bound
        if is_range_bound:
            if regime in ["bullish", "strong_bullish"]:
                regime = "bullish_range"
            elif regime in ["bearish", "strong_bearish"]:
                regime = "bearish_range"
            else:
                regime = "neutral_range"
        
        # Create market state
        market_state = MarketState(
            date=date,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            is_range_bound=is_range_bound
        )
        
        self.logger.debug(f"Market regime: {regime}, Volatility: {volatility:.4f}, Trend strength: {trend_strength:.4f}")
        
        return market_state
    
    def calculate_regime_filter_score(self, market_state: MarketState, signal_direction: str) -> float:
        """
        Calculate a score for the current market regime's compatibility with a signal direction
        
        Args:
            market_state: Current market state
            signal_direction: Direction of the signal ("long" or "short")
            
        Returns:
            Score between 0.0 and 1.0 indicating compatibility
        """
        regime = market_state.regime
        volatility = market_state.volatility
        trend_strength = market_state.trend_strength
        is_range_bound = market_state.is_range_bound
        
        # Base score
        score = 0.5
        
        # Adjust score based on regime and signal direction
        if signal_direction == "long":
            if regime in ["strong_bullish", "bullish"]:
                score -= 0.2  # Less favorable for mean reversion
            elif regime in ["strong_bearish", "bearish"]:
                score += 0.2  # More favorable for mean reversion (buying dips)
            elif regime in ["neutral", "transitional"]:
                score += 0.1  # Slightly favorable
            
            # Range-bound markets are good for mean reversion
            if is_range_bound:
                score += 0.2
                
            # High volatility can be good for mean reversion if not too extreme
            if volatility > self.params['low_volatility_threshold'] and volatility < self.params['high_volatility_threshold']:
                score += 0.1
            elif volatility >= self.params['high_volatility_threshold']:
                score -= 0.1  # Too volatile might be risky
        
        elif signal_direction == "short":
            if regime in ["strong_bearish", "bearish"]:
                score -= 0.2  # Less favorable for mean reversion
            elif regime in ["strong_bullish", "bullish"]:
                score += 0.2  # More favorable for mean reversion (shorting rallies)
            elif regime in ["neutral", "transitional"]:
                score += 0.1  # Slightly favorable
            
            # Range-bound markets are good for mean reversion
            if is_range_bound:
                score += 0.2
                
            # High volatility can be good for mean reversion if not too extreme
            if volatility > self.params['low_volatility_threshold'] and volatility < self.params['high_volatility_threshold']:
                score += 0.1
            elif volatility >= self.params['high_volatility_threshold']:
                score -= 0.1  # Too volatile might be risky
        
        # Ensure score is between 0.0 and 1.0
        score = max(0.0, min(1.0, score))
        
        self.logger.debug(f"Regime filter score for {signal_direction}: {score:.4f}")
        
        return score
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return data[-1]
        return np.mean(data[-period:])
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Average True Range without using TA-Lib"""
        if len(high) < period + 1:
            return (high[-1] - low[-1])
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        return np.mean(tr[-period:])
    
    def _calculate_trend_strength(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate trend strength (ADX-like) without using TA-Lib"""
        if len(high) < period + 1:
            return 0.0
        
        # Calculate +DM and -DM
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Calculate ATR
        atr = self._calculate_atr(high, low, close, period)
        
        # Calculate +DI and -DI
        plus_di = np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
        
        # Calculate DX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        return dx * 100  # Scale to 0-100 range like ADX
