#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Mean Reversion Strategy Implementation
Includes dynamic stop-loss placement, time-based exits, partial profit taking,
and volatility-adjusted position sizing
"""

import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CandleData:
    """Data class for candle data"""
    timestamp: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class MarketState:
    """Data class for market state"""
    date: datetime.datetime
    regime: str  # bullish, bearish, neutral
    volatility: float
    trend_strength: float
    is_range_bound: bool

@dataclass
class Signal:
    """Data class for trading signals"""
    symbol: str
    timestamp: datetime.datetime
    direction: str  # long, short
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    strength: str  # strong, medium, weak
    is_crypto: bool = False
    max_holding_days: Optional[int] = None
    partial_exit_level: Optional[float] = None
    trailing_stop_activation_level: Optional[float] = None
    quality_score: Optional[float] = None  # ML-predicted quality score
    regime_score: Optional[float] = None   # Market regime compatibility score

@dataclass
class Trade:
    """Data class for trades"""
    symbol: str
    entry_date: datetime.datetime
    entry_price: float
    direction: str  # long, short
    position_size: float
    stop_loss: float
    take_profit: float
    is_crypto: bool = False
    strategy_name: str = "EnhancedMeanReversion"
    exit_date: Optional[datetime.datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_holding_days: Optional[int] = None
    partial_exit_level: Optional[float] = None
    trailing_stop_activation_level: Optional[float] = None
    partial_exit_executed: bool = False
    trailing_stop_active: bool = False
    current_trailing_stop: Optional[float] = None
    
    def to_dict(self):
        """Convert Trade object to a dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'is_crypto': self.is_crypto,
            'strategy_name': self.strategy_name,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'max_holding_days': self.max_holding_days,
            'partial_exit_level': self.partial_exit_level,
            'trailing_stop_activation_level': self.trailing_stop_activation_level,
            'partial_exit_executed': self.partial_exit_executed,
            'trailing_stop_active': self.trailing_stop_active,
            'current_trailing_stop': self.current_trailing_stop
        }

class EnhancedMeanReversionStrategy:
    """Enhanced Mean Reversion Strategy with dynamic stop-loss and time-based exits"""
    
    def __init__(self, config: Dict):
        """Initialize the strategy with the given configuration"""
        self.config = config
        self.logger = logging.getLogger("EnhancedMeanReversionStrategy")
        
        # Extract parameters from config
        self.params = config.get('mean_reversion_params', {})
        
        # Debug logging for parameters
        self.logger.debug(f"Config keys: {list(config.keys())}")
        self.logger.debug(f"Mean reversion params: {self.params}")
        
        # Signal generation parameters
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std_dev = self.params.get('bb_std_dev', 2.0)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_overbought = self.params.get('rsi_overbought', 70)
        self.rsi_oversold = self.params.get('rsi_oversold', 30)
        self.require_reversal = self.params.get('require_reversal', True)
        self.min_reversal_candles = self.params.get('min_reversal_candles', 1)
        self.min_distance_to_band_pct = self.params.get('min_distance_to_band_pct', 0.01)  # Added parameter
        
        # Log the actual parameters being used
        self.logger.info(f"Initialized with parameters: BB period={self.bb_period}, BB std={self.bb_std_dev}, "
                        f"RSI period={self.rsi_period}, RSI thresholds={self.rsi_oversold}/{self.rsi_overbought}, "
                        f"Require reversal={self.require_reversal}")
        
        # Risk management parameters
        self.stop_loss_atr = self.params.get('stop_loss_atr', 1.8)
        self.take_profit_atr = self.params.get('take_profit_atr', 3.0)
        self.max_holding_days = self.params.get('max_holding_days', 10)
        
        # Dynamic stop-loss parameters
        self.use_support_resistance = self.params.get('use_support_resistance', True)
        self.support_lookback = self.params.get('support_lookback', 30)
        self.max_support_levels = self.params.get('max_support_levels', 3)
        
        # Partial profit taking parameters
        self.use_partial_exits = self.params.get('use_partial_exits', False)
        self.partial_exit_threshold = self.params.get('partial_exit_threshold', 0.5)
        self.trailing_stop_activation = self.params.get('trailing_stop_activation', 0.75)
        
        # Volatility-based position sizing parameters
        self.volatility_adjustment = self.params.get('volatility_adjustment', False)
        self.low_volatility_boost = self.params.get('low_volatility_boost', 1.2)
        self.high_volatility_reduction = self.params.get('high_volatility_reduction', 0.8)
        
        # Market regime filter parameters
        self.use_regime_filter = self.params.get('use_regime_filter', True)
        self.min_regime_score = self.params.get('min_regime_score', 0.3)
        self.regime_filter_strength = self.params.get('regime_filter_strength', 1.0)
        
        # ML signal quality parameters
        self.use_ml_filter = self.params.get('use_ml_filter', True)
        self.min_quality_score = self.params.get('min_quality_score', 0.4)
        self.ml_filter_strength = self.params.get('ml_filter_strength', 1.0)
        
        # Initialize market regime detector and ML classifier
        self.regime_detector = None
        self.ml_classifier = None
        
        # Try to import and initialize the market regime detector
        try:
            from market_regime_detector import MarketRegimeDetector
            self.regime_detector = MarketRegimeDetector(config)
            self.logger.info("Market regime detector initialized successfully")
        except ImportError:
            self.logger.warning("Market regime detector module not found, regime filtering disabled")
            self.use_regime_filter = False
        except Exception as e:
            self.logger.error(f"Error initializing market regime detector: {str(e)}")
            self.use_regime_filter = False
        
        # Try to import and initialize the ML signal classifier
        try:
            from ml_signal_classifier import MLSignalClassifier
            self.ml_classifier = MLSignalClassifier(config)
            self.logger.info("ML signal classifier initialized successfully")
        except ImportError:
            self.logger.warning("ML signal classifier module not found, ML filtering disabled")
            self.use_ml_filter = False
        except Exception as e:
            self.logger.error(f"Error initializing ML signal classifier: {str(e)}")
            self.use_ml_filter = False

    def calculate_bollinger_bands(self, candles: List[CandleData]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands for the given candles"""
        closes = [c.close for c in candles]
        
        # Calculate SMA
        sma = []
        for i in range(len(closes)):
            if i < self.bb_period - 1:
                sma.append(None)
            else:
                sma.append(np.mean(closes[i - self.bb_period + 1:i + 1]))
        
        # Calculate standard deviation
        std = []
        for i in range(len(closes)):
            if i < self.bb_period - 1:
                std.append(None)
            else:
                std.append(np.std(closes[i - self.bb_period + 1:i + 1]))
        
        # Calculate upper and lower bands
        upper_band = []
        lower_band = []
        for i in range(len(closes)):
            if sma[i] is None or std[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                upper_band.append(sma[i] + self.bb_std_dev * std[i])
                lower_band.append(sma[i] - self.bb_std_dev * std[i])
        
        return sma, upper_band, lower_band
    
    def calculate_rsi(self, candles: List[CandleData]) -> List[float]:
        """Calculate RSI for the given candles"""
        closes = [c.close for c in candles]
        
        # Calculate price changes
        changes = [0]
        for i in range(1, len(closes)):
            changes.append(closes[i] - closes[i - 1])
        
        # Calculate gains and losses
        gains = [max(0, change) for change in changes]
        losses = [abs(min(0, change)) for change in changes]
        
        # Calculate average gains and losses
        avg_gains = []
        avg_losses = []
        
        for i in range(len(changes)):
            if i < self.rsi_period:
                avg_gains.append(None)
                avg_losses.append(None)
            elif i == self.rsi_period:
                avg_gains.append(sum(gains[1:i + 1]) / self.rsi_period)
                avg_losses.append(sum(losses[1:i + 1]) / self.rsi_period)
            else:
                avg_gains.append((avg_gains[i - 1] * (self.rsi_period - 1) + gains[i]) / self.rsi_period)
                avg_losses.append((avg_losses[i - 1] * (self.rsi_period - 1) + losses[i]) / self.rsi_period)
        
        # Calculate RS and RSI
        rsi = []
        for i in range(len(changes)):
            if avg_gains[i] is None or avg_losses[i] is None:
                rsi.append(None)
            elif avg_losses[i] == 0:
                rsi.append(100)
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi.append(100 - (100 / (1 + rs)))
        
        return rsi
    
    def calculate_atr(self, candles: List[CandleData], period: int = 14) -> List[float]:
        """Calculate Average True Range for the given candles"""
        if len(candles) < period + 1:
            return [None] * len(candles)
        
        true_ranges = []
        
        # Calculate true range for each candle
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Calculate ATR
        atr = [None]  # First candle has no ATR
        
        # Simple average for the first ATR value
        if len(true_ranges) >= period:
            atr.append(sum(true_ranges[:period]) / period)
        else:
            # Not enough data for ATR
            return [None] * len(candles)
        
        # Exponential average for subsequent ATR values
        for i in range(period + 1, len(candles)):
            atr.append((atr[-1] * (period - 1) + true_ranges[i - 1]) / period)
        
        # Pad with None for candles without ATR
        while len(atr) < len(candles):
            atr.append(None)
        
        return atr
    
    def identify_support_resistance_levels(self, candles: List[CandleData], lookback: int = 30) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels from recent price action"""
        if len(candles) < lookback:
            return [], []
        
        # Get recent candles
        recent_candles = candles[-lookback:]
        
        # Find local minima (support) and maxima (resistance)
        support_levels = []
        resistance_levels = []
        
        for i in range(1, len(recent_candles) - 1):
            # Check for local minimum (support)
            if recent_candles[i].low < recent_candles[i - 1].low and recent_candles[i].low < recent_candles[i + 1].low:
                support_levels.append(recent_candles[i].low)
            
            # Check for local maximum (resistance)
            if recent_candles[i].high > recent_candles[i - 1].high and recent_candles[i].high > recent_candles[i + 1].high:
                resistance_levels.append(recent_candles[i].high)
        
        # Cluster similar levels (within 1% of each other)
        clustered_support = self._cluster_price_levels(support_levels)
        clustered_resistance = self._cluster_price_levels(resistance_levels)
        
        # Sort by strength (frequency of occurrence)
        clustered_support.sort(key=lambda x: x[1], reverse=True)
        clustered_resistance.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the price levels
        support_levels = [level[0] for level in clustered_support[:self.max_support_levels]]
        resistance_levels = [level[0] for level in clustered_resistance[:self.max_support_levels]]
        
        return support_levels, resistance_levels
    
    def _cluster_price_levels(self, levels: List[float]) -> List[Tuple[float, int]]:
        """Cluster similar price levels and return (level, count) tuples"""
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster similar levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If level is within 1% of the cluster average, add to cluster
            cluster_avg = sum(current_cluster) / len(current_cluster)
            if abs(level - cluster_avg) / cluster_avg < 0.01:
                current_cluster.append(level)
            else:
                # Start a new cluster
                clusters.append((sum(current_cluster) / len(current_cluster), len(current_cluster)))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clusters.append((sum(current_cluster) / len(current_cluster), len(current_cluster)))
        
        return clusters
    
    def detect_price_reversal(self, candles: List[CandleData], direction: str) -> bool:
        """Detect if price is showing reversal signs in the given direction"""
        if len(candles) < self.min_reversal_candles + 1:
            return False
        
        # Get recent candles
        recent_candles = candles[-(self.min_reversal_candles + 1):]
        
        if direction == "long":
            # For long positions, look for bullish reversal (price going up after decline)
            # Check if the most recent candle is bullish (close > open)
            if recent_candles[-1].close <= recent_candles[-1].open:
                return False
            
            # Check if the previous candle(s) showed downward movement
            for i in range(1, self.min_reversal_candles + 1):
                if recent_candles[-i - 1].close >= recent_candles[-i].open:
                    return False
            
            return True
        
        elif direction == "short":
            # For short positions, look for bearish reversal (price going down after rise)
            # Check if the most recent candle is bearish (close < open)
            if recent_candles[-1].close >= recent_candles[-1].open:
                return False
            
            # Check if the previous candle(s) showed upward movement
            for i in range(1, self.min_reversal_candles + 1):
                if recent_candles[-i - 1].close <= recent_candles[-i].open:
                    return False
            
            return True
        
        return False
    
    def calculate_dynamic_stop_loss(self, candles: List[CandleData], direction: str, entry_price: float, atr: float) -> float:
        """Calculate dynamic stop-loss level based on support/resistance and ATR"""
        if not self.use_support_resistance or len(candles) < self.support_lookback:
            # Fallback to ATR-based stop-loss
            if direction == "long":
                return entry_price - self.stop_loss_atr * atr
            else:  # short
                return entry_price + self.stop_loss_atr * atr
        
        # Identify support and resistance levels
        support_levels, resistance_levels = self.identify_support_resistance_levels(candles, self.support_lookback)
        
        if direction == "long":
            # For long positions, find the closest support level below entry price
            valid_supports = [level for level in support_levels if level < entry_price]
            
            if valid_supports:
                # Find the closest support level
                closest_support = max(valid_supports)
                
                # Calculate ATR-based buffer
                buffer = 0.5 * atr
                
                # Set stop-loss at support level minus buffer
                stop_loss = closest_support - buffer
                
                # Ensure stop-loss is not too far from entry (risk management)
                max_distance = self.stop_loss_atr * atr
                if entry_price - stop_loss > max_distance:
                    stop_loss = entry_price - max_distance
                
                return stop_loss
            else:
                # Fallback to ATR-based stop-loss
                return entry_price - self.stop_loss_atr * atr
        
        else:  # short
            # For short positions, find the closest resistance level above entry price
            valid_resistances = [level for level in resistance_levels if level > entry_price]
            
            if valid_resistances:
                # Find the closest resistance level
                closest_resistance = min(valid_resistances)
                
                # Calculate ATR-based buffer
                buffer = 0.5 * atr
                
                # Set stop-loss at resistance level plus buffer
                stop_loss = closest_resistance + buffer
                
                # Ensure stop-loss is not too far from entry (risk management)
                max_distance = self.stop_loss_atr * atr
                if stop_loss - entry_price > max_distance:
                    stop_loss = entry_price + max_distance
                
                return stop_loss
            else:
                # Fallback to ATR-based stop-loss
                return entry_price + self.stop_loss_atr * atr
    
    def calculate_volatility_adjustment(self, candles: List[CandleData]) -> float:
        """Calculate position size adjustment based on volatility"""
        if not self.volatility_adjustment or len(candles) < 20:
            return 1.0  # No adjustment
        
        # Calculate recent volatility (20-day standard deviation of returns)
        closes = [c.close for c in candles[-20:]]
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        volatility = np.std(returns)
        
        # Determine volatility regime
        avg_volatility = 0.015  # Average daily volatility (1.5%)
        
        if volatility < 0.8 * avg_volatility:
            # Low volatility - increase position size
            return self.low_volatility_boost
        elif volatility > 1.2 * avg_volatility:
            # High volatility - decrease position size
            return self.high_volatility_reduction
        else:
            # Normal volatility - no adjustment
            return 1.0
    
    def generate_signals(self, symbol: str, candles: List[CandleData], market_state: MarketState, is_crypto: bool = False) -> List[Signal]:
        """Generate trading signals based on Mean Reversion strategy"""
        if len(candles) < max(self.bb_period, self.rsi_period) + 10:
            self.logger.debug(f"Not enough candles for {symbol}: {len(candles)} < {max(self.bb_period, self.rsi_period) + 10}")
            return []
        
        signals = []
        
        # Calculate indicators
        middle_band, upper_band, lower_band = self.calculate_bollinger_bands(candles)
        rsi = self.calculate_rsi(candles)
        atr_values = self.calculate_atr(candles)
        
        # Get the most recent candle
        current_candle = candles[-1]
        
        # Check if we have valid ATR values
        if atr_values[-1] is None:
            # If ATR is None, calculate a simple volatility measure as fallback
            self.logger.warning(f"ATR is None for {symbol}, using fallback volatility calculation")
            closes = [c.close for c in candles[-20:]]
            if len(closes) >= 5:
                # Calculate average daily price change as a percentage of price
                price_changes = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                avg_price_change = sum(price_changes) / len(price_changes)
                fallback_atr = current_candle.close * avg_price_change
                atr_values[-1] = fallback_atr
                self.logger.info(f"Calculated fallback ATR for {symbol}: {fallback_atr:.2f}")
            else:
                # If we still can't calculate a meaningful ATR, use a conservative estimate
                fallback_atr = current_candle.close * 0.02  # Assume 2% volatility
                atr_values[-1] = fallback_atr
                self.logger.info(f"Using default fallback ATR for {symbol}: {fallback_atr:.2f} (2% of price)")
        
        # Skip if indicators are not available
        if (upper_band[-1] is None or lower_band[-1] is None or rsi[-1] is None):
            self.logger.debug(f"Missing indicators for {symbol}: BB={upper_band[-1]}, RSI={rsi[-1]}, ATR={atr_values[-1]}")
            return []
        
        # Log current values for debugging
        self.logger.info(f"{symbol} at {current_candle.timestamp}: Close={current_candle.close:.2f}, "
                    f"Lower BB={lower_band[-1]:.2f}, Upper BB={upper_band[-1]:.2f}, RSI={rsi[-1]:.2f}, ATR={atr_values[-1]:.2f}")
        
        # Add market trend detection
        trend_up = False
        trend_down = False
        if len(candles) >= 50:
            # Calculate 20-day EMA and 50-day EMA
            closes = [c.close for c in candles]
            ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
            
            # Determine trend direction
            trend_up = ema20 > ema50
            trend_down = ema20 < ema50
            
            self.logger.debug(f"{symbol}: EMA20={ema20:.2f}, EMA50={ema50:.2f}, Trend Up={trend_up}, Trend Down={trend_down}")
        
        # For long signals - price is close to or below lower band and RSI is below threshold
        long_signal_condition = False
        
        # Calculate how close price is to the lower band as a percentage
        lower_band_distance = (current_candle.close - lower_band[-1]) / current_candle.close
        
        # Check if price is within minimum distance of lower band or below it
        price_near_lower_band = lower_band_distance <= self.min_distance_to_band_pct or current_candle.close < lower_band[-1]
        self.logger.debug(f"{symbol}: Price to lower band distance: {lower_band_distance:.4f}, Near/below: {price_near_lower_band}")
        
        # Check if RSI is below threshold or has been below threshold in the last 3 candles
        rsi_condition = rsi[-1] < self.rsi_oversold or min(rsi[-3:]) < self.rsi_oversold
        self.logger.debug(f"{symbol}: Current RSI: {rsi[-1]:.2f}, Min RSI (last 3): {min(rsi[-3:]):.2f}, RSI condition: {rsi_condition}")
        
        # Check for volume confirmation
        volume_confirmation = False
        if len(candles) >= 20:
            avg_volume = sum(c.volume for c in candles[-20:-1]) / 19  # Average of last 19 candles excluding current
            volume_confirmation = current_candle.volume > avg_volume * 1.2  # 20% above average
            self.logger.debug(f"{symbol}: Current volume: {current_candle.volume}, Avg volume: {avg_volume}, Confirmation: {volume_confirmation}")
        
        # Check if price is showing a potential reversal pattern
        price_pattern = False
        if len(candles) >= 3:
            # Simple reversal pattern: lower low followed by higher low
            if candles[-3].low > candles[-2].low and candles[-1].low > candles[-2].low:
                price_pattern = True
                self.logger.debug(f"{symbol}: Potential bullish reversal pattern detected")
        
        # Improved condition: Price near band AND RSI condition, with optional trend filter
        if price_near_lower_band and rsi_condition and (not trend_down or price_pattern):
            long_signal_condition = True
            self.logger.info(f"{symbol}: Long signal condition met - price {current_candle.close:.2f} near/below lower BB {lower_band[-1]:.2f}, RSI {rsi[-1]:.2f}")
        
        if long_signal_condition:
            # Check for price reversal if required
            reversal_confirmed = True
            if self.require_reversal:
                reversal_confirmed = self.detect_price_reversal(candles, "long") or price_pattern
                self.logger.debug(f"{symbol}: Long reversal check result: {reversal_confirmed}")
                if not reversal_confirmed:
                    self.logger.info(f"{symbol}: Long signal conditions met but reversal not confirmed")
            
            if reversal_confirmed:
                # Calculate stop-loss and take-profit levels
                entry_price = current_candle.close
                stop_loss = self.calculate_dynamic_stop_loss(candles, "long", entry_price, atr_values[-1])
                
                # Ensure minimum distance for stop loss (at least 1.5% from entry)
                min_stop_distance = entry_price * 0.015
                if entry_price - stop_loss < min_stop_distance:
                    stop_loss = entry_price - min_stop_distance
                
                take_profit = entry_price + self.take_profit_atr * atr_values[-1]
                
                # Calculate partial exit level if enabled
                partial_exit_level = None
                trailing_stop_activation_level = None
                if self.use_partial_exits:
                    # Calculate partial exit level at 50% of the way to take profit
                    partial_exit_level = entry_price + (take_profit - entry_price) * self.partial_exit_threshold
                    # Calculate trailing stop activation level at 75% of the way to take profit
                    trailing_stop_activation_level = entry_price + (take_profit - entry_price) * self.trailing_stop_activation
                
                # Determine signal strength based on multiple factors
                strength = "medium"
                
                # Stronger signal if price is below the band and RSI is very low
                if current_candle.close < lower_band[-1] and rsi[-1] < self.rsi_oversold - 5:
                    strength = "strong"
                
                # Stronger signal if volume confirms
                if volume_confirmation:
                    if strength == "medium":
                        strength = "strong"
                
                # Weaker signal if against the trend
                if trend_down and strength == "strong":
                    strength = "medium"
                elif trend_down and strength == "medium":
                    strength = "weak"
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    timestamp=current_candle.timestamp,
                    direction="long",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_name="MeanReversion",
                    strength=strength,
                    is_crypto=is_crypto,
                    max_holding_days=self.max_holding_days,
                    partial_exit_level=partial_exit_level,
                    trailing_stop_activation_level=trailing_stop_activation_level
                )
                
                # Apply ML filter if enabled
                if self.use_ml_filter and self.ml_classifier is not None:
                    try:
                        quality_score = self.ml_classifier.predict_signal_quality(symbol, candles, "long")
                        signal.quality_score = quality_score
                        
                        if quality_score < self.min_quality_score:
                            self.logger.info(f"{symbol}: Long signal rejected by ML filter (score: {quality_score:.2f})")
                            return signals  # Skip this signal
                    except Exception as e:
                        self.logger.error(f"Error in ML signal filtering: {str(e)}")
                
                # Apply market regime filter if enabled
                if self.use_regime_filter and self.regime_detector is not None and market_state is not None:
                    try:
                        # Check if market regime is favorable for mean reversion long signals
                        regime_score = 0.5  # Neutral by default
                        
                        if market_state.regime == "neutral" or market_state.is_range_bound:
                            regime_score = 0.8  # Range-bound markets are good for mean reversion
                        elif market_state.regime == "bullish" and market_state.trend_strength < 0.5:
                            regime_score = 0.6  # Weak bullish trends can be ok
                        elif market_state.regime == "bearish":
                            regime_score = 0.3  # Bearish regimes are not ideal for long mean reversion
                        
                        signal.regime_score = regime_score
                        
                        if regime_score < self.min_regime_score:
                            self.logger.info(f"{symbol}: Long signal rejected by regime filter (score: {regime_score:.2f})")
                            return signals  # Skip this signal
                    except Exception as e:
                        self.logger.error(f"Error in market regime filtering: {str(e)}")
                
                signals.append(signal)
                self.logger.info(f"Generated LONG signal for {symbol} at {entry_price:.2f}, stop={stop_loss:.2f}, target={take_profit:.2f}")
        
        # For short signals - price is close to or above upper band and RSI is above threshold
        short_signal_condition = False
        
        # Calculate how close price is to the upper band as a percentage
        upper_band_distance = (upper_band[-1] - current_candle.close) / current_candle.close
        
        # Check if price is within minimum distance of upper band or above it
        price_near_upper_band = upper_band_distance <= self.min_distance_to_band_pct or current_candle.close > upper_band[-1]
        self.logger.debug(f"{symbol}: Price to upper band distance: {upper_band_distance:.4f}, Near/above: {price_near_upper_band}")
        
        # Check if RSI is above threshold or has been above threshold in the last 3 candles
        rsi_condition_short = rsi[-1] > self.rsi_overbought or max(rsi[-3:]) > self.rsi_overbought
        self.logger.debug(f"{symbol}: Current RSI: {rsi[-1]:.2f}, Max RSI (last 3): {max(rsi[-3:]):.2f}, RSI condition: {rsi_condition_short}")
        
        # Check if price is showing a potential reversal pattern
        price_pattern_short = False
        if len(candles) >= 3:
            # Simple reversal pattern: higher high followed by lower high
            if candles[-3].high < candles[-2].high and candles[-1].high < candles[-2].high:
                price_pattern_short = True
                self.logger.debug(f"{symbol}: Potential bearish reversal pattern detected")
        
        # Improved condition: Price near band AND RSI condition, with optional trend filter
        if price_near_upper_band and rsi_condition_short and (not trend_up or price_pattern_short):
            short_signal_condition = True
            self.logger.info(f"{symbol}: Short signal condition met - price {current_candle.close:.2f} near/above upper BB {upper_band[-1]:.2f}, RSI {rsi[-1]:.2f}")
        
        if short_signal_condition:
            # Check for price reversal if required
            reversal_confirmed = True
            if self.require_reversal:
                reversal_confirmed = self.detect_price_reversal(candles, "short") or price_pattern_short
                self.logger.debug(f"{symbol}: Short reversal check result: {reversal_confirmed}")
                if not reversal_confirmed:
                    self.logger.info(f"{symbol}: Short signal conditions met but reversal not confirmed")
            
            if reversal_confirmed:
                # Calculate stop-loss and take-profit levels
                entry_price = current_candle.close
                stop_loss = self.calculate_dynamic_stop_loss(candles, "short", entry_price, atr_values[-1])
                
                # Ensure minimum distance for stop loss (at least 1.5% from entry)
                min_stop_distance = entry_price * 0.015
                if stop_loss - entry_price < min_stop_distance:
                    stop_loss = entry_price + min_stop_distance
                    
                take_profit = entry_price - self.take_profit_atr * atr_values[-1]
                
                # Calculate partial exit level if enabled
                partial_exit_level = None
                trailing_stop_activation_level = None
                if self.use_partial_exits:
                    # Calculate partial exit level at 50% of the way to take profit
                    partial_exit_level = entry_price - (entry_price - take_profit) * self.partial_exit_threshold
                    # Calculate trailing stop activation level at 75% of the way to take profit
                    trailing_stop_activation_level = entry_price - (entry_price - take_profit) * self.trailing_stop_activation
                
                # Determine signal strength based on multiple factors
                strength = "medium"
                
                # Stronger signal if price is above the band and RSI is very high
                if current_candle.close > upper_band[-1] and rsi[-1] > self.rsi_overbought + 5:
                    strength = "strong"
                
                # Stronger signal if volume confirms
                if volume_confirmation:
                    if strength == "medium":
                        strength = "strong"
                
                # Weaker signal if against the trend
                if trend_up and strength == "strong":
                    strength = "medium"
                elif trend_up and strength == "medium":
                    strength = "weak"
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    timestamp=current_candle.timestamp,
                    direction="short",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_name="MeanReversion",
                    strength=strength,
                    is_crypto=is_crypto,
                    max_holding_days=self.max_holding_days,
                    partial_exit_level=partial_exit_level,
                    trailing_stop_activation_level=trailing_stop_activation_level
                )
                
                # Apply ML filter if enabled
                if self.use_ml_filter and self.ml_classifier is not None:
                    try:
                        quality_score = self.ml_classifier.predict_signal_quality(symbol, candles, "short")
                        signal.quality_score = quality_score
                        
                        if quality_score < self.min_quality_score:
                            self.logger.info(f"{symbol}: Short signal rejected by ML filter (score: {quality_score:.2f})")
                            return signals  # Skip this signal
                    except Exception as e:
                        self.logger.error(f"Error in ML signal filtering: {str(e)}")
                
                # Apply market regime filter if enabled
                if self.use_regime_filter and self.regime_detector is not None and market_state is not None:
                    try:
                        # Check if market regime is favorable for mean reversion short signals
                        regime_score = 0.5  # Neutral by default
                        
                        if market_state.regime == "neutral" or market_state.is_range_bound:
                            regime_score = 0.8  # Range-bound markets are good for mean reversion
                        elif market_state.regime == "bearish" and market_state.trend_strength < 0.5:
                            regime_score = 0.6  # Weak bearish trends can be ok
                        elif market_state.regime == "bullish":
                            regime_score = 0.3  # Bullish regimes are not ideal for short mean reversion
                        
                        signal.regime_score = regime_score
                        
                        if regime_score < self.min_regime_score:
                            self.logger.info(f"{symbol}: Short signal rejected by regime filter (score: {regime_score:.2f})")
                            return signals  # Skip this signal
                    except Exception as e:
                        self.logger.error(f"Error in market regime filtering: {str(e)}")
                
                signals.append(signal)
                self.logger.info(f"Generated SHORT signal for {symbol} at {entry_price:.2f}, stop={stop_loss:.2f}, target={take_profit:.2f}")
        
        return signals
