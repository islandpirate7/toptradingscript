#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Trade Management Module
-------------------------------
This module provides advanced trade management functionality including:
- Trailing stops
- Partial profit taking
- Time-based exits
- Kelly criterion position sizing
- Regime-based position sizing
"""

import logging
import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

from multi_strategy_system import (
    Signal, TradeDirection, SignalStrength, PositionState,
    CandleData, MarketState, MarketRegime
)

logger = logging.getLogger("EnhancedTradeManagement")

def calculate_kelly_position_size(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_factor: float = 0.3
) -> float:
    """
    Calculate position size using the Kelly Criterion with a fractional factor
    
    Args:
        win_rate: Historical win rate (0.0-1.0)
        avg_win_pct: Average win percentage
        avg_loss_pct: Average loss percentage
        kelly_factor: Fraction of full Kelly to use (0.0-1.0)
        
    Returns:
        float: Suggested position size as a fraction of portfolio
    """
    if win_rate <= 0 or avg_win_pct <= 0 or avg_loss_pct <= 0:
        return 0.0
    
    # Convert percentages to decimals if needed
    if avg_win_pct > 1:
        avg_win_pct = avg_win_pct / 100
    if avg_loss_pct > 1:
        avg_loss_pct = avg_loss_pct / 100
    
    # Calculate full Kelly
    q = 1 - win_rate
    b = avg_win_pct / avg_loss_pct
    kelly = (win_rate * b - q) / b
    
    # Apply fractional Kelly and ensure it's within reasonable bounds
    kelly = max(0, min(kelly * kelly_factor, 0.2))
    
    return kelly

def calculate_regime_based_size(
    base_size: float,
    market_state: MarketState,
    direction: TradeDirection
) -> float:
    """
    Adjust position size based on market regime and trade direction
    
    Args:
        base_size: Base position size
        market_state: Current market state
        direction: Trade direction (LONG or SHORT)
        
    Returns:
        float: Adjusted position size
    """
    regime = market_state.regime
    sub_regime = market_state.sub_regime
    
    # Default multiplier
    multiplier = 1.0
    
    # Adjust based on regime and direction alignment
    if regime == MarketRegime.TRENDING_BULLISH:
        if direction == TradeDirection.LONG:
            multiplier = 1.3
        else:
            multiplier = 0.7
    elif regime == MarketRegime.TRENDING_BEARISH:
        if direction == TradeDirection.SHORT:
            multiplier = 1.3
        else:
            multiplier = 0.7
    elif regime == MarketRegime.HIGH_VOLATILITY:
        multiplier = 0.8  # Reduce size in high volatility
    elif regime == MarketRegime.LOW_VOLATILITY:
        multiplier = 0.9  # Slightly reduce size in low volatility
    elif regime == MarketRegime.RANGE_BOUND:
        multiplier = 1.1  # Slightly increase for range-bound markets
    
    # Further adjust based on sub-regime if available
    if sub_regime:
        if "Bullish" in sub_regime and direction == TradeDirection.LONG:
            multiplier *= 1.1
        elif "Bearish" in sub_regime and direction == TradeDirection.SHORT:
            multiplier *= 1.1
        elif "Neutral" in sub_regime:
            multiplier *= 0.9
    
    # Ensure we don't increase size too much
    multiplier = max(0.5, min(multiplier, 1.5))
    
    return base_size * multiplier

def should_apply_trailing_stop(
    position: PositionState,
    current_price: float,
    activation_pct: float,
    distance_pct: float
) -> Tuple[bool, Optional[float]]:
    """
    Determine if a trailing stop should be applied and calculate the new stop level
    
    Args:
        position: Current position state
        current_price: Current market price
        activation_pct: Percentage profit needed to activate trailing stop
        distance_pct: Percentage distance for trailing stop
        
    Returns:
        Tuple[bool, Optional[float]]: (Should apply trailing stop, New stop level)
    """
    if not position:
        return False, None
    
    entry_price = position.entry_price
    direction = position.direction
    current_stop = position.stop_loss
    
    # Calculate current profit percentage
    if direction == TradeDirection.LONG:
        profit_pct = (current_price - entry_price) / entry_price * 100
        if profit_pct >= activation_pct:
            # Calculate trailing stop level
            new_stop = current_price * (1 - distance_pct / 100)
            # Only update if it would raise the stop level
            if new_stop > current_stop:
                return True, new_stop
    else:  # SHORT
        profit_pct = (entry_price - current_price) / entry_price * 100
        if profit_pct >= activation_pct:
            # Calculate trailing stop level
            new_stop = current_price * (1 + distance_pct / 100)
            # Only update if it would lower the stop level
            if new_stop < current_stop:
                return True, new_stop
    
    return False, None

def should_take_partial_profits(
    position: PositionState,
    current_price: float,
    profit_levels: Dict[float, float]
) -> Tuple[bool, float]:
    """
    Determine if partial profits should be taken
    
    Args:
        position: Current position state
        current_price: Current market price
        profit_levels: Dictionary mapping profit levels (in R) to percentage to exit
        
    Returns:
        Tuple[bool, float]: (Should take profits, Percentage to exit)
    """
    if not position or not profit_levels:
        return False, 0.0
    
    entry_price = position.entry_price
    stop_loss = position.stop_loss
    direction = position.direction
    
    # Calculate risk (R)
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return False, 0.0
    
    # Calculate current profit in terms of R
    if direction == TradeDirection.LONG:
        current_profit_r = (current_price - entry_price) / risk
    else:  # SHORT
        current_profit_r = (entry_price - current_price) / risk
    
    # Check if we've reached any profit levels
    for profit_level_r, exit_percentage in sorted(profit_levels.items()):
        if current_profit_r >= profit_level_r:
            return True, exit_percentage
    
    return False, 0.0

def should_exit_by_time(
    position: PositionState,
    current_time: dt.datetime,
    max_holding_period_days: int
) -> bool:
    """
    Determine if a position should be exited based on time
    
    Args:
        position: Current position state
        current_time: Current time
        max_holding_period_days: Maximum holding period in days
        
    Returns:
        bool: True if position should be exited
    """
    if not position:
        return False
    
    entry_time = position.entry_time
    holding_period = (current_time - entry_time).total_seconds() / (24 * 60 * 60)
    
    return holding_period >= max_holding_period_days

def filter_signals_by_quality(
    signals: List[Signal],
    min_profit_factor: float,
    min_sharpe_ratio: float,
    historical_performance: Dict[str, Dict[str, float]]
) -> List[Signal]:
    """
    Filter signals based on historical strategy performance
    
    Args:
        signals: List of signals to filter
        min_profit_factor: Minimum acceptable profit factor
        min_sharpe_ratio: Minimum acceptable Sharpe ratio
        historical_performance: Dictionary of historical performance metrics by strategy
        
    Returns:
        List[Signal]: Filtered signals
    """
    if not signals or not historical_performance:
        return signals
    
    filtered_signals = []
    
    for signal in signals:
        strategy = signal.strategy
        
        # Skip if we don't have historical data for this strategy
        if strategy not in historical_performance:
            filtered_signals.append(signal)
            continue
        
        # Get historical metrics
        metrics = historical_performance[strategy]
        profit_factor = metrics.get('profit_factor', 0.0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        
        # Filter based on minimum thresholds
        if profit_factor >= min_profit_factor and sharpe_ratio >= min_sharpe_ratio:
            filtered_signals.append(signal)
    
    return filtered_signals

def adjust_position_for_correlation(
    base_size: float,
    symbol: str,
    direction: TradeDirection,
    current_positions: List[PositionState],
    correlation_matrix: Dict[str, Dict[str, float]],
    max_correlation_exposure: float = 1.5
) -> float:
    """
    Adjust position size based on correlation with existing positions
    
    Args:
        base_size: Base position size
        symbol: Symbol for the new position
        direction: Direction of the new position
        current_positions: List of current positions
        correlation_matrix: Matrix of correlations between symbols
        max_correlation_exposure: Maximum exposure to correlated assets
        
    Returns:
        float: Adjusted position size
    """
    if not current_positions or not correlation_matrix or symbol not in correlation_matrix:
        return base_size
    
    # Calculate correlation-weighted exposure
    correlation_weighted_exposure = 0.0
    
    for position in current_positions:
        pos_symbol = position.symbol
        pos_direction = position.direction
        pos_size = position.size
        
        # Skip if same symbol or not in correlation matrix
        if pos_symbol == symbol or pos_symbol not in correlation_matrix[symbol]:
            continue
        
        # Get correlation between symbols
        correlation = correlation_matrix[symbol][pos_symbol]
        
        # Adjust for direction (negative correlation if opposite directions)
        direction_factor = 1.0 if direction == pos_direction else -1.0
        
        # Add to correlation-weighted exposure
        correlation_weighted_exposure += pos_size * correlation * direction_factor
    
    # Calculate adjustment factor
    if correlation_weighted_exposure > 0:
        # Reduce size if we already have correlated exposure
        adjustment_factor = max(0.2, 1.0 - (correlation_weighted_exposure / max_correlation_exposure))
    else:
        # Increase size if we have negative correlation (hedging)
        adjustment_factor = min(1.5, 1.0 + (abs(correlation_weighted_exposure) / max_correlation_exposure))
    
    return base_size * adjustment_factor
