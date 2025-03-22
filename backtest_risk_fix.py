#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Risk Management Fix
----------------------------
This script provides fixes for the risk management issues in the enhanced backtest system
that led to catastrophic losses. It includes:
1. Improved position sizing with stricter limits
2. Portfolio-level risk controls
3. Drawdown-based position size reduction
4. Better handling of losing strategies
"""

import os
import sys
import json
import logging
import datetime as dt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest_risk_fix.log')
    ]
)

logger = logging.getLogger("BacktestRiskFix")

class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

def safe_calculate_position_size(signal, market_state, candle_data, 
                                current_equity, position_sizing_config, 
                                current_exposure, max_portfolio_risk_pct,
                                drawdown_pct, logger):
    """
    Calculate position size with improved risk management
    
    Args:
        signal: The trading signal object
        market_state: Current market state/regime
        candle_data: Historical price data for the symbol
        current_equity: Current portfolio equity
        position_sizing_config: Configuration for position sizing
        current_exposure: Current portfolio exposure as percentage
        max_portfolio_risk_pct: Maximum portfolio risk percentage
        drawdown_pct: Current drawdown percentage
        logger: Logger object
        
    Returns:
        float: Position size in dollars with improved risk controls
    """
    try:
        # Base position size as percentage of portfolio
        base_risk = position_sizing_config["base_risk_per_trade"]
        
        # Reduce position size based on current drawdown
        # As drawdown increases, position size decreases exponentially
        drawdown_factor = max(0.2, 1.0 - (drawdown_pct * 2.5))
        
        # Reduce position size based on current exposure
        # As exposure increases, position size decreases linearly
        exposure_factor = max(0.1, 1.0 - (current_exposure / max_portfolio_risk_pct))
        
        # Apply strategy performance adjustment
        # Reduce position size for poorly performing strategies
        strategy_factor = 1.0
        if hasattr(signal, 'strategy') and hasattr(signal, 'metadata'):
            if 'strategy_win_rate' in signal.metadata:
                win_rate = signal.metadata['strategy_win_rate']
                # Scale from 0.5 to 1.0 based on win rate (50% to 100%)
                strategy_factor = max(0.5, win_rate / 100.0)
        
        # Calculate adjusted risk with much more conservative adjustments
        adjusted_risk = base_risk * drawdown_factor * exposure_factor * strategy_factor
        
        # Apply stricter min/max constraints
        adjusted_risk = max(
            position_sizing_config.get("min_position_size", 0.005),
            min(position_sizing_config.get("max_position_size", 0.05), adjusted_risk)
        )
        
        # Calculate dollar amount
        position_dollars = current_equity * adjusted_risk
        
        # Add safety cap based on absolute dollar amount
        max_dollars_per_trade = min(current_equity * 0.05, 5000)  # Max 5% or $5000
        position_dollars = min(position_dollars, max_dollars_per_trade)
        
        logger.info(f"Safe position size for {signal.symbol}: {adjusted_risk:.2%} of portfolio (${position_dollars:.2f})")
        logger.info(f"Adjustment factors - Drawdown: {drawdown_factor:.2f}, Exposure: {exposure_factor:.2f}, Strategy: {strategy_factor:.2f}")
        
        return position_dollars
    except Exception as e:
        logger.error(f"Error calculating safe position size: {str(e)}")
        # Fall back to very small position size
        return current_equity * 0.005  # 0.5% of portfolio

def calculate_portfolio_exposure(trade_history, current_equity):
    """
    Calculate current portfolio exposure from open trades
    
    Args:
        trade_history: List of trades
        current_equity: Current portfolio equity
        
    Returns:
        float: Current exposure as percentage of portfolio
    """
    total_exposure = 0.0
    
    for trade in trade_history:
        if trade.get("status") == "OPEN":
            position_size = trade.get("position_size", 0)
            total_exposure += position_size
    
    # Calculate as percentage of current equity
    if current_equity > 0:
        exposure_pct = (total_exposure / current_equity) * 100
    else:
        exposure_pct = 0  # Avoid division by zero
        
    return exposure_pct

def calculate_drawdown(current_equity, peak_equity):
    """
    Calculate current drawdown
    
    Args:
        current_equity: Current portfolio equity
        peak_equity: Peak portfolio equity
        
    Returns:
        float: Current drawdown as percentage
    """
    if peak_equity > 0:
        drawdown = 1.0 - (current_equity / peak_equity)
    else:
        drawdown = 0.0
        
    return drawdown

def apply_risk_management(signals, trade_history, current_equity, peak_equity, 
                         max_open_positions, max_portfolio_risk_pct, logger):
    """
    Apply enhanced risk management to filter signals
    
    Args:
        signals: List of trading signals
        trade_history: List of trades
        current_equity: Current portfolio equity
        peak_equity: Peak portfolio equity
        max_open_positions: Maximum open positions allowed
        max_portfolio_risk_pct: Maximum portfolio risk percentage
        logger: Logger object
        
    Returns:
        List: Filtered signals that passed risk management checks
    """
    if not signals:
        return []
    
    # Count open positions
    open_positions = sum(1 for trade in trade_history if trade.get("status") == "OPEN")
    
    # Calculate current exposure
    exposure_pct = calculate_portfolio_exposure(trade_history, current_equity)
    
    # Calculate current drawdown
    drawdown_pct = calculate_drawdown(current_equity, peak_equity)
    
    logger.info(f"Risk check - Open positions: {open_positions}, Exposure: {exposure_pct:.2f}%, Drawdown: {drawdown_pct:.2f}%")
    
    # Apply risk management rules
    filtered_signals = []
    
    # Stop trading if max positions reached
    if open_positions >= max_open_positions:
        logger.warning(f"Maximum open positions ({max_open_positions}) reached, no new trades allowed")
        return []
    
    # Stop trading if exposure too high
    if exposure_pct >= max_portfolio_risk_pct:
        logger.warning(f"Maximum portfolio exposure ({max_portfolio_risk_pct}%) reached, no new trades allowed")
        return []
    
    # Reduce trading during drawdowns
    max_new_positions = max_open_positions - open_positions
    if drawdown_pct > 0.1:  # More than 10% drawdown
        # Reduce max new positions based on drawdown severity
        reduction_factor = max(0.1, 1.0 - (drawdown_pct * 2))
        max_new_positions = max(1, int(max_new_positions * reduction_factor))
        logger.warning(f"Drawdown of {drawdown_pct:.2f}% detected, reducing max new positions to {max_new_positions}")
    
    # Take only the strongest signals up to the max new positions limit
    sorted_signals = sorted(signals, key=lambda s: getattr(s, 'score', 0.5), reverse=True)
    filtered_signals = sorted_signals[:max_new_positions]
    
    return filtered_signals

def analyze_strategy_performance(trade_history):
    """
    Analyze performance of each strategy to identify problematic ones
    
    Args:
        trade_history: List of trades
        
    Returns:
        Dict: Performance metrics by strategy
    """
    strategy_performance = {}
    
    # Group trades by strategy
    for trade in trade_history:
        if trade.get("status") != "CLOSED":
            continue
            
        strategy = trade.get("strategy", "Unknown")
        pnl = trade.get("pnl", 0)
        
        if strategy not in strategy_performance:
            strategy_performance[strategy] = {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0,
                "win_rate": 0
            }
        
        strategy_performance[strategy]["total_trades"] += 1
        if pnl > 0:
            strategy_performance[strategy]["winning_trades"] += 1
        strategy_performance[strategy]["total_pnl"] += pnl
    
    # Calculate win rates
    for strategy, perf in strategy_performance.items():
        if perf["total_trades"] > 0:
            perf["win_rate"] = (perf["winning_trades"] / perf["total_trades"]) * 100
    
    return strategy_performance

def should_disable_strategy(strategy_performance, min_trades=5, min_win_rate=40, max_loss_pct=10):
    """
    Determine if a strategy should be disabled based on performance
    
    Args:
        strategy_performance: Performance metrics for the strategy
        min_trades: Minimum trades to consider
        min_win_rate: Minimum acceptable win rate
        max_loss_pct: Maximum acceptable loss percentage
        
    Returns:
        bool: True if strategy should be disabled
    """
    if strategy_performance["total_trades"] < min_trades:
        return False  # Not enough data
    
    if strategy_performance["win_rate"] < min_win_rate:
        return True  # Poor win rate
    
    # Check if strategy is causing significant losses
    if strategy_performance["total_pnl"] < 0:
        # Calculate loss as percentage of initial capital
        # This would need the initial capital value to calculate properly
        # For now, we'll use a placeholder
        return True
    
    return False

def fix_enhanced_backtest(backtest_results_file, output_file=None):
    """
    Analyze backtest results and provide recommendations for fixing issues
    
    Args:
        backtest_results_file: Path to backtest results JSON file
        output_file: Path to output file for recommendations
        
    Returns:
        Dict: Recommendations for fixing backtest issues
    """
    try:
        # Load backtest results
        with open(backtest_results_file, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics
        initial_capital = results.get("initial_capital", 0)
        final_capital = results.get("final_capital", 0)
        total_return_pct = results.get("total_return_pct", 0)
        max_drawdown_pct = results.get("max_drawdown_pct", 0)
        trade_history = results.get("trade_history", [])
        strategy_performance = results.get("strategy_performance", {})
        
        # Analyze issues
        issues = []
        
        # Check for excessive losses
        if final_capital < 0:
            issues.append({
                "issue": "NEGATIVE_EQUITY",
                "description": "The system lost more than the initial capital, indicating excessive leverage or risk.",
                "severity": "CRITICAL",
                "recommendation": "Implement strict position sizing limits and portfolio-level risk controls."
            })
        
        # Check for excessive drawdown
        if max_drawdown_pct > 50:
            issues.append({
                "issue": "EXCESSIVE_DRAWDOWN",
                "description": f"Maximum drawdown of {max_drawdown_pct:.2f}% exceeds acceptable limits.",
                "severity": "HIGH",
                "recommendation": "Implement drawdown-based position size reduction and trading suspension."
            })
        
        # Analyze position sizes
        position_sizes = []
        for trade in trade_history:
            position_size = trade.get("position_size", 0)
            entry_price = trade.get("entry_price", 0)
            shares = trade.get("shares", 0)
            
            # Verify position size calculation
            calculated_size = entry_price * shares
            if abs(calculated_size - position_size) > 1:  # Allow for small rounding differences
                issues.append({
                    "issue": "POSITION_SIZE_MISMATCH",
                    "description": f"Position size mismatch for {trade.get('symbol')}: reported {position_size} vs calculated {calculated_size}",
                    "severity": "MEDIUM",
                    "recommendation": "Verify position size calculation logic."
                })
            
            # Check for excessive position sizes
            if position_size > initial_capital * 0.1:  # More than 10% of initial capital
                issues.append({
                    "issue": "EXCESSIVE_POSITION_SIZE",
                    "description": f"Position size of ${position_size:.2f} for {trade.get('symbol')} exceeds 10% of initial capital.",
                    "severity": "HIGH",
                    "recommendation": "Reduce maximum position size and implement stricter limits."
                })
            
            position_sizes.append(position_size)
        
        # Check for poorly performing strategies
        for strategy, perf in strategy_performance.items():
            win_rate = perf.get("win_rate", 0)
            total_pnl = perf.get("total_pnl", 0)
            
            if win_rate < 45 and total_pnl < 0:
                issues.append({
                    "issue": "POOR_STRATEGY_PERFORMANCE",
                    "description": f"Strategy {strategy} has poor performance: {win_rate:.2f}% win rate and ${total_pnl:.2f} PnL.",
                    "severity": "HIGH",
                    "recommendation": "Implement strategy performance monitoring and automatic disabling of poorly performing strategies."
                })
        
        # Generate recommendations
        recommendations = {
            "issues": issues,
            "fixes": [
                {
                    "type": "POSITION_SIZING",
                    "description": "Implement stricter position sizing with maximum 5% of portfolio per trade.",
                    "implementation": "Use safe_calculate_position_size function with drawdown and exposure adjustments."
                },
                {
                    "type": "RISK_MANAGEMENT",
                    "description": "Add portfolio-level risk controls to limit total exposure and prevent excessive leverage.",
                    "implementation": "Use apply_risk_management function to filter signals based on current exposure and drawdown."
                },
                {
                    "type": "STRATEGY_MONITORING",
                    "description": "Implement automatic monitoring and disabling of poorly performing strategies.",
                    "implementation": "Use analyze_strategy_performance and should_disable_strategy functions."
                },
                {
                    "type": "DRAWDOWN_PROTECTION",
                    "description": "Add drawdown-based position size reduction and trading suspension.",
                    "implementation": "Reduce position sizes and max positions during drawdowns."
                }
            ]
        }
        
        # Save recommendations to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            logger.info(f"Recommendations saved to {output_file}")
        
        return recommendations
    except Exception as e:
        logger.error(f"Error analyzing backtest results: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest Risk Management Fix")
    parser.add_argument("--input", "-i", default="enhanced_backtest_results.json", help="Path to backtest results JSON file")
    parser.add_argument("--output", "-o", default="backtest_fix_recommendations.json", help="Path to output file for recommendations")
    
    args = parser.parse_args()
    
    fix_enhanced_backtest(args.input, args.output)
