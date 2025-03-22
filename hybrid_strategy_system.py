#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Strategy System
---------------------
This module implements a hybrid trading system that combines the best features
from both the direct integration test and the further optimized configuration.
"""

import os
import sys
import logging
import datetime as dt
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union
import copy

# Import the multi-strategy system
from multi_strategy_system import (
    MultiStrategySystem, SystemConfig, Signal, MarketRegime, 
    BacktestResult, StockConfig, MarketState, TradeDirection,
    SignalStrength, PositionState, CandleData
)

# Import enhanced trade management
from enhanced_trade_management import (
    calculate_kelly_position_size,
    calculate_regime_based_size,
    should_apply_trailing_stop,
    should_take_partial_profits,
    should_exit_by_time,
    filter_signals_by_quality,
    adjust_position_for_correlation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_strategy_system.log')
    ]
)

logger = logging.getLogger("HybridStrategySystem")

class HybridStrategySystem(MultiStrategySystem):
    """
    Hybrid Strategy System that combines the best features of both approaches
    """
    
    def __init__(self, config):
        """Initialize the hybrid strategy system"""
        super().__init__(config)
        
        # Initialize additional parameters from config
        self.trade_management_config = config.trade_management if hasattr(config, 'trade_management') else {}
        
        # Historical performance metrics for strategies
        self.historical_performance = {
            "MeanReversion": {
                "win_rate": 0.69,
                "profit_factor": 6.33,
                "sharpe_ratio": 24.80,
                "avg_win_pct": 2.1,
                "avg_loss_pct": 1.0
            },
            "TrendFollowing": {
                "win_rate": 0.55,
                "profit_factor": 2.1,
                "sharpe_ratio": 1.8,
                "avg_win_pct": 2.5,
                "avg_loss_pct": 1.2
            },
            "VolatilityBreakout": {
                "win_rate": 0.60,
                "profit_factor": 3.2,
                "sharpe_ratio": 2.1,
                "avg_win_pct": 3.0,
                "avg_loss_pct": 1.5
            },
            "GapTrading": {
                "win_rate": 0.50,
                "profit_factor": 1.8,
                "sharpe_ratio": 1.5,
                "avg_win_pct": 2.8,
                "avg_loss_pct": 1.6
            }
        }
        
        # Initialize correlation matrix
        self.correlation_matrix = self._initialize_correlation_matrix()
        
        # Fix the sector performance error
        self._patch_market_analyzer()
        
        logger.info("Hybrid Strategy System initialized")
    
    def _initialize_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize correlation matrix between symbols"""
        symbols = [stock.symbol for stock in self.config.stocks]
        correlation_matrix = {}
        
        # Create empty matrix
        for symbol in symbols:
            correlation_matrix[symbol] = {}
            for other_symbol in symbols:
                if symbol == other_symbol:
                    correlation_matrix[symbol][other_symbol] = 1.0
                else:
                    # Default correlation of 0.5 for technology stocks, 0.3 for others
                    # This will be updated with real data during operation
                    correlation_matrix[symbol][other_symbol] = 0.5 if "Technology" in self._get_stock_sector(symbol) else 0.3
        
        return correlation_matrix
    
    def _get_stock_sector(self, symbol: str) -> str:
        """Get the sector for a given symbol"""
        for stock in self.config.stocks:
            if stock.symbol == symbol:
                return stock.sector
        return "Unknown"
    
    def _update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix with actual price data"""
        symbols = list(price_data.keys())
        
        # Create a DataFrame with close prices for all symbols
        close_prices = pd.DataFrame()
        for symbol in symbols:
            if symbol in price_data and not price_data[symbol].empty:
                close_prices[symbol] = price_data[symbol]['close']
        
        # Calculate correlation matrix if we have enough data
        if not close_prices.empty and len(close_prices.columns) > 1:
            correlation = close_prices.pct_change().corr()
            
            # Update our correlation matrix
            for symbol in symbols:
                if symbol in correlation.columns:
                    for other_symbol in symbols:
                        if other_symbol in correlation.columns:
                            self.correlation_matrix[symbol][other_symbol] = correlation.loc[symbol, other_symbol]
    
    def generate_signals(self, timestamp: dt.datetime, symbols: List[str]) -> Dict[str, List[Signal]]:
        """Generate trading signals with enhanced filtering"""
        # Get base signals from parent class
        base_signals = super().generate_signals(timestamp, symbols)
        
        # Apply enhanced filtering
        filtered_signals = {}
        
        for symbol, signals in base_signals.items():
            if not signals:
                filtered_signals[symbol] = []
                continue
            
            # Get signal quality filter parameters
            min_profit_factor = self.config.signal_quality_filters.get('min_profit_factor', 1.5)
            min_sharpe_ratio = self.config.signal_quality_filters.get('min_sharpe_ratio', 0.8)
            
            # Filter signals by quality
            quality_filtered = filter_signals_by_quality(
                signals,
                min_profit_factor,
                min_sharpe_ratio,
                self.historical_performance
            )
            
            filtered_signals[symbol] = quality_filtered
        
        return filtered_signals
    
    def calculate_position_size(self, signal: Signal, available_capital: float) -> float:
        """Calculate position size with enhanced methods"""
        # Get base position size from parent class
        base_size = super().calculate_position_size(signal, available_capital)
        
        # Get position sizing config
        position_sizing_config = self.config.position_sizing_config
        
        # Apply Kelly criterion if enabled
        if position_sizing_config.get('kelly_criterion_factor'):
            kelly_factor = position_sizing_config.get('kelly_criterion_factor', 0.3)
            strategy = signal.strategy
            
            if strategy in self.historical_performance:
                metrics = self.historical_performance[strategy]
                win_rate = metrics.get('win_rate', 0.5)
                avg_win_pct = metrics.get('avg_win_pct', 2.0)
                avg_loss_pct = metrics.get('avg_loss_pct', 1.0)
                
                kelly_size = calculate_kelly_position_size(
                    win_rate, avg_win_pct, avg_loss_pct, kelly_factor
                ) * available_capital
                
                # Blend with base size
                base_size = (base_size + kelly_size) / 2
        
        # Apply regime-based sizing if enabled
        if position_sizing_config.get('regime_based_sizing'):
            market_state = self.market_analyzer.get_current_market_state()
            base_size = calculate_regime_based_size(
                base_size, market_state, signal.direction
            )
        
        # Adjust for correlation with existing positions
        current_positions = list(self.positions.values())
        base_size = adjust_position_for_correlation(
            base_size,
            signal.symbol,
            signal.direction,
            current_positions,
            self.correlation_matrix,
            max_correlation_exposure=1.5
        )
        
        # Apply min/max constraints
        max_position_size = position_sizing_config.get('max_position_size', 0.05) * available_capital
        min_position_size = position_sizing_config.get('min_position_size', 0.002) * available_capital
        
        return max(min_position_size, min(base_size, max_position_size))
    
    def should_exit_position(self, position_id: str, current_price: float, timestamp: dt.datetime) -> Tuple[bool, str]:
        """Enhanced position exit logic with trailing stops and partial profit taking"""
        # Get position
        if position_id not in self.positions:
            return False, ""
        
        position = self.positions[position_id]
        symbol = position.symbol
        
        # Check basic exit conditions from parent class
        should_exit, reason = super().should_exit_position(position_id, current_price, timestamp)
        if should_exit:
            return True, reason
        
        # Get trade management config
        trade_mgmt = self.trade_management_config
        
        # Check for trailing stop
        if trade_mgmt.get('trailing_stop_activation_pct'):
            activation_pct = trade_mgmt.get('trailing_stop_activation_pct', 1.0)
            distance_pct = trade_mgmt.get('trailing_stop_distance_pct', 0.5)
            
            apply_trailing, new_stop = should_apply_trailing_stop(
                position, current_price, activation_pct, distance_pct
            )
            
            if apply_trailing and new_stop:
                # Update stop loss
                self.positions[position_id].stop_loss = new_stop
                logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
        
        # Check for partial profit taking
        if trade_mgmt.get('partial_profit_taking'):
            profit_levels = {
                trade_mgmt.get('partial_profit_level_1', 1.5): 0.25,
                trade_mgmt.get('partial_profit_level_2', 2.5): 0.25
            }
            
            take_profit, exit_percentage = should_take_partial_profits(
                position, current_price, profit_levels
            )
            
            if take_profit and exit_percentage > 0:
                # Reduce position size
                new_size = position.size * (1 - exit_percentage)
                old_size = position.size
                
                # Calculate profit
                if position.direction == TradeDirection.LONG:
                    profit = (current_price - position.entry_price) * (old_size - new_size)
                else:
                    profit = (position.entry_price - current_price) * (old_size - new_size)
                
                # Update position size
                self.positions[position_id].size = new_size
                
                # Update realized PnL
                self.positions[position_id].realized_pnl += profit
                
                logger.info(f"Took partial profits on {symbol}: {exit_percentage*100}%, Profit: ${profit:.2f}")
                
                # If position is completely closed
                if new_size <= 0:
                    return True, "Took full profits"
        
        # Check for time-based exit
        if trade_mgmt.get('time_based_exit'):
            max_days = trade_mgmt.get('max_holding_period_days', 10)
            
            if should_exit_by_time(position, timestamp, max_days):
                return True, f"Maximum holding period of {max_days} days reached"
        
        return False, ""
    
    def update_market_state(self, timestamp: dt.datetime):
        """Enhanced market state update with correlation matrix update"""
        # Update base market state
        super().update_market_state(timestamp)
        
        # Update correlation matrix if we have price data
        if hasattr(self, 'price_data') and self.price_data:
            self._update_correlation_matrix(self.price_data)
    
    def run_backtest(self, start_date: dt.datetime, end_date: dt.datetime) -> BacktestResult:
        """Run backtest with enhanced features"""
        logger.info(f"Starting hybrid backtest from {start_date} to {end_date}")
        
        # Run the backtest using the parent method
        result = super().run_backtest(start_date, end_date)
        
        # Additional post-processing and analysis
        if result:
            # Calculate strategy-specific metrics
            strategy_metrics = self._calculate_strategy_metrics(result)
            result.additional_metrics = strategy_metrics
            
            logger.info(f"Hybrid backtest completed with {result.total_trades} trades")
            logger.info(f"Total return: {result.total_return_pct:.2f}%")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
            
            # Update historical performance with new results
            self._update_historical_performance(result)
        
        return result
    
    def _calculate_strategy_metrics(self, result: BacktestResult) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for each strategy"""
        strategy_metrics = {}
        
        # Group trades by strategy
        strategy_trades = {}
        for trade in result.trades:
            strategy = trade.strategy
            if strategy not in strategy_trades:
                strategy_trades[strategy] = []
            strategy_trades[strategy].append(trade)
        
        # Calculate metrics for each strategy
        for strategy, trades in strategy_trades.items():
            if not trades:
                continue
            
            # Calculate basic metrics
            win_count = sum(1 for t in trades if t.pnl > 0)
            loss_count = sum(1 for t in trades if t.pnl < 0)
            total_count = len(trades)
            
            win_rate = win_count / total_count if total_count > 0 else 0
            
            total_profit = sum(t.pnl for t in trades if t.pnl > 0)
            total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate average win and loss percentages
            win_pcts = [(t.exit_price - t.entry_price) / t.entry_price * 100 if t.direction == TradeDirection.LONG 
                        else (t.entry_price - t.exit_price) / t.entry_price * 100 
                        for t in trades if t.pnl > 0]
            
            loss_pcts = [(t.entry_price - t.exit_price) / t.entry_price * 100 if t.direction == TradeDirection.LONG 
                         else (t.exit_price - t.entry_price) / t.entry_price * 100 
                         for t in trades if t.pnl < 0]
            
            avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0
            avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
            
            # Store metrics
            strategy_metrics[strategy] = {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "total_trades": total_count,
                "win_count": win_count,
                "loss_count": loss_count
            }
        
        return strategy_metrics
    
    def _update_historical_performance(self, result: BacktestResult):
        """Update historical performance metrics based on backtest results"""
        if not hasattr(result, 'additional_metrics') or not result.additional_metrics:
            return
        
        # Update metrics for each strategy
        for strategy, metrics in result.additional_metrics.items():
            if strategy in self.historical_performance:
                # Blend new metrics with historical data (70% historical, 30% new)
                for key, value in metrics.items():
                    if key in self.historical_performance[strategy]:
                        historical_value = self.historical_performance[strategy][key]
                        self.historical_performance[strategy][key] = historical_value * 0.7 + value * 0.3
                    else:
                        self.historical_performance[strategy][key] = value
            else:
                # Add new strategy
                self.historical_performance[strategy] = metrics
    
    def _patch_market_analyzer(self):
        """Fix the 'technology' sector error by patching the _determine_sub_regime method"""
        original_method = self.market_analyzer._determine_sub_regime
        
        def patched_method(self, base_regime, adx, vix, trend_direction, 
                          breadth_indicators, intermarket_indicators,
                          sector_performance, sentiment_indicators):
            """Patched method that checks if keys exist before accessing them"""
            if base_regime == MarketRegime.CONSOLIDATION:
                # Check if the required sector keys exist before accessing them
                if 'technology' in sector_performance and 'healthcare' in sector_performance:
                    if sector_performance['technology'] > 0 and sector_performance['healthcare'] > 0:
                        return "Bullish Consolidation"
                    elif sector_performance['technology'] < 0 and sector_performance['healthcare'] < 0:
                        return "Bearish Consolidation"
                    else:
                        return "Neutral Consolidation"
                else:
                    return "Neutral Consolidation"
            
            # Rest of the original method remains unchanged
            return original_method(self, base_regime, adx, vix, trend_direction, 
                                 breadth_indicators, intermarket_indicators,
                                 sector_performance, sentiment_indicators)
        
        # Replace the method
        self.market_analyzer._determine_sub_regime = patched_method.__get__(self.market_analyzer, type(self.market_analyzer))
        
        logger.info("Fixed sector performance error by patching _determine_sub_regime method")
