#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced MeanReversion Strategy Backtest Script
Implements dynamic stop-loss placement and time-based exits
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api
from alpaca_trade_api.rest import TimeFrame
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import math
from mean_reversion_enhanced import EnhancedMeanReversionStrategy, CandleData, Signal, Trade, MarketState

# Try to import market regime detector and ML signal classifier
try:
    from market_regime_detector import MarketRegimeDetector
    MARKET_REGIME_AVAILABLE = True
except ImportError:
    MARKET_REGIME_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Market regime detector module not available")

try:
    from ml_signal_classifier import MLSignalClassifier
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ML signal classifier module not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMeanReversionBacktest:
    """Backtest implementation for the Enhanced MeanReversion strategy"""
    
    def __init__(self, config):
        """
        Initialize the backtest
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy
        self.strategy = EnhancedMeanReversionStrategy(self.config)
        
        # Initialize portfolio
        self.initial_capital = self.config.get('global', {}).get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.signals = []
        self.signals_by_direction = {"long": 0, "short": 0}
        self.pending_trades = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Get stock and crypto configurations
        self.stock_configs = self.config.get('stocks', {})
        self.crypto_configs = self.config.get('cryptos', {})
        
        self.logger.info(f"Separated symbols: {len(self.stock_configs)} stocks, {len(self.crypto_configs)} cryptos")
        
        # Initialize historical data storage
        self.historical_data = {}
        self.spy_candles = []
        self.market_states = {}
        self.ml_training_data = []
        
        # Initialize market regime detector
        self.logger.info("Market regime detector initialized")
        try:
            self.market_regime_detector = MarketRegimeDetector(self.config.get('market_regime_params', {}))
            self.logger.info("Market regime detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing market regime detector: {str(e)}")
            self.market_regime_detector = None
            
        # Initialize ML signal classifier
        try:
            self.ml_signal_classifier = MLSignalClassifier(self.config.get('ml_classifier_params', {}))
            self.logger.info("ML signal classifier initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing ML signal classifier: {str(e)}")
            self.ml_signal_classifier = None

    def detect_market_regime(self, date: datetime.datetime) -> MarketState:
        """
        Detect the current market regime based on SPY price action
        
        Args:
            date: Current date
            
        Returns:
            MarketState object with regime information
        """
        # Use the market regime detector if available
        if self.market_regime_detector is not None:
            return self.market_regime_detector.detect_market_regime(self.spy_candles, date)
        
        # Fallback to simple regime detection
        if len(self.spy_candles) < 50:
            return MarketState(
                date=date,
                regime="neutral",
                volatility=0.0,
                trend_strength=0.0,
                is_range_bound=False
            )
        
        # Calculate 20-day and 50-day moving averages
        closes = [c.close for c in self.spy_candles]
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:])
        
        # Calculate volatility (20-day standard deviation)
        volatility = np.std(closes[-20:]) / ma20
        
        # Determine trend direction
        if ma20 > ma50 * 1.02:
            regime = "bullish"
        elif ma20 < ma50 * 0.98:
            regime = "bearish"
        else:
            regime = "neutral"
        
        # Calculate trend strength using ADX-like measure
        price_changes = np.abs(np.diff(closes[-20:]))
        avg_price_change = np.mean(price_changes)
        trend_strength = avg_price_change / ma20
        
        # Determine if market is range-bound
        is_range_bound = (volatility < 0.01) and (trend_strength < 0.005)
        
        market_state = MarketState(
            date=date,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            is_range_bound=is_range_bound
        )
        
        # Store market state for later analysis
        self.market_states.append(market_state)
        
        return market_state

    def generate_signals_for_symbol(self, symbol: str, candles: List[CandleData], market_state=None) -> List[Signal]:
        """
        Generate signals for a specific symbol using the strategy
        
        Args:
            symbol: Symbol to generate signals for
            candles: Historical candles for the symbol
            market_state: Current market state
            
        Returns:
            List of signals
        """
        if not candles or len(candles) < 20:  # Need at least 20 candles for indicators
            return []
            
        # Get the latest candle
        latest_candle = candles[-1]
        current_date = latest_candle.timestamp
        
        # Check if we should trade in the current market regime
        if market_state and self.market_regime_detector:
            # Skip if market regime is not favorable
            if market_state.regime == "bearish" and not self.config.get("trade_in_bear_market", True):
                self.logger.debug(f"Skipping signal generation for {symbol} in bearish market")
                return []
                
            # Skip if market is too volatile
            if market_state.volatility > self.config.get("max_market_volatility", 0.03):
                self.logger.debug(f"Skipping signal generation for {symbol} due to high market volatility: {market_state.volatility:.4f}")
                return []
        
        # Get symbol-specific configuration
        symbol_config = {}
        if symbol in self.stock_configs:
            symbol_config = self.stock_configs[symbol]
        elif symbol in self.crypto_configs:
            symbol_config = self.crypto_configs[symbol]
            
        # Use strategy to generate signals
        signals = self.strategy.generate_signals(candles, symbol, market_state)
        
        if signals:
            self.logger.info(f"Generated {len(signals)} raw signals for {symbol} at {current_date}")
            
        # Apply ML filtering if available
        if self.ml_signal_classifier and self.ml_signal_classifier.is_trained:
            filtered_signals = []
            for signal in signals:
                # Prepare features for the signal
                features = self.ml_signal_classifier.extract_features(candles, signal)
                
                # Get signal quality score
                quality_score = self.ml_signal_classifier.predict_signal_quality(features)
                
                # Set signal quality
                signal.quality_score = quality_score
                
                # Apply minimum quality threshold
                min_quality = self.config.get('strategies', {}).get('MeanReversion', {}).get('min_quality_score', 0.4)
                
                if quality_score >= min_quality:
                    filtered_signals.append(signal)
                else:
                    self.logger.debug(f"Filtered out {symbol} {signal.direction} signal with quality score {quality_score:.2f} < {min_quality:.2f}")
                    
            signals = filtered_signals
            
        # Apply symbol-specific weight multiplier if available
        for signal in signals:
            # Get strategy-specific weight multiplier for this symbol
            weight_multiplier = 1.0
            if 'strategies' in symbol_config:
                if 'MeanReversion' in symbol_config['strategies']:
                    weight_multiplier = symbol_config['strategies']['MeanReversion'].get('weight_multiplier', 1.0)
            
            signal.weight *= weight_multiplier
            
        return signals
    
    def execute_signal(self, signal, current_date):
        """
        Execute a trading signal by creating a position
        
        Args:
            signal: The signal to execute
            current_date: The current date
        """
        symbol = signal.symbol
        direction = signal.direction
        
        # Skip if we already have a position for this symbol
        if symbol in self.positions:
            self.logger.info(f"Already have position in {symbol}, skipping signal")
            return
            
        # Calculate position size
        position_size = self.calculate_position_size(signal)
        
        # Ensure position size is at least 1
        position_size = max(1, int(position_size))
        
        # Check if we have enough cash
        cost = signal.entry_price * position_size
        if cost > self.cash:
            self.logger.info(f"Not enough cash to enter {symbol} {direction} position: {cost:.2f} > {self.cash:.2f}")
            return
            
        # Create trade object
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_date=current_date.replace(tzinfo=None),  # Ensure naive datetime
            entry_price=signal.entry_price,
            position_size=position_size,  # Changed from quantity to position_size
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            exit_date=None,
            exit_price=None,
            pnl=0,
            exit_reason=None,
            max_holding_days=signal.max_holding_days,
            partial_exit_level=signal.partial_exit_level,
            trailing_stop_activation_level=signal.trailing_stop_activation_level,
            is_crypto='/' in symbol
        )
        
        # Create position
        self.positions[symbol] = {
            "direction": direction,
            "entry_price": signal.entry_price,
            "quantity": position_size,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "entry_date": current_date.replace(tzinfo=None),  # Ensure naive datetime
            "trade": trade
        }
        
        # Add to pending trades for ML training
        self.pending_trades[symbol] = {
            "signal": signal,
            "trade": trade
        }
        
        # Update cash
        self.cash -= cost
        
        self.logger.info(f"Entered {direction} position in {symbol} at {signal.entry_price:.2f}, quantity: {position_size}, stop: {signal.stop_loss:.2f}, target: {signal.take_profit:.2f}")
        
    def update_positions(self, current_date):
        """
        Update all positions for the current date
        
        Args:
            current_date: The current date
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            trade = position["trade"]
            
            # Get the latest candle for this symbol
            symbol_candles = [c for c in self.historical_data.get(symbol, []) if c.timestamp.replace(tzinfo=None) <= current_date.replace(tzinfo=None)]
            
            if not symbol_candles:
                continue
                
            latest_candle = symbol_candles[-1]
            
            # Check for stop loss
            if trade.direction == "long" and latest_candle.low <= trade.stop_loss:
                trade.exit_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "stop_loss"
                positions_to_close.append(symbol)
                
            elif trade.direction == "short" and latest_candle.high >= trade.stop_loss:
                trade.exit_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "stop_loss"
                positions_to_close.append(symbol)
                
            # Check for take profit
            elif trade.direction == "long" and latest_candle.high >= trade.take_profit:
                trade.exit_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
                trade.exit_price = trade.take_profit
                trade.exit_reason = "take_profit"
                positions_to_close.append(symbol)
                
            elif trade.direction == "short" and latest_candle.low <= trade.take_profit:
                trade.exit_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
                trade.exit_price = trade.take_profit
                trade.exit_reason = "take_profit"
                positions_to_close.append(symbol)
                
            # Check for max holding days
            elif trade.max_holding_days is not None:
                days_held = (current_date.date() - trade.entry_date.date()).days
                if days_held >= trade.max_holding_days:
                    trade.exit_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
                    trade.exit_price = latest_candle.close
                    trade.exit_reason = "max_holding_days"
                    positions_to_close.append(symbol)
        
        # Close positions
        for symbol in positions_to_close:
            self.close_position(symbol, current_date)
            
    def close_position(self, symbol, current_date):
        """
        Close a position
        
        Args:
            symbol: The symbol to close
            current_date: The current date
        """
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        trade = position["trade"]
        
        # Ensure exit_date is set
        if trade.exit_date is None:
            trade.exit_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
            
        # Ensure exit_price is set
        if trade.exit_price is None:
            # Get latest price
            symbol_candles = [c for c in self.historical_data.get(symbol, []) if c.timestamp.replace(tzinfo=None) <= current_date.replace(tzinfo=None)]
            if symbol_candles:
                trade.exit_price = symbol_candles[-1].close
            else:
                trade.exit_price = trade.entry_price
                
        # Calculate P&L
        if position["direction"] == "long":
            trade.pnl = (trade.exit_price - trade.entry_price) * position["quantity"]
        else:  # short
            trade.pnl = (trade.entry_price - trade.exit_price) * position["quantity"]
            
        # Update cash
        self.cash += position["quantity"] * trade.exit_price
        
        # Update trade statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Add to completed trades
        self.trades.append(trade)
        
        # Log trade
        self.logger.info(f"Closed {position['direction']} position in {symbol} at {trade.exit_price:.2f}, P&L: ${trade.pnl:.2f}, reason: {trade.exit_reason}")
        
        # Remove position
        del self.positions[symbol]
        
    def close_all_positions(self, date, reason="end_of_backtest"):
        """
        Close all open positions
        
        Args:
            date: The date to close positions
            reason: Reason for closing positions
        """
        current_date = datetime.datetime.combine(date, datetime.time()) if isinstance(date, datetime.date) else date
        current_date = current_date.replace(tzinfo=None)  # Ensure naive datetime
        
        symbols = list(self.positions.keys())
        for symbol in symbols:
            position = self.positions[symbol]
            trade = position["trade"]
            
            # Set exit details
            trade.exit_date = current_date
            
            # Get latest price
            symbol_candles = [c for c in self.historical_data.get(symbol, []) if c.timestamp.replace(tzinfo=None) <= current_date.replace(tzinfo=None)]
            if symbol_candles:
                trade.exit_price = symbol_candles[-1].close
            else:
                trade.exit_price = trade.entry_price
                
            trade.exit_reason = reason
            
            # Close the position
            self.close_position(symbol, current_date)
            
    def collect_ml_data_for_trade(self, symbol, trade):
        """
        Collect ML training data for a specific trade
        
        Args:
            symbol: The symbol of the trade
            trade: The trade object
        """
        if not self.ml_signal_classifier:
            return
            
        # Get the signal from pending trades
        if symbol not in self.pending_trades:
            return
            
        signal = self.pending_trades[symbol]["signal"]
        
        # Calculate outcome (1 for profit, 0 for loss)
        outcome = 1 if trade.pnl > 0 else 0
        
        # Get market state at entry
        entry_date = trade.entry_date
        if entry_date.tzinfo is not None:
            entry_date = entry_date.replace(tzinfo=None)  # Ensure naive datetime
            
        market_state = self.market_states.get(entry_date)
        
        # Get candles at entry time
        candles = [c for c in self.historical_data.get(symbol, []) if c.timestamp.replace(tzinfo=None) <= entry_date.replace(tzinfo=None)]
        
        if not candles or not market_state:
            return
            
        # Extract features for ML training
        features = self.ml_signal_classifier.extract_features(candles, signal, market_state)
        
        # Add to training data
        self.ml_training_data.append((features, outcome))
        
        self.logger.info(f"Added ML training data for {symbol}: outcome={outcome}")
        
    def run_backtest(self, start_date, end_date):
        """
        Run the backtest from start_date to end_date
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")
        
        # Initialize Alpaca API
        self.initialize_alpaca_api()
        
        # Fetch SPY data for market regime detection
        self.logger.info(f"Fetching historical data for SPY from {start_date.date()} to {end_date.date()}")
        self.spy_candles = self.fetch_historical_data("SPY", start_date, end_date)
        
        if not self.spy_candles:
            self.logger.error("Failed to fetch SPY data, cannot determine market state")
        
        # Fetch data for each symbol
        symbols = list(self.stock_configs.keys()) + list(self.crypto_configs.keys())
        self.logger.debug(f"Symbols to process: {symbols}")
        
        for symbol in symbols:
            self.logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
            candles = self.fetch_historical_data(symbol, start_date, end_date)
            if candles:
                self.historical_data[symbol] = candles
        
        # Process each trading day
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                current_date += datetime.timedelta(days=1)
                continue
                
            date = current_date.date()
            self.logger.info(f"Processing date: {date}")
            
            # Detect market regime
            if self.spy_candles:
                market_state = self.detect_market_regime(current_date)
                self.market_states[date] = market_state
                self.logger.debug(f"Market regime for {date}: {market_state.regime}, volatility: {market_state.volatility:.4f}")
            else:
                # Default market state if SPY data not available
                market_state = MarketState(
                    date=current_date,
                    regime="neutral",
                    volatility=0.01,
                    trend_strength=0.0,
                    is_range_bound=False
                )
                
            # Generate signals for each symbol
            for symbol, candles in self.historical_data.items():
                # Filter candles up to current date
                current_candles = [c for c in candles if c.timestamp.replace(tzinfo=None) <= current_date.replace(tzinfo=None)]
                
                if len(current_candles) < 20:  # Need enough data for indicators
                    continue
                    
                # Generate signals
                signals = self.generate_signals_for_symbol(symbol, current_candles, market_state)
                
                if signals:
                    self.logger.info(f"Generated {len(signals)} signals for {symbol} on {date}")
                    
                    # Execute signals
                    for signal in signals:
                        self.logger.info(f"Signal for {symbol}: {signal.direction} at {signal.entry_price:.2f}, stop: {signal.stop_loss:.2f}, target: {signal.take_profit:.2f}")
                        self.execute_signal(signal, current_date)
            
            # Update positions
            self.update_positions(current_date)
            
            # Collect ML training data
            if self.ml_signal_classifier:
                self.collect_ml_training_data(current_date)
                
            # Calculate equity for this date
            equity = self.calculate_equity(current_date)
            self.equity_curve.append((current_date, equity))
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Close all positions at the end of the backtest
        self.close_all_positions(end_date, reason="end_of_backtest")
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        # Add additional information
        performance['start_date'] = start_date
        performance['end_date'] = end_date
        performance['initial_capital'] = self.initial_capital
        performance['symbols'] = list(self.historical_data.keys())
        
        # Log summary
        self.logger.info(f"Backtest completed with {len(self.trades)} trades")
        self.logger.info(f"Final equity: ${performance.get('final_equity', 0):.2f}")
        self.logger.info(f"Total return: {performance.get('return_pct', 0):.2f}%")
        
        return performance

    def calculate_equity(self, current_date):
        """
        Calculate the total equity at the current date
        
        Args:
            current_date: The current date
            
        Returns:
            Total equity value
        """
        equity = self.cash
        
        for symbol, position in self.positions.items():
            # Get latest price
            symbol_candles = [c for c in self.historical_data.get(symbol, []) if c.timestamp.replace(tzinfo=None) <= current_date.replace(tzinfo=None)]
            if symbol_candles:
                latest_price = symbol_candles[-1].close
                equity += position["quantity"] * latest_price
                
        return equity
        
    def calculate_position_size(self, signal):
        """
        Calculate position size based on risk and available capital
        
        Args:
            signal: The signal to calculate position size for
            
        Returns:
            Position size in number of shares/contracts
        """
        # Default risk per trade (1% of capital)
        risk_pct = self.config.get('risk_per_trade', 0.01)
        
        # Calculate risk amount
        risk_amount = self.cash * risk_pct
        
        # Calculate risk per share
        if signal.direction == "long":
            risk_per_share = signal.entry_price - signal.stop_loss
        else:
            risk_per_share = signal.stop_loss - signal.entry_price
            
        # Ensure risk_per_share is positive and non-zero
        risk_per_share = max(risk_per_share, 0.01)
        
        # Calculate position size
        position_size = risk_amount / risk_per_share
        
        # Apply symbol weight if available
        if hasattr(signal, 'weight') and signal.weight is not None:
            position_size *= signal.weight
            
        # Limit position size to a percentage of available cash
        max_position_pct = self.config.get('max_position_size', 0.2)
        max_position_size = self.cash * max_position_pct / signal.entry_price
        
        position_size = min(position_size, max_position_size)
        
        return position_size
        
    def generate_signals_for_symbol(self, symbol, candles, market_state):
        """
        Generate trading signals for a symbol
        
        Args:
            symbol: The symbol to generate signals for
            candles: List of candles for the symbol
            market_state: Current market state
            
        Returns:
            List of signals
        """
        # Check if it's a crypto symbol
        is_crypto = symbol in self.crypto_configs
        
        # Generate signals using the strategy
        signals = self.strategy.generate_signals(symbol, candles, market_state, is_crypto)
        
        # Apply ML filtering if available
        if self.ml_signal_classifier and self.ml_signal_classifier.is_trained:
            filtered_signals = []
            for signal in signals:
                # Get quality score from ML classifier
                quality_score = self.ml_signal_classifier.predict_signal_quality(candles, signal, market_state)
                signal.quality_score = quality_score
                
                # Apply quality threshold
                min_quality = self.config.get('min_signal_quality', 0.5)
                if quality_score >= min_quality:
                    filtered_signals.append(signal)
                    self.logger.info(f"Signal passed ML filter: {symbol} {signal.direction} with quality {quality_score:.2f}")
                else:
                    self.logger.info(f"Signal rejected by ML filter: {symbol} {signal.direction} with quality {quality_score:.2f}")
                    
            return filtered_signals
        else:
            return signals

    def collect_ml_training_data(self, current_date):
        """
        Collect ML training data from closed trades
        
        Args:
            current_date: The current date
        """
        # Process any pending trades that have been closed
        for symbol in list(self.pending_trades.keys()):
            if symbol not in self.positions:
                # Trade was closed
                trade_info = self.pending_trades[symbol]
                self.collect_ml_data_for_trade(symbol, trade_info["trade"])
                del self.pending_trades[symbol]
                
    def calculate_performance(self) -> Dict:
        """Calculate performance metrics for the backtest"""
        if not self.trade_history:
            self.logger.warning("No trades to calculate performance metrics")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_profit': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_consecutive_losses': 0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'win_loss_ratio': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'annualized_return': 0.0
            }
        
        # Calculate basic statistics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit metrics
        total_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_win = total_profit / len(winning_trades) if winning_trades else 0.0
        average_loss = total_loss / len(losing_trades) if losing_trades else 0.0
        
        win_loss_ratio = average_win / abs(average_loss) if average_loss != 0 else float('inf')
        
        average_profit = sum(t.pnl for t in self.trade_history) / total_trades
        
        # Calculate expectancy
        expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)
        
        # Calculate returns
        initial_equity = self.initial_capital
        final_equity = self.cash
        for symbol, trade in self.current_positions.items():
            final_equity += trade["quantity"] * trade["entry_price"]
        
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate drawdown
        # Ensure all keys in equity_curve are datetime objects
        equity_curve_fixed = {}
        for date, equity in self.equity_curve.items():
            if isinstance(date, str):
                try:
                    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    try:
                        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        self.logger.warning(f"Could not parse date string: {date}")
                        continue
                equity_curve_fixed[date_obj] = equity
            else:
                equity_curve_fixed[date] = equity
        
        # Convert to DataFrame for easier analysis
        if equity_curve_fixed:
            equity_df = pd.DataFrame(list(equity_curve_fixed.items()), columns=['date', 'equity'])
            equity_df.set_index('date', inplace=True)
            equity_df = equity_df.sort_index()
            
            # Calculate drawdown
            equity_df['previous_peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['previous_peak'] - equity_df['equity']) / equity_df['previous_peak']
            max_drawdown = equity_df['drawdown'].max()
            
            # Calculate daily returns
            equity_df['daily_return'] = equity_df['equity'].pct_change()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = equity_df['daily_return'].mean() / equity_df['daily_return'].std() * np.sqrt(252) if len(equity_df) > 1 else 0.0
            
            # Calculate Sortino ratio (downside deviation)
            negative_returns = equity_df['daily_return'][equity_df['daily_return'] < 0]
            sortino_ratio = equity_df['daily_return'].mean() / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 0.0
        else:
            max_drawdown = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Calculate max consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for trade in self.trade_history:
            if trade.pnl <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Calculate annualized return
        if self.trades:
            start_date = min(t.entry_date.replace(tzinfo=None) for t in self.trades)
            end_date = max((t.exit_date.replace(tzinfo=None) if t.exit_date else datetime.datetime.now()) for t in self.trades)
            
            # Calculate years
            days = (end_date - start_date).days
            years = days / 365.0
            
            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Return performance metrics
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_profit': average_profit,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'average_win': average_win,
            'average_loss': average_loss,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            'annualized_return': annualized_return
        }

    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the backtest
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_pnl": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "avg_trade_duration_days": 0,
                "final_equity": 0,
                "return_pct": 0,
                "symbol_metrics": {}
            }
            
        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        total_loss = sum(abs(t.pnl) for t in self.trades if t.pnl < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate maximum drawdown
        if self.equity_curve:
            equity_values = [e[1] for e in self.equity_curve]
            max_dd = 0
            peak = equity_values[0] if equity_values else self.initial_capital
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = [(self.equity_curve[i][1] / self.equity_curve[i-1][1]) - 1 for i in range(1, len(self.equity_curve))]
            avg_return = sum(returns) / len(returns) if returns else 0
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Calculate average trade duration
        durations = [(t.exit_date - t.entry_date).total_seconds() / (60 * 60 * 24) for t in self.trades if t.exit_date and t.entry_date]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate metrics by symbol
        symbol_metrics = {}
        for symbol in set(t.symbol for t in self.trades):
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            symbol_wins = sum(1 for t in symbol_trades if t.pnl > 0)
            symbol_win_rate = symbol_wins / len(symbol_trades) if symbol_trades else 0
            symbol_pnl = sum(t.pnl for t in symbol_trades)
            
            symbol_metrics[symbol] = {
                "trades": len(symbol_trades),
                "win_rate": symbol_win_rate,
                "total_pnl": symbol_pnl,
                "avg_pnl": symbol_pnl / len(symbol_trades) if symbol_trades else 0
            }
        
        # Create final metrics dictionary
        metrics = {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": self.total_pnl,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe_ratio,
            "avg_trade_duration_days": avg_duration,
            "final_equity": self.equity_curve[-1][1] if self.equity_curve else self.cash,
            "return_pct": ((self.equity_curve[-1][1] / self.initial_capital) - 1) * 100 if self.equity_curve else 0,
            "symbol_metrics": symbol_metrics
        }
        
        # Log the metrics
        self.logger.info(f"Backtest performance metrics:")
        self.logger.info(f"  Total trades: {metrics['total_trades']}")
        self.logger.info(f"  Win rate: {metrics['win_rate']:.2%}")
        self.logger.info(f"  Profit factor: {metrics['profit_factor']:.2f}")
        self.logger.info(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        self.logger.info(f"  Max drawdown: {metrics['max_drawdown']:.2f}%")
        self.logger.info(f"  Return: {metrics['return_pct']:.2f}%")
        
        return metrics

    def initialize_alpaca_api(self):
        """Initialize the Alpaca API client"""
        try:
            # Get Alpaca API credentials from config
            alpaca_config = self.config.get('alpaca', {})
            if not alpaca_config:
                self.logger.error("Alpaca API configuration not found in config")
                return False
                
            api_key = alpaca_config.get('api_key')
            api_secret = alpaca_config.get('api_secret')
            base_url = alpaca_config.get('base_url', 'https://paper-api.alpaca.markets/v2')
            
            if not api_key or not api_secret:
                self.logger.error("Alpaca API key or secret missing")
                return False
                
            # Initialize Alpaca API client
            self.alpaca = alpaca_trade_api.REST(
                key_id=api_key,
                secret_key=api_secret,
                base_url=base_url
            )
            
            self.logger.info("Alpaca API initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {str(e)}")
            return False

    def fetch_historical_data(self, symbol, start_date, end_date):
        """Fetch historical price data from Alpaca"""
        self.logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        try:
            # Make sure we have an Alpaca API instance
            if not hasattr(self, 'alpaca'):
                self.initialize_alpaca_api()
            
            # Ensure dates are in the correct format
            if isinstance(start_date, str):
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
            # Ensure we're using 2023 data or earlier due to Alpaca free tier limitations
            current_year = datetime.datetime.now().year
            if start_date.year >= current_year or end_date.year >= current_year:
                self.logger.warning(f"Adjusting dates to use 2023 data due to Alpaca free tier limitations")
                # Adjust to use 2023 data
                year_diff = current_year - 2023
                start_date = datetime.datetime(start_date.year - year_diff, start_date.month, start_date.day)
                end_date = datetime.datetime(end_date.year - year_diff, end_date.month, end_date.day)
                self.logger.info(f"Adjusted date range: {start_date.date()} to {end_date.date()}")
            
            # Format dates for Alpaca API
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Fetch data from Alpaca
            bars = self.alpaca.get_bars(
                symbol,
                alpaca_trade_api.TimeFrame.Day,
                start=start_str,
                end=end_str,
                adjustment='raw'
            ).df
            
            if bars.empty:
                self.logger.warning(f"No data returned for {symbol} from {start_str} to {end_str}")
                return []
            
            # Convert to CandleData objects
            candles = []
            for index, row in bars.iterrows():
                candle = CandleData(
                    timestamp=index.to_pydatetime(),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                candles.append(candle)
            
            self.logger.info(f"Fetched {len(candles)} candles for {symbol}")
            return candles
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return []

    def save_results(self, performance):
        """
        Save backtest results to a JSON file
        
        Args:
            performance: Dictionary with performance metrics
            
        Returns:
            Path to the saved results file
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Convert datetime objects to strings for JSON serialization
        serializable_performance = {}
        for key, value in performance.items():
            if isinstance(value, datetime.datetime):
                serializable_performance[key] = value.isoformat()
            elif isinstance(value, dict):
                serializable_performance[key] = {}
                for k, v in value.items():
                    if isinstance(v, datetime.datetime):
                        serializable_performance[key][k] = v.isoformat()
                    else:
                        serializable_performance[key][k] = v
            else:
                serializable_performance[key] = value
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_performance, f, indent=4)
            
        self.logger.info(f"Saved backtest results to {filepath}")
        return filepath
        
    def plot_equity_curve(self, save_path=None):
        """
        Plot the equity curve
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to the saved plot file
        """
        if not self.equity_curve:
            self.logger.warning("No equity curve data to plot")
            return None
            
        # Convert equity curve to DataFrame
        dates = [e[0] for e in self.equity_curve]
        equity = [e[1] for e in self.equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved equity curve plot to {save_path}")
            
        return save_path

def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='Run Enhanced Mean Reversion strategy backtest')
    parser.add_argument('--config', type=str, default='configuration_mean_reversion_hybrid.yaml',
                        help='Path to configuration file')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4], help='Quarter to backtest')
    parser.add_argument('--year', type=int, default=datetime.datetime.now().year,
                        help='Year to backtest')
    
    args = parser.parse_args()
    
    # Determine date range
    if args.quarter:
        if args.quarter == 1:
            start_date = datetime.datetime(args.year, 1, 1)
            end_date = datetime.datetime(args.year, 3, 31)
        elif args.quarter == 2:
            start_date = datetime.datetime(args.year, 4, 1)
            end_date = datetime.datetime(args.year, 6, 30)
        elif args.quarter == 3:
            start_date = datetime.datetime(args.year, 7, 1)
            end_date = datetime.datetime(args.year, 9, 30)
        else:  # quarter 4
            start_date = datetime.datetime(args.year, 10, 1)
            end_date = datetime.datetime(args.year, 12, 31)
    else:
        # Use provided dates or default to current year
        if args.start_date:
            start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.datetime(args.year, 1, 1)
        
        if args.end_date:
            end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.datetime(args.year, 12, 31)
    
    # Load the configuration file
    try:
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Load Alpaca credentials from JSON file
        try:
            with open('alpaca_credentials.json', 'r') as cred_file:
                alpaca_credentials = json.load(cred_file)
                
            # Use paper trading credentials by default
            paper_credentials = alpaca_credentials.get('paper', {})
            
            # Add Alpaca credentials to config
            if 'alpaca' not in config:
                config['alpaca'] = {}
                
            config['alpaca']['api_key'] = paper_credentials.get('api_key')
            config['alpaca']['api_secret'] = paper_credentials.get('api_secret')
            config['alpaca']['base_url'] = paper_credentials.get('base_url')
            
            logger.info("Added Alpaca API credentials to configuration")
        except Exception as e:
            logger.error(f"Error loading Alpaca credentials: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}")
        raise
        
    # Initialize backtest
    backtest = EnhancedMeanReversionBacktest(config)
    
    # Run backtest
    performance = backtest.run_backtest(start_date, end_date)
    
    # Print results
    print("\n=== Enhanced MeanReversion Strategy Backtest Results ===")
    print(f"Period: {performance.get('start_date')} to {performance.get('end_date')}")
    print(f"Initial Capital: ${performance.get('initial_capital', 0):.2f}")
    print(f"Final Equity: ${performance.get('final_equity', 0):.2f}")
    print(f"Total Return: {performance.get('total_return', 0):.2f}%")
    print(f"Annualized Return: {performance.get('annualized_return', 0):.2f}%")
    print(f"Maximum Drawdown: {performance.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {performance.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
    print(f"Total Trades: {performance.get('total_trades', 0)}")
    
    # Exit reason statistics
    exit_reasons = performance.get('exit_reasons', {})
    if exit_reasons:
        print("\nExit Reason Statistics:")
        for reason, stats in exit_reasons.items():
            win_rate = stats['profit'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f"  {reason}: {stats['count']} trades, {win_rate:.2f}% win rate")
    
    # Save results
    results_file = backtest.save_results(performance)
    print(f"\nResults saved to: {results_file}")
    
    # Plot equity curve
    equity_curve_file = None
    if backtest.config.get('save_equity_curve', False):
        equity_curve_file = results_file.replace('.json', '_equity.png')
        backtest.plot_equity_curve(save_path=equity_curve_file)
    
    print(f"Equity curve saved to: {equity_curve_file}")


if __name__ == "__main__":
    main()
