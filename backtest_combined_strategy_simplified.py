#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Combined Strategy Simplified
-------------------------------------
This script provides a simplified version of the backtest engine for the combined strategy.
"""

import datetime as dt
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestResults:
    """Results of a backtest"""
    
    def __init__(self, initial_capital, equity_curve, trades, daily_returns=None):
        """Initialize backtest results
        
        Args:
            initial_capital (float): Initial capital
            equity_curve (pd.Series): Equity curve
            trades (list): List of trades
            daily_returns (pd.Series, optional): Daily returns. Defaults to None.
        """
        self.initial_capital = initial_capital
        self.equity_curve = equity_curve
        self.trades = trades
        
        # Calculate final equity
        self.final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        
        # Calculate total return
        self.total_return_pct = ((self.final_equity / initial_capital) - 1) * 100
        
        # Calculate daily returns if not provided
        if daily_returns is None and len(equity_curve) > 1:
            self.daily_returns = equity_curve.pct_change().dropna()
        else:
            self.daily_returns = daily_returns
        
        # Calculate drawdowns
        self.drawdowns = self._calculate_drawdowns()
        
        # Calculate monthly returns
        if self.daily_returns is not None and len(self.daily_returns) > 0:
            self.monthly_returns = self.daily_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
        else:
            self.monthly_returns = None
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Process trade outcomes
        if trades:
            self.trade_outcomes = pd.DataFrame(trades)
        else:
            self.trade_outcomes = None
    
    def _calculate_drawdowns(self) -> pd.Series:
        """Calculate drawdowns
        
        Returns:
            pd.Series: Drawdowns
        """
        # Calculate running maximum
        running_max = self.equity_curve.cummax()
        
        # Calculate drawdowns
        drawdowns = (self.equity_curve - running_max) / running_max
        
        return drawdowns
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        # Calculate max drawdown
        self.max_drawdown_pct = abs(self.drawdowns.min() * 100) if len(self.drawdowns) > 0 else 0
        
        # Calculate Sharpe ratio
        if self.daily_returns is not None and len(self.daily_returns) > 0:
            mean_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            self.sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            self.sharpe_ratio = 0
        
        # Calculate annualized return
        if self.daily_returns is not None and len(self.daily_returns) > 0:
            days = len(self.daily_returns)
            self.annualized_return_pct = ((1 + self.total_return_pct / 100) ** (252 / days) - 1) * 100
        else:
            self.annualized_return_pct = 0
        
        # Calculate win rate and profit factor
        if self.trades:
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]
            
            self.win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
            
            gross_profit = sum([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades])) if losing_trades else 0
            
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            self.win_rate = 0
            self.profit_factor = 0

class BacktestCombinedStrategy:
    """Simplified backtest engine for the combined strategy"""
    
    def __init__(self, config):
        """Initialize the backtest engine
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Initialize parameters
        self.initial_capital = config.get('general', {}).get('initial_capital', 100000)
        self.max_positions = config.get('general', {}).get('max_positions', 5)
        self.max_risk_per_trade = config.get('general', {}).get('max_risk_per_trade', 0.02)
        
        # Initialize backtest state
        self.reset()
        
        logger.info(f"Initialized backtest engine with initial capital: ${self.initial_capital}")
    
    def reset(self):
        """Reset the backtest state"""
        self.current_equity = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = {}
        self.daily_returns = {}
    
    def set_initial_capital(self, initial_capital):
        """Set initial capital
        
        Args:
            initial_capital (float): Initial capital
        """
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.cash = initial_capital
    
    def set_max_positions(self, max_positions):
        """Set maximum number of positions
        
        Args:
            max_positions (int): Maximum number of positions
        """
        self.max_positions = max_positions
    
    def process_signals_for_date(self, signals, date):
        """Process signals for a given date
        
        Args:
            signals (list): List of signals
            date (datetime): Date to process signals for
        """
        # Update existing positions
        self._update_positions(date)
        
        # Check if we can open new positions
        available_positions = self.max_positions - len(self.positions)
        
        if available_positions > 0 and signals:
            logger.info(f"Processing {len(signals)} signals for {date}, available positions: {available_positions}")
            
            # Sort signals by weight (descending)
            sorted_signals = sorted(signals, key=lambda x: x.get('weight', 0), reverse=True)
            
            # Take top N signals
            top_signals = sorted_signals[:available_positions]
            logger.info(f"Selected top {len(top_signals)} signals for {date}")
            
            # Process each signal
            positions_opened = 0
            for signal in top_signals:
                # Skip signals for symbols we already have positions in
                symbol = signal.get('symbol')
                if symbol in self.positions:
                    logger.info(f"Already have a position for {symbol}, skipping signal")
                    continue
                    
                # Process the signal
                result = self._process_signal(signal, date)
                if result:
                    positions_opened += 1
                
                # Check if we've reached our limit
                if positions_opened >= available_positions:
                    break
                    
            logger.info(f"Opened {positions_opened} new positions for {date}")
        elif not signals:
            logger.info(f"No signals to process for {date}")
        else:
            logger.info(f"No available positions for {date}, current positions: {len(self.positions)}")
        
        # Update equity curve
        self._update_equity_curve(date)
    
    def _process_signal(self, signal, date):
        """Process a single signal
        
        Args:
            signal (dict): Signal dictionary
            date (datetime): Date to process signal for
        
        Returns:
            bool: Whether the signal was processed successfully
        """
        symbol = signal.get('symbol')
        direction = signal.get('direction', 'LONG')
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        # Validate signal data
        if not symbol:
            logger.warning(f"Signal missing symbol: {signal}")
            return False
            
        if not entry_price:
            logger.warning(f"Signal missing entry_price for {symbol}: {signal}")
            return False
            
        if not stop_loss:
            logger.warning(f"Signal missing stop_loss for {symbol}: {signal}")
            return False
            
        if not take_profit:
            # If take_profit is missing, calculate it based on the entry price and stop loss
            if direction == 'LONG':
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2)  # 2:1 reward-to-risk ratio
            else:  # SHORT
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2)  # 2:1 reward-to-risk ratio
            logger.info(f"Calculated take_profit for {symbol}: {take_profit}")
        
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            logger.info(f"Already have a position for {symbol}, skipping signal")
            return False
        
        # Calculate position size
        position_size = self._calculate_position_size(entry_price, stop_loss)
        
        # Check if we have enough cash
        if position_size * entry_price > self.cash:
            logger.warning(f"Not enough cash to open position for {symbol}, needed: ${position_size * entry_price:.2f}, available: ${self.cash:.2f}")
            # Try to reduce position size to fit available cash
            max_possible_size = int(self.cash / entry_price * 0.95)
            if max_possible_size >= 1:
                position_size = max_possible_size
                logger.info(f"Reduced position size for {symbol} to {position_size} shares")
            else:
                logger.warning(f"Cannot open position for {symbol}, insufficient cash")
                return False
        
        # Open position
        self.positions[symbol] = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_date': date,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'current_price': entry_price,
            'pnl': 0
        }
        
        # Update cash
        self.cash -= position_size * entry_price
        
        logger.info(f"Opened {direction} position for {symbol} at {entry_price} with size {position_size}, stop_loss: {stop_loss}, take_profit: {take_profit}")
        
        return True
    
    def _calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            
        Returns:
            float: Position size
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk per share: {risk_per_share}")
            return 0
        
        # Calculate risk amount
        risk_amount = self.cash * self.max_risk_per_trade
        
        # Calculate position size
        position_size = risk_amount / risk_per_share
        
        # Limit position size to available cash
        max_shares = int(self.cash / entry_price * 0.95)  # Use 95% of cash at most
        position_size = min(int(position_size), max_shares)
        
        # Ensure at least 1 share
        position_size = max(1, position_size)
        
        logger.info(f"Calculated position size: {position_size} shares at ${entry_price:.2f}, " +
                   f"risk per share: ${risk_per_share:.2f}, risk amount: ${risk_amount:.2f}")
        
        return position_size
    
    def _update_positions(self, date):
        """Update positions for a given date
        
        Args:
            date (datetime): Date to update positions for
        """
        # Check for positions to close
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            # In a real backtest, we would fetch the current price from historical data
            # For now, let's simulate some price movement
            entry_price = position.get('entry_price', 0)
            days_held = (date - position.get('entry_date')).days
            
            # Simulate price movement based on direction
            if position['direction'] == 'LONG':
                # Simulate a price that moves toward take profit (optimistic)
                take_profit = position.get('take_profit', entry_price * 1.05)
                stop_loss = position.get('stop_loss', entry_price * 0.95)
                price_range = take_profit - entry_price
                current_price = entry_price + (price_range * min(days_held / 10, 1))
            else:  # SHORT
                # Simulate a price that moves toward take profit (optimistic)
                take_profit = position.get('take_profit', entry_price * 0.95)
                stop_loss = position.get('stop_loss', entry_price * 1.05)
                price_range = entry_price - take_profit
                current_price = entry_price - (price_range * min(days_held / 10, 1))
            
            # Update current price
            position['current_price'] = current_price
            
            # Calculate P&L
            if position['direction'] == 'LONG':
                position['pnl'] = (current_price - position['entry_price']) * position['size']
            else:  # SHORT
                position['pnl'] = (position['entry_price'] - current_price) * position['size']
            
            # Check for stop loss or take profit
            if position['direction'] == 'LONG':
                if current_price <= position['stop_loss']:
                    # Stop loss hit
                    symbols_to_close.append((symbol, current_price, 'stop_loss'))
                elif current_price >= position['take_profit']:
                    # Take profit hit
                    symbols_to_close.append((symbol, current_price, 'take_profit'))
            else:  # SHORT
                if current_price >= position['stop_loss']:
                    # Stop loss hit
                    symbols_to_close.append((symbol, current_price, 'stop_loss'))
                elif current_price <= position['take_profit']:
                    # Take profit hit
                    symbols_to_close.append((symbol, current_price, 'take_profit'))
        
        # Close positions
        for symbol, price, reason in symbols_to_close:
            self._close_position(symbol, price, date, reason)
    
    def _close_position(self, symbol, price, date, reason):
        """Close a position
        
        Args:
            symbol (str): Symbol to close position for
            price (float): Exit price
            date (datetime): Exit date
            reason (str): Reason for closing position
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position['direction'] == 'LONG':
            pnl = (price - position['entry_price']) * position['size']
            pnl_pct = (price / position['entry_price'] - 1) * 100
        else:  # SHORT
            pnl = (position['entry_price'] - price) * position['size']
            pnl_pct = (position['entry_price'] / price - 1) * 100
        
        # Update cash
        self.cash += position['size'] * price
        
        # Record trade
        trade = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'entry_date': position['entry_date'],
            'exit_price': price,
            'exit_date': date,
            'size': position['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {position['direction']} position for {symbol} at {price} with P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    def _update_equity_curve(self, date):
        """Update equity curve for a given date
        
        Args:
            date (datetime): Date to update equity curve for
        """
        # Calculate current equity
        position_value = 0
        for symbol, pos in self.positions.items():
            current_price = pos.get('current_price', pos.get('entry_price', 0))
            position_value += pos.get('size', 0) * current_price
        
        self.current_equity = self.cash + position_value
        
        # Update equity curve
        self.equity_curve[date] = self.current_equity
        
        # Calculate daily return
        dates = sorted(self.equity_curve.keys())
        if len(dates) > 1:
            prev_date = dates[-2]
            prev_equity = self.equity_curve[prev_date]
            if prev_equity > 0:
                daily_return = (self.current_equity / prev_equity) - 1
                self.daily_returns[date] = daily_return
                
        # Log equity update
        if len(self.positions) > 0:
            logger.info(f"Updated equity for {date}: ${self.current_equity:.2f}, Cash: ${self.cash:.2f}, Positions: {len(self.positions)}")
    
    def finalize(self):
        """Finalize the backtest and return results
        
        Returns:
            BacktestResults: Backtest results
        """
        # Close all positions
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            self._close_position(symbol, position['current_price'], max(self.equity_curve.keys()), 'end_of_backtest')
        
        # Convert equity curve and daily returns to pandas Series
        equity_curve = pd.Series(self.equity_curve)
        daily_returns = pd.Series(self.daily_returns)
        
        # Create backtest results
        results = BacktestResults(
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            trades=self.trades,
            daily_returns=daily_returns
        )
        
        return results
