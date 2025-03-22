#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Management for Backtesting
-----------------------------------
This module implements a portfolio class for tracking trades and performance metrics.
"""

import logging
import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Position:
    """Class to represent a trading position"""
    
    def __init__(self, symbol, entry_price, entry_time, position_size, direction, stop_loss=None, take_profit=None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_size = position_size
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_time = None
        self.profit_loss = None
        self.exit_reason = None
    
    def close(self, exit_price, exit_time, reason="manual"):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate P/L
        if self.direction == 'long':
            self.profit_loss = (exit_price - self.entry_price) * self.position_size
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.position_size
        
        return self.profit_loss
    
    def get_current_value(self, current_price):
        """Get current value of the position"""
        if self.direction == 'long':
            return self.position_size * current_price
        else:  # short
            return self.position_size * (2 * self.entry_price - current_price)
    
    def get_unrealized_pl(self, current_price):
        """Get unrealized profit/loss"""
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - current_price) * self.position_size
    
    def to_dict(self):
        """Convert position to dictionary for reporting"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'position_size': self.position_size,
            'profit_loss': self.profit_loss,
            'reason': self.exit_reason
        }

class Portfolio:
    """Class to manage trading portfolio and positions"""
    
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset the portfolio to initial state"""
        self.cash = self.initial_capital
        self.open_positions = {}
        self.closed_positions = []
        self.equity_curve = []
        self.logger.info(f"Portfolio reset to initial capital: ${self.initial_capital:.2f}")
    
    def open_position(self, symbol, entry_price, entry_time, position_size, direction, stop_loss=None, take_profit=None):
        """Open a new position"""
        # Check if we already have a position for this symbol
        if symbol in self.open_positions:
            self.logger.warning(f"Already have an open position for {symbol}, cannot open another")
            return False
        
        # Check if we have enough cash
        position_cost = position_size * entry_price
        if position_cost > self.cash:
            self.logger.warning(f"Not enough cash to open position for {symbol}: need ${position_cost:.2f}, have ${self.cash:.2f}")
            return False
        
        # Open the position
        position = Position(symbol, entry_price, entry_time, position_size, direction, stop_loss, take_profit)
        self.open_positions[symbol] = position
        
        # Update cash
        self.cash -= position_cost
        
        self.logger.info(f"Opened {direction} position for {symbol}: {position_size} shares at ${entry_price:.2f}")
        return True
    
    def close_position(self, symbol, exit_price, exit_time, reason="manual"):
        """Close an open position"""
        if symbol not in self.open_positions:
            self.logger.warning(f"No open position for {symbol}")
            return False
        
        # Get the position
        position = self.open_positions[symbol]
        
        # Close the position
        profit_loss = position.close(exit_price, exit_time, reason)
        
        # Update cash
        self.cash += (position.position_size * exit_price)
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[symbol]
        
        self.logger.info(f"Closed {position.direction} position for {symbol}: {position.position_size} shares at ${exit_price:.2f}, P/L: ${profit_loss:.2f}")
        return True
    
    def update_equity_curve(self, timestamp):
        """Update the equity curve with current portfolio value"""
        equity = self.get_equity()
        self.equity_curve.append((timestamp, equity))
    
    def get_equity(self):
        """Calculate total portfolio value (cash + open positions)"""
        return self.cash
    
    def get_win_rate(self):
        """Calculate win rate from closed positions"""
        if not self.closed_positions:
            return 0.0
        
        winners = sum(1 for p in self.closed_positions if p.profit_loss > 0)
        return winners / len(self.closed_positions)
    
    def get_profit_factor(self):
        """Calculate profit factor (gross profits / gross losses)"""
        gross_profit = sum(p.profit_loss for p in self.closed_positions if p.profit_loss > 0)
        gross_loss = sum(abs(p.profit_loss) for p in self.closed_positions if p.profit_loss < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_max_drawdown(self):
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [eq for _, eq in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_performance_metrics(self):
        """Get all performance metrics"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.cash,
            'return': (self.cash / self.initial_capital) - 1,
            'win_rate': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'max_drawdown': self.get_max_drawdown(),
            'total_trades': len(self.closed_positions),
            'trades': [p.to_dict() for p in self.closed_positions],
            'equity_curve': self.equity_curve
        }
