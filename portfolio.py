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
    
    def __init__(self, symbol, entry_price, entry_time, shares, direction, stop_loss=None, take_profit=None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.shares = shares
        self.direction = direction  # 'LONG' or 'SHORT'
        self.exit_price = None
        self.exit_time = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.pnl = 0
        self.pnl_pct = 0
        self.status = "OPEN"
        self.exit_reason = None
        self.tier = None  # Added tier attribute
    
    def close(self, exit_price, exit_time, reason="manual"):
        """Close the position and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "CLOSED"
        self.exit_reason = reason
        
        # Calculate P&L
        if self.direction == "LONG":
            self.pnl = (exit_price - self.entry_price) * self.shares
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.shares
            self.pnl_pct = (self.entry_price / exit_price - 1) * 100
        
        return self.pnl
    
    def get_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L at current price"""
        if self.direction == "LONG":
            unrealized_pnl = (current_price - self.entry_price) * self.shares
            unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100
        else:  # SHORT
            unrealized_pnl = (self.entry_price - current_price) * self.shares
            unrealized_pnl_pct = (self.entry_price / current_price - 1) * 100
        
        return unrealized_pnl, unrealized_pnl_pct

class Portfolio:
    """Class to manage trading portfolio and positions"""
    
    def __init__(self, initial_capital=10000, cash_allocation=0.9, max_positions=10, position_size=0.1, stop_loss=0.02, take_profit=0.05):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = {}
        self.trade_history = []  # Added trade history
        self.logger = logging.getLogger(__name__)
        
        # Portfolio parameters
        self.cash_allocation = cash_allocation  # Percentage of capital to allocate to trades
        self.max_positions = max_positions  # Maximum number of open positions
        self.position_size = position_size  # Default position size as percentage of capital
        self.stop_loss = stop_loss  # Default stop loss percentage
        self.take_profit = take_profit  # Default take profit percentage
        
        # Signal tier thresholds
        self.tier1_threshold = 0.8  # Threshold for Tier 1 signals (highest quality)
        self.tier2_threshold = 0.6  # Threshold for Tier 2 signals
        self.tier3_threshold = 0.4  # Threshold for Tier 3 signals (lowest quality)
    
    def reset(self):
        """Reset the portfolio to initial state"""
        self.cash = self.initial_capital
        self.open_positions = {}
        self.closed_positions = []
        self.equity_curve = {}
        self.trade_history = []  # Reset trade history
    
    def open_position(self, symbol, entry_price, entry_time, shares, direction="LONG", tier=1):
        """
        Open a new position.
        
        Args:
            symbol (str): Symbol to trade
            entry_price (float): Entry price
            entry_time (datetime): Entry time
            shares (int): Number of shares
            direction (str): Trade direction (LONG or SHORT)
            tier (int): Signal tier (1, 2, or 3)
            
        Returns:
            bool: True if position was opened successfully, False otherwise
        """
        # Calculate position value
        position_value = shares * entry_price
        
        # Check if we have enough cash
        if position_value > self.cash:
            self.logger.error(f"Not enough cash to open position for {symbol}. Required: ${position_value:.2f}, Available: ${self.cash:.2f}")
            return False
        
        # Check if we've reached the maximum number of positions
        if len(self.open_positions) >= self.max_positions:
            self.logger.warning(f"Maximum number of positions reached ({self.max_positions})")
            return False
        
        # Create position
        position = Position(symbol, entry_price, entry_time, shares, direction, tier=tier)
        
        # Add position to open positions
        self.open_positions[symbol] = position
        
        # Deduct position value from cash
        self.cash -= position_value
        
        # Log position opening
        self.logger.info(f"Opened {direction.lower()} position for {symbol} at ${entry_price:.2f} with {shares} shares (${position_value:.2f})")
        
        # Add to trade history
        self.trade_history.append({
            'time': entry_time,
            'symbol': symbol,
            'action': 'OPEN',
            'direction': direction,
            'price': entry_price,
            'shares': shares,
            'value': position_value,
            'tier': tier
        })
        
        return True
    
    def close_position(self, symbol, exit_price, exit_time, reason="manual"):
        """
        Close an open position.
        
        Args:
            symbol (str): Symbol to close
            exit_price (float): Exit price
            exit_time (datetime): Exit time
            reason (str): Reason for closing (manual, stop_loss, take_profit)
            
        Returns:
            bool: True if position was closed successfully, False otherwise
        """
        if symbol not in self.open_positions:
            self.logger.warning(f"No open position for {symbol}")
            return False
        
        position = self.open_positions[symbol]
        
        # Calculate P&L
        position_value = position.shares * exit_price
        entry_value = position.shares * position.entry_price
        
        if position.direction == "LONG":
            pnl = position_value - entry_value
        else:  # SHORT
            pnl = entry_value - position_value
        
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
        
        # Update position with exit information
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.exit_reason = reason
        position.pnl = pnl
        position.pnl_pct = pnl_pct
        
        # Log the trade
        self.logger.info(f"Closed {position.direction.lower()} position for {symbol} at {exit_price:.2f} with P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        # Update cash - add the position value at exit
        self.cash += position_value
        
        # Move to closed positions
        self.closed_positions.append(position)
        
        # Add to trade history
        self.trade_history.append({
            'time': exit_time,
            'symbol': symbol,
            'action': 'CLOSE',
            'direction': position.direction,
            'price': exit_price,
            'shares': position.shares,
            'value': position_value,
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        return True
    
    def update_position_value(self, symbol, current_price):
        """
        Update the value of a position.
        
        Args:
            symbol (str): Symbol of the position to update
            current_price (float): Current price of the symbol
            
        Returns:
            bool: True if position was updated, False otherwise
        """
        if symbol not in self.open_positions:
            return False
            
        position = self.open_positions[symbol]
        position.current_price = current_price
        position.current_value = position.shares * current_price
        
        # Calculate unrealized P&L
        if position.direction == 'LONG':
            position.unrealized_pnl = (current_price - position.entry_price) * position.shares
            position.unrealized_pnl_pct = ((current_price / position.entry_price) - 1) * 100
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.shares
            position.unrealized_pnl_pct = ((position.entry_price / current_price) - 1) * 100
        
        # Increment days held counter if this is a new day
        if 'last_update_date' not in position.__dict__ or position.last_update_date.date() != datetime.datetime.now().date():
            position.days_held = position.days_held + 1 if 'days_held' in position.__dict__ else 1
            position.last_update_date = datetime.datetime.now()
            
        return True
        
    def get_equity(self, price_data=None):
        """
        Calculate the current equity value.
        
        Args:
            price_data (dict, optional): Dictionary of price data by symbol
            
        Returns:
            float: Current equity value
        """
        equity = self.cash
        
        # If no positions, just return cash
        if not self.open_positions:
            return equity
            
        # If no price data provided, use current values stored in positions
        if price_data is None:
            for symbol, position in self.open_positions.items():
                equity += position.current_value
            return equity
        
        # Calculate equity using provided price data
        for symbol, position in self.open_positions.items():
            # Skip if no price data for this symbol
            if symbol not in price_data:
                continue
                
            # Get current price - handle both dictionary and direct value formats
            if isinstance(price_data[symbol], dict) and 'close' in price_data[symbol]:
                current_price = price_data[symbol]['close']
            else:
                # If price_data just contains prices directly
                current_price = price_data[symbol]
            
            # Calculate position value
            position_value = position.shares * current_price
            
            # Add to equity
            equity += position_value
            
        return equity
        
    def update_equity_curve(self, date, equity=None):
        """
        Update the equity curve.
        
        Args:
            date (datetime): Current date
            equity (float, optional): Current equity value. If None, will be calculated.
            
        Returns:
            None
        """
        # Calculate total equity if not provided
        if equity is None:
            equity = self.get_equity()
        
        # Calculate percentage change from previous day
        prev_equity = self.initial_capital
        if len(self.equity_curve) > 0:
            prev_date = list(self.equity_curve.keys())[-1]
            prev_equity = self.equity_curve[prev_date]
        
        pct_change = 0 if prev_equity == 0 else (equity / prev_equity - 1) * 100
            
        # Update equity curve with both equity value and percentage change
        self.equity_curve[date] = {
            'equity': equity,
            'pct_change': pct_change
        }
    
    def execute_trade(self, signal, tier=3, price_data=None):
        """
        Execute a trade based on a signal.
        
        Args:
            signal (dict): Signal dictionary with symbol, direction, and score
            tier (int): Signal tier (1, 2, or 3)
            price_data (dict): Dictionary of price data for all symbols
            
        Returns:
            dict: Trade result with success status and details
        """
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', 'LONG')
        price = signal.get('price', None)
        
        # If price is not in the signal, try to get it from price_data
        if price is None or price <= 0:
            if price_data and symbol in price_data:
                price = price_data[symbol]
            else:
                self.logger.warning(f"No price available for {symbol}")
                return {
                    'success': False,
                    'symbol': symbol,
                    'message': "No price available"
                }
        
        # Check if we already have an open position for this symbol
        if symbol in self.open_positions:
            self.logger.warning(f"Already have an open position for {symbol}")
            return {
                'success': False,
                'symbol': symbol,
                'message': "Position already open"
            }
        
        # Determine position size based on signal tier and total equity
        total_equity = self.get_equity(price_data)
        self.logger.info(f"Total equity for position sizing: ${total_equity:.2f}")
        
        if 'position_size' in signal and signal['position_size'] > 0:
            shares = signal['position_size']
            position_value = shares * price
        else:
            # Determine position size based on tier
            if tier == 1:
                position_pct = 0.05  # 5% of portfolio for Tier 1 signals
            elif tier == 2:
                position_pct = 0.03  # 3% of portfolio for Tier 2 signals
            else:  # tier == 3
                position_pct = 0.02  # 2% of portfolio for Tier 3 signals
            
            # Calculate position value based on total equity
            position_value = total_equity * position_pct
            
            # Calculate shares based on position value
            shares = max(1, int(position_value / price))
            position_value = shares * price  # Recalculate based on whole shares
        
        # Ensure we don't use more than available cash
        max_position_value = self.cash * self.cash_allocation
        if position_value > max_position_value:
            old_shares = shares
            shares = max(1, int(max_position_value / price))
            position_value = shares * price
            self.logger.warning(f"Position size adjusted for {symbol} due to cash constraints. Using {shares} shares instead of {old_shares}.")
        
        # Check if we have enough cash
        if position_value > self.cash:
            self.logger.warning(f"Not enough cash to execute {direction} trade for {symbol}. Required: ${position_value:.2f}, Available: ${self.cash:.2f}")
            return {
                'success': False,
                'symbol': symbol,
                'message': f"Not enough cash (required: ${position_value:.2f}, available: ${self.cash:.2f})"
            }
        
        # Execute the trade by opening a position
        success = self.open_position(
            symbol=symbol,
            entry_price=price,
            entry_time=datetime.datetime.now(),
            shares=shares,
            direction=direction,
            tier=tier
        )
        
        if success:
            self.logger.info(f"Executed {direction} trade for {symbol} at ${price:.2f} with {shares:.0f} shares (Tier {tier})")
            return {
                'success': True,
                'symbol': symbol,
                'action': direction,
                'price': price,
                'shares': shares,
                'value': position_value,
                'tier': tier
            }
        else:
            self.logger.warning(f"Failed to open position for {symbol}")
            return {
                'success': False,
                'symbol': symbol,
                'message': "Failed to open position"
            }
    
    def update_positions(self, date, price_data):
        """
        Update all open positions with current prices and check for stop loss/take profit triggers.
        
        Args:
            date (datetime): Current date
            price_data (dict): Dictionary of current prices for all symbols
            
        Returns:
            dict: Update results with positions updated, closed, and total P&L
        """
        positions_updated = 0
        positions_closed = 0
        total_pnl = 0.0
        positions_to_close = []
        
        # Update each position
        for symbol, position in self.open_positions.items():
            # Skip if no price data available
            if symbol not in price_data:
                self.logger.warning(f"No price data for {symbol}, skipping position update")
                continue
            
            positions_updated += 1
            current_price = price_data[symbol]['close'] if isinstance(price_data[symbol], dict) else price_data[symbol]
            
            # Update position's current price and value
            old_value = position.current_value if 'current_value' in position.__dict__ else position.entry_price * position.shares
            position.current_price = current_price
            position.current_value = position.shares * current_price
            
            # Calculate unrealized P&L
            entry_value = position.entry_price * position.shares
            
            if position.direction == 'LONG':
                position.unrealized_pnl = position.current_value - entry_value
                position.unrealized_pnl_pct = ((current_price / position.entry_price) - 1) * 100
            else:  # SHORT
                position.unrealized_pnl = entry_value - position.current_value
                position.unrealized_pnl_pct = ((position.entry_price / current_price) - 1) * 100
            
            # Log the position update
            self.logger.debug(f"Updated position {symbol}: {position.shares} shares @ ${current_price:.2f} = ${position.current_value:.2f}, P&L: ${position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2f}%)")
            
            # Check if stop loss or take profit has been hit
            if position.direction == "LONG":
                if current_price <= position.entry_price * (1 - 0.05):  # 5% stop loss
                    # Stop loss hit
                    positions_to_close.append((symbol, current_price, date, "stop_loss"))
                elif current_price >= position.entry_price * (1 + 0.15):  # 15% take profit
                    # Take profit hit
                    positions_to_close.append((symbol, current_price, date, "take_profit"))
            else:  # SHORT
                if current_price >= position.entry_price * (1 + 0.05):  # 5% stop loss
                    # Stop loss hit
                    positions_to_close.append((symbol, current_price, date, "stop_loss"))
                elif current_price <= position.entry_price * (1 - 0.15):  # 15% take profit
                    # Take profit hit
                    positions_to_close.append((symbol, current_price, date, "take_profit"))
            
            # Check if position has been held for more than 3 days
            days_held = (date - position.entry_time).days
            if days_held >= 3:
                positions_to_close.append((symbol, current_price, date, "max_hold_time"))
        
        # Close positions that hit stop loss or take profit
        for symbol, price, date, reason in positions_to_close:
            if self.close_position(symbol, price, date, reason):
                positions_closed += 1
                position = self.closed_positions[-1]  # Get the position we just closed
                total_pnl += position.pnl
                self.logger.info(f"Closed position {symbol} due to {reason} at ${price:.2f}, P&L: ${position.pnl:.2f} ({position.pnl_pct:.2f}%)")
        
        return {
            'positions_updated': positions_updated,
            'positions_closed': positions_closed,
            'total_pnl': total_pnl
        }
    
    def calculate_performance(self, price_data=None):
        """
        Calculate performance metrics.
        
        Args:
            price_data (dict, optional): Dictionary of price data by symbol
            
        Returns:
            dict: Dictionary with performance metrics
        """
        # Calculate total return
        initial_value = self.initial_capital
        
        if price_data is not None:
            final_value = self.get_equity(price_data)
        else:
            # If no price data is provided, use the last equity curve value or just cash
            if self.equity_curve:
                final_value = self.equity_curve[max(self.equity_curve.keys())]
            else:
                final_value = self.cash
                
                # Add value of open positions using their current prices
                for symbol, position in self.open_positions.items():
                    if 'current_value' in position.__dict__:
                        final_value += position.current_value
        
        # Calculate return
        if initial_value > 0:
            total_return = ((final_value / initial_value) - 1) * 100
        else:
            total_return = 0
        
        # Calculate annualized return
        if self.equity_curve and len(self.equity_curve) > 1:
            days = (max(self.equity_curve.keys()) - min(self.equity_curve.keys())).days
            if days > 0:
                annualized_return = ((1 + (total_return / 100)) ** (365 / days) - 1) * 100
            else:
                annualized_return = 0
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio
        if self.equity_curve and len(self.equity_curve) > 1:
            daily_returns = []
            for date in sorted(self.equity_curve.keys())[1:]:
                prev_equity = self.equity_curve[sorted(self.equity_curve.keys())[sorted(self.equity_curve.keys()).index(date) - 1]]
                curr_equity = self.equity_curve[date]
                if prev_equity > 0:
                    daily_return = (curr_equity / prev_equity) - 1
                    daily_returns.append(daily_return)
            
            if daily_returns:
                avg_daily_return = sum(daily_returns) / len(daily_returns)
                std_daily_return = np.std(daily_returns) if len(daily_returns) > 1 else 0
                if std_daily_return > 0:
                    sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        if self.equity_curve and len(self.equity_curve) > 1:
            max_drawdown = 0
            peak = self.equity_curve[min(self.equity_curve.keys())]
            
            for date, equity in self.equity_curve.items():
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # Calculate trade statistics
        total_trades = len(self.trade_history) // 2  # Each complete trade has an open and close action
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        
        # Count closed positions with PnL
        for pos in self.closed_positions:
            if 'pnl' in pos.__dict__:
                total_pnl += pos.pnl
                if pos.pnl > 0:
                    winning_trades += 1
                elif pos.pnl < 0:
                    losing_trades += 1
        
        # Calculate win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average PnL
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(pos.pnl for pos in self.closed_positions if 'pnl' in pos.__dict__ and pos.pnl > 0)
        total_loss = abs(sum(pos.pnl for pos in self.closed_positions if 'pnl' in pos.__dict__ and pos.pnl < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Log detailed trade information
        self.logger.info(f"Performance: initial_capital={initial_value}, final_value={final_value}, return={total_return}%, trades={total_trades}, wins={winning_trades}, losses={losing_trades}")
        
        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
