#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Portfolio and Position Classes
---------------------------------
This script fixes the Portfolio and Position classes to properly handle
equity calculations and position closing.
"""

import os
import sys
import logging
import datetime as dt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Position:
    """
    Enhanced Position class with proper handling of long and short positions
    """
    
    def __init__(self, symbol, entry_price, entry_time, position_size, direction='long', stop_loss=None, take_profit=None):
        """Initialize a position"""
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_size = position_size
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_time = None
        self.profit_loss = 0
        self.status = 'open'
        self.exit_reason = None
        self.current_price = entry_price  # Track current price for equity calculation
    
    def close_position(self, exit_price, exit_time, reason="manual"):
        """Close the position and calculate profit/loss"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = 'closed'
        self.exit_reason = reason
        
        # Calculate profit/loss based on direction
        if self.direction == 'long':
            self.profit_loss = (exit_price - self.entry_price) * self.position_size
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.position_size
        
        return self.profit_loss
    
    def update_current_price(self, current_price):
        """Update the current price of the position"""
        self.current_price = current_price
    
    def get_current_value(self):
        """Get the current value of the position based on direction"""
        if self.direction == 'long':
            return self.current_price * self.position_size
        else:  # short
            # For short positions, the value increases as price decreases
            return (2 * self.entry_price - self.current_price) * self.position_size
    
    def get_unrealized_pnl(self):
        """Get the unrealized profit/loss of the position"""
        if self.direction == 'long':
            return (self.current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - self.current_price) * self.position_size

class Portfolio:
    """
    Enhanced Portfolio class with proper handling of equity calculations
    """
    
    def __init__(self, initial_capital=100000):
        """Initialize the portfolio"""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = []  # (timestamp, equity)
        self.max_positions = 5
        self.max_position_value = 0.2  # Max 20% of portfolio in one position
    
    def reset(self):
        """Reset the portfolio to initial state"""
        self.cash = self.initial_capital
        self.open_positions = {}
        self.closed_positions = []
        self.equity_curve = []
    
    def open_position(self, symbol, entry_price, entry_time, position_size, direction='long', stop_loss=None, take_profit=None):
        """Open a new position"""
        # Check if we already have a position for this symbol
        if symbol in self.open_positions:
            logger.warning(f"Already have an open position for {symbol}")
            return False
        
        # Check if we have reached the maximum number of positions
        if len(self.open_positions) >= self.max_positions:
            logger.warning(f"Maximum number of positions reached ({self.max_positions})")
            return False
        
        # Calculate position value
        position_value = entry_price * position_size
        
        # Check if we have enough cash (for long positions)
        if direction == 'long' and position_value > self.cash:
            logger.warning(f"Not enough cash to open long position for {symbol}")
            return False
        
        # Check if position value exceeds maximum allowed
        max_allowed = self.get_equity() * self.max_position_value
        if position_value > max_allowed:
            logger.warning(f"Position value exceeds maximum allowed ({position_value} > {max_allowed})")
            return False
        
        # Create the position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update cash based on position direction
        if direction == 'long':
            self.cash -= position_value
        else:  # short
            self.cash += position_value
        
        # Add to open positions
        self.open_positions[symbol] = position
        
        return True
    
    def close_position(self, symbol, exit_price, exit_time, reason="manual"):
        """Close an open position"""
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol}")
            return False
        
        position = self.open_positions[symbol]
        profit_loss = position.close_position(exit_price, exit_time, reason)
        
        # Update cash based on position direction
        if position.direction == 'long':
            self.cash += exit_price * position.position_size
        else:  # short
            self.cash -= exit_price * position.position_size
            # Add the profit/loss for short positions
            self.cash += profit_loss
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[symbol]
        
        return True
    
    def get_equity(self):
        """Calculate the total portfolio value (cash + positions)"""
        equity = self.cash
        
        # Add value of open positions
        for symbol, position in self.open_positions.items():
            equity += position.get_current_value()
        
        return equity
    
    def update_equity_curve(self, timestamp):
        """Update the equity curve with current equity"""
        equity = self.get_equity()
        self.equity_curve.append((timestamp, equity))
    
    def get_win_rate(self):
        """Calculate the win rate of closed positions"""
        if not self.closed_positions:
            return 0.0
        
        winners = sum(1 for p in self.closed_positions if p.profit_loss > 0)
        return winners / len(self.closed_positions)
    
    def get_profit_factor(self):
        """Calculate the profit factor (gross profit / gross loss)"""
        gross_profit = sum(p.profit_loss for p in self.closed_positions if p.profit_loss > 0)
        gross_loss = sum(abs(p.profit_loss) for p in self.closed_positions if p.profit_loss < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_max_drawdown(self):
        """Calculate the maximum drawdown from the equity curve"""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [equity for _, equity in self.equity_curve]
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

def apply_fixes():
    """Apply fixes to the existing codebase"""
    # Check if the file exists
    target_file = 'test_optimized_mean_reversion_alpaca.py'
    if not os.path.exists(target_file):
        logger.error(f"Target file {target_file} not found")
        return False
    
    # Read the file
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Check if the file already contains our fixes
    if 'direction=' in content and 'get_current_value' in content:
        logger.info("Fixes already applied")
        return True
    
    # Create backup
    backup_file = f"{target_file}.bak"
    with open(backup_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Created backup of {target_file} at {backup_file}")
    
    # Replace Position class
    position_class = """
class Position:
    \"\"\"
    Enhanced Position class with proper handling of long and short positions
    \"\"\"
    
    def __init__(self, symbol, entry_price, entry_time, position_size, direction='long', stop_loss=None, take_profit=None):
        \"\"\"Initialize a position\"\"\"
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_size = position_size
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_price = None
        self.exit_time = None
        self.profit_loss = 0
        self.status = 'open'
        self.exit_reason = None
        self.current_price = entry_price  # Track current price for equity calculation
    
    def close_position(self, exit_price, exit_time, reason="manual"):
        \"\"\"Close the position and calculate profit/loss\"\"\"
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = 'closed'
        self.exit_reason = reason
        
        # Calculate profit/loss based on direction
        if self.direction == 'long':
            self.profit_loss = (exit_price - self.entry_price) * self.position_size
        else:  # short
            self.profit_loss = (self.entry_price - exit_price) * self.position_size
        
        return self.profit_loss
    
    def update_current_price(self, current_price):
        \"\"\"Update the current price of the position\"\"\"
        self.current_price = current_price
    
    def get_current_value(self):
        \"\"\"Get the current value of the position based on direction\"\"\"
        if self.direction == 'long':
            return self.current_price * self.position_size
        else:  # short
            # For short positions, the value increases as price decreases
            return (2 * self.entry_price - self.current_price) * self.position_size
    
    def get_unrealized_pnl(self):
        \"\"\"Get the unrealized profit/loss of the position\"\"\"
        if self.direction == 'long':
            return (self.current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - self.current_price) * self.position_size
"""
    
    # Replace Portfolio class
    portfolio_class = """
class Portfolio:
    \"\"\"
    Enhanced Portfolio class with proper handling of equity calculations
    \"\"\"
    
    def __init__(self, initial_capital=100000):
        \"\"\"Initialize the portfolio\"\"\"
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions = {}  # symbol -> Position
        self.closed_positions = []
        self.equity_curve = []  # (timestamp, equity)
        self.max_positions = 5
        self.max_position_value = 0.2  # Max 20% of portfolio in one position
    
    def reset(self):
        \"\"\"Reset the portfolio to initial state\"\"\"
        self.cash = self.initial_capital
        self.open_positions = {}
        self.closed_positions = []
        self.equity_curve = []
    
    def open_position(self, symbol, entry_price, entry_time, position_size, direction='long', stop_loss=None, take_profit=None):
        \"\"\"Open a new position\"\"\"
        # Check if we already have a position for this symbol
        if symbol in self.open_positions:
            logger.warning(f"Already have an open position for {symbol}")
            return False
        
        # Check if we have reached the maximum number of positions
        if len(self.open_positions) >= self.max_positions:
            logger.warning(f"Maximum number of positions reached ({self.max_positions})")
            return False
        
        # Calculate position value
        position_value = entry_price * position_size
        
        # Check if we have enough cash (for long positions)
        if direction == 'long' and position_value > self.cash:
            logger.warning(f"Not enough cash to open long position for {symbol}")
            return False
        
        # Check if position value exceeds maximum allowed
        max_allowed = self.get_equity() * self.max_position_value
        if position_value > max_allowed:
            logger.warning(f"Position value exceeds maximum allowed ({position_value} > {max_allowed})")
            return False
        
        # Create the position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update cash based on position direction
        if direction == 'long':
            self.cash -= position_value
        else:  # short
            self.cash += position_value
        
        # Add to open positions
        self.open_positions[symbol] = position
        
        return True
    
    def close_position(self, symbol, exit_price, exit_time, reason="manual"):
        \"\"\"Close an open position\"\"\"
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol}")
            return False
        
        position = self.open_positions[symbol]
        profit_loss = position.close_position(exit_price, exit_time, reason)
        
        # Update cash based on position direction
        if position.direction == 'long':
            self.cash += exit_price * position.position_size
        else:  # short
            self.cash -= exit_price * position.position_size
            # Add the profit/loss for short positions
            self.cash += profit_loss
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[symbol]
        
        return True
    
    def get_equity(self):
        \"\"\"Calculate the total portfolio value (cash + positions)\"\"\"
        equity = self.cash
        
        # Add value of open positions
        for symbol, position in self.open_positions.items():
            equity += position.get_current_value()
        
        return equity
    
    def update_equity_curve(self, timestamp):
        \"\"\"Update the equity curve with current equity\"\"\"
        equity = self.get_equity()
        self.equity_curve.append((timestamp, equity))
    
    def get_win_rate(self):
        \"\"\"Calculate the win rate of closed positions\"\"\"
        if not self.closed_positions:
            return 0.0
        
        winners = sum(1 for p in self.closed_positions if p.profit_loss > 0)
        return winners / len(self.closed_positions)
    
    def get_profit_factor(self):
        \"\"\"Calculate the profit factor (gross profit / gross loss)\"\"\"
        gross_profit = sum(p.profit_loss for p in self.closed_positions if p.profit_loss > 0)
        gross_loss = sum(abs(p.profit_loss) for p in self.closed_positions if p.profit_loss < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_max_drawdown(self):
        \"\"\"Calculate the maximum drawdown from the equity curve\"\"\"
        if not self.equity_curve:
            return 0.0
        
        equity_values = [equity for _, equity in self.equity_curve]
        max_drawdown = 0.0
        peak = equity_values[0]
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
"""
    
    # Find and replace the Position class
    import re
    position_pattern = r'class Position:.*?def get_unrealized_pnl\(.*?\).*?return.*?\n'
    if not re.search(position_pattern, content, re.DOTALL):
        position_pattern = r'class Position:.*?def close_position\(.*?\).*?return.*?\n'
    
    # Find and replace the Portfolio class
    portfolio_pattern = r'class Portfolio:.*?def get_max_drawdown\(.*?\).*?return.*?\n'
    if not re.search(portfolio_pattern, content, re.DOTALL):
        portfolio_pattern = r'class Portfolio:.*?def get_profit_factor\(.*?\).*?return.*?\n'
    
    # Replace classes
    new_content = content
    if re.search(position_pattern, content, re.DOTALL):
        new_content = re.sub(position_pattern, position_class, new_content, flags=re.DOTALL)
    else:
        logger.warning("Could not find Position class to replace")
    
    if re.search(portfolio_pattern, content, re.DOTALL):
        new_content = re.sub(portfolio_pattern, portfolio_class, new_content, flags=re.DOTALL)
    else:
        logger.warning("Could not find Portfolio class to replace")
    
    # Write the updated file
    with open(target_file, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Applied fixes to {target_file}")
    
    # Update stop_loss_atr in test_combined_mean_reversion.py
    combined_file = 'test_combined_mean_reversion.py'
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            combined_content = f.read()
        
        # Create backup
        combined_backup = f"{combined_file}.bak"
        with open(combined_backup, 'w') as f:
            f.write(combined_content)
        
        # Update stop_loss_atr
        updated_content = re.sub(
            r"stop_loss_atr': 1.5",
            "stop_loss_atr': 1.8",
            combined_content
        )
        
        # Write the updated file
        with open(combined_file, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Updated stop_loss_atr in {combined_file}")
    
    return True

if __name__ == "__main__":
    apply_fixes()
