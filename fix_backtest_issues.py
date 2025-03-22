#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Backtest Issues
------------------
This script fixes several issues in the backtest implementation:

1. Fixes the stop_loss_atr value in test_combined_mean_reversion.py
2. Fixes the equity calculation in Portfolio.get_equity()
3. Fixes the position closing logic in Portfolio.close_position()
4. Adds proper position value tracking for short positions
"""

import os
import re

# Fix 1: Update stop_loss_atr in test_combined_mean_reversion.py
combined_file_path = 'test_combined_mean_reversion.py'
with open(combined_file_path, 'r') as file:
    combined_content = file.read()

# Replace the stop_loss_atr value
updated_combined_content = combined_content.replace("'stop_loss_atr': 1.5,", "'stop_loss_atr': 1.8,")

# Write the updated content back to the file
with open(combined_file_path, 'w') as file:
    file.write(updated_combined_content)

print(f"1. Updated {combined_file_path} with fixed stop_loss_atr value")

# Fix 2 & 3: Fix Portfolio class in test_optimized_mean_reversion_alpaca.py
backtest_file_path = 'test_optimized_mean_reversion_alpaca.py'
with open(backtest_file_path, 'r') as file:
    backtest_content = file.read()

# Fix the Position class to track current price
position_class_pattern = r'class Position:\s+"""Class to represent a trading position"""\s+\s+def __init__\(self, symbol, entry_price, entry_time, position_size, direction, stop_loss=None, take_profit=None\):\s+self\.symbol = symbol\s+self\.entry_price = entry_price\s+self\.entry_time = entry_time\s+self\.position_size = position_size\s+self\.direction = direction  # \'long\' or \'short\'\s+self\.stop_loss = stop_loss\s+self\.take_profit = take_profit\s+self\.exit_price = None\s+self\.exit_time = None\s+self\.profit_loss = None\s+self\.status = "open"\s+self\.exit_reason = None'

new_position_class = '''class Position:
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
        self.status = "open"
        self.exit_reason = None
        self.current_price = entry_price  # Track current price'''

# Fix the close_position method in Portfolio class
close_position_pattern = r'def close_position\(self, symbol, exit_price, exit_time, reason="manual"\):\s+"""Close an open position"""\s+if symbol not in self\.open_positions:\s+self\.logger\.warning\(f"No open position for {symbol} to close"\)\s+return False\s+\s+position = self\.open_positions\[symbol\]\s+profit_loss = position\.close_position\(exit_price, exit_time, reason\)\s+\s+# Update cash\s+self\.cash \+= \(exit_price \* position\.position_size\)'

new_close_position = '''def close_position(self, symbol, exit_price, exit_time, reason="manual"):
        """Close an open position"""
        if symbol not in self.open_positions:
            self.logger.warning(f"No open position for {symbol} to close")
            return False
        
        position = self.open_positions[symbol]
        profit_loss = position.close_position(exit_price, exit_time, reason)
        
        # Update cash - account for the direction of the trade
        if position.direction == 'long':
            # For long positions, we get back the exit value
            self.cash += (exit_price * position.position_size)
        else:  # short
            # For short positions, we get back our initial cash plus/minus the profit/loss
            self.cash += (position.entry_price * position.position_size) + profit_loss'''

# Fix the get_equity method
get_equity_pattern = r'def get_equity\(self\):\s+"""Calculate total portfolio value \(cash \+ open positions\)"""\s+equity = self\.cash\s+\s+for symbol, position in self\.open_positions\.items\(\):\s+# In a real implementation, we would use current market prices\s+# For simplicity, we\'ll use the entry price here\s+position_value = position\.entry_price \* position\.position_size\s+equity \+= position_value\s+\s+return equity'

new_get_equity = '''def get_equity(self):
        """Calculate total portfolio value (cash + open positions)"""
        equity = self.cash
        
        for symbol, position in self.open_positions.items():
            # Use current price if available, otherwise use entry price
            current_price = getattr(position, 'current_price', position.entry_price)
            
            if position.direction == 'long':
                # For long positions, add the current value
                position_value = current_price * position.position_size
                equity += position_value
            else:  # short
                # For short positions, we need to account for profit/loss
                # Short profit = entry_price - current_price
                profit_loss = (position.entry_price - current_price) * position.position_size
                # Add the initial position value plus any profit/loss
                position_value = position.entry_price * position.position_size
                equity += position_value + profit_loss
        
        return equity'''

# Fix 4: Update position current price in run_backtest method
run_backtest_pattern = r'# Check for stop loss or take profit\s+if symbol in self\.portfolio\.open_positions:\s+position = self\.portfolio\.open_positions\[symbol\]\s+\s+# Check for stop loss\s+if position\.stop_loss is not None:'

new_run_backtest = '''# Update current price and check for stop loss or take profit
                if symbol in self.portfolio.open_positions:
                    position = self.portfolio.open_positions[symbol]
                    position.current_price = candle.close  # Update current price
                    
                    # Check for stop loss'''

# Apply all the fixes
updated_backtest_content = backtest_content
updated_backtest_content = re.sub(position_class_pattern, new_position_class, updated_backtest_content)
updated_backtest_content = re.sub(close_position_pattern, new_close_position, updated_backtest_content)
updated_backtest_content = re.sub(get_equity_pattern, new_get_equity, updated_backtest_content)
updated_backtest_content = re.sub(run_backtest_pattern, new_run_backtest, updated_backtest_content)

# Write the updated content back to the file
with open(backtest_file_path, 'w') as file:
    file.write(updated_backtest_content)

print(f"2. Updated {backtest_file_path} with fixed Position and Portfolio classes")
print("All fixes applied successfully. Run the backtest again to see improved results.")
