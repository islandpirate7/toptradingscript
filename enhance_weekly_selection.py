#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhance the weekly stock selection refresh functionality in final_sp500_strategy.py
"""

import re

def fix_get_symbols_method():
    # Read the file
    with open('final_sp500_strategy.py', 'r') as f:
        content = f.readlines()
    
    # Find the get_symbols method
    start_line = 0
    end_line = 0
    for i, line in enumerate(content):
        if "def get_symbols(self):" in line:
            start_line = i
        if start_line > 0 and "return self.TOP_100_SYMBOLS" in line:
            end_line = i + 1
            break
    
    if start_line == 0 or end_line == 0:
        print("Could not find the get_symbols method")
        return
    
    # Extract the method
    method_lines = content[start_line:end_line]
    
    # Fix the indentation issue with the midcap symbols
    fixed_method = []
    in_midcap_block = False
    
    for line in method_lines:
        if "if include_midcap:" in line:
            in_midcap_block = True
            fixed_method.append(line)
        elif in_midcap_block and "# Add mid-cap symbols to the list" in line:
            in_midcap_block = False
            # Fix indentation for this line and the following lines
            fixed_method.append(line.replace("            # Add mid-cap symbols to the list", "                # Add mid-cap symbols to the list"))
        elif in_midcap_block:
            # Keep the indentation for lines inside the midcap block
            fixed_method.append(line)
        else:
            # For lines outside the midcap block, keep them as is
            fixed_method.append(line)
    
    # Replace the old method with the fixed one
    new_content = content[:start_line] + fixed_method + content[end_line:]
    
    # Write the updated content back to the file
    with open('final_sp500_strategy.py', 'w') as f:
        f.writelines(new_content)
    
    print("Enhanced weekly stock selection refresh functionality in final_sp500_strategy.py")

def add_weekly_selection_to_run_backtest():
    # Read the file
    with open('final_sp500_strategy.py', 'r') as f:
        content = f.read()
    
    # Find the run_backtest function
    pattern = r'def run_backtest\(start_date, end_date, mode=\'backtest\', max_signals=None, initial_capital=300, random_seed=42\):'
    
    # Check if we need to add weekly selection parameter
    if "weekly_selection" not in content[content.find(pattern):content.find(pattern) + 1000]:
        # Add weekly_selection parameter
        replacement = r'def run_backtest(start_date, end_date, mode=\'backtest\', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True):'
        content = content.replace(pattern, replacement)
        
        # Find where to add the weekly selection code
        setup_pattern = r'strategy = SP500Strategy\(api=None, config=config, mode=mode, backtest_mode=True, backtest_start_date=start_date, backtest_end_date=end_date\)'
        
        # Add weekly selection configuration
        replacement = r'strategy = SP500Strategy(api=None, config=config, mode=mode, backtest_mode=True, backtest_start_date=start_date, backtest_end_date=end_date)\n    \n    # Configure weekly stock selection refresh\n    if weekly_selection:\n        strategy.symbol_reselection_interval = 7  # Refresh stocks weekly\n        logger.info("Weekly stock selection refresh enabled")\n    else:\n        strategy.symbol_reselection_interval = float("inf")  # Disable automatic refresh\n        logger.info("Weekly stock selection refresh disabled")'
        
        content = content.replace(setup_pattern, replacement)
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.write(content)
        
        print("Added weekly selection parameter to run_backtest function")
    else:
        print("Weekly selection parameter already exists in run_backtest function")

if __name__ == "__main__":
    fix_get_symbols_method()
    add_weekly_selection_to_run_backtest()
