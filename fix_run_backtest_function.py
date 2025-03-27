#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update the run_backtest function in final_sp500_strategy.py to accept the weekly_selection parameter
"""

def update_run_backtest():
    # Read the file
    with open('final_sp500_strategy.py', 'r') as f:
        content = f.read()
    
    # Find the run_backtest function definition
    old_signature = "def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42):"
    new_signature = "def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True):"
    
    # Replace the function signature
    if old_signature in content:
        content = content.replace(old_signature, new_signature)
        
        # Find where to add the weekly selection code
        setup_pattern = "strategy = SP500Strategy(api=None, config=config, mode=mode, backtest_mode=True, backtest_start_date=start_date, backtest_end_date=end_date)"
        
        # Add weekly selection configuration
        replacement = "strategy = SP500Strategy(api=None, config=config, mode=mode, backtest_mode=True, backtest_start_date=start_date, backtest_end_date=end_date)\n    \n    # Configure weekly stock selection refresh\n    if weekly_selection:\n        strategy.symbol_reselection_interval = 7  # Refresh stocks weekly\n        logger.info(\"Weekly stock selection refresh enabled\")\n    else:\n        strategy.symbol_reselection_interval = float(\"inf\")  # Disable automatic refresh\n        logger.info(\"Weekly stock selection refresh disabled\")"
        
        content = content.replace(setup_pattern, replacement)
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.write(content)
        
        print("Updated run_backtest function to accept weekly_selection parameter")
    else:
        print("Could not find the run_backtest function signature")

if __name__ == "__main__":
    update_run_backtest()
