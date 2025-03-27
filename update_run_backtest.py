#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update the run_backtest function in final_sp500_strategy.py to accept tier threshold parameters
"""

import re
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_run_backtest_function():
    """Update the run_backtest function to accept tier threshold parameters"""
    try:
        # Path to the final_sp500_strategy.py file
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_sp500_strategy.py')
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Define the pattern to match the function definition
        old_function_def = r"def run_backtest\(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False\):"
        
        # Define the new function definition with tier threshold parameters
        new_function_def = "def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False, tier1_threshold=0.8, tier2_threshold=0.7, tier3_threshold=0.6):"
        
        # Replace the function definition
        updated_content = re.sub(old_function_def, new_function_def, content)
        
        # Find the SP500Strategy initialization in the run_backtest function
        strategy_init_pattern = r"strategy = SP500Strategy\(\s*api=api,\s*config=config,\s*mode=mode,\s*backtest_mode=True,\s*backtest_start_date=start_date,\s*backtest_end_date=end_date\s*\)"
        
        # Define the new strategy initialization with tier threshold parameters
        new_strategy_init = """strategy = SP500Strategy(
            api=api,
            config=config,
            mode=mode,
            backtest_mode=True,
            backtest_start_date=start_date,
            backtest_end_date=end_date
        )
        
        # Update tier thresholds in config
        if 'strategy' in config:
            if 'signal_thresholds' not in config['strategy']:
                config['strategy']['signal_thresholds'] = {}
            config['strategy']['signal_thresholds']['tier_1'] = tier1_threshold
            config['strategy']['signal_thresholds']['tier_2'] = tier2_threshold
            config['strategy']['signal_thresholds']['tier_3'] = tier3_threshold"""
        
        # Replace the strategy initialization
        updated_content = re.sub(strategy_init_pattern, new_strategy_init, updated_content)
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        
        logger.info(f"Successfully updated run_backtest function in {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating run_backtest function: {str(e)}")
        return False

if __name__ == "__main__":
    update_run_backtest_function()
