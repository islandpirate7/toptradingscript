#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the function signature in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def fix_function_signature():
    """
    Fix the run_backtest function signature in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Fix the function signature
        fixed_content = re.sub(
            r'def run_backtest\(start_date, end_date, mode=\'backtest\', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False\), continuous_capital=False\):',
            r'def run_backtest(start_date, end_date, mode=\'backtest\', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False):',
            content
        )
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.write(fixed_content)
        
        print("Successfully fixed run_backtest function signature")
        
    except Exception as e:
        print(f"Error fixing function signature: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fixing function signature in final_sp500_strategy.py...")
    fix_function_signature()
    
    print("\nFix complete. Please restart the web server to apply the changes.")
