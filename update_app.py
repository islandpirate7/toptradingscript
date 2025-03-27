#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update App Script
-----------------------------------
This script updates the app.py file to use the optimized backtest function.
"""

import os
import re

def update_app_py():
    """Update app.py to use the optimized backtest function"""
    app_py_path = os.path.join('web_interface', 'app.py')
    
    # Read the current content of app.py
    with open(app_py_path, 'r') as f:
        content = f.read()
    
    # Add import for optimized backtest
    import_pattern = r'import os\nimport sys\nimport json'
    optimized_import = 'import os\nimport sys\nimport json\n\n# Import optimized backtest if available\ntry:\n    from optimized_backtest import run_optimized_backtest\n    USE_OPTIMIZED_BACKTEST = True\n    print("Using optimized backtest function")\nexcept ImportError:\n    USE_OPTIMIZED_BACKTEST = False\n    print("Optimized backtest module not found, falling back to standard backtest")'
    
    if 'from optimized_backtest import run_optimized_backtest' not in content:
        content = re.sub(import_pattern, optimized_import, content)
    
    # Update run_backtest_thread function to use optimized backtest
    thread_pattern = r'def run_backtest_thread\(quarters, run_id, process_name, data\):.*?# Disable hot reload during backtest to avoid unnecessary file system operations.*?hot_reload_enabled = config_data\.get\(\'hot_reload\', \{\}\)\.get\(\'enabled\', False\)'
    thread_replacement = '''def run_backtest_thread(quarters, run_id, process_name, data):
    """Run backtest in a separate thread"""
    try:
        # Get parameters from data
        max_signals = int(data.get('max_signals', 10))
        initial_capital = float(data.get('initial_capital', 10000))
        continuous_capital = data.get('continuous_capital', False)
        weekly_selection = data.get('weekly_selection', False)
        
        # Log parameters
        logger.info(f"Backtest parameters: max_signals={max_signals}, initial_capital={initial_capital}, continuous_capital={continuous_capital}, weekly_selection={weekly_selection}")
        active_processes[process_name]['logs'].append(f"Backtest parameters: max_signals={max_signals}, initial_capital={initial_capital}, continuous_capital={continuous_capital}, weekly_selection={weekly_selection}")
        
        # Import the modify_backtest_results module
        try:
            from modify_backtest_results import modify_results_for_quarter
            logger.info("Successfully imported modify_backtest_results module")
        except ImportError:
            logger.error("Failed to import modify_backtest_results module. Results will not be modified for quarters.")
            # Define a dummy function that returns the input unchanged
            def modify_results_for_quarter(result_data, quarter):
                return result_data
        
        # Track previous capital for continuous capital mode
        previous_capital = initial_capital if continuous_capital else None
        
        # Get paths from config
        results_dir = config_data.get('paths', {}).get('backtest_results', './backtest_results')
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Dictionary to store results for each quarter
        results = {}
        
        # Disable hot reload during backtest to avoid unnecessary file system operations
        hot_reload_enabled = config_data.get('hot_reload', {}).get('enabled', False)'''
    
    content = re.sub(thread_pattern, thread_replacement, content, flags=re.DOTALL)
    
    # Update backtest call to use optimized version when available
    backtest_pattern = r'# Run the backtest with the custom date range.*?summary, trades = run_backtest\('
    backtest_replacement = '''# Run the backtest with the custom date range
                try:
                    # Use optimized backtest if available
                    if USE_OPTIMIZED_BACKTEST:
                        logger.info(f"Running optimized backtest with custom date range: start_date={start_date}, end_date={end_date}, max_signals={max_signals}")
                        summary, trades = run_optimized_backtest('''
    
    content = re.sub(backtest_pattern, backtest_replacement, content, flags=re.DOTALL)
    
    # Update the second backtest call
    second_backtest_pattern = r'# Run the backtest.*?summary, trades = run_backtest\('
    second_backtest_replacement = '''# Run the backtest
                    # Use optimized backtest if available
                    if USE_OPTIMIZED_BACKTEST:
                        logger.info(f"Running optimized backtest with parameters: start_date={start_date}, end_date={end_date}, max_signals={max_signals}, initial_capital={previous_capital if continuous_capital and previous_capital else initial_capital}")
                        summary, trades = run_optimized_backtest('''
    
    # Find all occurrences and replace only the second one
    matches = list(re.finditer(second_backtest_pattern, content, re.DOTALL))
    if len(matches) >= 2:
        start, end = matches[1].span()
        content = content[:start] + second_backtest_replacement + content[end:]
    
    # Write the updated content back to app.py
    with open(app_py_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {app_py_path} to use optimized backtest function")

if __name__ == "__main__":
    update_app_py()
