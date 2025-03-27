#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix path issues in final_sp500_strategy.py
"""

import os
import re
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fix_paths_in_run_backtest():
    """Fix path issues in the run_backtest function in final_sp500_strategy.py"""
    try:
        # Path to the final_sp500_strategy.py file
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_sp500_strategy.py')
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace relative path with absolute path for config_path
        pattern = r"config_path = 'sp500_config.yaml'"
        replacement = "config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sp500_config.yaml')"
        updated_content = re.sub(pattern, replacement, content)
        
        # Replace relative path with absolute path for alpaca_credentials.json
        pattern = r"with open\('alpaca_credentials.json', 'r'\) as f:"
        replacement = "with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca_credentials.json'), 'r') as f:"
        updated_content = re.sub(pattern, replacement, updated_content)
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        
        logger.info(f"Successfully updated paths in run_backtest function in {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating paths in run_backtest function: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    fix_paths_in_run_backtest()
