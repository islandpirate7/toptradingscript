#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Backtest Summary

This script adds missing fields to backtest result files to ensure compatibility with the web interface.
"""

import os
import json
import logging
import glob
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_backtest_summary_file(file_path):
    """Add missing fields to backtest summary file"""
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if summary exists
        if 'summary' not in data:
            logger.warning(f"No summary found in {file_path}")
            return False
        
        summary = data['summary']
        
        # Add missing fields with default values
        required_fields = {
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0
        }
        
        modified = False
        for field, default_value in required_fields.items():
            if field not in summary:
                summary[field] = default_value
                modified = True
                logger.info(f"Added missing field '{field}' to {file_path}")
        
        # Write the updated data back to the file if modified
        if modified:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Updated {file_path} with missing fields")
            return True
        else:
            logger.info(f"No updates needed for {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing backtest summary file {file_path}: {str(e)}")
        return False

def fix_all_backtest_files(directory):
    """Fix all backtest result files in the specified directory"""
    # Find all backtest result files
    backtest_files = glob.glob(os.path.join(directory, "backtest_*.json"))
    
    if not backtest_files:
        logger.warning(f"No backtest files found in {directory}")
        return 0
    
    count = 0
    for file_path in backtest_files:
        if fix_backtest_summary_file(file_path):
            count += 1
    
    logger.info(f"Fixed {count} out of {len(backtest_files)} backtest files")
    return count

if __name__ == "__main__":
    # Get the backtest results directory
    backtest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
    
    if not os.path.exists(backtest_dir):
        logger.error(f"Backtest directory {backtest_dir} does not exist")
        exit(1)
    
    # Fix all backtest files
    fixed_count = fix_all_backtest_files(backtest_dir)
    
    if fixed_count > 0:
        logger.info(f"Successfully fixed {fixed_count} backtest files")
    else:
        logger.info("No backtest files needed fixing")
