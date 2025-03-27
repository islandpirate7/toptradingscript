#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze continuous capital functionality in backtest results.
This script checks if the final capital from one quarter matches the initial capital of the next quarter.
"""

import os
import json
import glob
from datetime import datetime

def find_latest_backtest_files():
    """Find the most recent backtest result files for each quarter."""
    # Get all backtest result files
    result_files = glob.glob('backtest_results_*.json')
    
    # Group files by quarter
    quarter_files = {}
    for file in result_files:
        # Extract quarter from filename (e.g., 'backtest_results_Q1_2023_20250325_*.json')
        parts = file.split('_')
        if len(parts) >= 4 and parts[2].startswith('Q') and parts[3].isdigit():
            quarter = f"{parts[2]}_{parts[3]}"
            if quarter not in quarter_files:
                quarter_files[quarter] = []
            quarter_files[quarter].append(file)
    
    # Find the most recent file for each quarter
    latest_files = {}
    for quarter, files in quarter_files.items():
        if files:
            # Sort by timestamp in filename
            sorted_files = sorted(files, key=lambda x: x.split('_')[-2] + '_' + x.split('_')[-1].split('.')[0], reverse=True)
            latest_files[quarter] = sorted_files[0]
    
    return latest_files

def analyze_continuous_capital():
    """Analyze if continuous capital is working correctly."""
    latest_files = find_latest_backtest_files()
    
    # Sort quarters chronologically
    quarters = sorted(latest_files.keys())
    
    print("=== Continuous Capital Analysis ===")
    
    # Check if final capital from one quarter matches initial capital of next quarter
    for i in range(len(quarters) - 1):
        current_quarter = quarters[i]
        next_quarter = quarters[i + 1]
        
        # Load result files
        with open(latest_files[current_quarter], 'r') as f:
            current_data = json.load(f)
        
        with open(latest_files[next_quarter], 'r') as f:
            next_data = json.load(f)
        
        # Extract capital values
        current_final_capital = current_data.get('summary', {}).get('final_capital')
        next_initial_capital = next_data.get('summary', {}).get('initial_capital')
        
        if current_final_capital is not None and next_initial_capital is not None:
            # Round to 2 decimal places for comparison
            current_final_capital = round(current_final_capital, 2)
            next_initial_capital = round(next_initial_capital, 2)
            
            if current_final_capital == next_initial_capital:
                print(f" {current_quarter} Final Capital MATCHES {next_quarter} Initial Capital")
                print(f"  {current_quarter} Final: ${current_final_capital:.2f}, {next_quarter} Initial: ${next_initial_capital:.2f}")
            else:
                print(f" {current_quarter} Final Capital does NOT match {next_quarter} Initial Capital")
                print(f"  {current_quarter} Final: ${current_final_capital:.2f}, {next_quarter} Initial: ${next_initial_capital:.2f}")
        else:
            print(f" Could not compare {current_quarter} and {next_quarter} - missing capital values")

if __name__ == "__main__":
    analyze_continuous_capital()
