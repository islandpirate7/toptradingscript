#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to read and analyze backtest log files
"""

import os
import sys
import re
from datetime import datetime

def read_log_file(log_file_path):
    """Read and print the contents of a log file"""
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            print(f"Log file contents ({log_file_path}):")
            print("=" * 80)
            print(content)
            print("=" * 80)
            return content
    except Exception as e:
        print(f"Error reading log file: {str(e)}")
        return None

def analyze_log(content):
    """Analyze the log content to identify key events and issues"""
    if not content:
        return
    
    # Count different types of log entries
    info_count = content.count(" - INFO - ")
    warning_count = content.count(" - WARNING - ")
    error_count = content.count(" - ERROR - ")
    
    print(f"\nLog Analysis:")
    print(f"INFO entries: {info_count}")
    print(f"WARNING entries: {warning_count}")
    print(f"ERROR entries: {error_count}")
    
    # Extract positions opened
    positions_opened = re.findall(r"Opened (LONG|SHORT) position for ([A-Z\-]+) at ([\d\.]+) with ([\d\.]+) shares", content)
    
    if positions_opened:
        print("\nPositions Opened:")
        for position in positions_opened:
            direction, symbol, price, shares = position
            print(f"  {direction} {symbol}: {shares} shares at ${price}")
    else:
        print("\nNo positions were opened during the backtest.")
    
    # Extract not enough cash warnings
    not_enough_cash = re.findall(r"Not enough cash to open position for ([A-Z\-]+)", content)
    
    if not_enough_cash:
        print(f"\nNot enough cash for {len(not_enough_cash)} positions:")
        for symbol in not_enough_cash:
            print(f"  {symbol}")
    
    # Extract API errors
    api_errors = re.findall(r"ERROR - Error getting bars: (.+)", content)
    
    if api_errors:
        print("\nAPI Errors:")
        for error in api_errors:
            print(f"  {error}")
    
    # Extract performance metrics
    final_value_match = re.search(r"Final portfolio value: \$([\d\.]+)", content)
    return_match = re.search(r"Return: ([\d\.\-]+)%", content)
    
    if final_value_match and return_match:
        final_value = float(final_value_match.group(1))
        return_pct = float(return_match.group(1))
        print(f"\nPerformance Metrics:")
        print(f"  Final Portfolio Value: ${final_value:.2f}")
        print(f"  Return: {return_pct:.2f}%")
    
    # Check for any specific errors or issues
    if "403" in content:
        print("\nWARNING: 403 Forbidden errors detected - API access issue!")
    
    # Check if backtest completed
    if "Backtest completed" in content:
        print("\nBacktest completed successfully.")
    else:
        print("\nBacktest may not have completed properly.")

def main():
    """Main function to read and analyze log files"""
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
    else:
        # Find the most recent log file
        logs_dir = os.path.join(os.getcwd(), "logs")
        strategy_logs = [f for f in os.listdir(logs_dir) if f.startswith("strategy_")]
        
        if not strategy_logs:
            print("No strategy log files found.")
            return
        
        # Sort by creation time (most recent first)
        strategy_logs.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
        log_file_path = os.path.join(logs_dir, strategy_logs[0])
        print(f"Analyzing most recent log file: {log_file_path}")
    
    content = read_log_file(log_file_path)
    analyze_log(content)

if __name__ == "__main__":
    main()
