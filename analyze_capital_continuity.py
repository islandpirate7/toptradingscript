#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze capital continuity in backtest results.
This script finds the most recent backtest results and checks if the continuous capital is working correctly.
"""

import os
import json
import glob
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_all_backtest_files():
    """Find all backtest result files."""
    # Get all backtest result files
    result_files = glob.glob('backtest_results_*.json')
    result_files += glob.glob('combined_backtest_results_*.json')
    
    # Also check in the web_interface directory
    web_interface_files = glob.glob('web_interface/backtest_results_*.json')
    web_interface_files += glob.glob('web_interface/combined_backtest_results_*.json')
    
    result_files += web_interface_files
    
    # Sort by modification time (newest first)
    sorted_files = sorted(result_files, key=os.path.getmtime, reverse=True)
    
    return sorted_files

def analyze_backtest_results():
    """Analyze backtest results for capital continuity."""
    result_files = find_all_backtest_files()
    
    print("=== Backtest Results Analysis ===")
    print(f"Found {len(result_files)} backtest result files")
    
    if not result_files:
        print("No backtest result files found")
        return
    
    # Analyze each result file
    for file in result_files[:5]:  # Limit to the 5 most recent files
        print(f"\nAnalyzing file: {file}")
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Check if it's a combined result file
            if 'quarters' in data:
                print("This is a combined result file")
                for quarter, quarter_data in data['quarters'].items():
                    print(f"  Quarter: {quarter}")
                    analyze_quarter_data(quarter_data)
            else:
                # Check for quarter info in the summary
                quarter = data.get('summary', {}).get('quarter', 'Unknown')
                print(f"  Quarter: {quarter}")
                analyze_quarter_data(data)
        except Exception as e:
            print(f"  Error analyzing file {file}: {str(e)}")

def analyze_quarter_data(data):
    """Analyze quarter data for capital continuity."""
    summary = data.get('summary', {})
    
    # Extract capital values
    initial_capital = summary.get('initial_capital')
    final_capital = summary.get('final_capital')
    
    if initial_capital is not None:
        print(f"  Initial Capital: ${initial_capital:.2f}")
    else:
        print("  Initial Capital: Not found")
    
    if final_capital is not None:
        print(f"  Final Capital: ${final_capital:.2f}")
    else:
        print("  Final Capital: Not found")
    
    # Calculate profit/loss
    if initial_capital is not None and final_capital is not None:
        profit_loss = final_capital - initial_capital
        profit_loss_pct = (profit_loss / initial_capital) * 100
        print(f"  Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
    
    # Check for other relevant metrics
    metrics = summary.get('metrics', {})
    if metrics:
        print("  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    logger.info("Starting analysis of backtest results...")
    analyze_backtest_results()
    logger.info("Analysis completed")
