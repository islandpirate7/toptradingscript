#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Results Viewer

This script provides a simple way to view backtest results without using the web interface.
It reads JSON result files from the backtest_results directory and displays their contents.
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from tabulate import tabulate

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='View backtest results')
    parser.add_argument('--list', action='store_true', help='List all available backtest result files')
    parser.add_argument('--file', type=str, help='View a specific backtest result file')
    parser.add_argument('--latest', action='store_true', help='View the latest backtest result file')
    parser.add_argument('--quarter', type=str, help='View results for a specific quarter (e.g., Q1_2023)')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to backtest results directory
    results_dir = os.path.join(script_dir, 'backtest_results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Backtest results directory not found: {results_dir}")
        sys.exit(1)
    
    # Get list of backtest result files
    result_files = []
    for file in glob.glob(os.path.join(results_dir, '*.json')):
        if os.path.isfile(file):
            result_files.append(file)
    
    if not result_files:
        print("No backtest result files found.")
        sys.exit(0)
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if args.list:
        # List all backtest result files
        print("\nAvailable Backtest Result Files:")
        print("=" * 80)
        
        table_data = []
        for i, file in enumerate(result_files):
            file_name = os.path.basename(file)
            file_size = os.path.getsize(file)
            file_time = datetime.fromtimestamp(os.path.getmtime(file))
            
            # Try to extract quarter information from filename
            quarter = "Unknown"
            if "Q1" in file_name:
                quarter = "Q1"
            elif "Q2" in file_name:
                quarter = "Q2"
            elif "Q3" in file_name:
                quarter = "Q3"
            elif "Q4" in file_name:
                quarter = "Q4"
            
            table_data.append([
                i + 1,
                file_name,
                f"{file_size / 1024:.1f} KB",
                file_time.strftime("%Y-%m-%d %H:%M:%S"),
                quarter
            ])
        
        print(tabulate(table_data, headers=["#", "Filename", "Size", "Modified", "Quarter"], tablefmt="grid"))
        print("\nTo view a specific file, use: python view_backtest_results.py --file <filename>")
        print("To view the latest result, use: python view_backtest_results.py --latest")
        
    elif args.file:
        # View a specific backtest result file
        file_path = None
        
        # Check if the file exists directly
        if os.path.exists(args.file):
            file_path = args.file
        elif os.path.exists(os.path.join(results_dir, args.file)):
            file_path = os.path.join(results_dir, args.file)
        else:
            # Try to find a file that contains the specified name
            for file in result_files:
                if args.file in os.path.basename(file):
                    file_path = file
                    break
        
        if not file_path:
            print(f"Error: Backtest result file not found: {args.file}")
            sys.exit(1)
        
        view_result_file(file_path)
        
    elif args.latest:
        # View the latest backtest result file
        if result_files:
            view_result_file(result_files[0])
        else:
            print("No backtest result files found.")
            
    elif args.quarter:
        # View results for a specific quarter
        quarter_files = []
        for file in result_files:
            if args.quarter in os.path.basename(file):
                quarter_files.append(file)
        
        if not quarter_files:
            print(f"No backtest result files found for quarter: {args.quarter}")
            sys.exit(0)
        
        # Sort by modification time (newest first)
        quarter_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # View the latest result for the specified quarter
        view_result_file(quarter_files[0])
        
    else:
        # No arguments provided, show help
        parser.print_help()

def view_result_file(file_path):
    """View a backtest result file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        file_name = os.path.basename(file_path)
        print(f"\nBacktest Result: {file_name}")
        print("=" * 80)
        
        # Print basic information
        print(f"Date: {data.get('date', 'Unknown')}")
        print(f"Period: {data.get('start_date', 'Unknown')} to {data.get('end_date', 'Unknown')}")
        print(f"Initial Capital: ${data.get('initial_capital', 0):.2f}")
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print("-" * 80)
        
        metrics = [
            ("Final Value", f"${data.get('final_value', 0):.2f}"),
            ("Return", f"{data.get('return', 0) * 100:.2f}%"),
            ("Annualized Return", f"{data.get('annualized_return', 0) * 100:.2f}%"),
            ("Sharpe Ratio", f"{data.get('sharpe_ratio', 0):.2f}"),
            ("Max Drawdown", f"{data.get('max_drawdown', 0) * 100:.2f}%"),
            ("Win Rate", f"{data.get('win_rate', 0) * 100:.2f}%"),
            ("Profit Factor", f"{data.get('profit_factor', 0):.2f}"),
            ("Total Trades", f"{data.get('total_trades', 0)}"),
            ("Winning Trades", f"{data.get('winning_trades', 0)}"),
            ("Losing Trades", f"{data.get('losing_trades', 0)}")
        ]
        
        print(tabulate([list(m) for m in metrics], tablefmt="simple"))
        
        # Print trades if available
        if 'trades' in data and data['trades']:
            print("\nTrades Summary:")
            print("-" * 80)
            
            # Get a sample of trades (first 10)
            sample_trades = data['trades'][:10]
            
            # Prepare table data
            trade_data = []
            for trade in sample_trades:
                trade_data.append([
                    trade.get('symbol', 'Unknown'),
                    trade.get('direction', 'Unknown'),
                    trade.get('entry_date', 'Unknown'),
                    trade.get('exit_date', 'Unknown'),
                    f"${trade.get('entry_price', 0):.2f}",
                    f"${trade.get('exit_price', 0):.2f}",
                    f"{trade.get('profit_loss_pct', 0) * 100:.2f}%",
                    f"${trade.get('profit_loss', 0):.2f}"
                ])
            
            print(tabulate(
                trade_data,
                headers=["Symbol", "Direction", "Entry Date", "Exit Date", "Entry Price", "Exit Price", "P/L %", "P/L $"],
                tablefmt="grid"
            ))
            
            if len(data['trades']) > 10:
                print(f"\n... and {len(data['trades']) - 10} more trades (showing first 10 only)")
        
        # Print daily performance if available
        if 'daily_performance' in data and data['daily_performance']:
            print("\nDaily Performance Summary:")
            print("-" * 80)
            
            # Get a sample of daily performance (first and last 5 days)
            daily_perf = data['daily_performance']
            sample_days = []
            
            if len(daily_perf) <= 10:
                sample_days = daily_perf
            else:
                sample_days = daily_perf[:5] + daily_perf[-5:]
            
            # Prepare table data
            daily_data = []
            for day in sample_days:
                daily_data.append([
                    day.get('date', 'Unknown'),
                    f"${day.get('portfolio_value', 0):.2f}",
                    f"{day.get('daily_return', 0) * 100:.2f}%",
                    day.get('positions', 0),
                    f"${day.get('cash', 0):.2f}"
                ])
            
            print(tabulate(
                daily_data,
                headers=["Date", "Portfolio Value", "Daily Return", "Positions", "Cash"],
                tablefmt="grid"
            ))
            
            if len(daily_perf) > 10:
                print(f"\n... and {len(daily_perf) - 10} more days (showing first 5 and last 5 only)")
        
    except Exception as e:
        print(f"Error viewing result file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
