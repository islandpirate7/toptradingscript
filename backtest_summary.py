#!/usr/bin/env python
"""
Backtest Summary Generator
--------------------------
This script provides a clean, formatted summary of backtest results.
It can be used to quickly analyze the performance of a backtest without
having to parse through verbose log output.
"""

import os
import json
import argparse
from datetime import datetime
import glob
from tabulate import tabulate
import yaml
import matplotlib.pyplot as plt
import pandas as pd

def load_config():
    """Load configuration from YAML file."""
    try:
        with open('sp500_config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}

def find_result_files(results_dir=None):
    """Find all backtest result files."""
    if results_dir is None:
        config = load_config()
        results_dir = config.get('backtest_results_dir', '')
    
    # Look in multiple possible locations
    search_paths = [
        results_dir,
        'results',
        'backtest_results',
        '.',  # Current directory
    ]
    
    all_files = []
    for path in search_paths:
        if path and os.path.exists(path):
            json_files = glob.glob(os.path.join(path, '*.json'))
            all_files.extend(json_files)
    
    # Also search for any JSON files with 'backtest' in the name
    backtest_files = glob.glob('*backtest*.json')
    all_files.extend(backtest_files)
    
    # Remove duplicates and sort by modification time
    unique_files = list(set(all_files))
    return sorted(unique_files, key=os.path.getmtime, reverse=True)

def load_result_file(filename):
    """Load a specific result file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading result file {filename}: {e}")
        return None

def format_currency(value):
    """Format a value as currency."""
    return f"${value:.2f}"

def format_percentage(value):
    """Format a value as percentage."""
    return f"{value:.2f}%"

def generate_summary(result_data):
    """Generate a summary of the backtest results."""
    if not result_data:
        return "No result data available."
    
    # Extract basic information based on the actual format
    summary = []
    
    # Basic information
    initial_capital = result_data.get('initial_capital', 0)
    final_equity = result_data.get('final_equity', 0)
    total_return = result_data.get('return', 0) * 100  # Convert to percentage
    
    summary.append(["Initial Capital", format_currency(initial_capital)])
    summary.append(["Final Equity", format_currency(final_equity)])
    summary.append(["Total Return", format_percentage(total_return)])
    
    # Calculate additional metrics
    trade_history = result_data.get('trade_history', [])
    
    # Calculate win rate
    winning_trades = [t for t in trade_history if t.get('profit', 0) > 0]
    win_rate = len(winning_trades) / len(trade_history) * 100 if trade_history else 0
    summary.append(["Total Trades", len(trade_history)])
    summary.append(["Winning Trades", len(winning_trades)])
    summary.append(["Win Rate", format_percentage(win_rate)])
    
    # Calculate profit factor
    total_profit = sum(t.get('profit', 0) for t in trade_history if t.get('profit', 0) > 0)
    total_loss = abs(sum(t.get('profit', 0) for t in trade_history if t.get('profit', 0) < 0))
    profit_factor = total_profit / total_loss if total_loss else float('inf')
    summary.append(["Total Profit", format_currency(total_profit)])
    summary.append(["Total Loss", format_currency(total_loss)])
    summary.append(["Profit Factor", f"{profit_factor:.2f}"])
    
    # Calculate average trade
    avg_trade = sum(t.get('profit', 0) for t in trade_history) / len(trade_history) if trade_history else 0
    summary.append(["Average Trade", format_currency(avg_trade)])
    
    # Extract date range if available
    equity_curve = result_data.get('equity_curve', {})
    if equity_curve:
        dates = list(equity_curve.keys())
        if dates:
            start_date = min(dates)
            end_date = max(dates)
            summary.insert(0, ["Backtest Period", f"{start_date} to {end_date}"])
    
    # Calculate max drawdown
    if equity_curve:
        values = list(equity_curve.values())
        max_drawdown = 0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        summary.append(["Max Drawdown", format_percentage(max_drawdown)])
    
    return tabulate(summary, headers=["Metric", "Value"], tablefmt="grid")

def plot_equity_curve(result_data, output_file=None):
    """Plot the equity curve from backtest results."""
    if not result_data or 'equity_curve' not in result_data:
        print("No equity curve data available.")
        return
    
    equity_curve = result_data['equity_curve']
    
    if not equity_curve:
        print("Equity curve data is empty.")
        return
    
    # Convert to DataFrame
    dates = list(equity_curve.keys())
    values = list(equity_curve.values())
    
    # Create DataFrame
    df = pd.DataFrame({'Date': dates, 'Value': values})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Value'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Equity curve saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate a summary of backtest results')
    parser.add_argument('--file', help='Specific result file to analyze')
    parser.add_argument('--latest', action='store_true', help='Analyze the latest result file')
    parser.add_argument('--list', action='store_true', help='List all available result files')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    parser.add_argument('--output', help='Output file for the equity curve plot')
    
    args = parser.parse_args()
    
    # Find all result files
    result_files = find_result_files()
    
    if not result_files:
        print("No result files found.")
        return
    
    if args.list:
        print("Available result files:")
        for i, file in enumerate(result_files):
            file_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"{i+1}. {os.path.basename(file)} - {file_time}")
        return
    
    # Determine which file to analyze
    target_file = None
    if args.file:
        if os.path.exists(args.file):
            target_file = args.file
        else:
            # Try to find the file by name
            matching_files = [f for f in result_files if os.path.basename(f) == args.file]
            if matching_files:
                target_file = matching_files[0]
            else:
                print(f"File not found: {args.file}")
                return
    elif args.latest:
        if result_files:
            target_file = result_files[0]  # First file (most recent)
    else:
        # If no specific file is requested, use the latest
        if result_files:
            target_file = result_files[0]
    
    if not target_file:
        print("No target file specified or found.")
        return
    
    # Load and analyze the target file
    result_data = load_result_file(target_file)
    if not result_data:
        return
    
    print(f"\nAnalyzing: {os.path.basename(target_file)}")
    print("\nBACKTEST SUMMARY")
    print("===============")
    print(generate_summary(result_data))
    
    # Plot equity curve if requested
    if args.plot:
        output_file = args.output if args.output else None
        plot_equity_curve(result_data, output_file)

if __name__ == "__main__":
    main()
