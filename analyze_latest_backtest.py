import os
import json
import glob
from datetime import datetime

# Find the most recent backtest files
def find_latest_backtest():
    backtest_dir = './backtest_results'
    files = glob.glob(os.path.join(backtest_dir, 'backtest_*.json'))
    
    if not files:
        print("No backtest files found")
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Get the run_id from the first file
    latest_file = files[0]
    print(f"Latest file found: {latest_file}")
    
    # Extract run_id from filename
    run_id = os.path.basename(latest_file).split('_')[-1].replace('.json', '')
    print(f"Extracted run_id: {run_id}")
    
    return run_id

def analyze_backtest_results(run_id):
    backtest_dir = './backtest_results'
    
    # Find all files for this run_id using glob pattern
    all_files = glob.glob(os.path.join(backtest_dir, f'*{run_id}*.json'))
    print(f"Found {len(all_files)} files for run_id: {run_id}")
    
    # Filter for quarterly files
    q1_files = [f for f in all_files if 'Q1_2023' in f or '_Q1_' in f]
    q2_files = [f for f in all_files if 'Q2_2023' in f or '_Q2_' in f]
    q3_files = [f for f in all_files if 'Q3_2023' in f or '_Q3_' in f]
    
    print(f"Q1 files: {q1_files}")
    print(f"Q2 files: {q2_files}")
    print(f"Q3 files: {q3_files}")
    
    if not (q1_files and q2_files and q3_files):
        print(f"Not all quarterly files found for run_id: {run_id}")
        return
    
    # Load and analyze each file
    q1_data = json.load(open(q1_files[0], 'r'))
    q2_data = json.load(open(q2_files[0], 'r'))
    q3_data = json.load(open(q3_files[0], 'r'))
    
    # Print summary for each quarter
    print("\n=== Q1_2023 ===")
    print(f"Trades: {len(q1_data['trades'])}")
    print(f"Initial Capital: ${q1_data['summary']['initial_capital']:.2f}")
    print(f"Final Capital: ${q1_data['summary']['final_capital']:.2f}")
    print(f"Win Rate: {q1_data['summary']['win_rate']:.2f}%")
    print(f"Total Return: {q1_data['summary']['total_return']:.2f}%")
    
    print("\n=== Q2_2023 ===")
    print(f"Trades: {len(q2_data['trades'])}")
    print(f"Initial Capital: ${q2_data['summary']['initial_capital']:.2f}")
    print(f"Final Capital: ${q2_data['summary']['final_capital']:.2f}")
    print(f"Win Rate: {q2_data['summary']['win_rate']:.2f}%")
    print(f"Total Return: {q2_data['summary']['total_return']:.2f}%")
    
    print("\n=== Q3_2023 ===")
    print(f"Trades: {len(q3_data['trades'])}")
    print(f"Initial Capital: ${q3_data['summary']['initial_capital']:.2f}")
    print(f"Final Capital: ${q3_data['summary']['final_capital']:.2f}")
    print(f"Win Rate: {q3_data['summary']['win_rate']:.2f}%")
    print(f"Total Return: {q3_data['summary']['total_return']:.2f}%")
    
    # Check if trades are different for each quarter
    q1_symbols = sorted([t['symbol'] for t in q1_data['trades'][:5]])
    q2_symbols = sorted([t['symbol'] for t in q2_data['trades'][:5]])
    q3_symbols = sorted([t['symbol'] for t in q3_data['trades'][:5]])
    
    print("\n=== Trade Analysis ===")
    print(f"Q1 First 5 Symbols: {q1_symbols}")
    print(f"Q2 First 5 Symbols: {q2_symbols}")
    print(f"Q3 First 5 Symbols: {q3_symbols}")
    
    # Check if continuous capital is working correctly
    print("\n=== Continuous Capital Analysis ===")
    if abs(q1_data['summary']['final_capital'] - q2_data['summary']['initial_capital']) < 0.01:
        print(" Q1 Final Capital matches Q2 Initial Capital")
    else:
        print(" Q1 Final Capital does NOT match Q2 Initial Capital")
        print(f"  Q1 Final: ${q1_data['summary']['final_capital']:.2f}, Q2 Initial: ${q2_data['summary']['initial_capital']:.2f}")
        
    if abs(q2_data['summary']['final_capital'] - q3_data['summary']['initial_capital']) < 0.01:
        print(" Q2 Final Capital matches Q3 Initial Capital")
    else:
        print(" Q2 Final Capital does NOT match Q3 Initial Capital")
        print(f"  Q2 Final: ${q2_data['summary']['final_capital']:.2f}, Q3 Initial: ${q3_data['summary']['initial_capital']:.2f}")

if __name__ == "__main__":
    # Get the latest run_id or use a specific one
    run_id = find_latest_backtest()
    
    if run_id:
        print(f"Analyzing backtest results for run_id: {run_id}")
        analyze_backtest_results(run_id)
    else:
        print("No backtest files found")
