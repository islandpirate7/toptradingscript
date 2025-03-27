import os
import json
import glob
from datetime import datetime

def check_backtest_results():
    """Check if backtest results are being saved correctly"""
    print("Checking backtest results...")
    
    # Check both relative and absolute paths
    results_dirs = [
        os.path.join('.', 'backtest_results'),  # Relative path
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'backtest_results'))  # Absolute path
    ]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            print(f"\nFound backtest results directory: {results_dir}")
            
            # Get all JSON files in the directory
            json_files = glob.glob(os.path.join(results_dir, "*.json"))
            
            if not json_files:
                print(f"No JSON files found in {results_dir}")
                continue
                
            print(f"Found {len(json_files)} JSON files")
            
            # Sort files by modification time (newest first)
            json_files.sort(key=os.path.getmtime, reverse=True)
            
            # Check the most recent file
            most_recent = json_files[0]
            mod_time = datetime.fromtimestamp(os.path.getmtime(most_recent))
            print(f"\nMost recent file: {os.path.basename(most_recent)}")
            print(f"Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check file contents
            try:
                with open(most_recent, 'r') as f:
                    data = json.load(f)
                    
                # Check if it has the expected structure
                if 'summary' in data:
                    print("File has the expected 'summary' key")
                    
                    # Print some summary data
                    summary = data['summary']
                    print(f"\nSummary data:")
                    if 'win_rate' in summary:
                        print(f"Win Rate: {summary['win_rate']}%")
                    if 'profit_factor' in summary:
                        print(f"Profit Factor: {summary['profit_factor']}")
                    if 'total_return' in summary:
                        print(f"Total Return: {summary['total_return']}%")
                    if 'initial_capital' in summary:
                        print(f"Initial Capital: ${summary['initial_capital']}")
                    if 'final_capital' in summary:
                        print(f"Final Capital: ${summary['final_capital']}")
                else:
                    print("WARNING: File does not have the expected 'summary' key")
                    print(f"Keys in file: {list(data.keys())}")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"Directory not found: {results_dir}")

if __name__ == "__main__":
    check_backtest_results()
