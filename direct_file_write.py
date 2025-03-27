import os
import sys
import argparse
from datetime import datetime

def run_backtest_with_direct_logging():
    """
    Run a backtest with direct file writing for logs.
    This script will call the original backtest function but will
    capture and log all output to a file using direct file writing.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run backtest with direct file logging')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date for backtest')
    parser.add_argument('--end-date', type=str, default='2023-01-15', help='End date for backtest')
    parser.add_argument('--initial-capital', type=float, default=300, help='Initial capital for backtest')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"direct_backtest_{timestamp}.log")
    
    print(f"Creating log file: {log_file}")
    
    # Open the log file for writing
    with open(log_file, 'w') as log_file_handle:
        # Write initial log message
        log_file_handle.write(f"{datetime.now()} - INFO - Starting backtest with direct file logging\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Start date: {args.start_date}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - End date: {args.end_date}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Initial capital: ${args.initial_capital}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Verbose: {args.verbose}\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Import the original strategy module
        log_file_handle.write(f"{datetime.now()} - INFO - Importing strategy module\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        try:
            # Import the strategy module
            import final_sp500_strategy
            
            # Run the backtest
            log_file_handle.write(f"{datetime.now()} - INFO - Running backtest\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            # Capture the start time
            start_time = datetime.now()
            
            # Run the backtest
            result = final_sp500_strategy.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                mode='backtest'
            )
            
            # Capture the end time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log the results
            log_file_handle.write(f"{datetime.now()} - INFO - Backtest completed in {duration:.2f} seconds\n")
            
            if isinstance(result, dict) and 'performance' in result:
                performance = result['performance']
                log_file_handle.write(f"{datetime.now()} - INFO - Final portfolio value: ${performance.get('final_value', 0):.2f}\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Return: {performance.get('return', 0):.2f}%\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Annualized return: {performance.get('annualized_return', 0):.2f}%\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Max drawdown: {performance.get('max_drawdown', 0):.2f}%\n")
                log_file_handle.write(f"{datetime.now()} - INFO - Win rate: {performance.get('win_rate', 0):.2f}%\n")
            else:
                log_file_handle.write(f"{datetime.now()} - INFO - No performance data available\n")
            
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            print(f"Backtest completed successfully")
            print(f"Log file: {log_file}")
            
            # Check log file size
            file_size = os.path.getsize(log_file)
            print(f"Log file size: {file_size} bytes")
            
            if file_size > 0:
                print("SUCCESS: Log file contains content")
            else:
                print("ERROR: Log file is empty (0 bytes)")
                
        except Exception as e:
            # Log the error
            log_file_handle.write(f"{datetime.now()} - ERROR - {str(e)}\n")
            import traceback
            log_file_handle.write(f"{datetime.now()} - ERROR - {traceback.format_exc()}\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            
            print(f"ERROR: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    run_backtest_with_direct_logging()
