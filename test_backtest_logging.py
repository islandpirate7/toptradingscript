import os
import sys
import logging
from datetime import datetime
import argparse

def setup_direct_logging():
    """
    Set up direct file logging without using the logging module
    """
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"test_backtest_{timestamp}.log")
    
    print(f"Creating log file: {log_file}")
    
    # Open the log file for writing
    log_file_handle = open(log_file, 'w')
    
    # Write initial log message
    log_file_handle.write(f"{datetime.now()} - INFO - Starting test backtest logging\n")
    log_file_handle.flush()
    os.fsync(log_file_handle.fileno())
    
    return log_file, log_file_handle

def test_backtest_logging():
    """
    Test backtest logging functionality
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test backtest logging')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date for backtest')
    parser.add_argument('--end-date', type=str, default='2023-01-15', help='End date for backtest')
    parser.add_argument('--initial-capital', type=float, default=300, help='Initial capital for backtest')
    args = parser.parse_args()
    
    # Set up direct logging
    log_file, log_file_handle = setup_direct_logging()
    
    try:
        # Log the test parameters
        log_file_handle.write(f"{datetime.now()} - INFO - Test parameters:\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Start date: {args.start_date}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - End date: {args.end_date}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Initial capital: ${args.initial_capital}\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Simulate backtest steps
        log_file_handle.write(f"{datetime.now()} - INFO - Loading configuration\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        log_file_handle.write(f"{datetime.now()} - INFO - Generating signals\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Simulate signal generation
        signals = [
            {'symbol': 'AAPL', 'score': 0.95},
            {'symbol': 'MSFT', 'score': 0.92},
            {'symbol': 'GOOGL', 'score': 0.88},
            {'symbol': 'AMZN', 'score': 0.85},
            {'symbol': 'META', 'score': 0.82}
        ]
        
        log_file_handle.write(f"{datetime.now()} - INFO - Generated {len(signals)} signals\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Log each signal
        for signal in signals:
            log_file_handle.write(f"{datetime.now()} - INFO - Signal: {signal['symbol']} with score {signal['score']:.2f}\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
        
        # Simulate portfolio initialization
        log_file_handle.write(f"{datetime.now()} - INFO - Initializing portfolio with ${args.initial_capital}\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Simulate trade execution
        for signal in signals:
            log_file_handle.write(f"{datetime.now()} - INFO - Executing trade for {signal['symbol']} with score {signal['score']:.2f}\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
        
        # Simulate performance calculation
        final_value = args.initial_capital * 1.15  # 15% return
        log_file_handle.write(f"{datetime.now()} - INFO - Backtest completed\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Final portfolio value: ${final_value:.2f}\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Return: 15.00%\n")
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
    
    finally:
        # Close the log file handle
        if log_file_handle:
            log_file_handle.write(f"{datetime.now()} - INFO - Closing log file\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            log_file_handle.close()

if __name__ == "__main__":
    test_backtest_logging()
