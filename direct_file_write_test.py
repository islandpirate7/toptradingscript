import os
import time
from datetime import datetime

def test_direct_file_write():
    """
    Test direct file writing without using the logging module
    to isolate the issue.
    """
    # Create a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a log file path
    log_file = os.path.join('logs', f"direct_write_{timestamp}.log")
    
    # Write directly to the file
    with open(log_file, 'w') as f:
        f.write(f"{datetime.now()} - INFO - Starting direct file write test\n")
        f.write(f"{datetime.now()} - INFO - This is a test message\n")
        f.flush()  # Explicitly flush the file buffer
        os.fsync(f.fileno())  # Force the OS to write to disk
        
        # Write more messages with delays
        for i in range(5):
            f.write(f"{datetime.now()} - INFO - Test message {i+1}\n")
            f.flush()
            os.fsync(f.fileno())
            time.sleep(0.1)
    
    # Verify the file was created and has content
    try:
        file_size = os.path.getsize(log_file)
        print(f"Log file created: {log_file}")
        print(f"Log file size: {file_size} bytes")
        
        if file_size > 0:
            print("SUCCESS: Log file contains content")
        else:
            print("ERROR: Log file is empty (0 bytes)")
    except Exception as e:
        print(f"ERROR: {str(e)}")
    
    # Now let's create a modified version of the run_backtest function
    # that uses direct file writing instead of the logging module
    create_modified_run_backtest()

def create_modified_run_backtest():
    """
    Create a modified version of the run_backtest function that uses
    direct file writing instead of the logging module.
    """
    # Create the modified function in a new file
    with open('direct_write_run_backtest.py', 'w') as f:
        f.write("""
import os
import sys
import time
from datetime import datetime

def direct_write_run_backtest(start_date, end_date, mode='backtest', initial_capital=10000, 
                              random_seed=None, continuous_capital=False, previous_capital=None,
                              config_path='sp500_config.yaml'):
    \"\"\"
    A modified version of run_backtest that uses direct file writing
    instead of the logging module.
    \"\"\"
    # Create a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a log file path
    log_file = os.path.join('logs', f"direct_write_backtest_{timestamp}.log")
    
    # Open the log file for writing
    log_file_handle = open(log_file, 'w')
    
    try:
        # Write initial log messages
        log_file_handle.write(f"{datetime.now()} - INFO - Starting backtest from {start_date} to {end_date}\\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Backtest log file created: {log_file}\\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Initial capital: ${initial_capital}\\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Backtest mode: {mode}\\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Random seed: {random_seed}\\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Continuous capital: {continuous_capital}\\n")
        
        if continuous_capital and previous_capital is not None:
            log_file_handle.write(f"{datetime.now()} - INFO - Previous capital: ${previous_capital}\\n")
        
        log_file_handle.write(f"{datetime.now()} - INFO - Running backtest from {start_date} to {end_date} with initial capital ${initial_capital} (Seed: {random_seed})\\n")
        
        # Force flush to disk
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Simulate some backtest operations
        log_file_handle.write(f"{datetime.now()} - INFO - Simulating backtest operations...\\n")
        for i in range(5):
            log_file_handle.write(f"{datetime.now()} - INFO - Backtest operation {i+1}\\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
            time.sleep(0.1)
        
        # Finish up
        log_file_handle.write(f"{datetime.now()} - INFO - Backtest completed\\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        print(f"Direct write backtest completed. Check {log_file} for the log entries.")
        
        # Return a dummy result
        return {'success': True, 'log_file': log_file}
    
    finally:
        # Make sure to close the file handle
        log_file_handle.close()

if __name__ == "__main__":
    # Run a test backtest
    direct_write_run_backtest('2023-01-01', '2023-01-15', initial_capital=300)
""")
    
    print("\nCreated direct_write_run_backtest.py")
    print("This file contains a modified version of run_backtest that uses direct file writing")
    print("Run it with: python direct_write_run_backtest.py")

if __name__ == "__main__":
    test_direct_file_write()
