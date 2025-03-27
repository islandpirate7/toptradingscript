import os
import re

def apply_direct_write_fix():
    """
    Apply a fix to the run_backtest function that uses direct file writing
    instead of the logging module to ensure logs are properly created and populated.
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the current file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Make sure the necessary imports are present
    if "import time" not in content:
        content = content.replace("import random", "import random\nimport time")
    
    # Find the run_backtest function
    run_backtest_pattern = r'def run_backtest\(.*?def '
    run_backtest_match = re.search(run_backtest_pattern, content, re.DOTALL)
    
    if not run_backtest_match:
        print("ERROR: Could not find run_backtest function")
        return
    
    run_backtest_content = run_backtest_match.group(0)
    
    # Replace the logging setup with direct file writing
    logging_setup_pattern = r'# Set up logging specifically for this backtest run.*?# Log the start of the backtest'
    direct_write_setup = '''# Set up direct file writing for this backtest run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join('logs', f"strategy_{timestamp}.log")
        
        # Make sure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Open the log file for writing
        log_file_handle = open(log_file, 'w')
        
        # Store the original logging configuration
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        
        # Log the start of the backtest'''
    
    modified_run_backtest = re.sub(logging_setup_pattern, direct_write_setup, run_backtest_content, flags=re.DOTALL)
    
    # Replace logging calls with direct file writes
    log_patterns = [
        (r'logger\.info\(f"Starting backtest from {start_date} to {end_date}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Starting backtest from {start_date} to {end_date}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Backtest log file created: {log_file}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Backtest log file created: {log_file}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Initial capital: \${initial_capital}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Initial capital: ${initial_capital}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Backtest mode: {mode}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Backtest mode: {mode}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Random seed: {random_seed}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Random seed: {random_seed}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Continuous capital: {continuous_capital}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Continuous capital: {continuous_capital}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Previous capital: \${previous_capital}"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Previous capital: ${previous_capital}\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())'),
        (r'logger\.info\(f"Running backtest from {start_date} to {end_date} with initial capital \${initial_capital} \(Seed: {random_seed}\)"\)', 
         r'log_file_handle.write(f"{datetime.now()} - INFO - Running backtest from {start_date} to {end_date} with initial capital ${initial_capital} (Seed: {random_seed})\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())')
    ]
    
    for pattern, replacement in log_patterns:
        modified_run_backtest = re.sub(pattern, replacement, modified_run_backtest)
    
    # Replace other logger.info calls with direct file writes
    modified_run_backtest = re.sub(
        r'logger\.info\((.*?)\)', 
        r'log_file_handle.write(f"{datetime.now()} - INFO - " + str(\1) + "\\n")\n        log_file_handle.flush()\n        os.fsync(log_file_handle.fileno())', 
        modified_run_backtest
    )
    
    # Replace the cleanup code before returning
    cleanup_pattern = r'# Restore original logging configuration.*?return results'
    direct_write_cleanup = '''# Close the log file handle
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        log_file_handle.close()
        
        # Restore original logging configuration
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        for handler in original_handlers:
            root_logger.addHandler(handler)
        
        root_logger.setLevel(original_level)
        
        return results'''
    
    modified_run_backtest = re.sub(cleanup_pattern, direct_write_cleanup, modified_run_backtest, flags=re.DOTALL)
    
    # Fix variable references from midcap_symbols to midcap_signals
    modified_run_backtest = modified_run_backtest.replace("len(midcap_symbols)", "len(midcap_signals)")
    modified_run_backtest = modified_run_backtest.replace("midcap_symbols = sorted(", "midcap_signals = sorted(")
    modified_run_backtest = modified_run_backtest.replace("selected_mid_cap = midcap_symbols[", "selected_mid_cap = midcap_signals[")
    
    # Update the content with the modified run_backtest function
    content = content.replace(run_backtest_content, modified_run_backtest)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print("Successfully applied the direct write fix to run_backtest function:")
    print("1. Replaced logging setup with direct file writing")
    print("2. Replaced logging calls with direct file writes")
    print("3. Added proper flush and close calls")
    print("4. Fixed variable reference issues (midcap_symbols -> midcap_signals)")
    print("\nPlease run a backtest to verify that the logs are now being properly created and populated.")

if __name__ == "__main__":
    apply_direct_write_fix()
