import re
import os
import sys

def fix_run_backtest_function():
    """
    This script fixes the run_backtest function in final_sp500_strategy.py:
    1. Improves the logging setup to ensure logs are properly written to disk
    2. Fixes variable reference issues (midcap_symbols -> midcap_signals)
    3. Adds explicit flush calls at key points
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the current file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix 1: Update the logging setup
    logging_setup_pattern = r'# Set up logging specifically for this backtest run.*?# Log the start of the backtest'
    improved_logging_setup = '''# Set up logging specifically for this backtest run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join('logs', f"strategy_{timestamp}.log")
        
        # Make sure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Reset the root logger completely
        root_logger = logging.getLogger()
        
        # Store existing handlers to restore later
        existing_handlers = root_logger.handlers.copy()
        original_level = root_logger.level
        
        # Remove all handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Log the start of the backtest'''
    
    content = re.sub(logging_setup_pattern, improved_logging_setup, content, flags=re.DOTALL)
    
    # Fix 2: Replace midcap_symbols with midcap_signals in the run_backtest function
    # We need to be careful to only replace within the run_backtest function
    run_backtest_pattern = r'def run_backtest\(.*?def '
    run_backtest_content = re.search(run_backtest_pattern, content, re.DOTALL).group(0)
    
    # Replace midcap_symbols with midcap_signals, but only in specific contexts
    # We don't want to replace variable declarations or function calls
    fixed_run_backtest = run_backtest_content
    
    # Replace in len() calls
    fixed_run_backtest = re.sub(r'len\(midcap_symbols\)', 'len(midcap_signals)', fixed_run_backtest)
    
    # Replace in sorting and slicing operations
    fixed_run_backtest = re.sub(r'midcap_symbols = sorted\(midcap_symbols', 'midcap_signals = sorted(midcap_signals', fixed_run_backtest)
    fixed_run_backtest = re.sub(r'selected_mid_cap = midcap_symbols\[', 'selected_mid_cap = midcap_signals[', fixed_run_backtest)
    
    # Update the content with the fixed run_backtest function
    content = content.replace(run_backtest_content, fixed_run_backtest)
    
    # Fix 3: Add explicit flush calls after important log messages
    # Find the pattern for the initial log messages
    log_messages_pattern = r'logger\.info\(f"Running backtest from {start_date} to {end_date} with initial capital \${initial_capital} \(Seed: {random_seed}\)"\)\s*\n'
    flush_code = '''        logger.info(f"Running backtest from {start_date} to {end_date} with initial capital ${initial_capital} (Seed: {random_seed})")
        
        # Force flush the logs to disk
        for handler in logging.getLogger().handlers:
            handler.flush()
'''
    content = re.sub(log_messages_pattern, flush_code, content)
    
    # Add flush before returning results
    return_pattern = r'# Restore original logging configuration.*?return {'
    flush_before_return = '''        # Force flush the logs to disk before returning
        for handler in logging.getLogger().handlers:
            handler.flush()
            
        # Restore original logging configuration'''
    content = re.sub(return_pattern, flush_before_return, content)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print("Successfully fixed the run_backtest function in final_sp500_strategy.py:")
    print("1. Improved logging setup to ensure logs are properly written to disk")
    print("2. Fixed variable reference issues (midcap_symbols -> midcap_signals)")
    print("3. Added explicit flush calls at key points")
    print("\nPlease run a backtest to verify that the logs are now being properly created and populated.")

if __name__ == "__main__":
    fix_run_backtest_function()
