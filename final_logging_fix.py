import os
import sys
import re

def apply_final_logging_fix():
    """
    Apply the final logging fix to the run_backtest function based on the 
    successful standalone logging test.
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the current file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Make sure sys is imported
    if "import sys" not in content:
        content = content.replace("import random", "import random\nimport sys")
    
    # Fix 1: Update the logging setup
    logging_setup_pattern = r'# Set up logging specifically for this backtest run.*?# Log the start of the backtest'
    improved_logging_setup = '''# Set up logging specifically for this backtest run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join('logs', f"strategy_{timestamp}.log")
        
        # Make sure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Get the logger
        logger = logging.getLogger()
        
        # Log the start of the backtest'''
    
    content = re.sub(logging_setup_pattern, improved_logging_setup, content, flags=re.DOTALL)
    
    # Fix 2: Add explicit flush calls after important log messages
    # Find the pattern for the initial log messages
    log_messages_pattern = r'logger\.info\(f"Running backtest from {start_date} to {end_date} with initial capital \${initial_capital} \(Seed: {random_seed}\)"\)\s*\n'
    flush_code = '''        logger.info(f"Running backtest from {start_date} to {end_date} with initial capital ${initial_capital} (Seed: {random_seed})")
        
        # Force flush the logs to disk
        for handler in logger.handlers:
            handler.flush()
'''
    content = re.sub(log_messages_pattern, flush_code, content)
    
    # Fix 3: Add explicit close and flush before returning
    return_pattern = r'# Restore original logging configuration.*?for handler in root_logger\.handlers\[:\]:'
    flush_before_return = '''        # Force flush and close the logs before returning
        for handler in logger.handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()
        
        # Restore original logging configuration
        for handler in root_logger.handlers[:]:'''
    
    content = re.sub(return_pattern, flush_before_return, content)
    
    # Fix 4: Replace midcap_symbols with midcap_signals in specific contexts
    # We need to be careful to only replace within the run_backtest function
    run_backtest_pattern = r'def run_backtest\(.*?def '
    run_backtest_content = re.search(run_backtest_pattern, content, re.DOTALL).group(0)
    
    # Replace in specific contexts
    fixed_run_backtest = run_backtest_content
    fixed_run_backtest = re.sub(r'len\(midcap_symbols\)', 'len(midcap_signals)', fixed_run_backtest)
    fixed_run_backtest = re.sub(r'midcap_symbols = sorted\(midcap_symbols', 'midcap_signals = sorted(midcap_signals', fixed_run_backtest)
    fixed_run_backtest = re.sub(r'selected_mid_cap = midcap_symbols\[', 'selected_mid_cap = midcap_signals[', fixed_run_backtest)
    fixed_run_backtest = re.sub(r'logger\.info\(f"Using all {len\(signals\)} signals \({len\(largecap_signals\)} large-cap, {len\(midcap_symbols\)} mid-cap\)"\)', 
                              'logger.info(f"Using all {len(signals)} signals ({len(largecap_signals)} large-cap, {len(midcap_signals)} mid-cap)")', fixed_run_backtest)
    
    # Update the content with the fixed run_backtest function
    content = content.replace(run_backtest_content, fixed_run_backtest)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print("Successfully applied the final logging fix to run_backtest function:")
    print("1. Improved logging setup using the successful standalone approach")
    print("2. Added explicit flush calls after important log messages")
    print("3. Added explicit close and flush before returning")
    print("4. Fixed variable reference issues (midcap_symbols -> midcap_signals)")
    print("\nPlease run a backtest to verify that the logs are now being properly created and populated.")

if __name__ == "__main__":
    apply_final_logging_fix()
