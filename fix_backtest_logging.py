import os
import sys
import logging
from datetime import datetime

def fix_run_backtest():
    """
    This function contains the improved logging setup for the run_backtest function.
    Copy and paste this code into the run_backtest function to fix the logging issues.
    """
    # Set up logging specifically for this backtest run
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
    
    # Log the start of the backtest
    logger = logging.getLogger()
    logger.info(f"Starting backtest from [START_DATE] to [END_DATE]")
    logger.info(f"Backtest log file created: {log_file}")
    logger.info(f"Initial capital: $[INITIAL_CAPITAL]")
    logger.info(f"Backtest mode: [MODE]")
    logger.info(f"Random seed: [RANDOM_SEED]")
    logger.info(f"Continuous capital: [CONTINUOUS_CAPITAL]")
    
    # Force flush the logs to disk
    for handler in logger.handlers:
        handler.flush()
    
    # Important: Add these flush calls at key points in the run_backtest function
    # For example, after important log messages or before returning results
    
    # At the end of the function, restore original logging configuration
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)
    # 
    # for handler in existing_handlers:
    #     root_logger.addHandler(handler)
    # 
    # root_logger.setLevel(original_level)

print("Fix for run_backtest function created. Copy the code from fix_backtest_logging.py into the run_backtest function.")
print("Make sure to add flush calls at key points in the function, especially after important log messages.")
print("Replace the placeholder values like [START_DATE], [END_DATE], etc. with the actual variables.")
