import os
import sys
import logging
import time
from datetime import datetime

def test_standalone_logging():
    """
    A standalone test of the logging functionality that mimics the structure
    of the run_backtest function but is much simpler.
    """
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"standalone_{timestamp}.log")
    
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
    
    # Log some test messages
    logger.info("Starting standalone logging test")
    logger.info(f"Log file created: {log_file}")
    
    # Log some more messages with delays to ensure they're written
    for i in range(5):
        logger.info(f"Test message {i+1}")
        time.sleep(0.1)  # Small delay to ensure messages are processed
    
    # Force flush the logs to disk
    for handler in logger.handlers:
        handler.flush()
    
    # Explicitly close the file handler to ensure everything is written
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
    
    print(f"\nStandalone logging test complete. Check {log_file} for the log entries.")
    
    # Verify the log file has content
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            print(f"Log file size: {len(content)} bytes")
            if content:
                print("SUCCESS: Log file contains content")
            else:
                print("ERROR: Log file is empty (0 bytes)")
    except Exception as e:
        print(f"ERROR reading log file: {str(e)}")

if __name__ == "__main__":
    test_standalone_logging()
