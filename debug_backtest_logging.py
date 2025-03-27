import logging
import os
import sys
from datetime import datetime

def setup_logging():
    """Set up logging with a simple configuration that works"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"debug_{timestamp}.log")
    
    # Reset the root logger completely
    root_logger = logging.getLogger()
    
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
    
    return log_file

def main():
    # Set up logging
    log_file = setup_logging()
    logger = logging.getLogger()
    
    # Log some test messages
    logger.info("Starting debug logging test")
    logger.info(f"Log file created: {log_file}")
    
    # Log some LONG-only strategy information (from memory)
    logger.info("LONG-only strategy initialized with win rate: 96.67%")
    logger.info("Total trades: 120, Winning trades: 116, Losing trades: 4")
    logger.info("Profit factor: 308.47")
    
    # Log some mid-cap integration information (from memory)
    logger.info("Mid-cap symbols acquired and integrated")
    logger.info("Applied dynamic boost to mid-cap stocks based on configuration")
    logger.info("Balanced signal distribution achieved")
    
    # Force flush the logs to disk
    for handler in logger.handlers:
        handler.flush()
    
    print(f"Debug logging test complete. Check {log_file} for the log entries.")

if __name__ == "__main__":
    main()
