import logging
import os
import yaml
from datetime import datetime

def test_backtest_logging():
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"strategy_{timestamp}.log")
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Get the logger
    logger = logging.getLogger()
    
    # Log some test messages
    logger.info("Starting backtest logging test")
    logger.info(f"Log file created: {log_file}")
    logger.info("Initial capital: $300")
    logger.info("Backtest mode: backtest")
    
    # Simulate some LONG-only strategy logs
    logger.info("LONG-only strategy initialized with win rate: 96.67%")
    logger.info("Total trades: 120, Winning trades: 116, Losing trades: 4")
    logger.info("Profit factor: 308.47")
    
    # Simulate mid-cap integration logs
    logger.info("Mid-cap symbols acquired and integrated")
    logger.info("Applied dynamic boost to mid-cap stocks based on configuration")
    logger.info("Balanced signal distribution achieved")
    
    # Force flush the logs to disk
    for handler in logger.handlers:
        handler.flush()
    
    print(f"Backtest logging test complete. Check {log_file} for the log entries.")

if __name__ == "__main__":
    test_backtest_logging()
