import logging
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f"test_logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Generate some log entries
logger.info("Starting test logging script")
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

# Log some information relevant to your LONG-only strategy
logger.info("LONG-only strategy initialized with win rate: 96.67%")
logger.info("Total trades: 120, Winning trades: 116, Losing trades: 4")
logger.info("Profit factor: 308.47")
logger.warning("Low number of signals generated for 2023-01-15")
logger.error("Failed to fetch data for symbol AAPL on 2023-01-20")

logger.info("Test logging complete")

print("Logging test complete. Check the logs directory for the new log file.")
