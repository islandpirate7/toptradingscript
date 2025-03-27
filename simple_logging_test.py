import logging
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('logs', f"simple_test_{timestamp}.log")

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
logger.info("Starting simple logging test")
logger.info(f"Log file created: {log_file}")
logger.warning("This is a warning message")
logger.error("This is an error message")

# Force flush the logs to disk
for handler in logger.handlers:
    handler.flush()

print(f"Simple logging test complete. Check {log_file} for the log entries.")
