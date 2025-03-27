#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launch Web Interface Debug

This script launches the web interface with debug logging
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/web_interface_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

def main():
    """Main function"""
    logger.info("Starting web interface with debug logging")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    sys.path.insert(0, script_dir)
    
    try:
        # Import the Flask app
        from new_web_interface.app_fixed import app
        
        # Run the app with debug logging
        logger.info("Running web interface on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Error running web interface: {str(e)}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    main()
