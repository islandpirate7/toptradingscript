#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launch Updated Web Interface
---------------------------
This script launches the updated web interface that uses the new backtest engine.
"""

import os
import sys
import logging
from datetime import datetime

def main():
    # Set up logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = os.path.join('logs', f"updated_web_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting updated web interface")
    
    try:
        # Import the updated web interface
        from web_interface.app_updated import app
        
        # Start the web server
        logger.info("Starting web server on 127.0.0.1:5000")
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start updated web interface: {str(e)}")
        raise

if __name__ == "__main__":
    main()
