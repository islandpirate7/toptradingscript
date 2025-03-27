#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Start Web Interface (Updated)
---------------------
Launcher for the S&P 500 Trading Strategy Web Interface (Updated Version)
"""

import os
import argparse
import logging
from datetime import datetime

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Start the S&P 500 Trading Strategy Web Interface')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web interface on')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the web interface on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Set up logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = os.path.join('logs', f"web_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting web interface on {args.host}:{args.port}")
    
    # Import the web interface module
    try:
        from web_interface.app_updated import app
        logger.info("Web interface module loaded successfully")
        
        # Start the web server
        logger.info(f"Starting web server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start web interface: {str(e)}")
        raise

if __name__ == "__main__":
    main()
