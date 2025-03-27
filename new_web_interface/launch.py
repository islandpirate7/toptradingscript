#!/usr/bin/env python
"""
Launcher script for the S&P 500 Trading Strategy Web Interface
"""

import os
import sys
import logging
from app import app

# Setup logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, 'web_interface.log')
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == '__main__':
    # Setup logging
    logger = setup_logging()
    
    try:
        # Create backtest results directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), 'backtest_results'), exist_ok=True)
        
        # Create static and templates directories if they don't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
        
        # Log startup information
        logger.info("Starting S&P 500 Trading Strategy Web Interface")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Run the app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error starting web interface: {str(e)}", exc_info=True)
        sys.exit(1)
