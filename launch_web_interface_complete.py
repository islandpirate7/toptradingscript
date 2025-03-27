#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launch script for the fixed web interface
This script will:
1. Set up proper logging
2. Import and run the fixed web interface
3. Run on the standard port 5000
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"web_interface_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to launch the web interface"""
    try:
        logger.info("Starting fixed web interface")
        
        # Add the parent directory to the path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        
        # Import the fixed web interface
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        sys.path.append(web_interface_dir)
        
        # Check if app_fixed.py exists
        app_fixed_path = os.path.join(web_interface_dir, 'app_fixed.py')
        if not os.path.exists(app_fixed_path):
            logger.error(f"Fixed web interface not found: {app_fixed_path}")
            logger.info("Please run fix_web_interface_complete.py first")
            return
        
        # Import the app from app_fixed.py
        sys.path.insert(0, os.path.dirname(app_fixed_path))
        from app_fixed import app
        
        # Run the app
        logger.info("Running web interface on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    except Exception as e:
        logger.error(f"Error launching web interface: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
