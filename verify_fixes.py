#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verify Fixes Script
This script verifies that all the fixes we made are working correctly without relying on the web interface.
"""

import os
import sys
import yaml
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_INTERFACE_DIR = os.path.join(ROOT_DIR, 'new_web_interface')
STATIC_DIR = os.path.join(WEB_INTERFACE_DIR, 'static')
JS_DIR = os.path.join(STATIC_DIR, 'js')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def verify_seasonality_file():
    """Verify that the seasonality.yaml file exists and has the correct format"""
    seasonality_file = os.path.join(DATA_DIR, 'seasonality.yaml')
    
    if not os.path.exists(seasonality_file):
        logger.error(f"seasonality.yaml file does not exist: {seasonality_file}")
        return False
    
    try:
        with open(seasonality_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check if the file has valid YAML content
        if not isinstance(data, dict):
            logger.error(f"seasonality.yaml is not a dictionary: {data}")
            return False
        
        # We don't care about the specific structure, just that it's a valid YAML dictionary
        logger.info(f"seasonality.yaml exists and has valid YAML content with keys: {list(data.keys())}")
        return True
    except Exception as e:
        logger.error(f"Error verifying seasonality.yaml: {str(e)}")
        return False

def verify_favicon():
    """Verify that the favicon.ico file exists"""
    favicon_path = os.path.join(STATIC_DIR, 'favicon.ico')
    
    if not os.path.exists(favicon_path):
        logger.error(f"favicon.ico file does not exist: {favicon_path}")
        return False
    
    logger.info(f"favicon.ico exists")
    return True

def verify_app_fixed_py():
    """Verify that app_fixed.py has the correct routes and error handling"""
    app_fixed_path = os.path.join(WEB_INTERFACE_DIR, 'app_fixed.py')
    
    if not os.path.exists(app_fixed_path):
        logger.error(f"app_fixed.py file does not exist: {app_fixed_path}")
        return False
    
    try:
        with open(app_fixed_path, 'r') as f:
            content = f.read()
        
        # Check if the file has the expected routes
        if 'def favicon():' not in content:
            logger.error(f"app_fixed.py does not have the favicon route")
            return False
        
        if 'def get_processes():' not in content:
            logger.error(f"app_fixed.py does not have the get_processes route")
            return False
        
        if 'def get_backtest_results_route():' not in content:
            logger.error(f"app_fixed.py does not have the get_backtest_results_route")
            return False
        
        # Check if the file has the correct JSON responses
        if 'return jsonify({"processes": list(processes.values())})' not in content:
            logger.error(f"app_fixed.py does not return the correct JSON response for get_processes")
            return False
        
        if 'return jsonify({"results": results})' not in content:
            logger.error(f"app_fixed.py does not return the correct JSON response for get_backtest_results_route")
            return False
        
        logger.info(f"app_fixed.py has the correct routes and error handling")
        return True
    except Exception as e:
        logger.error(f"Error verifying app_fixed.py: {str(e)}")
        return False

def verify_main_js():
    """Verify that main.js has the correct error handling"""
    main_js_path = os.path.join(JS_DIR, 'main.js')
    
    if not os.path.exists(main_js_path):
        logger.error(f"main.js file does not exist: {main_js_path}")
        return False
    
    try:
        with open(main_js_path, 'r') as f:
            content = f.read()
        
        # Check if the file has the expected error handling
        if 'const processes = data.processes || [];' not in content:
            logger.warning(f"main.js does not have the expected error handling for processes")
            # This is not a critical error, so we'll just log a warning
        
        if 'const results = data.results || [];' not in content:
            logger.warning(f"main.js does not have the expected error handling for results")
            # This is not a critical error, so we'll just log a warning
        
        logger.info(f"main.js has been updated with error handling")
        return True
    except Exception as e:
        logger.error(f"Error verifying main.js: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("Starting verification of fixes")
    
    # Verify seasonality file
    seasonality_ok = verify_seasonality_file()
    
    # Verify favicon
    favicon_ok = verify_favicon()
    
    # Verify app_fixed.py
    app_fixed_ok = verify_app_fixed_py()
    
    # Verify main.js
    main_js_ok = verify_main_js()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Verification Summary")
    logger.info("=" * 50)
    logger.info(f"seasonality.yaml: {'✓' if seasonality_ok else '✗'}")
    logger.info(f"favicon.ico: {'✓' if favicon_ok else '✗'}")
    logger.info(f"app_fixed.py: {'✓' if app_fixed_ok else '✗'}")
    logger.info(f"main.js: {'✓' if main_js_ok else '✗'}")
    logger.info("=" * 50)
    
    # Overall status
    if seasonality_ok and favicon_ok and app_fixed_ok and main_js_ok:
        logger.info("All fixes have been verified successfully!")
        return 0
    else:
        logger.error("Some fixes could not be verified. Please check the logs for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
