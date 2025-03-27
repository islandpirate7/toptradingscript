#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for running a comprehensive backtest directly
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/test_comprehensive_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run a test of the comprehensive backtest script"""
    try:
        # Define the command to run
        cmd = [
            sys.executable,
            'run_comprehensive_backtest.py',
            'Q1_2023',  # Quarter to test
            '--max_signals', '40',
            '--initial_capital', '300',
            '--multiple_runs',
            '--num_runs', '2',  # Use a small number for quick testing
            '--weekly_selection'
        ]
        
        # Log the command we're about to run
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        logger.info("Backtest output:")
        for line in process.stdout:
            print(line.strip())
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info("Comprehensive backtest completed successfully")
        else:
            logger.error(f"Comprehensive backtest failed with return code {process.returncode}")
            
    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
