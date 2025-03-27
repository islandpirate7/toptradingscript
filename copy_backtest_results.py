#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to copy backtest results from the root directory to the web interface directory
"""

import os
import shutil
import glob
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_backtest_results():
    """
    Copy backtest results from the root directory to the web interface directory
    """
    # Define the source and destination directories
    source_dir = os.path.join('backtest_results')
    dest_dir = os.path.join('web_interface', 'backtest_results')
    
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all JSON files in the source directory
    json_files = glob.glob(os.path.join(source_dir, '*.json'))
    
    # Counter for copied files
    copied_count = 0
    
    for file_path in json_files:
        try:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # Skip if the file already exists in the destination
            if os.path.exists(dest_path):
                continue
                
            # Check if this is a valid backtest result file
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Only copy files that have a summary or are combined results
                if 'summary' in data or 'Q1_2023' in data or 'Q2_2023' in data:
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"Copied {filename} to web interface directory")
                    copied_count += 1
            except json.JSONDecodeError:
                logger.warning(f"Skipping {filename} - not a valid JSON file")
                continue
            except Exception as e:
                logger.warning(f"Error reading {filename}: {str(e)}")
                continue
            
        except Exception as e:
            logger.error(f"Error copying {file_path}: {str(e)}")
    
    logger.info(f"Copied {copied_count} backtest result files to web interface directory")
    return copied_count

if __name__ == "__main__":
    copy_backtest_results()
