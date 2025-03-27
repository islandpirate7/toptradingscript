#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix combined backtest result files
"""

import os
import json
import glob
import logging
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_combined_results():
    """
    Fix combined backtest result files by combining individual quarter results
    """
    # Define the directory where backtest result files are stored
    results_dir = os.path.join('web_interface', 'backtest_results')
    
    # Get all combined JSON files
    combined_files = glob.glob(os.path.join(results_dir, 'backtest_combined_*.json'))
    
    # Counter for fixed files
    fixed_count = 0
    
    for combined_file in combined_files:
        try:
            # Extract timestamp from filename
            timestamp_match = re.search(r'(\d{8}_\d{6})', combined_file)
            if not timestamp_match:
                logger.warning(f"Could not extract timestamp from {combined_file}")
                continue
                
            timestamp = timestamp_match.group(1)
            
            # Find all quarter files with the same timestamp
            quarter_files = glob.glob(os.path.join(results_dir, f'backtest_Q*_{timestamp}.json'))
            
            if not quarter_files:
                logger.warning(f"No quarter files found for {combined_file}")
                continue
                
            # Create a combined results dictionary
            combined_results = {}
            
            for quarter_file in quarter_files:
                try:
                    # Extract quarter from filename
                    quarter_match = re.search(r'backtest_(Q\d_\d{4})_', quarter_file)
                    if not quarter_match:
                        logger.warning(f"Could not extract quarter from {quarter_file}")
                        continue
                        
                    quarter = quarter_match.group(1)
                    
                    # Read the quarter file
                    with open(quarter_file, 'r') as f:
                        quarter_data = json.load(f)
                    
                    # Add to combined results
                    combined_results[quarter] = quarter_data
                    
                except Exception as e:
                    logger.error(f"Error processing quarter file {quarter_file}: {str(e)}")
            
            # Save the combined results
            with open(combined_file, 'w') as f:
                json.dump(combined_results, f, indent=4, default=str)
            
            logger.info(f"Fixed combined result file: {combined_file}")
            fixed_count += 1
            
        except Exception as e:
            logger.error(f"Error fixing combined result file {combined_file}: {str(e)}")
    
    logger.info(f"Fixed {fixed_count} combined result files")
    return fixed_count

if __name__ == "__main__":
    fix_combined_results()
