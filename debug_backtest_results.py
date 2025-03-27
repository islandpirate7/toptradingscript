#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to debug backtest results display in the web interface
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

def debug_backtest_results():
    """
    Debug backtest results display in the web interface
    """
    # Define the directories where backtest result files are stored
    results_dirs = [
        os.path.join('web_interface', 'backtest_results'),  # Web interface directory
        os.path.join('backtest_results'),  # Root directory
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'web_interface', 'backtest_results')),  # Absolute path to web interface directory
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'backtest_results'))  # Absolute path to root directory
    ]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            logger.info(f"Checking backtest results in: {results_dir}")
            
            # Count files by type
            json_files = glob.glob(os.path.join(results_dir, '*.json'))
            csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
            
            logger.info(f"Found {len(json_files)} JSON files and {len(csv_files)} CSV files in {results_dir}")
            
            # Check for specific file patterns
            combined_files = glob.glob(os.path.join(results_dir, 'backtest_combined_*.json'))
            quarter_files = glob.glob(os.path.join(results_dir, 'backtest_Q*_*.json'))
            
            logger.info(f"Found {len(combined_files)} combined files and {len(quarter_files)} quarter files in {results_dir}")
            
            # Check file permissions
            for file_path in json_files[:5]:  # Check first 5 files
                try:
                    readable = os.access(file_path, os.R_OK)
                    writable = os.access(file_path, os.W_OK)
                    logger.info(f"File {os.path.basename(file_path)}: Readable={readable}, Writable={writable}")
                    
                    # Try to read the file
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Check for expected keys
                            if 'summary' in data:
                                logger.info(f"File {os.path.basename(file_path)} has summary key")
                            elif any(key.startswith('Q') for key in data.keys()):
                                logger.info(f"File {os.path.basename(file_path)} has quarter keys: {list(data.keys())}")
                            else:
                                logger.warning(f"File {os.path.basename(file_path)} does not have expected keys: {list(data.keys())}")
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error checking permissions for {file_path}: {str(e)}")
        else:
            logger.warning(f"Directory {results_dir} does not exist")
    
    # Now let's simulate what the web interface does to find backtest results
    logger.info("Simulating web interface get_backtest_results function...")
    
    results = []
    
    # Check both relative and absolute paths to ensure we find the files
    results_dirs = [
        os.path.join('..', 'backtest_results'),  # Relative path
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backtest_results'))  # Absolute path
    ]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            logger.info(f"Checking backtest results in: {results_dir}")
            files = os.listdir(results_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
            
            # Group files by quarter/date range to avoid duplicates
            quarter_groups = {}
            
            for file in files:
                file_path = os.path.join(results_dir, file)
                if os.path.isfile(file_path):
                    # Extract quarter or date range from filename
                    quarter_key = None
                    
                    # Try to identify quarter from filename
                    if "_Q" in file:
                        for part in file.split('_'):
                            if part.startswith('Q') and len(part) <= 3:
                                year_part = next((p for p in file.split('_') if p.startswith('20') and len(p) == 4), "")
                                if year_part:
                                    quarter_key = f"{part}_{year_part}"
                                    break
                    
                    # Try to identify by date range
                    if not quarter_key:
                        date_match = re.findall(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', file)
                        if date_match:
                            start_date, end_date = date_match[0]
                            quarter_key = f"{start_date}_to_{end_date}"
                            
                            # Try to extract quarter info if available
                            quarter_match = re.search(r'Q\d_\d{4}', file)
                            if quarter_match:
                                quarter_key = quarter_match.group(0)
                    
                    # If we couldn't identify a quarter, use the filename
                    if not quarter_key:
                        quarter_key = file
                    
                    # Only add if this is a new quarter or a newer file for the same quarter
                    if quarter_key not in quarter_groups:
                        quarter_groups[quarter_key] = []
                    
                    # Check if it's a JSON file (our backtest results are in JSON format)
                    if file.endswith('.json'):
                        try:
                            with open(file_path, 'r') as f:
                                file_data = json.load(f)
                                # Verify this is a backtest result file by checking for expected keys
                                if 'summary' in file_data:
                                    quarter_groups[quarter_key].append({
                                        'name': file,
                                        'path': file_path,
                                        'date': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
                                        'quarter_key': quarter_key
                                    })
                                    logger.info(f"Found valid backtest result: {file}")
                        except Exception as e:
                            logger.warning(f"Error reading backtest result file {file}: {str(e)}")
            
            # Take only the most recent file from each quarter group
            for quarter, files in quarter_groups.items():
                # Sort by date (newest first)
                files.sort(key=lambda x: x['date'], reverse=True)
                # Add only the most recent file for each quarter
                if files:
                    results.append(files[0])
    
    # Sort all results by date (newest first)
    results.sort(key=lambda x: x['date'], reverse=True)
    logger.info(f"Found {len(results)} backtest results")
    
    # Print the first few results
    for i, result in enumerate(results[:5]):
        logger.info(f"Result {i+1}: {result['name']} ({result['date']})")
    
    return results

if __name__ == "__main__":
    debug_backtest_results()
