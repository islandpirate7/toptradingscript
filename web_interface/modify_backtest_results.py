#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper module to modify backtest results for different quarters
"""

import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def modify_results_for_quarter(result_data, quarter):
    """
    Add quarter information to backtest results without modifying the actual backtest data
    
    Args:
        result_data (dict): The backtest result data
        quarter (str): Quarter identifier (e.g., 'Q1_2023')
        
    Returns:
        dict: Modified backtest result data with quarter information
    """
    try:
        # Parse quarter to get quarter number and year
        parts = quarter.split('_')
        if len(parts) != 2 or not parts[0].startswith('Q') or not parts[1].isdigit():
            logger.warning(f"Invalid quarter format: {quarter}. Expected format: Q1_2023")
            return result_data
        
        # Make a copy of the result data
        modified_data = result_data.copy()
        
        # Add quarter info to the summary without modifying the actual backtest metrics
        if 'summary' in modified_data and modified_data['summary']:
            summary = modified_data['summary']
            
            # Add quarter info to the summary
            summary['quarter'] = quarter
        
        # Add quarter info to parameters
        if 'parameters' in modified_data:
            modified_data['parameters']['quarter'] = quarter
        else:
            modified_data['parameters'] = {
                'quarter': quarter
            }
        
        return modified_data
    
    except Exception as e:
        logger.error(f"Error modifying results for quarter {quarter}: {str(e)}")
        return result_data

def get_quarter_from_dates(start_date, end_date):
    """
    Get quarter identifier from date range
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        str: Quarter identifier (e.g., 'Q1_2023')
    """
    try:
        # Parse start date
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Extract year and quarter
        year = start_dt.year
        quarter = (start_dt.month - 1) // 3 + 1
        
        return f"Q{quarter}_{year}"
    
    except Exception as e:
        logger.error(f"Error getting quarter from dates: {str(e)}")
        return "Unknown"

def get_quarter_from_filename(filename):
    """
    Extract quarter from filename
    
    Args:
        filename (str): Filename (e.g., 'backtest_Q1_2023_20250324_123456.json')
        
    Returns:
        str: Quarter identifier (e.g., 'Q1_2023')
    """
    try:
        # Extract quarter from filename
        if 'backtest_Q' in filename:
            parts = filename.split('_')
            quarter_idx = -1
            
            for i, part in enumerate(parts):
                if part.startswith('Q'):
                    quarter_idx = i
                    break
            
            if quarter_idx >= 0 and quarter_idx + 1 < len(parts):
                return f"{parts[quarter_idx]}_{parts[quarter_idx+1]}"
        
        return "Unknown"
    
    except Exception as e:
        logger.error(f"Error extracting quarter from filename: {str(e)}")
        return "Unknown"
