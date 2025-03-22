#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Seasonal Opportunities Converter
-------------------------------------
This script converts seasonal opportunities with NumPy values to standard Python types
for better compatibility with YAML and JSON serialization.
"""

import os
import logging
import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_numpy_to_python(obj: Any) -> Any:
    """Convert NumPy types to Python native types
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    else:
        return obj

def convert_seasonal_opportunities(input_file: str, output_file: str) -> None:
    """Convert seasonal opportunities file to use Python native types
    
    Args:
        input_file (str): Path to input seasonal opportunities file
        output_file (str): Path to output seasonal opportunities file
    """
    try:
        # Read the input file as text
        with open(input_file, 'r') as f:
            file_content = f.read()
            
        # Parse the data manually to extract the opportunities
        import re
        
        # Find all opportunities in the file
        opportunity_blocks = re.findall(r'- .*?(?=- |$)', file_content, re.DOTALL)
        
        opportunities = []
        
        for block in opportunity_blocks:
            # Extract key information from each opportunity block
            symbol_match = re.search(r'symbol: ([A-Z]+)', block)
            season_match = re.search(r'season: ([A-Za-z]+)', block)
            direction_match = re.search(r'direction: ([A-Z]+)', block)
            win_rate_match = re.search(r'win_rate: ([-\d\.]+)', block)
            avg_return_match = re.search(r'avg_return: ([-\d\.]+)', block)
            correlation_match = re.search(r'correlation: ([-\d\.]+)', block)
            current_price_match = re.search(r'current_price: ([-\d\.]+)', block)
            expected_return_match = re.search(r'expected_return: ([-\d\.]+)', block)
            trade_count_match = re.search(r'trade_count: ([-\d\.]+)', block)
            
            if symbol_match and season_match:
                opportunity = {
                    'symbol': symbol_match.group(1),
                    'season': season_match.group(1),
                    'direction': direction_match.group(1) if direction_match else 'LONG',
                    'win_rate': float(win_rate_match.group(1)) if win_rate_match else 0.0,
                    'avg_return': float(avg_return_match.group(1)) if avg_return_match else 0.0,
                    'correlation': float(correlation_match.group(1)) if correlation_match else 0.0
                }
                
                # Add optional fields if present
                if current_price_match:
                    opportunity['current_price'] = float(current_price_match.group(1))
                if expected_return_match:
                    opportunity['expected_return'] = float(expected_return_match.group(1))
                if trade_count_match:
                    opportunity['trade_count'] = int(float(trade_count_match.group(1)))
                
                opportunities.append(opportunity)
        
        # Create output data structure
        output_data = {
            'opportunities': opportunities
        }
        
        # Write to output file
        with open(output_file, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False)
            
        logging.info(f"Converted {len(opportunities)} opportunities to {output_file}")
        
    except Exception as e:
        logging.error(f"Error converting seasonal opportunities: {e}")

def main():
    """Main function to convert seasonal opportunities"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert seasonal opportunities to use Python native types')
    parser.add_argument('--input', type=str, default='output/seasonal_opportunities.yaml',
                      help='Path to input seasonal opportunities file')
    parser.add_argument('--output', type=str, default='output/seasonal_opportunities_converted.yaml',
                      help='Path to output seasonal opportunities file')
    args = parser.parse_args()
    
    # Convert seasonal opportunities
    convert_seasonal_opportunities(args.input, args.output)
    
    logging.info(f"Seasonal opportunities conversion completed successfully")

if __name__ == "__main__":
    main()
