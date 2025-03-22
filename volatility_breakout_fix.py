#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VolatilityBreakout Strategy Fix
------------------------------
This script patches the VolatilityBreakout strategy to fix the premature return issue
and improves signal generation.
"""

import sys
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import datetime as dt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('volatility_breakout_fix.log')
    ]
)

logger = logging.getLogger("VolatilityBreakoutFix")

def patch_volatility_breakout_strategy():
    """
    Patch the VolatilityBreakout strategy in the multi_strategy_system.py file
    to fix the premature return issue.
    """
    try:
        # Read the multi_strategy_system.py file
        with open('multi_strategy_system.py', 'r') as file:
            lines = file.readlines()
        
        # Find the line with the premature return
        target_line_index = None
        for i, line in enumerate(lines):
            if "return signals" in line and "# Bearish breakout" in lines[i-45:i]:
                target_line_index = i
                break
        
        if target_line_index is None:
            logger.error("Could not find the line with the premature return")
            return False
        
        # Comment out the premature return
        lines[target_line_index] = "                        # " + lines[target_line_index].lstrip()
        
        # Write the modified file
        with open('multi_strategy_system.py', 'w') as file:
            file.writelines(lines)
        
        logger.info("Successfully patched the VolatilityBreakout strategy")
        return True
    
    except Exception as e:
        logger.error(f"Error patching VolatilityBreakout strategy: {str(e)}")
        return False

def update_compare_models_script():
    """
    Update the compare_models.py script to use the further_optimized_config.yaml file.
    """
    try:
        # Read the compare_models.py file
        with open('compare_models.py', 'r') as file:
            lines = file.readlines()
        
        # Find the line with the optimized_config.yaml
        target_line_index = None
        for i, line in enumerate(lines):
            if "optimized_config.yaml" in line:
                target_line_index = i
                break
        
        if target_line_index is None:
            logger.error("Could not find the line with optimized_config.yaml")
            return False
        
        # Replace optimized_config.yaml with further_optimized_config.yaml
        lines[target_line_index] = lines[target_line_index].replace(
            'optimized_config.yaml', 'further_optimized_config.yaml'
        )
        
        # Write the modified file
        with open('compare_models.py', 'w') as file:
            file.writelines(lines)
        
        logger.info("Successfully updated the compare_models.py script")
        return True
    
    except Exception as e:
        logger.error(f"Error updating compare_models.py: {str(e)}")
        return False

def main():
    """Main function to apply fixes"""
    logger.info("Starting VolatilityBreakout strategy fix")
    
    # Patch the VolatilityBreakout strategy
    if patch_volatility_breakout_strategy():
        logger.info("VolatilityBreakout strategy patched successfully")
    else:
        logger.error("Failed to patch VolatilityBreakout strategy")
    
    # Update the compare_models.py script
    if update_compare_models_script():
        logger.info("compare_models.py updated successfully")
    else:
        logger.error("Failed to update compare_models.py")
    
    logger.info("VolatilityBreakout strategy fix completed")

if __name__ == "__main__":
    main()
