#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix indentation errors in final_sp500_strategy.py
"""

import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_indentation():
    """Fix indentation errors in final_sp500_strategy.py"""
    try:
        # Path to the file
        file_path = 'final_sp500_strategy.py'
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find and fix the indentation error
        fixed_lines = []
        for i, line in enumerate(lines):
            if "return summary, signals" in line and line.startswith(" "):
                # Remove any extra indentation
                fixed_line = "        return summary, signals\n"
                fixed_lines.append(fixed_line)
                logger.info(f"Fixed indentation error on line {i+1}")
            else:
                fixed_lines.append(line)
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
        
        logger.info("Successfully fixed indentation errors in final_sp500_strategy.py")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_indentation()
