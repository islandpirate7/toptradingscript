#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Hybrid Model
---------------
This script fixes the VolatilityBreakout strategy issue and runs the hybrid model comparison.
"""

import os
import sys
import logging
import datetime as dt
import yaml
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('run_hybrid_model.log')
    ]
)

logger = logging.getLogger("RunHybridModel")

def fix_volatility_breakout_strategy():
    """
    Fix the premature return statement in the VolatilityBreakout strategy's generate_signals method
    """
    import re
    
    # Path to the multi_strategy_system.py file
    file_path = 'multi_strategy_system.py'
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the VolatilityBreakoutStrategy class and its generate_signals method
    pattern = r'(class VolatilityBreakoutStrategy.*?def generate_signals.*?signals\.append\(signal\))\s*\n\s*return signals\s*\n(.*?)def'
    
    # Replace the premature return with a commented version
    modified_content = re.sub(pattern, r'\1\n            # return signals  # Commented out premature return\n\2def', content, flags=re.DOTALL)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(modified_content)
    
    logger.info("Fixed VolatilityBreakout strategy by commenting out premature return statement")

def run_comparison():
    """
    Run the hybrid model comparison
    """
    from compare_hybrid_model import main as compare_main
    
    logger.info("Running hybrid model comparison...")
    compare_main()
    logger.info("Hybrid model comparison completed")

def main():
    """Main function"""
    # Fix the VolatilityBreakout strategy
    fix_volatility_breakout_strategy()
    
    # Run the comparison
    run_comparison()
    
    logger.info("All tasks completed successfully")

if __name__ == "__main__":
    main()
