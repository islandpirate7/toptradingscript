#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug patch for final_sp500_strategy.py
This script adds debug logging to the run_backtest function
"""

import os
import re
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_file():
    """Patch the final_sp500_strategy.py file to add debug logging"""
    try:
        # Path to the file
        file_path = 'final_sp500_strategy.py'
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the run_backtest function
        pattern = r'def run_backtest\(start_date, end_date, mode=\'backtest\', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True\):'
        match = re.search(pattern, content)
        
        if not match:
            logger.error("Could not find run_backtest function in final_sp500_strategy.py")
            return False
        
        # Find the position to insert the debug logging
        pos = match.end()
        
        # Find the next line after the function definition
        next_line_pos = content.find('\n', pos) + 1
        
        # Find the try block
        try_pos = content.find('try:', next_line_pos)
        
        if try_pos == -1:
            logger.error("Could not find try block in run_backtest function")
            return False
        
        # Find the position after the try block
        after_try_pos = content.find('\n', try_pos) + 1
        
        # Debug logging to insert
        debug_logging = """
        # Debug logging
        logger.info("[DEBUG] Starting run_backtest in final_sp500_strategy.py")
        logger.info(f"[DEBUG] Parameters: start_date={start_date}, end_date={end_date}, mode={mode}, max_signals={max_signals}, initial_capital={initial_capital}, random_seed={random_seed}, weekly_selection={weekly_selection}")
        start_time = time.time()
        """
        
        # Insert the debug logging after the try block
        new_content = content[:after_try_pos] + debug_logging + content[after_try_pos:]
        
        # Find the position to add timing information before the return statement
        return_pattern = r'return summary, signals'
        return_match = re.search(return_pattern, new_content)
        
        if not return_match:
            logger.error("Could not find return statement in run_backtest function")
            return False
        
        return_pos = return_match.start()
        
        # Find the position before the return statement
        before_return_pos = new_content.rfind('\n', 0, return_pos) + 1
        
        # Timing information to insert
        timing_info = """
        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"[DEBUG] run_backtest in final_sp500_strategy.py execution time: {execution_time:.2f} seconds")
        logger.info(f"[DEBUG] Returning {len(signals) if signals else 0} signals")
        if signals and len(signals) > 0:
            logger.info(f"[DEBUG] First few signals: {signals[:3]}")
        
        """
        
        # Insert the timing information before the return statement
        new_content = new_content[:before_return_pos] + timing_info + new_content[before_return_pos:]
        
        # Add import for time module if not already present
        if "import time" not in new_content:
            import_pos = new_content.find("import")
            if import_pos != -1:
                # Find the end of the import block
                import_block_end = new_content.find("\n\n", import_pos)
                if import_block_end != -1:
                    new_content = new_content[:import_block_end] + "\nimport time" + new_content[import_block_end:]
        
        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Successfully patched final_sp500_strategy.py with debug logging")
        return True
    
    except Exception as e:
        logger.error(f"Error patching file: {str(e)}")
        return False

if __name__ == "__main__":
    patch_file()
