#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for continuous capital issue in the backtest process.
This script creates a backup of app.py and modifies the run_backtest_thread function
to ensure that the continuous capital is correctly passed between quarters.
"""

import os
import re
import shutil
import time
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the specified file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def fix_continuous_capital_issue():
    """Fix the continuous capital issue in app.py."""
    app_py_path = os.path.join('web_interface', 'app.py')
    
    # Create a backup
    backup_path = backup_file(app_py_path)
    
    # Read the file
    with open(app_py_path, 'r') as f:
        content = f.read()
    
    # Find the run_backtest_thread function
    pattern = r'def run_backtest_thread\(quarters, run_id, process_name, data\):(.*?)# Create a combined result file'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find run_backtest_thread function in app.py")
        return False
    
    function_code = match.group(1)
    
    # Check if the function already contains the fix
    if "# Store the updated initial capital for the next quarter" in function_code:
        print("Fix already applied")
        return True
    
    # Find the section where previous_capital is updated
    pattern = r'if continuous_capital and summary and \'final_capital\' in summary:(.*?)previous_capital = summary\[\'final_capital\'\]'
    match = re.search(pattern, function_code, re.DOTALL)
    
    if not match:
        print("Could not find the section to update previous_capital")
        return False
    
    update_section = match.group(0)
    
    # Create the modified section with explicit logging
    modified_section = update_section + """
                        # Store the updated initial capital for the next quarter
                        logger.info(f"Storing final capital {previous_capital} from {quarter} for next quarter")
                        # Ensure we're using the exact value without any floating point precision issues
                        previous_capital = round(previous_capital, 2)
                        logger.info(f"Rounded previous_capital to {previous_capital}")
                        """
    
    # Replace the section in the function code
    modified_function_code = function_code.replace(update_section, modified_section)
    
    # Replace the function in the content
    modified_content = content.replace(match.group(1), modified_function_code)
    
    # Write the modified content back to the file
    with open(app_py_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully updated {app_py_path} to fix continuous capital issue")
    return True

if __name__ == "__main__":
    print("Starting fix for continuous capital issue...")
    if fix_continuous_capital_issue():
        print("Fix completed successfully")
    else:
        print("Fix failed")
