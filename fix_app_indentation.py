#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix App Indentation

This script fixes all indentation errors in the app_fixed.py file
by properly aligning return statements inside try blocks.
"""

import os
import re
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_app_indentation():
    """Fix all indentation errors in the app_fixed.py file"""
    file_path = os.path.join('new_web_interface', 'app_fixed.py')
    
    # Create backup
    backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Read the current file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix indentation errors in return statements inside try blocks
    # This regex finds return statements with extra indentation inside try blocks
    pattern = r'(\s+)try\s*:\s*\n(\s+).*?\n(\s+)(\s+)return'
    
    # Replace with properly indented return statements
    fixed_content = re.sub(pattern, r'\1try:\n\2\3\n\3return', content, flags=re.DOTALL)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.write(fixed_content)
    
    logger.info(f"Fixed indentation errors in {file_path}")
    
    # Additional manual fix for any remaining issues
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    fixed_lines = []
    for line in lines:
        # Fix any lines with 'return jsonify' that have extra indentation
        if '            return jsonify' in line:
            fixed_lines.append(line.replace('            return jsonify', '        return jsonify'))
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(fixed_lines)
    
    logger.info(f"Completed additional fixes in {file_path}")
    return True

def fix_get_processes_function():
    """Fix the get_processes function specifically"""
    file_path = os.path.join('new_web_interface', 'app_fixed.py')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the get_processes function and fix it
    in_get_processes = False
    for i, line in enumerate(lines):
        if '@app.route(\'/get_processes\')' in line:
            in_get_processes = True
        elif in_get_processes and 'return jsonify' in line:
            # Fix the return statement to use proper JSON format
            lines[i] = '        return jsonify({"processes": list(processes.values())})\n'
            in_get_processes = False
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    logger.info(f"Fixed get_processes function in {file_path}")
    return True

def fix_get_backtest_results_function():
    """Fix the get_backtest_results_route function specifically"""
    file_path = os.path.join('new_web_interface', 'app_fixed.py')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the get_backtest_results_route function and fix it
    in_get_backtest_results = False
    for i, line in enumerate(lines):
        if '@app.route(\'/get_backtest_results\')' in line:
            in_get_backtest_results = True
        elif in_get_backtest_results and 'return jsonify' in line and 'results' in line:
            # Fix the return statement to use proper JSON format
            lines[i] = '        return jsonify({"results": results})\n'
            in_get_backtest_results = False
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    logger.info(f"Fixed get_backtest_results_route function in {file_path}")
    return True

def add_cache_headers():
    """Add cache-busting headers to the app"""
    file_path = os.path.join('new_web_interface', 'app_fixed.py')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Check if cache headers already exist
    if any('@app.after_request\ndef add_header(' in line for line in lines):
        logger.info("Cache headers already exist, skipping")
        return True
    
    # Add cache headers after CORS headers
    cache_headers = """
# Add cache-busting headers to prevent browser caching of static files
@app.after_request
def add_header(response):
    \"\"\"Add headers to prevent caching\"\"\"
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
"""
    
    # Find the right spot to insert the cache headers
    for i, line in enumerate(lines):
        if '@app.after_request' in line and 'def add_cors_headers' in lines[i+1]:
            # Insert after the CORS headers function
            for j in range(i+1, len(lines)):
                if 'return response' in lines[j] and j+1 < len(lines) and lines[j+1].strip() == '':
                    lines.insert(j+2, cache_headers)
                    break
            break
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    logger.info(f"Added cache headers to {file_path}")
    return True

def fix_run_comprehensive_backtest():
    """Fix the run_comprehensive_backtest function"""
    file_path = os.path.join('new_web_interface', 'app_fixed.py')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the run_comprehensive_backtest function and fix it
    in_function = False
    for i, line in enumerate(lines):
        if '@app.route(\'/run_comprehensive_backtest\'' in line:
            in_function = True
        elif in_function and 'flash(' in line and not line.strip().startswith('#'):
            # Ensure proper indentation for flash statements
            if not line.startswith('        '):
                lines[i] = '        ' + line.lstrip()
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    logger.info(f"Fixed run_comprehensive_backtest function in {file_path}")
    return True

def main():
    """Main function"""
    logger.info("Starting app indentation fix")
    
    # Fix all indentation errors
    if not fix_app_indentation():
        logger.error("Failed to fix app indentation")
        return False
    
    # Fix specific functions
    if not fix_get_processes_function():
        logger.error("Failed to fix get_processes function")
        return False
    
    if not fix_get_backtest_results_function():
        logger.error("Failed to fix get_backtest_results function")
        return False
    
    if not add_cache_headers():
        logger.error("Failed to add cache headers")
        return False
    
    if not fix_run_comprehensive_backtest():
        logger.error("Failed to fix run_comprehensive_backtest function")
        return False
    
    logger.info("App indentation fix complete")
    return True

if __name__ == "__main__":
    main()
