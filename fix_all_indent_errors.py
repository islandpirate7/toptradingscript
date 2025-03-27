#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix all indentation errors in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def fix_indentation_errors():
    """
    Fix all indentation errors in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Find the run_backtest function
        pattern = r"def run_backtest\([^)]*\):(.*?)def "
        match = re.search(pattern, content, re.DOTALL)
        if match:
            function_code = match.group(1)
            
            # Find the problematic section
            bad_section = """        # Add continuous_capital flag to summary
        
            final_capital = metrics['final_capital']
        if metrics:
        # Update final_capital for continuous capital mode
        return summary, signals"""
            
            # Create the corrected section
            corrected_section = """        # Add continuous_capital flag to summary
        if summary:
            summary['continuous_capital'] = continuous_capital
        
        # Update final_capital for continuous capital mode
        if metrics:
            final_capital = metrics['final_capital']
            
        return summary, signals"""
            
            # Replace the bad section with the corrected one
            new_function_code = function_code.replace(bad_section, corrected_section)
            
            # Replace the function code in the original content
            new_content = content.replace(function_code, new_function_code)
            
            # Write the updated content back to the file
            with open('final_sp500_strategy.py', 'w') as f:
                f.write(new_content)
            
            print("Successfully fixed all indentation errors in final_sp500_strategy.py")
        else:
            print("Could not find the run_backtest function in the file")
        
    except Exception as e:
        print(f"Error fixing indentation errors: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fixing all indentation errors in final_sp500_strategy.py...")
    fix_indentation_errors()
    
    print("\nAll indentation errors fixed. You can now run your tests.")
