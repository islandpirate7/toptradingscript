#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix indentation error in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def fix_indentation_error():
    """
    Fix indentation error in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.readlines()
        
        # Find the problematic lines
        for i, line in enumerate(content):
            if "# Track final capital for continuous capital mode" in line:
                # Check indentation of this line and the next line
                current_indent = len(line) - len(line.lstrip())
                
                # Get the correct indentation from the previous line
                prev_line = content[i-1]
                correct_indent = len(prev_line) - len(prev_line.lstrip())
                
                # Fix indentation for this line and the next line
                if current_indent != correct_indent:
                    content[i] = ' ' * correct_indent + line.lstrip()
                    if i+1 < len(content):
                        content[i+1] = ' ' * correct_indent + content[i+1].lstrip()
                
                print(f"Fixed indentation for lines {i+1} and {i+2}")
                break
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.writelines(content)
        
        print("Successfully fixed indentation error in final_sp500_strategy.py")
        
    except Exception as e:
        print(f"Error fixing indentation error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fixing indentation error in final_sp500_strategy.py...")
    fix_indentation_error()
    
    print("\nIndentation error fixed. You can now run your tests.")
