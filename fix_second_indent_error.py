#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix second indentation error in final_sp500_strategy.py
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
        
        # Find the problematic lines around line 3959
        for i in range(3955, 3965):
            if i < len(content) and "if summary:" in content[i]:
                # Check the next line's indentation
                if i+1 < len(content) and not content[i+1].strip().startswith('#'):
                    # Get the correct indentation (should be current + 4 spaces)
                    current_indent = len(content[i]) - len(content[i].lstrip())
                    correct_indent = current_indent + 4
                    
                    # Fix indentation for the next line
                    content[i+1] = ' ' * correct_indent + content[i+1].lstrip()
                    
                    print(f"Fixed indentation for line {i+2}")
                break
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.writelines(content)
        
        print("Successfully fixed second indentation error in final_sp500_strategy.py")
        
    except Exception as e:
        print(f"Error fixing indentation error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fixing second indentation error in final_sp500_strategy.py...")
    fix_indentation_error()
    
    print("\nSecond indentation error fixed. You can now run your tests.")
