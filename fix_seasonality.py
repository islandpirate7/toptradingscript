#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the seasonality score calculation error in final_sp500_strategy.py
"""

import re

def fix_seasonality_score():
    # Read the file
    with open('final_sp500_strategy.py', 'r') as f:
        content = f.read()
    
    # Find and replace the problematic line
    pattern = r'date_key = f"{month:02d}-{day:02d}"'
    replacement = 'date_key = f"{int(month):02d}-{int(day):02d}"'
    
    # Replace the pattern
    new_content = content.replace(pattern, replacement)
    
    # Write the updated content back to the file
    with open('final_sp500_strategy.py', 'w') as f:
        f.write(new_content)
    
    print("Fixed seasonality score calculation in final_sp500_strategy.py")

if __name__ == "__main__":
    fix_seasonality_score()
