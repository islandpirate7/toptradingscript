#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update the run_comprehensive_backtest.py script to add the weekly selection parameter
"""

def add_weekly_selection_parameter():
    # Read the file
    with open('run_comprehensive_backtest.py', 'r') as f:
        content = f.readlines()
    
    # Find the argument parser section
    arg_parser_line = 0
    continuous_capital_line = 0
    
    for i, line in enumerate(content):
        if "parser.add_argument('--continuous_capital'" in line:
            continuous_capital_line = i
            break
    
    if continuous_capital_line == 0:
        print("Could not find the continuous_capital argument")
        return
    
    # Add the weekly_selection parameter after the continuous_capital parameter
    weekly_selection_param = "        parser.add_argument('--weekly_selection', action='store_true', help='Enable weekly stock selection refresh')\n"
    content.insert(continuous_capital_line + 1, weekly_selection_param)
    
    # Find all occurrences of run_comprehensive_backtest function calls
    for i, line in enumerate(content):
        if "run_comprehensive_backtest(" in line:
            # Find the closing parenthesis
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # If we found the closing parenthesis, add the weekly_selection parameter
            if j < len(content) and ")" in content[j]:
                # Replace the closing parenthesis with the new parameter
                content[j] = content[j].replace(")", ", weekly_selection=args.weekly_selection)")
        
        if "run_all_quarters_backtest(" in line:
            # Find the closing parenthesis
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # If we found the closing parenthesis, add the weekly_selection parameter
            if j < len(content) and ")" in content[j]:
                # Replace the closing parenthesis with the new parameter
                content[j] = content[j].replace(")", ", weekly_selection=args.weekly_selection)")
    
    # Update the function signatures
    for i, line in enumerate(content):
        if "def run_comprehensive_backtest(" in line:
            # Find the closing parenthesis
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # If we found the closing parenthesis, add the weekly_selection parameter
            if j < len(content) and ")" in content[j]:
                # Replace the closing parenthesis with the new parameter
                content[j] = content[j].replace(")", ", weekly_selection=False)")
        
        if "def run_all_quarters_backtest(" in line:
            # Find the closing parenthesis
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # If we found the closing parenthesis, add the weekly_selection parameter
            if j < len(content) and ")" in content[j]:
                # Replace the closing parenthesis with the new parameter
                content[j] = content[j].replace(")", ", weekly_selection=False)")
    
    # Update the run_backtest function calls
    for i, line in enumerate(content):
        if "run_backtest(" in line and "weekly_selection" not in line:
            # Find the closing parenthesis
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # If we found the closing parenthesis, add the weekly_selection parameter
            if j < len(content) and ")" in content[j]:
                # Replace the closing parenthesis with the new parameter
                content[j] = content[j].replace(")", ", weekly_selection=weekly_selection)")
    
    # Write the updated content back to the file
    with open('run_comprehensive_backtest.py', 'w') as f:
        f.writelines(content)
    
    print("Added weekly selection parameter to run_comprehensive_backtest.py")

if __name__ == "__main__":
    add_weekly_selection_parameter()
