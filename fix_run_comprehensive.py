#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the run_comprehensive_backtest.py script to properly handle weekly selection
"""

def fix_run_comprehensive():
    # Read the file
    with open('run_comprehensive_backtest.py', 'r') as f:
        content = f.readlines()
    
    # Add the weekly_selection parameter to the argument parser
    for i, line in enumerate(content):
        if "--continuous_capital" in line and "--weekly_selection" not in content[i+1]:
            content.insert(i+1, "        parser.add_argument('--weekly_selection', action='store_true', help='Enable weekly stock selection refresh')\n")
            break
    
    # Find the run_comprehensive_backtest function definition
    for i, line in enumerate(content):
        if "def run_comprehensive_backtest(" in line:
            # Update the function signature
            content[i] = "def run_comprehensive_backtest(quarter, max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):\n"
            
            # Find the run_backtest call
            for j in range(i, len(content)):
                if "run_backtest(" in content[j]:
                    # Find the end of the run_backtest call
                    k = j
                    while k < len(content) and ")" not in content[k]:
                        k += 1
                    
                    # Update the run_backtest call to include weekly_selection
                    if k < len(content):
                        # Check if weekly_selection is already included
                        if "weekly_selection" not in content[k]:
                            content[k] = content[k].replace(")", ", weekly_selection=weekly_selection)")
                    break
            break
    
    # Find the run_all_quarters_backtest function definition
    for i, line in enumerate(content):
        if "def run_all_quarters_backtest(" in line:
            # Update the function signature
            content[i] = "def run_all_quarters_backtest(max_signals=100, initial_capital=300, multiple_runs=False, num_runs=5, continuous_capital=False, weekly_selection=False):\n"
            break
    
    # Find the main function to update the function calls
    for i, line in enumerate(content):
        if "run_comprehensive_backtest(" in line and "weekly_selection" not in line:
            # Find the end of the function call
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # Update the function call to include weekly_selection
            if j < len(content):
                content[j] = content[j].replace(")", ", weekly_selection=args.weekly_selection)")
        
        if "run_all_quarters_backtest(" in line and "weekly_selection" not in line:
            # Find the end of the function call
            j = i
            while j < len(content) and ")" not in content[j]:
                j += 1
            
            # Update the function call to include weekly_selection
            if j < len(content):
                content[j] = content[j].replace(")", ", weekly_selection=args.weekly_selection)")
    
    # Write the updated content back to the file
    with open('run_comprehensive_backtest.py', 'w') as f:
        f.writelines(content)
    
    print("Fixed run_comprehensive_backtest.py")

if __name__ == "__main__":
    fix_run_comprehensive()
