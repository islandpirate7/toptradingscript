#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the continuous capital parameter in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def fix_continuous_capital():
    """
    Add continuous_capital parameter to run_backtest function in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.readlines()
        
        # Find the run_backtest function definition
        for i, line in enumerate(content):
            if "def run_backtest(" in line:
                # Update the function signature to include continuous_capital parameter
                old_signature = line
                new_signature = line.replace(
                    "def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True):",
                    "def run_backtest(start_date, end_date, mode='backtest', max_signals=None, initial_capital=300, random_seed=42, weekly_selection=True, continuous_capital=False):"
                )
                
                # If the signature didn't match exactly, try a more flexible approach
                if old_signature == new_signature:
                    # Try a regex-based approach
                    pattern = r"def run_backtest\([^)]*\):"
                    match = re.search(pattern, line)
                    if match:
                        current_sig = match.group(0)
                        # Remove the trailing colon for parameter insertion
                        params_part = current_sig[:-1]
                        # Add the continuous_capital parameter
                        new_sig = f"{params_part}, continuous_capital=False):"
                        new_signature = line.replace(current_sig, new_sig)
                
                content[i] = new_signature
                print(f"Updated function signature from:\n{old_signature}to:\n{new_signature}")
                
                # Now find the part where remaining_capital is initialized
                for j in range(i, len(content)):
                    if "remaining_capital = initial_capital" in content[j]:
                        # Insert code to track the initial and final capital
                        content.insert(j+1, "        # Track final capital for continuous capital mode\n")
                        content.insert(j+2, "        final_capital = remaining_capital\n")
                        break
                
                # Find the part where the function returns
                for j in range(i, len(content)):
                    if "return summary, signals" in content[j]:
                        # Update the final_capital before returning
                        content.insert(j, "        # Update final_capital for continuous capital mode\n")
                        content.insert(j, "        if metrics:\n")
                        content.insert(j, "            final_capital = metrics['final_capital']\n")
                        content.insert(j, "        \n")
                        content.insert(j, "        # Add continuous_capital flag to summary\n")
                        content.insert(j, "        if summary:\n")
                        content.insert(j, "            summary['continuous_capital'] = continuous_capital\n")
                        content.insert(j, "        \n")
                        break
                
                break
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.writelines(content)
        
        print("Successfully added continuous_capital parameter to run_backtest function")
        
    except Exception as e:
        print(f"Error fixing continuous_capital parameter: {str(e)}")
        traceback.print_exc()

def fix_run_backtest_for_web():
    """
    Update run_backtest_for_web function to pass continuous_capital parameter
    """
    try:
        # Read the file
        with open('run_comprehensive_backtest.py', 'r') as f:
            content = f.readlines()
        
        # Find the run_backtest call in run_backtest_for_web function
        in_function = False
        for i, line in enumerate(content):
            if "def run_backtest_for_web(" in line:
                in_function = True
            
            if in_function and "summary, signals = run_backtest(" in line:
                # Check if the next lines contain the parameters
                j = i
                while j < len(content) and ")" not in content[j]:
                    j += 1
                
                # If we found the closing parenthesis
                if j < len(content):
                    # Check if continuous_capital is already passed
                    continuous_capital_passed = False
                    for k in range(i, j+1):
                        if "continuous_capital=" in content[k]:
                            continuous_capital_passed = True
                            break
                    
                    # If not passed, add it before the closing parenthesis
                    if not continuous_capital_passed:
                        # Find the line with the closing parenthesis
                        for k in range(i, j+1):
                            if ")" in content[k]:
                                # Add the continuous_capital parameter
                                closing_idx = content[k].find(")")
                                if closing_idx > 0:
                                    content[k] = content[k][:closing_idx] + ",\n            continuous_capital=continuous_capital" + content[k][closing_idx:]
                                    print(f"Added continuous_capital parameter to run_backtest call")
                                break
                break
        
        # Write the updated content back to the file
        with open('run_comprehensive_backtest.py', 'w') as f:
            f.writelines(content)
        
        print("Successfully updated run_backtest_for_web function to pass continuous_capital parameter")
        
    except Exception as e:
        print(f"Error updating run_backtest_for_web function: {str(e)}")
        traceback.print_exc()

def fix_quarters_issue():
    """
    Fix the issue with identical results for different quarters
    """
    try:
        # Read the file
        with open('web_interface/app.py', 'r') as f:
            content = f.readlines()
        
        # Find the run_comprehensive_backtest route
        in_route = False
        for i, line in enumerate(content):
            if "@app.route('/run_comprehensive_backtest', methods=['POST'])" in line:
                in_route = True
            
            # Find where quarters are processed
            if in_route and "quarters = request.form.get('quarters', 'Q1_2023')" in line:
                # Check how quarters are handled
                for j in range(i, min(i+50, len(content))):
                    if "quarters_list = quarters.split(',')" in content[j]:
                        # Already handling multiple quarters correctly
                        break
                else:
                    # Need to add code to handle multiple quarters
                    content.insert(i+1, "        # Split quarters into a list if multiple are provided\n")
                    content.insert(i+2, "        quarters_list = quarters.split(',')\n")
                    content.insert(i+3, "        quarters_list = [q.strip() for q in quarters_list]\n")
                    content.insert(i+4, "        logger.info(f\"Processing quarters: {quarters_list}\")\n")
                    print("Added code to handle multiple quarters")
                break
        
        # Write the updated content back to the file
        with open('web_interface/app.py', 'w') as f:
            f.writelines(content)
        
        print("Successfully fixed quarters handling in web interface")
        
    except Exception as e:
        print(f"Error fixing quarters issue: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fixing continuous capital parameter in final_sp500_strategy.py...")
    fix_continuous_capital()
    
    print("\nUpdating run_backtest_for_web function to pass continuous_capital parameter...")
    fix_run_backtest_for_web()
    
    print("\nFixing quarters issue in web interface...")
    fix_quarters_issue()
    
    print("\nAll fixes applied. Please restart the web server to apply the changes.")
