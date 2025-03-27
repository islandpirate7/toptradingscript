#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the continuous capital implementation in the web interface
"""

import os
import re
import sys
import traceback

def fix_web_interface_continuous_capital():
    """
    Update the web interface to properly implement continuous capital across quarters
    """
    try:
        # Read the file
        with open('web_interface/app.py', 'r') as f:
            content = f.readlines()
        
        # Find the run_comprehensive_backtest route
        in_route = False
        previous_capital_added = False
        
        for i, line in enumerate(content):
            if "@app.route('/run_comprehensive_backtest', methods=['POST'])" in line:
                in_route = True
            
            # Add previous_capital variable before the loop
            if in_route and "# Run backtest for each quarter" in line:
                # Add previous_capital variable to track capital between quarters
                content.insert(i, "        # Track previous capital for continuous capital mode\n")
                content.insert(i+1, "        previous_capital = initial_capital if continuous_capital else None\n")
                content.insert(i+2, "\n")
                previous_capital_added = True
                break
        
        # Find where run_backtest_for_web is called
        for i, line in enumerate(content):
            if "result = run_backtest_for_web(" in line:
                # Look for the closing parenthesis
                j = i
                while j < len(content) and ")" not in content[j]:
                    j += 1
                
                if j < len(content):
                    # Check if we need to update initial_capital
                    if previous_capital_added:
                        # Update initial_capital parameter if continuous_capital is enabled
                        for k in range(i, j+1):
                            if "initial_capital=initial_capital" in content[k]:
                                content[k] = content[k].replace(
                                    "initial_capital=initial_capital",
                                    "initial_capital=previous_capital if continuous_capital and previous_capital else initial_capital"
                                )
                                print("Updated initial_capital parameter in run_backtest_for_web call")
                                break
                    
                    # Add code to update previous_capital after the backtest
                    for k in range(j, min(j+20, len(content))):
                        if "logger.info(f\"Backtest completed successfully" in content[k]:
                            # Add code to update previous_capital
                            content.insert(k+1, "                    # Update previous_capital for next quarter if continuous_capital is enabled\n")
                            content.insert(k+2, "                    if continuous_capital and result and 'summary' in result and result['summary']:\n")
                            content.insert(k+3, "                        if 'final_capital' in result['summary']:\n")
                            content.insert(k+4, "                            previous_capital = result['summary']['final_capital']\n")
                            content.insert(k+5, "                            logger.info(f\"Updated previous_capital to {previous_capital} for next quarter\")\n")
                            print("Added code to update previous_capital after backtest")
                            break
        
        # Write the updated content back to the file
        with open('web_interface/app.py', 'w') as f:
            f.writelines(content)
        
        print("Successfully updated web interface to implement continuous capital")
        
    except Exception as e:
        print(f"Error updating web interface: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Updating web interface to properly implement continuous capital...")
    fix_web_interface_continuous_capital()
    
    print("\nUpdate complete. Please restart the web server to apply the changes.")
