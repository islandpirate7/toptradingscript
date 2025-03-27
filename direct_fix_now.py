#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Directly fix the problematic section in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def direct_fix():
    """
    Directly fix the problematic section in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            lines = f.readlines()
        
        # Find the problematic section (around lines 3950-3965)
        start_line = 3950
        end_line = 3965
        
        # Create the corrected section
        corrected_lines = [
            '            logger.info(f"[DEBUG] First few signals: {signals[:3]}")\n',
            '        \n',
            '        # Add continuous_capital flag to summary\n',
            '        if summary:\n',
            '            summary[\'continuous_capital\'] = continuous_capital\n',
            '        \n',
            '        # Update final_capital for continuous capital mode\n',
            '        if metrics:\n',
            '            final_capital = metrics[\'final_capital\']\n',
            '        \n',
            '        return summary, signals\n',
            '    \n',
            '    except Exception as e:\n',
            '        logger.error(f"Error running backtest: {str(e)}")\n',
            '        traceback.print_exc()\n',
            '        return None, []\n'
        ]
        
        # Replace the problematic section with the corrected one
        lines[start_line-1:end_line] = corrected_lines
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.writelines(lines)
        
        print(f"Successfully fixed lines {start_line}-{end_line} in final_sp500_strategy.py")
        
    except Exception as e:
        print(f"Error fixing problematic section: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Directly fixing problematic section in final_sp500_strategy.py...")
    direct_fix()
    
    print("\nFix applied. You can now run your tests.")
