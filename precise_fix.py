#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Precisely fix the indentation issues in final_sp500_strategy.py
"""

import os
import sys
import traceback

def precise_fix():
    """
    Precisely fix the indentation issues in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            lines = f.readlines()
        
        # Fix line 3949 (unexpected indent)
        if len(lines) > 3949:
            current_line = lines[3948]
            if "logger.info" in current_line and "First few signals" in current_line:
                # Fix the indentation to match the previous line
                prev_line = lines[3947]
                correct_indent = len(prev_line) - len(prev_line.lstrip())
                lines[3948] = ' ' * correct_indent + current_line.lstrip()
                print(f"Fixed indentation for line 3949")
        
        # Ensure the rest of the section has correct indentation
        correct_section = [
            '        logger.info(f"[DEBUG] Returning {len(signals) if signals else 0} signals")\n',
            '        if signals and len(signals) > 0:\n',
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
        lines[3947:3965] = correct_section
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.writelines(lines)
        
        print(f"Successfully fixed lines 3948-3965 in final_sp500_strategy.py")
        
    except Exception as e:
        print(f"Error fixing problematic section: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Precisely fixing indentation issues in final_sp500_strategy.py...")
    precise_fix()
    
    print("\nFix applied. You can now run your tests.")
