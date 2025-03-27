#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the run_backtest function in final_sp500_strategy.py
"""

import os
import re
import sys
import traceback

def fix_run_backtest_function():
    """
    Fix the run_backtest function in final_sp500_strategy.py
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            content = f.read()
        
        # Find the problematic section and fix it
        pattern = r"# Add continuous_capital flag to summary\s+if summary:\s+summary\['continuous_capital'\] = continuous_capital\s+# Update final_capital for continuous capital mode\s+if metrics:\s+final_capital = metrics\['final_capital'\]\s+\s+summary\['continuous_capital'\] = continuous_capital\s+if summary:\s+# Add continuous_capital flag to summary\s+\s+final_capital = metrics\['final_capital'\]\s+if metrics:\s+# Update final_capital for continuous capital mode"
        
        replacement = """# Add continuous_capital flag to summary
        if summary:
            summary['continuous_capital'] = continuous_capital
            if metrics and 'final_capital' in metrics:
                summary['final_capital'] = metrics['final_capital']
        
        # Update final_capital for continuous capital mode
        if metrics and 'final_capital' in metrics:
            final_capital = metrics['final_capital']"""
        
        # Use regex to replace the problematic section
        fixed_content = re.sub(pattern, replacement, content)
        
        # Write the updated content back to the file
        with open('final_sp500_strategy.py', 'w') as f:
            f.write(fixed_content)
        
        print("Successfully fixed run_backtest function in final_sp500_strategy.py")
        
    except Exception as e:
        print(f"Error fixing run_backtest function: {str(e)}")
        traceback.print_exc()

def add_final_capital_to_summary():
    """
    Add final_capital to the summary object in run_backtest function
    """
    try:
        # Read the file
        with open('final_sp500_strategy.py', 'r') as f:
            lines = f.readlines()
        
        # Find the section where the summary is created
        summary_section_start = None
        for i, line in enumerate(lines):
            if "summary = {" in line:
                summary_section_start = i
                break
        
        if summary_section_start is not None:
            # Find the end of the summary dictionary
            summary_section_end = None
            for i in range(summary_section_start, len(lines)):
                if "}" in lines[i]:
                    summary_section_end = i
                    break
            
            if summary_section_end is not None:
                # Check if final_capital is already in the summary
                final_capital_exists = False
                for i in range(summary_section_start, summary_section_end + 1):
                    if "'final_capital':" in lines[i]:
                        final_capital_exists = True
                        break
                
                # If final_capital doesn't exist, add it before the closing brace
                if not final_capital_exists:
                    # Find the line with the closing brace
                    for i in range(summary_section_start, len(lines)):
                        if "}" in lines[i]:
                            # Add final_capital before the closing brace
                            indent = lines[i].split("}")[0]
                            lines.insert(i, f"{indent}    'final_capital': metrics['final_capital'] if metrics and 'final_capital' in metrics else initial_capital,\n")
                            break
                
                # Write the updated content back to the file
                with open('final_sp500_strategy.py', 'w') as f:
                    f.writelines(lines)
                
                print("Successfully added final_capital to summary in run_backtest function")
        
    except Exception as e:
        print(f"Error adding final_capital to summary: {str(e)}")
        traceback.print_exc()

def fix_continuous_capital_in_web_interface():
    """
    Fix the continuous capital implementation in the web interface
    """
    try:
        # Read the file
        with open('web_interface/app.py', 'r') as f:
            content = f.read()
        
        # Add debug logging for previous_capital
        pattern = r"logger\.info\(f\"Updated previous_capital to \{previous_capital\} for next quarter\"\)"
        replacement = """logger.info(f"Updated previous_capital to {previous_capital} for next quarter")
                            # Add final_capital to the result summary for display
                            if 'summary' in result and result['summary']:
                                result['summary']['initial_capital'] = previous_capital if continuous_capital and previous_capital else initial_capital
                                logger.info(f"Set initial_capital in result summary to {result['summary'].get('initial_capital')}")"""
        
        # Use regex to replace the pattern
        fixed_content = re.sub(pattern, replacement, content)
        
        # Write the updated content back to the file
        with open('web_interface/app.py', 'w') as f:
            f.write(fixed_content)
        
        print("Successfully fixed continuous capital implementation in web interface")
        
    except Exception as e:
        print(f"Error fixing continuous capital in web interface: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Fixing run_backtest function in final_sp500_strategy.py...")
    fix_run_backtest_function()
    
    print("\nAdding final_capital to summary in run_backtest function...")
    add_final_capital_to_summary()
    
    print("\nFixing continuous capital implementation in web interface...")
    fix_continuous_capital_in_web_interface()
    
    print("\nAll fixes applied. Please restart the web server to apply the changes.")
