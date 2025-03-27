import re

def fix_indentation_error():
    """Fix the indentation error in the run_backtest function"""
    file_path = 'final_sp500_strategy.py'
    
    # Read the current file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the problematic line and fix its indentation
    for i in range(len(lines)):
        if "logger.info(f\"Running backtest from {start_date} to {end_date}" in lines[i]:
            # Check if the line has incorrect indentation
            if lines[i].startswith('                '):
                # Fix indentation to match surrounding code
                lines[i] = '        ' + lines[i].lstrip()
                print(f"Fixed indentation on line {i+1}")
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("Fixed indentation error in final_sp500_strategy.py")
    print("Please run a backtest to verify that the logs are now being properly created and populated.")

if __name__ == "__main__":
    fix_indentation_error()
