def fix_indentation():
    """Fix any indentation issues in the final_sp500_strategy.py file"""
    file_path = 'final_sp500_strategy.py'
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Look for indentation issues
    fixed_lines = []
    for line in lines:
        # Fix any lines with unexpected indentation
        if "logger.info(f\"Running backtest from {start_date}" in line and line.startswith('    '):
            # Fix the indentation to match the surrounding code
            fixed_line = '        ' + line.lstrip()
            fixed_lines.append(fixed_line)
            print(f"Fixed indentation in line: {line.strip()}")
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(fixed_lines)
    
    print("Fixed indentation issues in final_sp500_strategy.py")

if __name__ == "__main__":
    fix_indentation()
