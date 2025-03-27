def manual_fix():
    """
    Manually fix specific problematic lines in final_sp500_strategy.py
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Fix specific problematic lines
    fixes = [
        # Line 3393: Fix f-string syntax
        (3393, '        log_file_handle.write(f"{datetime.now()} - INFO - Final signals: {len(signals)}\\n")\n'),
        # Line 3394-3395: Ensure proper indentation for flush calls
        (3394, '        log_file_handle.flush()\n'),
        (3395, '        os.fsync(log_file_handle.fileno())\n'),
        # Line 3396: Fix indentation for else statement
        (3396, '        else:\n'),
        # Line 3400: Fix f-string syntax
        (3400, '            log_file_handle.write(f"{datetime.now()} - INFO - Using all {len(signals)}\\n")\n'),
        # Add flush calls after line 3400
        (3401, '            log_file_handle.flush()\n'),
        (3402, '            os.fsync(log_file_handle.fileno())\n'),
    ]
    
    # Apply the fixes (adjusting for 0-based indexing)
    for line_num, new_content in fixes:
        if line_num - 1 < len(lines):
            lines[line_num - 1] = new_content
            print(f"Fixed line {line_num}")
        else:
            # If the line doesn't exist (for adding new lines), append it
            lines.append(new_content)
            print(f"Added line {line_num}")
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Manually fixed specific problematic lines in {file_path}")
    print("Please run a backtest to verify that the fixes are working correctly.")

if __name__ == "__main__":
    manual_fix()
