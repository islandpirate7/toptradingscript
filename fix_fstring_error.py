def fix_fstring_error():
    """Fix the syntax error in the f-string conversion in final_sp500_strategy.py"""
    file_path = 'final_sp500_strategy.py'
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find and fix the problematic lines
    for i in range(len(lines)):
        if 'log_file_handle.write(f"{datetime.now()} - INFO - " + str(f"Generated {len(signals) + "' in lines[i]:
            # Fix the syntax error
            lines[i] = '        log_file_handle.write(f"{datetime.now()} - INFO - Generated {len(signals)} total signals: {len(largecap_signals)} large-cap, {len(midcap_signals)} mid-cap\\n")\n'
            # Remove the next two lines that are part of the broken statement
            if i+1 < len(lines) and 'log_file_handle.flush()' in lines[i+1]:
                lines[i+1] = '        log_file_handle.flush()\n'
            if i+2 < len(lines) and 'os.fsync(log_file_handle.fileno())}' in lines[i+2]:
                lines[i+2] = '        os.fsync(log_file_handle.fileno())\n'
            print(f"Fixed f-string syntax error around line {i+1}")
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("Fixed f-string syntax error in final_sp500_strategy.py")

if __name__ == "__main__":
    fix_fstring_error()
