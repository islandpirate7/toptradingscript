def fix_all_fstring_errors():
    """Fix all f-string syntax errors in final_sp500_strategy.py"""
    file_path = 'final_sp500_strategy.py'
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # List of problematic patterns and their fixes
    error_patterns = [
        # Pattern for "Generated {len(signals)..." error
        {
            'pattern': 'log_file_handle.write(f"{datetime.now()} - INFO - " + str(f"Generated {len(signals)',
            'fix': '        log_file_handle.write(f"{datetime.now()} - INFO - Generated {len(signals)} total signals: {len(largecap_signals)} large-cap, {len(midcap_signals)} mid-cap\\n")\n'
        },
        # Pattern for "Limiting signals to top..." error
        {
            'pattern': 'log_file_handle.write(f"{datetime.now()} - INFO - " + str(f"Limiting signals to top {max_signals}',
            'fix': '        log_file_handle.write(f"{datetime.now()} - INFO - Limiting signals to top {max_signals} (from {len(signals)})\\n")\n'
        }
    ]
    
    # Find and fix all problematic lines
    fixed_count = 0
    i = 0
    while i < len(lines):
        fixed = False
        for error in error_patterns:
            if error['pattern'] in lines[i]:
                # Fix the syntax error
                lines[i] = error['fix']
                # Remove the next two lines that are part of the broken statement
                if i+1 < len(lines) and 'log_file_handle.flush()' in lines[i+1]:
                    lines[i+1] = '        log_file_handle.flush()\n'
                if i+2 < len(lines) and 'os.fsync(log_file_handle.fileno())' in lines[i+2]:
                    lines[i+2] = '        os.fsync(log_file_handle.fileno())\n'
                print(f"Fixed f-string syntax error around line {i+1}")
                fixed_count += 1
                fixed = True
                break
        
        # Check for other potential f-string errors
        if not fixed and 'log_file_handle.write(f"{datetime.now()} - INFO - " + str(f"' in lines[i]:
            # This is likely another f-string error with the same pattern
            # Extract the message content
            try:
                message_start = lines[i].index('str(f"') + 6
                message_content = lines[i][message_start:].strip()
                # Create a fixed version
                fixed_line = f'        log_file_handle.write(f"{{datetime.now()}} - INFO - {message_content}\\n")\n'
                lines[i] = fixed_line
                # Remove the next two lines that are part of the broken statement
                if i+1 < len(lines) and 'log_file_handle.flush()' in lines[i+1]:
                    lines[i+1] = '        log_file_handle.flush()\n'
                if i+2 < len(lines) and 'os.fsync(log_file_handle.fileno())' in lines[i+2]:
                    lines[i+2] = '        os.fsync(log_file_handle.fileno())\n'
                print(f"Fixed generic f-string syntax error around line {i+1}")
                fixed_count += 1
            except:
                print(f"Attempted to fix line {i+1} but couldn't parse it properly")
        
        i += 1
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Fixed {fixed_count} f-string syntax errors in final_sp500_strategy.py")

if __name__ == "__main__":
    fix_all_fstring_errors()
