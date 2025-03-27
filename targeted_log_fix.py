import os
import re
from datetime import datetime

def apply_targeted_log_fix():
    """
    Apply a targeted fix to ensure log files are properly created and written to.
    This script focuses on fixing specific issues with log file handling.
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the section where log files are created and written to
    log_file_creation_line = -1
    log_file_handle_line = -1
    
    for i, line in enumerate(lines):
        if 'log_file = os.path.join' in line and 'logs' in line:
            log_file_creation_line = i
        if 'log_file_handle = open' in line:
            log_file_handle_line = i
    
    if log_file_creation_line == -1 or log_file_handle_line == -1:
        print("ERROR: Could not find log file creation or handle opening lines")
        return
    
    # Create a backup of the original file
    backup_file = f"{file_path}.bak"
    with open(backup_file, 'w') as file:
        file.writelines(lines)
    print(f"Created backup of original file: {backup_file}")
    
    # Create a test function that we know works
    test_function = """
def test_log_creation():
    \"\"\"
    Test function to verify log file creation and writing.
    \"\"\"
    import os
    from datetime import datetime
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"test_log_{timestamp}.log")
    
    print(f"Creating test log file: {log_file}")
    
    # Open the log file for writing
    with open(log_file, 'w') as log_file_handle:
        # Write some test messages
        log_file_handle.write(f"{datetime.now()} - INFO - This is a test message\\n")
        log_file_handle.write(f"{datetime.now()} - INFO - Testing log file creation\\n")
        log_file_handle.flush()
        os.fsync(log_file_handle.fileno())
        
        # Write more messages with explicit flush calls
        for i in range(5):
            log_file_handle.write(f"{datetime.now()} - INFO - Test message {i+1}\\n")
            log_file_handle.flush()
            os.fsync(log_file_handle.fileno())
    
    # Verify the file was created and has content
    file_size = os.path.getsize(log_file)
    print(f"Test log file created: {log_file}")
    print(f"Test log file size: {file_size} bytes")
    
    if file_size > 0:
        print("SUCCESS: Test log file contains content")
    else:
        print("ERROR: Test log file is empty (0 bytes)")
    
    return log_file
"""
    
    # Add the test function to the end of the file
    lines.append(test_function)
    
    # Modify the run_backtest function to use our test function
    for i, line in enumerate(lines):
        if 'def run_backtest(' in line:
            run_backtest_start = i
            break
    
    # Find where we set up logging in run_backtest
    for i in range(run_backtest_start, len(lines)):
        if 'log_file = os.path.join' in lines[i]:
            # Replace the log file creation code with a call to our test function
            indent = re.match(r'^\s*', lines[i]).group(0)
            lines[i] = f"{indent}# Use our test function to create a log file\n"
            lines[i+1] = f"{indent}log_file = test_log_creation()\n"
            
            # Skip the next few lines that deal with log file creation
            j = i + 2
            while j < len(lines) and ('log_file_handle' not in lines[j] or 'open' not in lines[j]):
                lines[j] = f"{indent}# Skipped: {lines[j].strip()}\n"
                j += 1
            
            # Replace the log file handle opening with a new open call
            if j < len(lines):
                lines[j] = f"{indent}# Re-open the log file for appending\n"
                lines[j+1] = f"{indent}log_file_handle = open(log_file, 'a')\n"
            
            break
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("Successfully applied targeted log fix")
    print("Added a test function to verify log file creation and modified run_backtest to use it")
    print("Please run a backtest to verify that logs are now being properly created and populated")

if __name__ == "__main__":
    apply_targeted_log_fix()
