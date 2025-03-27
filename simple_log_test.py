import os
import time
from datetime import datetime

def test_simple_log():
    """
    A very simple test to verify we can create and write to log files.
    This uses the most basic file operations without any logging framework.
    """
    # Create a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a log file path
    log_file = os.path.join('logs', f"simple_test_{timestamp}.log")
    
    print(f"Creating log file: {log_file}")
    
    # Write directly to the file
    with open(log_file, 'w') as f:
        f.write(f"{datetime.now()} - INFO - Starting simple log test\n")
        f.write(f"{datetime.now()} - INFO - This is a test message\n")
        f.flush()  # Explicitly flush the file buffer
        os.fsync(f.fileno())  # Force the OS to write to disk
        
        # Write more messages with delays
        for i in range(5):
            f.write(f"{datetime.now()} - INFO - Test message {i+1}\n")
            f.flush()
            os.fsync(f.fileno())
            time.sleep(0.1)
    
    # Verify the file was created and has content
    try:
        file_size = os.path.getsize(log_file)
        print(f"Log file created: {log_file}")
        print(f"Log file size: {file_size} bytes")
        
        if file_size > 0:
            print("SUCCESS: Log file contains content")
            
            # Read and display the content
            with open(log_file, 'r') as f:
                content = f.read()
                print("\nLog file content:")
                print(content)
        else:
            print("ERROR: Log file is empty (0 bytes)")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_simple_log()
