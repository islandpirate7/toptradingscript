import os
import sys
import logging
from datetime import datetime

def setup_logging():
    """Set up logging with a simple configuration that works"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f"direct_fix_{timestamp}.log")
    
    # Reset the root logger completely
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove all handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and add file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create and add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file

def apply_fix_to_run_backtest():
    """Apply the direct logging fix to the run_backtest function"""
    file_path = 'final_sp500_strategy.py'
    
    # First, test that our logging setup works
    log_file = setup_logging()
    logger = logging.getLogger()
    
    logger.info("Testing direct logging fix")
    logger.info(f"Log file created: {log_file}")
    
    # Force flush the logs to disk
    for handler in logger.handlers:
        handler.flush()
    
    # Check if the log file has content
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if content:
                print(f"SUCCESS: Log file created with content ({len(content)} bytes)")
            else:
                print("ERROR: Log file created but is empty (0 bytes)")
                return
    except Exception as e:
        print(f"ERROR: Could not read log file: {str(e)}")
        return
    
    # Now apply the fix to run_backtest
    print("\nApplying the direct logging fix to run_backtest function...")
    
    # Read the current file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the run_backtest function
    start_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def run_backtest('):
            start_line = i
            break
    
    if start_line == -1:
        print("ERROR: Could not find run_backtest function")
        return
    
    # Find the logging setup section
    logging_start = -1
    for i in range(start_line, len(lines)):
        if "# Set up logging specifically for this backtest run" in lines[i]:
            logging_start = i
            break
    
    if logging_start == -1:
        print("ERROR: Could not find logging setup section")
        return
    
    # Find the end of the logging setup section
    logging_end = -1
    for i in range(logging_start, len(lines)):
        if "# Log the start of the backtest" in lines[i]:
            logging_end = i
            break
    
    if logging_end == -1:
        print("ERROR: Could not find end of logging setup section")
        return
    
    # Replace the logging setup with our working version
    new_logging_setup = [
        "        # Set up logging specifically for this backtest run\n",
        "        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
        "        log_file = os.path.join('logs', f\"strategy_{timestamp}.log\")\n",
        "        \n",
        "        # Make sure logs directory exists\n",
        "        os.makedirs('logs', exist_ok=True)\n",
        "        \n",
        "        # Reset the root logger completely\n",
        "        root_logger = logging.getLogger()\n",
        "        root_logger.setLevel(logging.INFO)\n",
        "        \n",
        "        # Store existing handlers to restore later\n",
        "        existing_handlers = root_logger.handlers.copy()\n",
        "        original_level = root_logger.level\n",
        "        \n",
        "        # Remove all handlers\n",
        "        for handler in root_logger.handlers[:]:\n",
        "            root_logger.removeHandler(handler)\n",
        "        \n",
        "        # Create and add file handler\n",
        "        file_handler = logging.FileHandler(log_file, mode='w')\n",
        "        file_handler.setLevel(logging.INFO)\n",
        "        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')\n",
        "        file_handler.setFormatter(formatter)\n",
        "        root_logger.addHandler(file_handler)\n",
        "        \n",
        "        # Create and add console handler\n",
        "        console_handler = logging.StreamHandler(sys.stdout)\n",
        "        console_handler.setLevel(logging.INFO)\n",
        "        console_handler.setFormatter(formatter)\n",
        "        root_logger.addHandler(console_handler)\n",
        "        \n"
    ]
    
    # Replace the logging setup section
    lines[logging_start:logging_end] = new_logging_setup
    
    # Add flush calls after important log messages
    for i in range(start_line, len(lines)):
        if "logger.info(f\"Running backtest from {start_date} to {end_date}" in lines[i]:
            # Add flush calls after this line
            flush_lines = [
                "\n",
                "        # Force flush the logs to disk\n",
                "        for handler in logging.getLogger().handlers:\n",
                "            handler.flush()\n"
            ]
            lines.insert(i + 1, "".join(flush_lines))
            break
    
    # Add flush calls before returning results
    for i in range(start_line, len(lines)):
        if "# Restore original logging configuration" in lines[i]:
            # Add flush calls before this line
            flush_lines = [
                "        # Force flush the logs to disk before returning\n",
                "        for handler in logging.getLogger().handlers:\n",
                "            handler.flush()\n",
                "\n"
            ]
            lines.insert(i, "".join(flush_lines))
            break
    
    # Fix variable references from midcap_symbols to midcap_signals
    for i in range(start_line, len(lines)):
        if "midcap_symbols = sorted(" in lines[i] and "key=lambda" in lines[i]:
            lines[i] = lines[i].replace("midcap_symbols = sorted(", "midcap_signals = sorted(")
            lines[i] = lines[i].replace("midcap_symbols,", "midcap_signals,")
        
        if "selected_mid_cap = midcap_symbols[" in lines[i]:
            lines[i] = lines[i].replace("selected_mid_cap = midcap_symbols[", "selected_mid_cap = midcap_signals[")
        
        if "len(midcap_symbols)" in lines[i]:
            lines[i] = lines[i].replace("len(midcap_symbols)", "len(midcap_signals)")
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    # Make sure sys is imported
    with open(file_path, 'r') as file:
        content = file.read()
    
    if "import sys" not in content:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Add import sys after the other imports
        for i, line in enumerate(lines):
            if line.startswith('import random'):
                lines.insert(i + 1, 'import sys\n')
                break
        
        with open(file_path, 'w') as file:
            file.writelines(lines)
    
    print("SUCCESS: Applied direct logging fix to run_backtest function")
    print("Please run a backtest to verify that the logs are now being properly created and populated.")

if __name__ == "__main__":
    apply_fix_to_run_backtest()
