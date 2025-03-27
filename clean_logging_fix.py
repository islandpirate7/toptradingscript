import os
import re

def clean_and_fix_logging():
    """
    Clean up the run_backtest function and apply a proper logging fix
    that ensures logs are written to disk.
    """
    file_path = 'final_sp500_strategy.py'
    
    # Read the current file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the run_backtest function
    start_line = -1
    end_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def run_backtest('):
            start_line = i
        elif start_line != -1 and line.strip().startswith('def '):
            end_line = i
            break
    
    if start_line == -1:
        print("ERROR: Could not find run_backtest function")
        return
    
    if end_line == -1:
        end_line = len(lines)
    
    # Extract the run_backtest function
    run_backtest_lines = lines[start_line:end_line]
    
    # Clean up the logging setup and flush calls
    cleaned_lines = []
    skip_next_lines = 0
    
    for i, line in enumerate(run_backtest_lines):
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue
        
        # Skip duplicate flush calls
        if "# Force flush the logs to disk" in line and i < len(run_backtest_lines) - 1:
            if "# Force flush the logs to disk" in run_backtest_lines[i+1]:
                skip_next_lines = 3  # Skip this and the next 3 lines (comment + for loop + handler.flush())
                continue
        
        cleaned_lines.append(line)
    
    # Find the logging setup section
    logging_start = -1
    logging_end = -1
    
    for i, line in enumerate(cleaned_lines):
        if "# Set up logging specifically for this backtest run" in line:
            logging_start = i
        elif logging_start != -1 and "# Log the start of the backtest" in line:
            logging_end = i
            break
    
    if logging_start == -1 or logging_end == -1:
        print("ERROR: Could not find logging setup section")
        return
    
    # Replace with a clean logging setup
    clean_logging_setup = [
        "        # Set up logging specifically for this backtest run\n",
        "        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
        "        log_file = os.path.join('logs', f\"strategy_{timestamp}.log\")\n",
        "        \n",
        "        # Make sure logs directory exists\n",
        "        os.makedirs('logs', exist_ok=True)\n",
        "        \n",
        "        # Reset the root logger completely\n",
        "        root_logger = logging.getLogger()\n",
        "        \n",
        "        # Store existing handlers to restore later\n",
        "        existing_handlers = root_logger.handlers.copy()\n",
        "        original_level = root_logger.level\n",
        "        \n",
        "        # Remove all handlers\n",
        "        for handler in root_logger.handlers[:]:\n",
        "            root_logger.removeHandler(handler)\n",
        "        \n",
        "        # Create file handler\n",
        "        file_handler = logging.FileHandler(log_file, mode='w')\n",
        "        file_handler.setLevel(logging.INFO)\n",
        "        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')\n",
        "        file_handler.setFormatter(formatter)\n",
        "        \n",
        "        # Create console handler\n",
        "        console_handler = logging.StreamHandler()\n",
        "        console_handler.setLevel(logging.INFO)\n",
        "        console_handler.setFormatter(formatter)\n",
        "        \n",
        "        # Add handlers to root logger\n",
        "        root_logger.addHandler(file_handler)\n",
        "        root_logger.addHandler(console_handler)\n",
        "        root_logger.setLevel(logging.INFO)\n",
        "        \n"
    ]
    
    # Replace the logging setup
    cleaned_lines[logging_start:logging_end] = clean_logging_setup
    
    # Find where to add the flush call
    for i, line in enumerate(cleaned_lines):
        if "Running backtest from {start_date} to {end_date}" in line:
            # Add a flush call after this line if it doesn't already exist
            if i + 1 < len(cleaned_lines) and "# Force flush the logs to disk" not in cleaned_lines[i+1]:
                flush_lines = [
                    "        \n",
                    "        # Force flush the logs to disk\n",
                    "        for handler in logging.getLogger().handlers:\n",
                    "            handler.flush()\n"
                ]
                for j, flush_line in enumerate(flush_lines):
                    cleaned_lines.insert(i + 1 + j, flush_line)
            break
    
    # Find where to add the close and flush call before returning
    for i, line in enumerate(cleaned_lines):
        if "# Restore original logging configuration" in line:
            # Add a close and flush call before this line if it doesn't already exist
            if "# Force flush and close the logs before returning" not in cleaned_lines[i-1]:
                close_lines = [
                    "        # Force flush and close the logs before returning\n",
                    "        for handler in logging.getLogger().handlers:\n",
                    "            handler.flush()\n",
                    "            if isinstance(handler, logging.FileHandler):\n",
                    "                handler.close()\n",
                    "        \n"
                ]
                for j, close_line in enumerate(close_lines):
                    cleaned_lines.insert(i, close_line)
            break
    
    # Fix variable references from midcap_symbols to midcap_signals
    for i, line in enumerate(cleaned_lines):
        if "midcap_symbols = sorted(" in line and "key=lambda" in line:
            cleaned_lines[i] = line.replace("midcap_symbols = sorted(", "midcap_signals = sorted(")
            cleaned_lines[i] = cleaned_lines[i].replace("midcap_symbols,", "midcap_signals,")
        
        if "selected_mid_cap = midcap_symbols[" in line:
            cleaned_lines[i] = line.replace("selected_mid_cap = midcap_symbols[", "selected_mid_cap = midcap_signals[")
        
        if "len(midcap_symbols)" in line:
            cleaned_lines[i] = line.replace("len(midcap_symbols)", "len(midcap_signals)")
    
    # Update the run_backtest function in the original file
    lines[start_line:end_line] = cleaned_lines
    
    # Make sure sys is imported
    import_sys_found = False
    for line in lines:
        if line.strip() == 'import sys':
            import_sys_found = True
            break
    
    if not import_sys_found:
        for i, line in enumerate(lines):
            if line.strip() == 'import random':
                lines.insert(i + 1, 'import sys\n')
                break
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("Successfully cleaned and fixed the logging in run_backtest function:")
    print("1. Removed duplicate flush calls")
    print("2. Applied a clean logging setup")
    print("3. Added proper flush and close calls")
    print("4. Fixed variable reference issues (midcap_symbols -> midcap_signals)")
    print("\nPlease run a backtest to verify that the logs are now being properly created and populated.")

if __name__ == "__main__":
    clean_and_fix_logging()
