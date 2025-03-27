#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web Interface for S&P 500 Trading Strategy
This Flask application provides a web interface to control the trading strategy
"""

import os
import sys
import json

# Import turbo backtest if available
try:
    from turbo_backtest import run_turbo_backtest
    USE_TURBO_BACKTEST = True
    print("Using turbo backtest function")
except ImportError:
    USE_TURBO_BACKTEST = False
    print("Turbo backtest module not found, falling back to standard backtest")

# Import optimized backtest if available and turbo is not available
if not USE_TURBO_BACKTEST:
    try:
        from optimized_backtest import run_optimized_backtest
        USE_OPTIMIZED_BACKTEST = True
        print("Using optimized backtest function")
    except ImportError:
        USE_OPTIMIZED_BACKTEST = False
        print("Optimized backtest module not found, falling back to standard backtest")

# Import optimized backtest if available
try:
    from optimized_backtest import run_optimized_backtest
    USE_OPTIMIZED_BACKTEST = True
    print("Using optimized backtest function")
except ImportError:
    USE_OPTIMIZED_BACKTEST = False
    print("Optimized backtest module not found, falling back to standard backtest")
import yaml
import time
import logging
import subprocess
import threading
import traceback  # Add missing import for traceback
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import re

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hot reload module
from web_interface.hot_reload import start_hot_reload

# Configure logging
def setup_logging(log_file=None):
    """Set up logging for the application"""
    global logger
    
    # Create logger
    logger = logging.getLogger('web_interface')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # If log file is specified, create file handler
    if log_file:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger

# Initialize logging
setup_logging()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Enhanced CORS configuration

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Handle OPTIONS requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 200

# Global variables
config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sp500_config.yaml')
active_processes = {}
process_logs = {}
process_status = {}
config_data = None

def load_config():
    """Load configuration from YAML file"""
    global config_data
    try:
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config_data
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def save_config(config):
    """Save configuration to YAML file"""
    try:
        with open(config_file, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        logger.info(f"Successfully saved configuration to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

def get_backtest_results():
    """Get backtest results"""
    results = []
    
    # Track already processed files by absolute path to avoid duplicates
    processed_files = set()
    
    # Check multiple possible locations for backtest results
    results_dirs = [
        os.path.join(os.path.dirname(__file__), 'backtest_results'),  # Web interface directory
        os.path.join('..', 'backtest_results'),  # Relative path to root directory
        'backtest_results'  # Direct path in case the app is run from the root directory
    ]
    
    # Use a cache to avoid repeated file system operations
    # This is a static variable that persists between function calls
    if not hasattr(get_backtest_results, 'cache'):
        get_backtest_results.cache = {'results': [], 'last_update': 0}
    
    # Only refresh the cache every 60 seconds to avoid excessive file system operations
    current_time = time.time()
    if current_time - get_backtest_results.cache['last_update'] < 60:
        return get_backtest_results.cache['results']
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            # Convert to absolute path for consistent tracking
            abs_results_dir = os.path.abspath(results_dir)
            logger.info(f"Checking backtest results in: {abs_results_dir}")
            
            # Skip if we've already processed this directory
            if abs_results_dir in processed_files:
                logger.info(f"Skipping already processed directory: {abs_results_dir}")
                continue
                
            # Mark this directory as processed
            processed_files.add(abs_results_dir)
            
            try:
                files = os.listdir(abs_results_dir)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(abs_results_dir, x)), reverse=True)
                
                # Group files by quarter/date range to avoid duplicates
                quarter_groups = {}
                
                # Limit the number of files to process to avoid excessive processing
                max_files_to_process = 100
                files = files[:max_files_to_process]
                
                for file in files:
                    file_path = os.path.join(abs_results_dir, file)
                    abs_file_path = os.path.abspath(file_path)
                    
                    # Skip if we've already processed this file
                    if abs_file_path in processed_files:
                        continue
                    
                    # Mark this file as processed
                    processed_files.add(abs_file_path)
                    
                    if os.path.isfile(file_path):
                        # Extract quarter or date range from filename
                        quarter_key = None
                        
                        # Try to identify quarter from filename (multiple formats)
                        if "_Q" in file:
                            # Format: backtest_Q1_2023_timestamp.json or results_Q1_2023_timestamp.json
                            for part in file.split('_'):
                                if part.startswith('Q') and len(part) <= 3:
                                    year_part = next((p for p in file.split('_') if p.startswith('20') and len(p) == 4), "")
                                    if year_part:
                                        quarter_key = f"{part}_{year_part}"
                                        break
                        
                        # Try to identify by date range
                        if not quarter_key:
                            date_match = re.findall(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', file)
                            if date_match:
                                start_date, end_date = date_match[0]
                                quarter_key = f"{start_date}_to_{end_date}"
                                
                                # Try to extract quarter info if available
                                quarter_match = re.search(r'Q\d_\d{4}', file)
                                if quarter_match:
                                    quarter_key = quarter_match.group(0)
                        
                        # Check for results_YYYY_QX.json format
                        if not quarter_key and file.startswith('results_') and file.endswith('.json'):
                            parts = file.split('_')
                            if len(parts) >= 3 and parts[1].startswith('20') and parts[2].startswith('Q'):
                                year = parts[1]
                                quarter = parts[2].split('.')[0]  # Remove .json if it's part of the quarter
                                quarter_key = f"{quarter}_{year}"
                        
                        # If we couldn't identify a quarter, use the filename
                        if not quarter_key:
                            quarter_key = file
                        
                        # Only add if this is a new quarter or a newer file for the same quarter
                        if quarter_key not in quarter_groups:
                            quarter_groups[quarter_key] = []
                        
                        # Check if it's a JSON file (our backtest results are in JSON format)
                        if file.endswith('.json'):
                            try:
                                # Use a simple check to avoid loading large files
                                file_size = os.path.getsize(file_path)
                                if file_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                                    logger.warning(f"Skipping large file {file} ({file_size / (1024*1024):.2f} MB)")
                                    continue
                                    
                                with open(file_path, 'r') as f:
                                    # Read just enough to check if it's a valid JSON
                                    start_content = f.read(1024)  # Read first 1KB
                                    if not start_content.strip().startswith('{'):
                                        continue  # Not a JSON object
                                    
                                    # Reset file pointer and load the JSON
                                    f.seek(0)
                                    file_data = json.load(f)
                                    
                                    # Verify this is a backtest result file by checking for expected keys
                                    if 'summary' in file_data or any(key.startswith('Q') for key in file_data.keys()):
                                        quarter_groups[quarter_key].append({
                                            'name': file,
                                            'path': file_path,
                                            'date': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
                                            'quarter_key': quarter_key
                                        })
                                        logger.info(f"Found valid backtest result: {file}")
                            except Exception as e:
                                logger.warning(f"Error reading backtest result file {file}: {str(e)}")
                                # Still add the file even if we can't read it
                                quarter_groups[quarter_key].append({
                                    'name': file,
                                    'path': file_path,
                                    'date': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
                                    'quarter_key': quarter_key
                                })
                
                # Take only the most recent file from each quarter group
                for quarter, files in quarter_groups.items():
                    # Sort by date (newest first)
                    files.sort(key=lambda x: x['date'], reverse=True)
                    # Add only the most recent file for each quarter
                    if files:
                        results.append(files[0])
            except Exception as e:
                logger.error(f"Error processing directory {abs_results_dir}: {str(e)}")
    
    # Sort all results by date (newest first)
    results.sort(key=lambda x: x['date'], reverse=True)
    logger.info(f"Found {len(results)} backtest results")
    
    # Update the cache
    get_backtest_results.cache['results'] = results[:25]
    get_backtest_results.cache['last_update'] = current_time
    
    return results[:25]  # Return the 25 most recent results

def get_open_positions():
    """Get list of open positions from CSV files"""
    try:
        if not config_data:
            load_config()
        
        trades_dir = os.path.join('..', config_data.get('paths', {}).get('trades', 'trades'))
        if not os.path.exists(trades_dir):
            return []
        
        position_files = []
        for file in os.listdir(trades_dir):
            if 'position' in file.lower() and file.endswith('.csv'):
                file_path = os.path.join(trades_dir, file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                position_files.append({
                    'name': file,
                    'path': file_path,
                    'date': mod_time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by date (newest first)
        position_files.sort(key=lambda x: x['date'], reverse=True)
        
        # Get positions from the most recent file
        if position_files:
            try:
                df = pd.read_csv(position_files[0]['path'])
                positions = df.to_dict('records')
                return positions
            except Exception as e:
                logger.error(f"Error reading positions file: {str(e)}")
                return []
        
        return []
    except Exception as e:
        logger.error(f"Error getting open positions: {str(e)}")
        return []

def run_process(script, args, process_name):
    """Run a Python script as a subprocess and capture its output"""
    try:
        # Ensure the logs directory exists
        os.makedirs('./logs', exist_ok=True)
        
        # Create a unique log file for this process
        log_file = f"./logs/{process_name}.log"
        
        # Determine the correct working directory
        if script == 'run_comprehensive_backtest.py':
            # Use the parent directory for the comprehensive backtest script
            cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            script_path = os.path.join(cwd, script)
        else:
            # Use the current directory for other scripts
            cwd = os.getcwd()
            script_path = script
        
        # Construct the command
        cmd = ['python', script_path] + args
        
        # Log the command being executed
        logger.info(f"Executing command: {' '.join(cmd)} in directory: {cwd}")
        
        # Start the process with unbuffered output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            cwd=cwd,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Ensure Python output is unbuffered
        )
        
        # Store the process in the active processes dictionary
        active_processes[process_name] = {
            'process': process,
            'log_file': log_file,
            'status': 'Running',
            'logs': []
        }
        
        # Start a thread to capture the process output
        def capture_output():
            # Flag to track if we've written anything to the log file
            has_output = False
            log_file_handle = None
            
            try:
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        # Only open the log file when we actually have output
                        if not log_file_handle:
                            log_file_handle = open(log_file, 'w')
                            has_output = True
                        
                        # Write to log file
                        log_file_handle.write(line)
                        log_file_handle.flush()
                        
                        # Store in memory (limited to last 100 lines)
                        active_processes[process_name]['logs'].append(line.strip())
                        if len(active_processes[process_name]['logs']) > 100:
                            active_processes[process_name]['logs'].pop(0)
                        
                        # Log to application logger
                        logger.debug(f"[{process_name}] {line.strip()}")
                
                # Make sure we read any remaining output
                remaining_output, _ = process.communicate()
                if remaining_output:
                    # Open log file if we haven't already but now have output
                    if not log_file_handle:
                        log_file_handle = open(log_file, 'w')
                        has_output = True
                        
                    for line in remaining_output.splitlines():
                        line_with_newline = line + '\n'
                        log_file_handle.write(line_with_newline)
                        log_file_handle.flush()
                        active_processes[process_name]['logs'].append(line)
                        if len(active_processes[process_name]['logs']) > 100:
                            active_processes[process_name]['logs'].pop(0)
            except Exception as e:
                logger.error(f"Error capturing output for {process_name}: {str(e)}")
            finally:
                # Close the log file if it was opened
                if log_file_handle:
                    log_file_handle.close()
                
                # If no output was generated, remove the empty log file if it exists
                if not has_output and os.path.exists(log_file) and os.path.getsize(log_file) == 0:
                    try:
                        os.remove(log_file)
                        logger.info(f"Removed empty log file: {log_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove empty log file {log_file}: {str(e)}")
            
            # Update process status when completed
            active_processes[process_name]['status'] = 'Completed'
            logger.info(f"Process {process_name} completed with exit code {process.returncode}")
        
        # Start the output capture thread
        thread = threading.Thread(target=capture_output, daemon=True)
        thread.start()
        
        return {'status': 'success', 'message': f"Process {process_name} started", 'process_name': process_name}
    
    except Exception as e:
        logger.error(f"Error starting process: {str(e)}")
        return {'status': 'error', 'message': f"Error starting process: {str(e)}"}

def stop_process(process_name):
    """Stop a running process"""
    global active_processes, process_status
    
    if process_name in active_processes:
        try:
            process = active_processes[process_name]['process']
            process.terminate()
            process_status[process_name] = 'Terminated'
            logger.info(f"Process {process_name} terminated")
            return True
        except Exception as e:
            logger.error(f"Error terminating process {process_name}: {str(e)}")
            return False
    else:
        logger.warning(f"Process {process_name} not found in active processes")
        return False

def emergency_stop():
    """Stop all running processes and close all positions"""
    # Stop all active processes
    for process_name in list(active_processes.keys()):
        stop_process(process_name)
    
    # Run emergency script to close all positions
    try:
        # Create a script file with the emergency shutdown code
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emergency_shutdown.py')
        with open(script_path, 'w') as f:
            f.write('''
import sys
import os
import logging
import yaml
import time
from datetime import datetime

# Set up logging
log_file = f"emergency_shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from file"""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to the project root
        project_root = os.path.dirname(script_dir)
        
        # Construct the path to the config file
        config_path = os.path.join(project_root, 'sp500_config.yaml')
        
        # Check if the file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return None
        
        # Load the configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def close_all_positions():
    """Close all open positions"""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Import required modules
        try:
            from alpaca_api import AlpacaAPI
        except ImportError:
            logger.error("Failed to import AlpacaAPI")
            return
        
        # Initialize API
        api = AlpacaAPI(
            api_key=config.get('alpaca', {}).get('api_key', ''),
            api_secret=config.get('alpaca', {}).get('api_secret', ''),
            paper=config.get('alpaca', {}).get('paper', True)
        )
        
        # Get all positions
        positions = api.get_positions()
        
        # Log positions
        logger.info(f"Found {len(positions)} open positions")
        
        # Close each position
        for position in positions:
            symbol = position.symbol
            qty = abs(int(float(position.qty)))
            side = 'sell' if float(position.qty) > 0 else 'buy'
            
            logger.info(f"Closing position: {symbol} - {qty} shares - {side}")
            
            try:
                # Close position
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )
                
                logger.info(f"Successfully closed position: {symbol}")
                
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {str(e)}")
        
        # Wait for orders to complete
        logger.info("Waiting for orders to complete...")
        time.sleep(5)
        
        # Check if all positions are closed
        remaining_positions = api.get_positions()
        if remaining_positions:
            logger.warning(f"Still have {len(remaining_positions)} open positions")
        else:
            logger.info("All positions closed successfully")
    
    except Exception as e:
        logger.error(f"Error in close_all_positions: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting emergency shutdown procedure")
    close_all_positions()
    logger.info("Emergency shutdown completed")
''')
        
        # Run the script
        cmd = [sys.executable, script_path]
        logger.info(f"Running emergency shutdown script: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.info("Emergency shutdown completed successfully")
            return True, "Emergency shutdown completed successfully"
        else:
            logger.error(f"Emergency shutdown failed: {process.stderr}")
            return False, f"Emergency shutdown failed: {process.stderr}"
        
    except Exception as e:
        logger.error(f"Error during emergency shutdown: {str(e)}")
        return False, f"Error during emergency shutdown: {str(e)}"

@app.route('/')
def index():
    """Render the main dashboard"""
    # Load configuration if not loaded
    if not config_data:
        load_config()
    
    # Get backtest results
    backtest_results = get_backtest_results()
    
    # Get open positions
    open_positions = get_open_positions()
    
    return render_template(
        'index.html',
        config=config_data,
        backtest_results=backtest_results,
        open_positions=open_positions,
        active_processes=active_processes,
        process_status=process_status
    )

@app.route('/config')
def config():
    """Render the configuration page"""
    # Load configuration if not loaded
    if not config_data:
        load_config()
    
    return render_template('config.html', config=config_data)

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Load current configuration
        current_config = load_config()
        
        # Update configuration
        for key, value in form_data.items():
            if '.' in key:
                # Handle nested keys
                parts = key.split('.')
                
                # Navigate to the correct nested dictionary
                current_dict = current_config
                for i in range(len(parts) - 1):
                    part = parts[i]
                    
                    # Handle special case for dictionary keys with quotes
                    if part.startswith("'") and part.endswith("'"):
                        part = part[1:-1]  # Remove quotes
                    elif part.startswith("[") and part.endswith("]"):
                        # Handle array index or dictionary key with brackets
                        part = part[1:-1]  # Remove brackets
                    
                    # Create nested dictionary if it doesn't exist
                    if part not in current_dict:
                        current_dict[part] = {}
                    
                    current_dict = current_dict[part]
                
                # Set the value in the deepest level
                final_key = parts[-1]
                
                # Convert value to appropriate type
                if value.lower() == 'true':
                    current_dict[final_key] = True
                elif value.lower() == 'false':
                    current_dict[final_key] = False
                elif value.isdigit():
                    current_dict[final_key] = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                    current_dict[final_key] = float(value)
                else:
                    # Handle comma-separated lists
                    if ',' in value and not (value.startswith('{') or value.startswith('[')):
                        current_dict[final_key] = [item.strip() for item in value.split(',') if item.strip()]
                    else:
                        current_dict[final_key] = value
            else:
                # Handle top-level keys
                if value.lower() == 'true':
                    current_config[key] = True
                elif value.lower() == 'false':
                    current_config[key] = False
                elif value.isdigit():
                    current_config[key] = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                    current_config[key] = float(value)
                else:
                    # Handle comma-separated lists
                    if ',' in value and not (value.startswith('{') or value.startswith('[')):
                        current_config[key] = [item.strip() for item in value.split(',') if item.strip()]
                    else:
                        current_config[key] = value
        
        # Save configuration
        if save_config(current_config):
            return jsonify({'success': True, 'message': 'Configuration updated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Error saving configuration'})
    
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_backtest_results')
def get_backtest_results_route():
    """Get backtest results"""
    return jsonify(get_backtest_results())

@app.route('/get_processes')
def get_processes():
    """Get list of active processes"""
    try:
        processes = []
        
        for name, process_info in active_processes.items():
            if process_info is not None:
                # Get the most recent logs (limited to last 3 for display)
                logs = process_info.get('logs', [])[-3:]
                
                processes.append({
                    'name': name,
                    'status': process_info.get('status', 'Unknown'),
                    'logs': logs
                })
        
        return jsonify(processes)
    except Exception as e:
        logger.error(f"Error getting processes: {str(e)}")
        return jsonify([])

@app.route('/run_comprehensive_backtest', methods=['POST'])
def run_comprehensive_backtest():
    """Run a comprehensive backtest for multiple quarters"""
    try:
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        # Get form data - support both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
            # Handle checkbox values
            for key in ['continuous_capital', 'weekly_selection']:
                data[key] = key in request.form
            
            # Handle quarters which may be comma-separated
            if 'quarters' in data and isinstance(data['quarters'], str):
                # Handle empty string case
                if not data['quarters'].strip():
                    data['quarters'] = []
                elif ',' in data['quarters']:
                    data['quarters'] = [q.strip() for q in data['quarters'].split(',') if q.strip()]
                else:
                    data['quarters'] = [data['quarters']]  # Convert single string to list
        
        # Get parameters from form data
        quarters = data.get('quarters', [])
        
        # If no quarters selected, use the global start/end dates from the configuration
        if not quarters:
            # Get start and end dates from backtest configuration
            backtest_config = config_data.get('backtest', {})
            start_date = backtest_config.get('start_date')
            end_date = backtest_config.get('end_date')
            
            if not start_date or not end_date:
                return jsonify({'success': False, 'message': 'No quarters selected and no start/end dates in configuration'})
            
            # Create a custom quarter identifier
            quarters = ['custom_range']
            
            # Add start and end dates to data for the backtest thread
            data['custom_start_date'] = start_date
            data['custom_end_date'] = end_date
            
            logger.info(f"No quarters selected, using configuration dates: {start_date} to {end_date}")
        
        # Create a unique ID for this backtest run
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a log file for this run
        log_file = f"test_comprehensive_backtest_{run_id}.log"
        log_path = os.path.join(config_data.get('paths', {}).get('logs', './logs'), log_file)
        
        # Set up logging
        setup_logging(log_path)
        
        # Log the start of the backtest
        logger.info(f"Starting comprehensive backtest for quarters: {quarters}")
        
        # Update the active processes
        process_name = f"test_comprehensive_backtest_{run_id}"
        active_processes[process_name] = {
            'status': 'running',
            'logs': [f"Starting comprehensive backtest for quarters: {quarters}"]
        }
        
        # Create a thread to run the backtest
        thread = threading.Thread(
            target=run_backtest_thread,
            args=(quarters, run_id, process_name, data)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Backtest started for quarters: {quarters}',
            'log_file': log_file,
            'run_id': run_id,
            'process_name': process_name
        })
    
    except Exception as e:
        logger.error(f"Error starting backtest: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

def run_backtest_thread(quarters, run_id, process_name, data):
    """Run backtest in a separate thread"""
    try:
        # Get parameters from data
        max_signals = int(data.get('max_signals', 10))
        initial_capital = float(data.get('initial_capital', 10000))
        continuous_capital = data.get('continuous_capital', False)
        weekly_selection = data.get('weekly_selection', False)
        
        # Log parameters
        logger.info(f"Backtest parameters: max_signals={max_signals}, initial_capital={initial_capital}, continuous_capital={continuous_capital}, weekly_selection={weekly_selection}")
        active_processes[process_name]['logs'].append(f"Backtest parameters: max_signals={max_signals}, initial_capital={initial_capital}, continuous_capital={continuous_capital}, weekly_selection={weekly_selection}")
        
        # Import the modify_backtest_results module
        try:
            from modify_backtest_results import modify_results_for_quarter
            logger.info("Successfully imported modify_backtest_results module")
        except ImportError:
            logger.error("Failed to import modify_backtest_results module. Results will not be modified for quarters.")
            # Define a dummy function that returns the input unchanged
            def modify_results_for_quarter(result_data, quarter):
                return result_data
        
        # Track previous capital for continuous capital mode
        previous_capital = initial_capital if continuous_capital else None
        
        # Get paths from config
        results_dir = config_data.get('paths', {}).get('backtest_results', './backtest_results')
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Dictionary to store results for each quarter
        results = {}
        
        # Disable hot reload during backtest to avoid unnecessary file system operations
        hot_reload_enabled = config_data.get('hot_reload', {}).get('enabled', False)
        if hot_reload_enabled:
            config_data['hot_reload']['enabled'] = False
            logger.info("Temporarily disabled hot reload during backtest")
            
        # Run backtest for each quarter
        for quarter in quarters:
            logger.info(f"Running backtest for quarter: {quarter}")
            active_processes[process_name]['logs'].append(f"Running backtest for quarter: {quarter}")
            
            # Handle custom_range quarter
            if quarter == 'custom_range':
                start_date = data.get('custom_start_date')
                end_date = data.get('custom_end_date')
                
                if not start_date or not end_date:
                    error_msg = f"Missing custom date range parameters for custom_range quarter"
                    logger.error(error_msg)
                    active_processes[process_name]['logs'].append(error_msg)
                    
                    # Create an error result file
                    error_result = {
                        'success': False,
                        'quarter': quarter,
                        'error': error_msg,
                        'summary': {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0,
                            'win_rate': 0,
                            'avg_win': 0,
                            'avg_loss': 0,
                            'profit_factor': 0,
                            'total_pnl': 0,
                            'max_drawdown': 0,
                            'sharpe_ratio': 0,
                            'sortino_ratio': 0
                        }
                    }
                    
                    # Save error result
                    error_file = f"backtest_error_{quarter}_{run_id}.json"
                    error_path = os.path.join(results_dir, error_file)
                    
                    with open(error_path, 'w') as f:
                        json.dump(error_result, f, indent=4)
                    
                    results[quarter] = error_result
                    continue
                
                # Extract year for file naming
                year = start_date.split('-')[0]
                
                # Create a unique filename for this custom range
                result_file = f"backtest_custom_{start_date}_to_{end_date}_{run_id}.json"
                compat_file = f"results_{year}_custom_{run_id}.json"
                trades_file = f"trades_{year}_custom_{run_id}.csv"
                
                logger.info(f"Using custom date range: {start_date} to {end_date}")
                
                # Set up file paths
                result_path = os.path.join(results_dir, result_file)
                compat_path = os.path.join(results_dir, compat_file)
                trades_path = os.path.join(results_dir, trades_file)
                
                # Run the backtest with the custom date range
                try:
                    # Use turbo backtest if available
                    if 'USE_TURBO_BACKTEST' in globals() and USE_TURBO_BACKTEST:
                        logger.info(f"Running turbo backtest with custom date range: start_date={start_date}, end_date={end_date}, max_signals={max_signals}")
                        summary, trades = run_turbo_backtest(
                    # Use optimized backtest if available and turbo is not available
                    elif 'USE_OPTIMIZED_BACKTEST' in globals() and USE_OPTIMIZED_BACKTEST:
                        logger.info(f"Running optimized backtest with custom date range: start_date={start_date}, end_date={end_date}, max_signals={max_signals}")
                        summary, trades = run_optimized_backtest(
                    # Fall back to standard backtest
                    else:
                        logger.info(f"Running standard backtest with custom date range: start_date={start_date}, end_date={end_date}, max_signals={max_signals}")
                        from final_sp500_strategy import run_backtest
                        summary, trades = run_backtest(
                        start_date, 
                        end_date, 
                        mode='backtest',
                        max_signals=max_signals,
                        initial_capital=previous_capital if continuous_capital and previous_capital else initial_capital,
                        weekly_selection=weekly_selection,
                        random_seed=random_seed  # Pass unique random seed for each quarter
                    )
                    
                    # Create result structure
                    result = {
                        'summary': summary,
                        'trades': trades if trades else []
                    }
                    
                    logger.info(f"Backtest completed successfully, got result with {len(trades if trades else [])} trades")
                    # Update previous_capital for next quarter if continuous_capital is enabled
                    if continuous_capital and summary and 'final_capital' in summary:
                        previous_capital = summary['final_capital']
                        # Round to avoid floating point precision issues
                        previous_capital = round(previous_capital, 2)
                        logger.info(f"Updated previous_capital to {previous_capital} for next quarter")
                        # Add initial_capital to the result summary for display
                        if summary:
                            summary['initial_capital'] = previous_capital if continuous_capital and previous_capital else initial_capital
                            logger.info(f"Set initial_capital in result summary to {summary.get('initial_capital')}")
                    
                    # Add quarter information to the result
                    logger.info(f"Adding quarter information for: {quarter}")
                    result = modify_results_for_quarter(result, quarter)
                    logger.info(f"Quarter information added successfully for: {quarter}")
                    
                    # Store the result
                    results[quarter] = result
                    
                    # Save the result to a file
                    logger.info(f"Saving backtest results to {result_file}")
                    with open(os.path.join(results_dir, result_file), 'w') as f:
                        json.dump(result, f, indent=4, default=str)
                    logger.info(f"Results saved successfully to {result_file}")
                    
                    # Also save a file with the compatibility format
                    logger.info(f"Saving compatibility results to {compat_file}")
                    with open(os.path.join(results_dir, compat_file), 'w') as f:
                        json.dump(result, f, indent=4, default=str)
                    logger.info(f"Compatibility results saved successfully to {compat_file}")
                    
                    # Save trades to CSV
                    logger.info(f"Saving trades to {trades_file}")
                    trades_df = pd.DataFrame(trades if trades else [])
                    if not trades_df.empty:
                        trades_df.to_csv(os.path.join(results_dir, trades_file), index=False)
                    else:
                        # Create an empty CSV file
                        with open(os.path.join(results_dir, trades_file), 'w') as f:
                            f.write("symbol,entry_date,exit_date,entry_price,exit_price,shares,pnl,trade_type\n")
                    
                    logger.info(f"All files saved successfully for quarter {quarter}")
                    
                except Exception as e:
                    error_msg = f"Error running backtest for quarter {quarter}: {str(e)}"
                    logger.error(error_msg)
                    active_processes[process_name]['logs'].append(error_msg)
                    logger.error(traceback.format_exc())
                    
                    # Save error to result file
                    with open(os.path.join(results_dir, f"backtest_error_{quarter}_{run_id}.json"), 'w') as f:
                        error_result = {
                            'success': False,
                            'quarter': quarter,
                            'error': str(e),
                            'summary': {
                                'total_trades': 0,
                                'winning_trades': 0,
                                'losing_trades': 0,
                                'win_rate': 0,
                                'avg_win': 0,
                                'avg_loss': 0,
                                'profit_factor': 0,
                                'total_pnl': 0,
                                'max_drawdown': 0,
                                'sharpe_ratio': 0,
                                'sortino_ratio': 0
                            }
                        }
                        json.dump(error_result, f, indent=4)
            
                results[quarter] = error_result
        
        # Create a combined result file with all quarters
        combined_results = {
            'run_id': run_id,
            'quarters': quarters,
            'results': results
        }
        
        # Save combined results
        combined_file = f"backtest_combined_{run_id}.json"
        combined_path = os.path.join(results_dir, combined_file)
        
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=4, default=str)
        
        logger.info(f"Combined results saved to: {combined_path}")
        active_processes[process_name]['logs'].append(f"Combined results saved to: {combined_path}")
        active_processes[process_name]['status'] = 'completed'
        active_processes[process_name]['logs'].append("Backtest process completed")
        
        # Re-enable hot reload if it was enabled before
        if hot_reload_enabled:
            config_data['hot_reload']['enabled'] = True
            logger.info("Re-enabled hot reload after backtest")
    
    except Exception as e:
        error_msg = f"Error in backtest thread: {str(e)}"
        logger.error(error_msg)
        if process_name in active_processes:
            active_processes[process_name]['status'] = 'failed'
            active_processes[process_name]['logs'].append(error_msg)
            active_processes[process_name]['logs'].append(traceback.format_exc())

@app.route('/view_backtest_result/<path:filename>')
def view_backtest_result(filename):
    """View backtest result"""
    try:
        # Initialize default_summary at the beginning to avoid UnboundLocalError
        default_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0
        }
        
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        # Get backtest results directory from config
        results_dir = config_data.get('paths', {}).get('backtest_results', './backtest_results')
        
        # Construct full path
        result_path = os.path.join(results_dir, filename)
        
        # Log the file path for debugging
        logger.info(f"Attempting to access backtest result file: {result_path}")
        
        # Check if file exists
        if not os.path.exists(result_path):
            logger.error(f"File not found: {result_path}")
            
            # Try to find the file with a similar name
            similar_files = []
            for file in os.listdir(results_dir):
                if filename.split('_')[0] in file and filename.split('_')[1] in file:
                    similar_files.append(file)
            
            if similar_files:
                logger.info(f"Found similar files: {similar_files}")
                return render_template('error.html', 
                                      error=f'Backtest result file not found: {filename}. Similar files found: {", ".join(similar_files[:5])}')
            else:
                return render_template('error.html', error=f'Backtest result file not found: {filename}')
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        # Extract quarter information from filename
        # Try different patterns:
        # 1. backtest_Q1_2023_timestamp.json
        # 2. results_2024_Q1_timestamp.json
        # 3. backtest_custom_2023-01-01_to_2023-12-31_timestamp.json
        
        quarter_info = "Unknown Quarter"
        
        # Pattern 1: backtest_Q1_2023_...
        pattern1_match = re.search(r'backtest_(Q\d)_(\d{4})', filename)
        if pattern1_match:
            quarter = pattern1_match.group(1)
            year = pattern1_match.group(2)
            quarter_info = f"{year} {quarter}"
        else:
            # Pattern 2: results_2024_Q1_...
            pattern2_match = re.search(r'results_(\d{4})_(Q\d)', filename)
            if pattern2_match:
                year = pattern2_match.group(1)
                quarter = pattern2_match.group(2)
                quarter_info = f"{year} {quarter}"
            else:
                # Pattern 3: Custom date range
                custom_match = re.search(r'custom_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', filename)
                if custom_match:
                    start_date = custom_match.group(1)
                    end_date = custom_match.group(2)
                    quarter_info = f"Custom Range: {start_date} to {end_date}"
                else:
                    # Try to extract combined results pattern
                    combined_match = re.search(r'combined_(\d{8}_\d{6})', filename)
                    if combined_match:
                        run_id = combined_match.group(1)
                        quarter_info = f"Combined Results (Run ID: {run_id})"
        
        # Process based on file type
        if ext.lower() == '.csv':
            try:
                # Read CSV file
                df = pd.read_csv(result_path)
                
                # Calculate summary statistics
                total_trades = len(df)
                winning_trades = len(df[df['pnl'] > 0])
                losing_trades = len(df[df['pnl'] < 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                total_pnl = df['pnl'].sum()
                
                # Create summary dictionary
                summary = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win * winning_trades) / abs(avg_loss * losing_trades) if avg_loss != 0 and losing_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'max_drawdown': 0,  # Placeholder, would need to calculate
                    'sharpe_ratio': 0,  # Placeholder, would need to calculate
                    'sortino_ratio': 0  # Placeholder, would need to calculate
                }
                
                # Convert DataFrame to HTML
                trades_html = df.to_html(classes='table table-striped table-sm')
                
                return render_template(
                    'backtest_result.html',
                    filename=filename,
                    quarter_info=quarter_info,
                    summary=summary,
                    trades_html=trades_html
                )
                
            except Exception as e:
                logger.error(f"Error processing CSV file: {str(e)}")
                return render_template('error.html', error=f'Error processing CSV file: {str(e)}')
                
        elif ext.lower() == '.json':
            try:
                # Read JSON file
                with open(result_path, 'r') as f:
                    data = json.load(f)
                
                # Format the JSON data with proper indentation for display
                formatted_json = json.dumps(data, indent=4, sort_keys=True, default=str)
                
                # Check if this is a combined results file
                if 'combined' in filename:
                    # For combined results, we'll display each quarter's summary
                    quarters_html = ""
                    
                    for quarter, quarter_data in data.items():
                        # Extract summary for this quarter
                        quarter_summary = quarter_data.get('summary', default_summary)
                        
                        # Create HTML for this quarter
                        quarters_html += f"<h4>{quarter}</h4>"
                        quarters_html += "<table class='table table-striped table-sm'>"
                        quarters_html += "<tr><th>Metric</th><th>Value</th></tr>"
                        
                        for key, value in quarter_summary.items():
                            # Format the value based on type
                            if isinstance(value, float):
                                formatted_value = f"{value:.4f}"
                            else:
                                formatted_value = str(value)
                            
                            quarters_html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
                        
                        quarters_html += "</table>"
                    
                    return render_template(
                        'backtest_result.html',
                        filename=filename,
                        quarter_info=quarter_info,
                        summary=default_summary,  # Not used for combined results
                        trades_html="",  # Not used for combined results
                        quarters_html=quarters_html,
                        json_data=formatted_json
                    )
                else:
                    # For single quarter results
                    # Extract summary from the data
                    summary = data.get('summary', default_summary)
                    
                    # If summary is None or empty, use default_summary
                    if not summary:
                        summary = default_summary
                    
                    # Create HTML table for trades if available
                    trades_html = ""
                    trades = data.get('trades', [])
                    
                    if trades:
                        trades_html = "<table class='table table-striped table-sm'>"
                        trades_html += "<tr><th>Symbol</th><th>Direction</th><th>Signal Score</th><th>Price</th><th>Sector</th><th>Market Regime</th></tr>"
                        
                        for trade in trades:
                            trades_html += f"<tr>"
                            trades_html += f"<td>{trade.get('symbol', '')}</td>"
                            trades_html += f"<td>{trade.get('direction', '')}</td>"
                            trades_html += f"<td>{trade.get('score', '')}</td>"
                            trades_html += f"<td>${trade.get('price', ''):.2f}</td>" if trade.get('price') else f"<td></td>"
                            trades_html += f"<td>{trade.get('sector', '')}</td>"
                            trades_html += f"<td>{trade.get('market_regime', '')}</td>"
                            trades_html += f"</tr>"
                        
                        trades_html += "</table>"
                    
                    return render_template(
                        'backtest_result.html',
                        filename=filename,
                        quarter_info=quarter_info,
                        summary=summary,
                        trades_html=trades_html,
                        json_data=formatted_json
                    )
                
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON file: {str(e)}")
                return render_template('error.html', error=f'Error decoding JSON file: {str(e)}')
                
            except Exception as e:
                logger.error(f"Error processing JSON file: {str(e)}")
                traceback.print_exc()
                return render_template('error.html', error=f'Error processing JSON file: {str(e)}')
                
        else:
            return render_template('error.html', error=f'Unsupported file type: {ext}')
            
    except Exception as e:
        logger.error(f"Error viewing backtest result: {str(e)}")
        traceback.print_exc()
        return render_template('error.html', error=f'Error: {str(e)}')

@app.route('/view_logs')
def view_logs():
    """View application logs"""
    try:
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        # Get logs directory from config
        logs_dir = config_data.get('paths', {}).get('logs', './logs')
        
        # Create logs directory if it doesn't exist
        os.makedirs(logs_dir, exist_ok=True)
        
        # Get list of log files
        logs = []
        if os.path.exists(logs_dir):
            for file in os.listdir(logs_dir):
                if file.endswith('.log'):
                    file_path = os.path.join(logs_dir, file)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    size = os.path.getsize(file_path)
                    logs.append({
                        'filename': file,
                        'path': file_path,
                        'modified': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'size_kb': f"{size / 1024:.1f}"
                    })
        
        # Sort by date (newest first)
        logs.sort(key=lambda x: x['modified'], reverse=True)
        
        # Get current log settings
        log_level = config_data.get('logging', {}).get('level', 'INFO')
        log_to_file = config_data.get('logging', {}).get('to_file', True)
        
        return render_template('logs.html', logs=logs, log_level=log_level, log_to_file=log_to_file)
    
    except Exception as e:
        logger.error(f"Error viewing logs: {str(e)}")
        return render_template('logs.html', logs=[], log_level='INFO', log_to_file=True, error=str(e))

@app.route('/view_log/<path:log_file>')
def view_log_file(log_file):
    """View contents of a log file"""
    try:
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        # Get logs directory from config
        logs_dir = config_data.get('paths', {}).get('logs', './logs')
        
        # Construct full path (ensure it's within logs directory for security)
        log_path = os.path.join(logs_dir, os.path.basename(log_file))
        
        # Log debugging information
        logger.info(f"Attempting to read log file: {log_path}")
        
        if not os.path.exists(log_path):
            logger.warning(f"Log file not found: {log_path}")
            return jsonify({'success': False, 'message': 'Log file not found'})
        
        # Get file size
        file_size = os.path.getsize(log_path)
        logger.info(f"Log file size: {file_size} bytes")
        
        # Read log file content
        try:
            if file_size == 0:
                # For empty files, check if there's a process with the same name
                process_name = os.path.basename(log_file).replace('.log', '')
                
                # Check if this is a running process
                if process_name in active_processes:
                    process_status = active_processes[process_name]['status']
                    process_logs = active_processes[process_name]['logs']
                    
                    if process_logs:
                        content = '\n'.join(process_logs)
                    else:
                        content = f"Process {process_name} is {process_status} but no logs have been captured yet."
                else:
                    # Try to find the most recent non-empty log file with a similar name
                    similar_logs = []
                    for log_filename in os.listdir(logs_dir):
                        if process_name in log_filename and os.path.getsize(os.path.join(logs_dir, log_filename)) > 0:
                            similar_logs.append((log_filename, os.path.getmtime(os.path.join(logs_dir, log_filename))))
                    
                    if similar_logs:
                        # Sort by modification time (newest first)
                        similar_logs.sort(key=lambda x: x[1], reverse=True)
                        newest_log = similar_logs[0][0]
                        
                        # Read the content of the newest similar log
                        with open(os.path.join(logs_dir, newest_log), 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        
                        content = f"NOTE: This log file is empty. Showing content from similar log file: {newest_log}\n\n{content}"
                    else:
                        content = "Log file exists but is empty. The process may have completed without generating any output."
            else:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
            
            # Return the content
            return jsonify({
                'success': True, 
                'content': content,
                'file_size': file_size
            })
        except Exception as e:
            logger.error(f"Error reading log file: {str(e)}")
            return jsonify({'success': False, 'message': f'Error reading log file: {str(e)}'})
    
    except Exception as e:
        logger.error(f"Error viewing log file: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/debug_log/<path:log_file>')
def debug_log_file(log_file):
    """Debug view for log file contents"""
    try:
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        # Get logs directory from config
        logs_dir = config_data.get('paths', {}).get('logs', './logs')
        
        # Construct full path
        log_path = os.path.join(logs_dir, os.path.basename(log_file))
        
        # Log debugging information
        logger.info(f"Debug: Attempting to read log file: {log_path}")
        
        if not os.path.exists(log_path):
            logger.warning(f"Log file not found: {log_path}")
            return jsonify({'success': False, 'message': 'Log file not found'})
        
        # Get file size
        file_size = os.path.getsize(log_path)
        
        if file_size == 0:
            # For empty files, check if there's a process with the same name
            process_name = os.path.basename(log_file).replace('.log', '')
            
            # Check if this is a running process
            if process_name in active_processes:
                process_status = active_processes[process_name]['status']
                process_logs = active_processes[process_name]['logs']
                
                if process_logs:
                    content = '\n'.join(process_logs)
                else:
                    content = f"Process {process_name} is {process_status} but no logs have been captured yet."
            else:
                # Try to find the most recent non-empty log file with a similar name
                similar_logs = []
                for log_filename in os.listdir(logs_dir):
                    if process_name in log_filename and os.path.getsize(os.path.join(logs_dir, log_filename)) > 0:
                        similar_logs.append((log_filename, os.path.getmtime(os.path.join(logs_dir, log_filename))))
                
                if similar_logs:
                    # Sort by modification time (newest first)
                    similar_logs.sort(key=lambda x: x[1], reverse=True)
                    newest_log = similar_logs[0][0]
                    
                    # Read the content of the newest similar log
                    with open(os.path.join(logs_dir, newest_log), 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    content = f"NOTE: This log file is empty. Showing content from similar log file: {newest_log}\n\n{content}"
                else:
                    content = "Log file exists but is empty. The process may have completed without generating any output."
        else:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
        
        # Return the content
        return jsonify({
            'success': True, 
            'content': content,
            'file_size': file_size
        })
    except Exception as e:
        logger.error(f"Error viewing log file: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/update_log_settings', methods=['POST'])
def update_log_settings():
    """Update log settings"""
    try:
        # Get form data
        log_level = request.form.get('log_level', 'INFO')
        log_to_file = 'log_to_file' in request.form
        
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        # Update config
        if 'logging' not in config_data:
            config_data['logging'] = {}
        
        config_data['logging']['level'] = log_level
        config_data['logging']['to_file'] = log_to_file
        
        # Save updated config
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Updated log settings: level={log_level}, to_file={log_to_file}")
        
        # Apply new log level to current logger
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        flash('Log settings updated successfully', 'success')
        return redirect(url_for('view_logs'))
    
    except Exception as e:
        logger.error(f"Error updating log settings: {str(e)}")
        flash(f'Error updating log settings: {str(e)}', 'danger')
        return redirect(url_for('view_logs'))

@app.route('/run_paper_trading', methods=['POST'])
def run_paper_trading():
    """Run paper trading"""
    try:
        # Get form data
        max_signals = request.form.get('max_signals', '20')
        duration = request.form.get('duration', '1')
        interval = request.form.get('interval', '5')
        
        # Create process name
        process_name = f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start process in a thread
        args = ['--max_signals', max_signals, '--duration', duration, '--interval', interval]
        thread = threading.Thread(
            target=run_process,
            args=('run_paper_trading.py', args, process_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Paper trading started with process name: {process_name}',
            'process_name': process_name
        })
    except Exception as e:
        logger.error(f"Error starting paper trading: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run market simulation"""
    try:
        # Get form data
        days = request.form.get('days', '30')
        capital = request.form.get('capital', '100000')
        max_signals = request.form.get('max_signals', '20')
        interval = request.form.get('interval', '5')
        
        # Create process name
        process_name = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start process in a thread
        args = ['--days', days, '--capital', capital, '--max_signals', max_signals, '--interval', interval]
        thread = threading.Thread(
            target=run_process,
            args=('run_market_simulation.py', args, process_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Market simulation started with process name: {process_name}',
            'process_name': process_name
        })
    except Exception as e:
        logger.error(f"Error starting market simulation: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/run_live_trading', methods=['POST'])
def run_live_trading():
    """Run live trading"""
    try:
        # Get form data
        max_signals = request.form.get('max_signals', '10')
        check_interval = request.form.get('check_interval', '5')
        max_capital = request.form.get('max_capital', '50000')
        risk_level = request.form.get('risk_level', 'medium')
        
        # Create process name
        process_name = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if credentials file exists
        credentials_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alpaca_credentials.json')
        if not os.path.exists(credentials_file):
            return jsonify({
                'success': False,
                'message': 'Error: Alpaca credentials file not found. Please create alpaca_credentials.json with your API keys.'
            })
        
        # Start process in a thread
        args = [
            '--max_signals', max_signals,
            '--check_interval', check_interval,
            '--max_capital', max_capital,
            '--risk_level', risk_level,
            '--live'  # Important flag to indicate live trading
        ]
        thread = threading.Thread(
            target=run_process,
            args=('run_live_trading.py', args, process_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Live trading started with process name: {process_name}',
            'process_name': process_name
        })
    except Exception as e:
        logger.error(f"Error starting live trading: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/stop_process/<process_name>', methods=['POST'])
def stop_process_route(process_name):
    """Stop a running process"""
    if stop_process(process_name):
        return jsonify({'success': True, 'message': f'Process {process_name} stopped'})
    else:
        return jsonify({'success': False, 'message': f'Error stopping process {process_name}'})

@app.route('/process_logs/<process_name>')
def get_process_logs(process_name):
    """Get logs for a process"""
    if process_name in active_processes:
        return jsonify({
            'success': True,
            'logs': active_processes[process_name]['logs'],
            'status': active_processes[process_name]['status']
        })
    else:
        return jsonify({'success': False, 'message': f'Process {process_name} not found'})

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop_route():
    """Emergency stop all processes and close all positions"""
    success, message = emergency_stop()
    return jsonify({'success': success, 'message': message})

@app.route('/get_positions')
def get_positions_route():
    """Get open positions"""
    positions = get_open_positions()
    return jsonify({'success': True, 'positions': positions})

@app.route('/restart_server', methods=['POST', 'OPTIONS'])
def restart_server():
    """Restart the Flask server"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        logger.info("Restart server request received")
        
        # Create a simple batch file to kill and restart the server
        restart_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'restart_server.bat')
        
        # Get the current process ID
        current_pid = os.getpid()
        
        # Get the path to the Python executable and the app.py file
        python_exe = sys.executable
        app_path = os.path.abspath(__file__)
        app_dir = os.path.dirname(app_path)
        parent_dir = os.path.dirname(app_dir)
        
        # Create the batch file content
        batch_content = f"""@echo off
echo Restarting Flask server...
timeout /t 2 /nobreak > nul
taskkill /F /PID {current_pid} > nul 2>&1
cd /d {parent_dir}
start "" "{python_exe}" "{app_path}"
exit
"""
        
        # Write the batch file
        with open(restart_script, 'w') as f:
            f.write(batch_content)
        
        # Make the batch file executable (not needed on Windows)
        
        # Start the batch file in a new process
        subprocess.Popen(['cmd', '/c', restart_script], 
                        shell=True,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        
        # Return success response
        return jsonify({"success": True, "message": "Server restart initiated"})
    except Exception as e:
        logger.error(f"Error restarting server: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

if __name__ == '__main__':
    # Load configuration
    load_config()
    
    # Create required directories
    if config_data:
        for path_key, path_value in config_data.get('paths', {}).items():
            # Skip if the path is a file path (contains file extension)
            if '.' in os.path.basename(path_value):
                continue
            os.makedirs(path_value, exist_ok=True)
    
    # Start hot reload for configuration files
    hot_reloader = start_hot_reload(app, config_file)
    logger.info("Hot reload enabled for configuration files")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

@app.route('/emergency_shutdown', methods=['POST'])
def emergency_shutdown():
    """Emergency shutdown endpoint to close all positions"""
    try:
        # Create a script file with the emergency shutdown code
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emergency_shutdown.py')
        with open(script_path, 'w') as f:
            f.write('''
import sys
import os
import logging
import yaml
import time
from datetime import datetime

# Set up logging
log_file = f"emergency_shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from file"""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to the project root
        project_root = os.path.dirname(script_dir)
        
        # Construct the path to the config file
        config_path = os.path.join(project_root, 'sp500_config.yaml')
        
        # Check if the file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return None
        
        # Load the configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def close_all_positions():
    """Close all open positions"""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Import required modules
        try:
            from alpaca_api import AlpacaAPI
        except ImportError:
            logger.error("Failed to import AlpacaAPI")
            return
        
        # Initialize API
        api = AlpacaAPI(
            api_key=config.get('alpaca', {}).get('api_key', ''),
            api_secret=config.get('alpaca', {}).get('api_secret', ''),
            paper=config.get('alpaca', {}).get('paper', True)
        )
        
        # Get all positions
        positions = api.get_positions()
        
        # Log positions
        logger.info(f"Found {len(positions)} open positions")
        
        # Close each position
        for position in positions:
            symbol = position.symbol
            qty = abs(int(float(position.qty)))
            side = 'sell' if float(position.qty) > 0 else 'buy'
            
            logger.info(f"Closing position: {symbol} - {qty} shares - {side}")
            
            try:
                # Close position
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='gtc'
                )
                
                logger.info(f"Successfully closed position: {symbol}")
                
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {str(e)}")
        
        # Wait for orders to complete
        logger.info("Waiting for orders to complete...")
        time.sleep(5)
        
        # Check if all positions are closed
        remaining_positions = api.get_positions()
        if remaining_positions:
            logger.warning(f"Still have {len(remaining_positions)} open positions")
        else:
            logger.info("All positions closed successfully")
    
    except Exception as e:
        logger.error(f"Error in close_all_positions: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting emergency shutdown procedure")
    close_all_positions()
    logger.info("Emergency shutdown completed")
''')
        
        # Run the script
        cmd = [sys.executable, script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Emergency shutdown completed successfully'})
        else:
            return jsonify({'success': False, 'message': f'Emergency shutdown failed: {result.stderr}'})
    except Exception as e:
        logger.error(f"Error in emergency shutdown: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    results_dir = config_data.get('paths', {}).get('backtest_results', './backtest_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    logs_dir = config_data.get('paths', {}).get('logs', './logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
