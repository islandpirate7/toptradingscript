#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web Interface for S&P 500 Trading Strategy (Updated)
This Flask application provides a web interface to control the trading strategy
with an updated backtest engine that doesn't rely on the Alpaca Trade API
"""

import os
import sys
import json
import yaml
import time
import logging
import subprocess
import threading
import traceback
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import re

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our updated backtest engine
try:
    from backtest_engine_updated import run_backtest
    USE_UPDATED_BACKTEST = True
    print("Using updated backtest engine")
except ImportError:
    USE_UPDATED_BACKTEST = False
    print("Updated backtest engine not found, falling back to standard backtest")

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
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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
                    
                    # Skip if not a JSON file
                    if not file.endswith('.json'):
                        continue
                    
                    # Extract quarter information from filename
                    quarter_match = re.search(r'backtest_([^_]+)_', file)
                    date_range_match = re.search(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', file)
                    
                    quarter = None
                    date_range = None
                    
                    if quarter_match:
                        quarter = quarter_match.group(1)
                    
                    if date_range_match:
                        date_range = f"{date_range_match.group(1)} to {date_range_match.group(2)}"
                    
                    # Skip if we can't determine quarter or date range
                    if not quarter and not date_range:
                        continue
                    
                    # Create a unique key for this result
                    result_key = f"{quarter}_{date_range}" if quarter and date_range else (quarter or date_range)
                    
                    # Skip if we've already processed a file for this quarter/date range
                    if result_key in quarter_groups:
                        # Only replace if this file is newer
                        existing_file = quarter_groups[result_key]['file']
                        existing_time = os.path.getmtime(os.path.join(abs_results_dir, existing_file))
                        current_time = os.path.getmtime(file_path)
                        
                        if current_time <= existing_time:
                            continue
                    
                    # Try to load the JSON file
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Extract summary information
                        summary = data.get('summary', {})
                        
                        # Skip if summary is empty or None
                        if not summary:
                            continue
                        
                        # Create result object
                        result = {
                            'file': file,
                            'path': file_path,
                            'quarter': quarter,
                            'date_range': date_range,
                            'win_rate': summary.get('win_rate', 0),
                            'profit_factor': summary.get('profit_factor', 0),
                            'total_return': summary.get('total_return', 0),
                            'num_trades': summary.get('num_trades', 0),
                            'timestamp': os.path.getmtime(file_path)
                        }
                        
                        # Store in quarter groups
                        quarter_groups[result_key] = result
                    
                    except Exception as e:
                        logger.error(f"Error loading backtest result file {file}: {str(e)}")
                        continue
                
                # Add all quarter groups to results
                results.extend(quarter_groups.values())
            
            except Exception as e:
                logger.error(f"Error processing backtest results directory {results_dir}: {str(e)}")
    
    # Sort results by timestamp (newest first)
    results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    
    # Update cache
    get_backtest_results.cache = {'results': results, 'last_update': current_time}
    
    return results

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
                    # Use our updated backtest engine if available
                    if USE_UPDATED_BACKTEST:
                        logger.info(f"Running updated backtest with custom date range: start_date={start_date}, end_date={end_date}, max_signals={max_signals}")
                        summary, trades = run_backtest(
                            start_date, 
                            end_date, 
                            mode='backtest',
                            max_signals=max_signals,
                            initial_capital=previous_capital if continuous_capital and previous_capital else initial_capital,
                            weekly_selection=weekly_selection
                        )
                    # Fall back to standard backtest if updated engine is not available
                    else:
                        logger.info(f"Running standard backtest with custom date range: start_date={start_date}, end_date={end_date}, max_signals={max_signals}")
                        from final_sp500_strategy import run_backtest as original_run_backtest
                        summary, trades = original_run_backtest(
                            start_date, 
                            end_date, 
                            mode='backtest',
                            max_signals=max_signals,
                            initial_capital=previous_capital if continuous_capital and previous_capital else initial_capital,
                            weekly_selection=weekly_selection
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
        
        logger.info(f"Combined results saved to {combined_file}")
        
        # Re-enable hot reload if it was enabled before
        if hot_reload_enabled:
            config_data['hot_reload']['enabled'] = True
            logger.info("Re-enabled hot reload after backtest")
        
        # Update process status
        active_processes[process_name]['status'] = 'completed'
        active_processes[process_name]['logs'].append(f"Backtest completed for quarters: {quarters}")
        
        logger.info(f"Backtest completed for quarters: {quarters}")
    
    except Exception as e:
        logger.error(f"Error in backtest thread: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update process status
        active_processes[process_name]['status'] = 'error'
        active_processes[process_name]['logs'].append(f"Error: {str(e)}")
        
        # Re-enable hot reload if it was enabled before
        if 'hot_reload_enabled' in locals() and hot_reload_enabled:
            config_data['hot_reload']['enabled'] = True
            logger.info("Re-enabled hot reload after backtest error")

@app.route('/view_backtest_result', methods=['GET'])
def view_backtest_result():
    """View backtest result"""
    try:
        # Initialize default_summary to avoid the "cannot access local variable" error
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
            'sortino_ratio': 0,
            'total_return': 0,
            'num_trades': 0
        }
        
        # Get file parameter
        file = request.args.get('file')
        
        if not file:
            return jsonify({'success': False, 'message': 'No file specified'})
        
        # Check multiple possible locations for backtest results
        results_dirs = [
            os.path.join(os.path.dirname(__file__), 'backtest_results'),  # Web interface directory
            os.path.join('..', 'backtest_results'),  # Relative path to root directory
            'backtest_results'  # Direct path in case the app is run from the root directory
        ]
        
        # Initialize variables
        result_data = None
        file_path = None
        
        # Try to find the file in each results directory
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                temp_path = os.path.join(results_dir, file)
                if os.path.exists(temp_path):
                    file_path = temp_path
                    break
        
        # If file not found, return error
        if not file_path:
            logger.error(f"Backtest result file not found: {file}")
            return jsonify({
                'success': False, 
                'message': f'Backtest result file not found: {file}',
                'summary': default_summary,
                'trades': []
            })
        
        # Load the result file
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
            
            # Extract summary and trades
            summary = result_data.get('summary', {})
            trades = result_data.get('trades', [])
            
            # If summary is None or empty, use default summary
            if not summary:
                logger.warning(f"No summary found in result file: {file}")
                summary = default_summary
            
            # Return result
            return jsonify({
                'success': True,
                'message': 'Backtest result loaded successfully',
                'summary': summary,
                'trades': trades
            })
        
        except Exception as e:
            logger.error(f"Error loading backtest result file {file}: {str(e)}")
            return jsonify({
                'success': False, 
                'message': f'Error loading backtest result file: {str(e)}',
                'summary': default_summary,
                'trades': []
            })
    
    except Exception as e:
        logger.error(f"Error viewing backtest result: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error: {str(e)}',
            'summary': default_summary,
            'trades': []
        })

@app.route('/get_backtest_results', methods=['GET'])
def get_backtest_results_route():
    """Get backtest results"""
    try:
        results = get_backtest_results()
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}', 'results': []})

@app.route('/get_config', methods=['GET'])
def get_config():
    """Get configuration"""
    try:
        # Load configuration if not loaded
        if not config_data:
            load_config()
        
        return jsonify({'success': True, 'config': config_data})
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        # Get form data
        data = request.get_json()
        
        # Update configuration
        config_data.update(data)
        
        # Save configuration
        save_config(config_data)
        
        return jsonify({'success': True, 'message': 'Configuration updated successfully'})
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_process_status', methods=['GET'])
def get_process_status():
    """Get process status"""
    try:
        # Get process name
        process_name = request.args.get('process_name')
        
        if not process_name:
            return jsonify({'success': False, 'message': 'No process name specified'})
        
        # Get process status
        if process_name in active_processes:
            return jsonify({
                'success': True,
                'status': active_processes[process_name]['status'],
                'logs': active_processes[process_name]['logs']
            })
        else:
            return jsonify({'success': False, 'message': f'Process {process_name} not found'})
    except Exception as e:
        logger.error(f"Error getting process status: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/templates/<path:path>')
def serve_templates(path):
    """Serve template files"""
    return send_from_directory('templates', path)

@app.route('/backtest_results/<path:path>')
def serve_backtest_results(path):
    """Serve backtest result files"""
    # Check multiple possible locations for backtest results
    results_dirs = [
        os.path.join(os.path.dirname(__file__), 'backtest_results'),  # Web interface directory
        os.path.join('..', 'backtest_results'),  # Relative path to root directory
        'backtest_results'  # Direct path in case the app is run from the root directory
    ]
    
    # Try to find the file in each results directory
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            file_path = os.path.join(results_dir, path)
            if os.path.exists(file_path):
                return send_from_directory(results_dir, path)
    
    # If file not found, return 404
    return "File not found", 404

# Main function
def main():
    """Main function"""
    # Load configuration
    load_config()
    
    # Start hot reload if enabled
    if config_data.get('hot_reload', {}).get('enabled', False):
        start_hot_reload()
    
    # Run the app
    app.run(
        host=config_data.get('web_interface', {}).get('host', '127.0.0.1'),
        port=config_data.get('web_interface', {}).get('port', 5000),
        debug=config_data.get('web_interface', {}).get('debug', False)
    )

# Run the app if this file is executed directly
if __name__ == '__main__':
    main()
