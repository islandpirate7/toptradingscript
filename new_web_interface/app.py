#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
New Web Interface for S&P 500 Trading Strategy
This Flask application provides a web interface to control the trading strategy
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
from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
def setup_logging(log_file=None):
    """Set up logging for the application"""
    # Create a logger
    logger = logging.getLogger('new_web_interface')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create a file handler if a log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logging with a daily log file
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"new_web_interface_{datetime.now().strftime('%Y%m%d')}.log")
logger = setup_logging(log_file)

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
        return {}

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
    
    # Check multiple possible locations for backtest results
    results_dirs = [
        os.path.join(os.path.dirname(__file__), 'backtest_results'),  # Web interface directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backtest_results'),  # Root directory
        'backtest_results'  # Direct path in case the app is run from the root directory
    ]
    
    # Use a cache to avoid repeated file system operations
    if not hasattr(get_backtest_results, 'cache'):
        get_backtest_results.cache = {'results': [], 'last_update': 0}
    
    # Only refresh the cache every 60 seconds
    current_time = time.time()
    if current_time - get_backtest_results.cache['last_update'] < 60:
        return get_backtest_results.cache['results']
    
    # Track already processed files by absolute path to avoid duplicates
    processed_files = set()
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            # Convert to absolute path for consistent tracking
            abs_results_dir = os.path.abspath(results_dir)
            logger.info(f"Checking backtest results in: {abs_results_dir}")
            
            # Skip if we've already processed this directory
            if abs_results_dir in processed_files:
                continue
                
            # Mark this directory as processed
            processed_files.add(abs_results_dir)
            
            try:
                files = os.listdir(abs_results_dir)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(abs_results_dir, x)), reverse=True)
                
                # Limit the number of files to process
                max_files_to_process = 100
                files = files[:max_files_to_process]
                
                for file in files:
                    if file.endswith('.json') and file.startswith('backtest_'):
                        file_path = os.path.join(abs_results_dir, file)
                        abs_file_path = os.path.abspath(file_path)
                        
                        # Skip if we've already processed this file
                        if abs_file_path in processed_files:
                            continue
                        
                        # Mark this file as processed
                        processed_files.add(abs_file_path)
                        
                        try:
                            # Get file modification time
                            mod_time = os.path.getmtime(file_path)
                            mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Extract quarter and date range from filename
                            # Format: backtest_Q1_2023_2023-01-01_to_2023-03-31_20250325_123456.json
                            parts = file.replace('backtest_', '').replace('.json', '').split('_')
                            
                            quarter = None
                            date_range = None
                            
                            # Try to extract quarter and date range
                            if len(parts) >= 2 and parts[0].startswith('Q') and parts[1].isdigit():
                                quarter = f"{parts[0]}_{parts[1]}"
                                
                                # Try to extract date range
                                if 'to' in file:
                                    date_range_parts = file.split('_to_')
                                    if len(date_range_parts) >= 2:
                                        start_date = date_range_parts[0].split('_')[-1]
                                        end_date = date_range_parts[1].split('_')[0]
                                        date_range = f"{start_date} to {end_date}"
                            
                            # Add result to list
                            results.append({
                                'filename': file,
                                'path': file_path,
                                'quarter': quarter,
                                'date_range': date_range,
                                'modified': mod_time_str
                            })
                        except Exception as e:
                            logger.error(f"Error processing backtest result file {file}: {str(e)}")
            except Exception as e:
                logger.error(f"Error listing backtest results in {results_dir}: {str(e)}")
    
    # Sort results by modification time (newest first)
    results.sort(key=lambda x: x.get('modified', ''), reverse=True)
    
    # Update cache
    get_backtest_results.cache = {
        'results': results,
        'last_update': current_time
    }
    
    return results

def run_process(script, args, process_name):
    """Run a Python script as a subprocess and capture its output"""
    try:
        # Check if process is already running
        if process_name in active_processes and active_processes[process_name]['process'].poll() is None:
            logger.warning(f"Process {process_name} is already running")
            return False, f"Process {process_name} is already running"
        
        # Prepare command
        cmd = [sys.executable, script]
        cmd.extend(args)
        
        # Log command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Create process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Store process information
        active_processes[process_name] = {
            'process': process,
            'command': ' '.join(cmd),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'logs': []
        }
        
        # Start thread to read output
        def read_output():
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    logger.info(f"[{process_name}] {line}")
                    active_processes[process_name]['logs'].append(line)
            
            # Process completed
            return_code = process.poll()
            logger.info(f"Process {process_name} completed with return code {return_code}")
            active_processes[process_name]['return_code'] = return_code
            active_processes[process_name]['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            active_processes[process_name]['status'] = 'completed' if return_code == 0 else 'failed'
        
        # Start output reading thread
        threading.Thread(target=read_output, daemon=True).start()
        
        return True, f"Started process {process_name}"
    except Exception as e:
        logger.error(f"Error running process: {str(e)}")
        return False, f"Error running process: {str(e)}"

def stop_process(process_name):
    """Stop a running process"""
    try:
        if process_name in active_processes and active_processes[process_name]['process'].poll() is None:
            # Process is running, terminate it
            active_processes[process_name]['process'].terminate()
            logger.info(f"Terminated process {process_name}")
            active_processes[process_name]['status'] = 'terminated'
            active_processes[process_name]['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return True, f"Terminated process {process_name}"
        else:
            logger.warning(f"Process {process_name} is not running")
            return False, f"Process {process_name} is not running"
    except Exception as e:
        logger.error(f"Error stopping process: {str(e)}")
        return False, f"Error stopping process: {str(e)}"

def emergency_stop():
    """Stop all running processes"""
    try:
        stopped_processes = []
        
        # Stop all running processes
        for process_name in list(active_processes.keys()):
            if active_processes[process_name]['process'].poll() is None:
                success, message = stop_process(process_name)
                if success:
                    stopped_processes.append(process_name)
        
        if stopped_processes:
            logger.info(f"Emergency stop completed. Stopped processes: {', '.join(stopped_processes)}")
            return True, f"Emergency stop completed. Stopped processes: {', '.join(stopped_processes)}"
        else:
            logger.info("Emergency stop completed. No running processes to stop.")
            return True, "Emergency stop completed. No running processes to stop."
    except Exception as e:
        logger.error(f"Error during emergency stop: {str(e)}")
        return False, f"Error during emergency stop: {str(e)}"

# Flask routes
@app.route('/')
def index():
    """Render the main dashboard"""
    # Load configuration if not loaded
    if not config_data:
        load_config()
    
    return render_template('index.html', config=config_data)

@app.route('/config')
def config_page():
    """Render the configuration page"""
    # Load configuration if not loaded
    if not config_data:
        load_config()
    
    return render_template('config.html', config=config_data)

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        # Load current configuration
        current_config = load_config()
        
        # Get form data
        form_data = request.form
        
        # Update configuration
        for key in form_data:
            if key.startswith('config.'):
                # Split key into parts
                parts = key.split('.')
                
                # Skip first part (config)
                parts = parts[1:]
                
                # Get value
                value = form_data[key]
                
                # Convert value to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)
                
                # Update configuration
                if len(parts) == 1:
                    current_config[parts[0]] = value
                elif len(parts) == 2:
                    if parts[0] not in current_config:
                        current_config[parts[0]] = {}
                    current_config[parts[0]][parts[1]] = value
                elif len(parts) == 3:
                    if parts[0] not in current_config:
                        current_config[parts[0]] = {}
                    if parts[1] not in current_config[parts[0]]:
                        current_config[parts[0]][parts[1]] = {}
                    current_config[parts[0]][parts[1]][parts[2]] = value
        
        # Save configuration
        save_config(current_config)
        
        return jsonify({'success': True, 'message': 'Configuration updated successfully'})
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return jsonify({'success': False, 'message': f'Error updating configuration: {str(e)}'})

@app.route('/get_backtest_results')
def get_backtest_results_route():
    # Get backtest results
    try:
        results = get_backtest_results()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_processes')
def get_processes():
    """Get list of active processes"""
    try:
        # Prepare process information
        processes = {}
        
        for process_name, process_info in active_processes.items():
            # Check if process is still running
            is_running = process_info['process'].poll() is None
            
            # Get process status
            status = 'running' if is_running else process_info.get('status', 'unknown')
            
            # Get process logs (last 100 lines)
            logs = process_info.get('logs', [])[-100:]
            
            # Add process information
            processes[process_name] = {
                'name': process_name,
                'command': process_info.get('command', ''),
                'start_time': process_info.get('start_time', ''),
                'end_time': process_info.get('end_time', '') if not is_running else '',
                'status': status,
                'logs': logs,
                'return_code': process_info.get('return_code', None) if not is_running else None
            }
        
        return jsonify(processes)
    except Exception as e:
        logger.error(f"Error getting processes: {str(e)}")
        return jsonify({})

@app.route('/run_comprehensive_backtest', methods=['POST'])
def run_comprehensive_backtest():
    try:
        # Get form data
        quarters = request.form.get('quarters', 'Q1_2023,Q2_2023')
        max_signals = int(request.form.get('max_signals', 40))
        initial_capital = int(request.form.get('initial_capital', 300))
        multiple_runs = request.form.get('multiple_runs') == 'on'
        num_runs = int(request.form.get('num_runs', 5))
        random_seed = int(request.form.get('random_seed', 42))
        continuous_capital = request.form.get('continuous_capital') == 'on'
        weekly_selection = request.form.get('weekly_selection') == 'on'
        
        # Get tier thresholds
        tier1_threshold = float(request.form.get('tier1_threshold', 0.8))
        tier2_threshold = float(request.form.get('tier2_threshold', 0.7))
        tier3_threshold = float(request.form.get('tier3_threshold', 0.6))
        
        # Check if custom date range is provided
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        use_custom_dates = request.form.get('use_custom_dates') == 'on'
        
        # Validate custom date range if selected
        if use_custom_dates:
            if not start_date or not end_date:
                flash('Both start date and end date must be provided for custom date range', 'danger')
                return redirect(url_for('index'))
            
            # Format for custom date range
            quarters_list = ['custom']
            
            # Create command for custom date range
            args = [
                '--start_date', start_date,
                '--end_date', end_date,
                '--max_signals', str(max_signals),
                '--initial_capital', str(initial_capital),
                '--tier1_threshold', str(tier1_threshold),
                '--tier2_threshold', str(tier2_threshold),
                '--tier3_threshold', str(tier3_threshold)
            ]
            
            if multiple_runs:
                args.append('--multiple_runs')
                args.extend(['--num_runs', str(num_runs)])
            
            if random_seed:
                args.extend(['--random_seed', str(random_seed)])
            
            if continuous_capital:
                args.append('--continuous_capital')
            
            if weekly_selection:
                args.append('--weekly_selection')
            
            # Add 'custom' as a positional argument
            args.append('custom')
        else:
            # Parse quarters
            quarters_list = [q.strip() for q in quarters.split(',') if q.strip()]
            
            # Create command for regular quarters
            args = [
                '--max_signals', str(max_signals),
                '--initial_capital', str(initial_capital),
                '--tier1_threshold', str(tier1_threshold),
                '--tier2_threshold', str(tier2_threshold),
                '--tier3_threshold', str(tier3_threshold)
            ]
            
            if multiple_runs:
                args.append('--multiple_runs')
                args.extend(['--num_runs', str(num_runs)])
            
            if random_seed:
                args.extend(['--random_seed', str(random_seed)])
            
            if continuous_capital:
                args.append('--continuous_capital')
            
            if weekly_selection:
                args.append('--weekly_selection')
            
            # Add quarters as positional arguments
            args.extend(quarters_list)
        
        # Run the process
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_comprehensive_backtest.py')
        process_name = f"comprehensive_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_process(script_path, args, process_name)
        
        flash(f'Comprehensive backtest started for {", ".join(quarters_list)}', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}", exc_info=True)
        flash(f'Error running comprehensive backtest: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest for a specific date range"""
    try:
        # Get form data
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        max_signals = request.form.get('max_signals', '40')
        initial_capital = request.form.get('initial_capital', '300')
        continuous_capital = request.form.get('continuous_capital', 'false').lower() == 'true'
        weekly_selection = request.form.get('weekly_selection', 'false').lower() == 'true'
        
        # Validate dates
        if not start_date or not end_date:
            return jsonify({'success': False, 'message': 'Start date and end date are required'})
        
        # Generate a unique run ID
        run_id = f"backtest_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare arguments
        args = [
            '--start-date', start_date,
            '--end-date', end_date,
            '--max-signals', max_signals,
            '--initial-capital', initial_capital
        ]
        
        if continuous_capital:
            args.append('--continuous-capital')
        
        if weekly_selection:
            args.append('--weekly-selection')
        
        # Run the process
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_comprehensive_backtest.py')
        success, message = run_process(script_path, args, run_id)
        
        if success:
            return jsonify({'success': True, 'message': message, 'run_id': run_id})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return jsonify({'success': False, 'message': f'Error running backtest: {str(e)}'})

@app.route('/view_backtest_result/<filename>')
def view_backtest_result(filename):
    """View backtest result"""
    try:
        # Initialize default_summary at the beginning to avoid UnboundLocalError
        default_summary = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'initial_capital': 0,
            'final_capital': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_holding_period': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'tier1_threshold': 0.8,
            'tier2_threshold': 0.7,
            'tier3_threshold': 0.6,
            'trading_parameters': {
                'position_sizing': {
                    'base_position_pct': 5,
                    'tier1_factor': 3.0,
                    'tier2_factor': 1.5,
                    'midcap_factor': 0.8
                },
                'stop_loss_pct': 5,
                'take_profit_pct': 10,
                'max_drawdown_pct': 15,
                'large_cap_percentage': 70,
                'avg_holding_period': {
                    'win': 12,
                    'loss': 5
                },
                'win_rate_adjustments': {
                    'base_long_win_rate': 0.62,
                    'market_regime_adjustments': {
                        'STRONG_BULLISH': 0.15,
                        'BULLISH': 0.10,
                        'NEUTRAL': 0.00,
                        'BEARISH': -0.10,
                        'STRONG_BEARISH': -0.20
                    }
                }
            },
            'trades': []
        }
        
        # Find the result file
        results = get_backtest_results()
        result_file = None
        
        for result in results:
            if result['filename'] == filename:
                result_file = result
                break
        
        if not result_file:
            return render_template('error.html', error=f'Backtest result file not found: {filename}')
        
        # Get the file path
        result_path = result_file['path']
        
        # Check if file exists
        if not os.path.exists(result_path):
            return render_template('error.html', error=f'Backtest result file not found: {result_path}')
        
        # Get file extension
        ext = os.path.splitext(result_path)[1]
        
        # Parse file based on extension
        if ext.lower() == '.json':
            try:
                # Read JSON file
                with open(result_path, 'r') as f:
                    data = json.load(f)
                
                # Get summary and signals
                summary = data.get('summary', default_summary)
                signals = data.get('signals', [])
                
                # Return template with data
                return render_template('backtest_result.html', 
                                      summary=summary, 
                                      signals=signals, 
                                      filename=filename)
            except Exception as e:
                logger.error(f"Error parsing JSON file: {str(e)}")
                return render_template('error.html', error=f'Error parsing JSON file: {str(e)}')
        else:
            return render_template('error.html', error=f'Unsupported file format: {ext}')
    except Exception as e:
        logger.error(f"Error viewing backtest result: {str(e)}")
        traceback.print_exc()
        return render_template('error.html', error=f'Error: {str(e)}')

@app.route('/view_logs')
def view_logs():
    """View application logs"""
    try:
        # Get log files
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_files = []
        
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                log_files.append({
                    'filename': file,
                    'path': os.path.join(log_dir, file),
                    'modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(log_dir, file))).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort log files by modification time (newest first)
        log_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return render_template('logs.html', log_files=log_files)
    except Exception as e:
        logger.error(f"Error viewing logs: {str(e)}")
        return render_template('error.html', error=f'Error viewing logs: {str(e)}')

@app.route('/view_log_file/<log_file>')
def view_log_file(log_file):
    """View contents of a log file"""
    try:
        # Get log file path
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(log_dir, log_file)
        
        # Check if file exists
        if not os.path.exists(log_path):
            return render_template('error.html', error=f'Log file not found: {log_file}')
        
        # Read log file
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        return render_template('log_file.html', log_file=log_file, log_content=log_content)
    except Exception as e:
        logger.error(f"Error viewing log file: {str(e)}")
        return render_template('error.html', error=f'Error viewing log file: {str(e)}')

@app.route('/run_paper_trading', methods=['POST'])
def run_paper_trading():
    """Run paper trading"""
    try:
        # Get form data
        max_signals = request.form.get('max_signals', '20')
        duration = request.form.get('duration', '1')
        interval = request.form.get('interval', '5')
        
        # Generate a unique run ID
        run_id = f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare arguments
        args = [
            '--max-signals', max_signals,
            '--duration', duration,
            '--interval', interval,
            '--mode', 'paper'
        ]
        
        # Run the process
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_sp500_strategy.py')
        success, message = run_process(script_path, args, run_id)
        
        if success:
            return jsonify({'success': True, 'message': message, 'run_id': run_id})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        logger.error(f"Error running paper trading: {str(e)}")
        return jsonify({'success': False, 'message': f'Error running paper trading: {str(e)}'})

@app.route('/run_live_trading', methods=['POST'])
def run_live_trading():
    """Run live trading"""
    try:
        # Get form data
        max_signals = request.form.get('max_signals', '10')
        check_interval = request.form.get('check_interval', '5')
        max_capital = request.form.get('max_capital', '50000')
        risk_level = request.form.get('risk_level', 'medium')
        
        # Generate a unique run ID
        run_id = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare arguments
        args = [
            '--max-signals', max_signals,
            '--check-interval', check_interval,
            '--max-capital', max_capital,
            '--risk-level', risk_level,
            '--mode', 'live'
        ]
        
        # Run the process
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_sp500_strategy.py')
        success, message = run_process(script_path, args, run_id)
        
        if success:
            return jsonify({'success': True, 'message': message, 'run_id': run_id})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        logger.error(f"Error running live trading: {str(e)}")
        return jsonify({'success': False, 'message': f'Error running live trading: {str(e)}'})

@app.route('/stop_process/<process_name>', methods=['POST'])
def stop_process_route(process_name):
    """Stop a running process"""
    success, message = stop_process(process_name)
    return jsonify({'success': success, 'message': message})

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop_route():
    """Emergency stop all processes"""
    success, message = emergency_stop()
    return jsonify({'success': success, 'message': message})

# Main entry point
if __name__ == '__main__':
    # Load configuration
    load_config()
    
    # Create backtest results directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'backtest_results'), exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
