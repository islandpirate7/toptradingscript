#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Web Interface for S&P 500 Trading Strategy
"""

import os
import sys
import json
import yaml
import logging
import datetime
import threading
import time
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, abort
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/web_interface_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to web interface directory
web_interface_dir = os.path.join(script_dir, 'new_web_interface')

# Create Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(web_interface_dir, 'templates'),
    static_folder=os.path.join(web_interface_dir, 'static')
)
app.secret_key = 'sp500_trading_strategy'

# Enable CORS
CORS(app)

# Dictionary to store active processes
active_processes = {}

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Index route
@app.route('/')
def index():
    """Render the index page"""
    try:
        # Add timestamp for cache-busting
        now = int(time.time())
        
        # Load configuration
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        config = {}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error reading configuration file: {str(e)}", exc_info=True)
                flash(f'Error reading configuration file: {str(e)}', 'danger')
        
        # Ensure all required config sections exist
        if 'initial_capital' not in config:
            config['initial_capital'] = 300
            
        if 'backtest' not in config:
            config['backtest'] = {
                'max_signals_per_day': 40,
                'tier1_threshold': 0.8,
                'tier2_threshold': 0.7,
                'tier3_threshold': 0.6,
                'weekly_selection': True,
                'continuous_capital': False,
                'random_seed': 42
            }
        
        if 'position_sizing' not in config:
            config['position_sizing'] = {
                'tier1_size': 0.3,
                'tier2_size': 0.2,
                'tier3_size': 0.1
            }
        
        return render_template('index.html', config=config, now=now)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", 500

# Get active processes
@app.route('/get_active_processes', methods=['GET'])
def get_active_processes():
    """Get active processes"""
    try:
        # Prepare process information
        process_list = []
        for name, process_info in active_processes.items():
            # Add to list
            process_list.append({
                'name': name,
                'type': process_info['type'],
                'status': process_info['status'],
                'start_time': process_info['start_time'],
                'description': process_info.get('description', ''),
                'params': process_info.get('params', {})
            })
        
        return jsonify({"processes": process_list})
    except Exception as e:
        logger.error(f"Error getting active processes: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': f'Error getting active processes: {str(e)}'}), 500

# Add endpoint for get_processes to match what the frontend expects
@app.route('/get_processes', methods=['GET'])
def get_processes():
    """Alias for get_active_processes to match frontend expectations"""
    return get_active_processes()

# Get backtest results
@app.route('/get_backtest_results', methods=['GET'])
def get_backtest_results():
    """Get backtest results"""
    try:
        # Get list of backtest result files
        results_dir = os.path.join(script_dir, 'backtest_results')
        result_files = []
        
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(results_dir, file)
                    try:
                        with open(file_path, 'r') as f:
                            result_data = json.load(f)
                            
                            # Extract key information
                            result_files.append({
                                'filename': file,
                                'date': result_data.get('date', ''),
                                'start_date': result_data.get('start_date', ''),
                                'end_date': result_data.get('end_date', ''),
                                'initial_capital': result_data.get('initial_capital', 0),
                                'final_value': result_data.get('final_value', 0),
                                'return': result_data.get('return', 0),
                                'sharpe_ratio': result_data.get('sharpe_ratio', 0),
                                'max_drawdown': result_data.get('max_drawdown', 0),
                                'win_rate': result_data.get('win_rate', 0),
                                'file_path': file_path
                            })
                    except Exception as e:
                        logger.error(f"Error reading result file {file}: {str(e)}", exc_info=True)
        
        # Sort by date (newest first)
        result_files.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return jsonify({"results": result_files})
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': f'Error getting backtest results: {str(e)}'}), 500

# Run backtest
@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest"""
    try:
        # Get form data
        quarter = request.form.get('quarter')
        use_custom_date_range = 'use_custom_date_range' in request.form
        max_signals_per_day = int(request.form.get('max_signals_per_day', 40))
        initial_capital = float(request.form.get('initial_capital', 300))
        tier1_threshold = float(request.form.get('tier1_threshold', 0.8))
        tier1_size = float(request.form.get('tier1_size', 0.3))
        tier2_threshold = float(request.form.get('tier2_threshold', 0.7))
        tier2_size = float(request.form.get('tier2_size', 0.2))
        tier3_threshold = float(request.form.get('tier3_threshold', 0.6))
        tier3_size = float(request.form.get('tier3_size', 0.1))
        weekly_selection = 'weekly_selection' in request.form
        multiple_runs = 'multiple_runs' in request.form
        num_runs = int(request.form.get('num_runs', 5))
        random_seed = int(request.form.get('random_seed', 42))
        
        # Log the parameters
        logger.info(f"Running backtest with parameters: quarter={quarter}, use_custom_date_range={use_custom_date_range}, max_signals_per_day={max_signals_per_day}, initial_capital={initial_capital}")
        
        # Define quarter date ranges
        quarter_dates = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31'),
            'Q2_2024': ('2024-04-01', '2024-06-30'),
            'Q3_2024': ('2024-07-01', '2024-09-30'),
            'Q4_2024': ('2024-10-01', '2024-12-31')
        }
        
        # Create a unique process ID for tracking
        process_id = f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if use_custom_date_range:
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            
            if not start_date or not end_date:
                return jsonify({
                    'success': False,
                    'message': 'Start date and end date are required for custom date range'
                })
            
            # Add to active processes
            active_processes[process_id] = {
                'type': 'backtest',
                'status': 'running',
                'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': f'Custom backtest from {start_date} to {end_date}',
                'params': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'max_signals': max_signals_per_day,
                    'weekly_selection': weekly_selection,
                    'random_seed': random_seed
                }
            }
            
            # Run the backtest in a separate thread
            thread = threading.Thread(
                target=run_backtest_thread,
                args=(
                    process_id,
                    start_date,
                    end_date,
                    max_signals_per_day,
                    initial_capital,
                    random_seed,
                    weekly_selection,
                    False,  # continuous_capital
                    tier1_threshold,
                    tier2_threshold,
                    tier3_threshold
                )
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'message': f'Started backtest with custom date range: {start_date} to {end_date}',
                'process_id': process_id
            })
        else:
            if not quarter or quarter not in quarter_dates:
                return jsonify({
                    'success': False,
                    'message': 'Valid quarter is required'
                })
            
            start_date, end_date = quarter_dates[quarter]
            
            # Add to active processes
            active_processes[process_id] = {
                'type': 'backtest',
                'status': 'running',
                'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': f'Backtest for {quarter}',
                'params': {
                    'quarter': quarter,
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'max_signals': max_signals_per_day,
                    'weekly_selection': weekly_selection,
                    'random_seed': random_seed
                }
            }
            
            # Run the backtest in a separate thread
            thread = threading.Thread(
                target=run_backtest_thread,
                args=(
                    process_id,
                    start_date,
                    end_date,
                    max_signals_per_day,
                    initial_capital,
                    random_seed,
                    weekly_selection,
                    False,  # continuous_capital
                    tier1_threshold,
                    tier2_threshold,
                    tier3_threshold
                )
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'message': f'Started backtest for quarter: {quarter}',
                'process_id': process_id
            })
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error running backtest: {str(e)}'
        })

# Run comprehensive backtest
@app.route('/run_comprehensive_backtest', methods=['POST'])
def run_comprehensive_backtest():
    """Run a comprehensive backtest"""
    try:
        # Get quarters from form data
        quarters_str = request.form.get('quarters', '')
        max_signals_per_day = int(request.form.get('max_signals_per_day', 40))
        initial_capital = float(request.form.get('initial_capital', 300))
        weekly_selection = 'weekly_selection' in request.form
        continuous_capital = 'continuous_capital' in request.form
        tier1_threshold = float(request.form.get('tier1_threshold', 0.8))
        tier2_threshold = float(request.form.get('tier2_threshold', 0.7))
        tier3_threshold = float(request.form.get('tier3_threshold', 0.6))
        random_seed = int(request.form.get('random_seed', 42))
        
        # Log the parameters
        logger.info(f"Running comprehensive backtest with parameters: quarters={quarters_str}, max_signals_per_day={max_signals_per_day}, initial_capital={initial_capital}, weekly_selection={weekly_selection}, continuous_capital={continuous_capital}")
        
        if not quarters_str:
            return jsonify({
                'success': False,
                'message': 'No quarters specified for backtest'
            })
        
        # Parse quarters from the string
        quarters = [q.strip() for q in quarters_str.split(',') if q.strip()]
        
        if not quarters:
            return jsonify({
                'success': False,
                'message': 'No valid quarters specified for backtest'
            })
        
        # Define quarter date ranges
        quarter_dates = {
            'Q1_2023': ('2023-01-01', '2023-03-31'),
            'Q2_2023': ('2023-04-01', '2023-06-30'),
            'Q3_2023': ('2023-07-01', '2023-09-30'),
            'Q4_2023': ('2023-10-01', '2023-12-31'),
            'Q1_2024': ('2024-01-01', '2024-03-31'),
            'Q2_2024': ('2024-04-01', '2024-06-30'),
            'Q3_2024': ('2024-07-01', '2024-09-30'),
            'Q4_2024': ('2024-10-01', '2024-12-31')
        }
        
        # Validate quarters
        valid_quarters = []
        for quarter in quarters:
            if quarter in quarter_dates:
                valid_quarters.append(quarter)
        
        if not valid_quarters:
            return jsonify({
                'success': False,
                'message': 'No valid quarters found in the specified list'
            })
        
        # Sort quarters chronologically
        valid_quarters.sort()
        
        # Get start and end dates for the entire period
        start_date = quarter_dates[valid_quarters[0]][0]
        end_date = quarter_dates[valid_quarters[-1]][1]
        
        # Create a unique process ID for tracking
        process_id = f"comprehensive_backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add to active processes
        active_processes[process_id] = {
            'type': 'comprehensive_backtest',
            'status': 'running',
            'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': f'Comprehensive backtest for {", ".join(valid_quarters)}',
            'params': {
                'quarters': valid_quarters,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'max_signals': max_signals_per_day,
                'weekly_selection': weekly_selection,
                'continuous_capital': continuous_capital,
                'random_seed': random_seed
            }
        }
        
        # Run the comprehensive backtest in a separate thread
        thread = threading.Thread(
            target=run_backtest_thread,
            args=(
                process_id,
                start_date,
                end_date,
                max_signals_per_day,
                initial_capital,
                random_seed,
                weekly_selection,
                continuous_capital,
                tier1_threshold,
                tier2_threshold,
                tier3_threshold
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Started comprehensive backtest for quarters: {", ".join(valid_quarters)}',
            'process_id': process_id
        })
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error running comprehensive backtest: {str(e)}'
        })

# View logs
@app.route('/view_logs', methods=['GET'])
def view_logs():
    """View logs page"""
    try:
        # Get list of log files
        log_files = []
        logs_dir = os.path.join(script_dir, 'logs')
        
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            log_files.sort(reverse=True)  # Most recent first
        
        # If a specific log file is requested
        selected_log = request.args.get('file')
        log_content = ''
        
        if selected_log and selected_log in log_files:
            log_path = os.path.join(logs_dir, selected_log)
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
            except Exception as e:
                logger.error(f"Error reading log file {selected_log}: {str(e)}", exc_info=True)
                flash(f'Error reading log file: {str(e)}', 'danger')
        
        # Add timestamp for cache-busting
        now = int(time.time())
        
        return render_template('logs.html', log_files=log_files, selected_log=selected_log, log_content=log_content, now=now)
    except Exception as e:
        logger.error(f"Error viewing logs: {str(e)}", exc_info=True)
        flash(f'Error viewing logs: {str(e)}', 'danger')
        return redirect(url_for('index'))

# View configuration
@app.route('/view_configuration', methods=['GET', 'POST'])
def view_configuration():
    """View and edit configuration"""
    try:
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        config = {}
        
        # Load existing configuration
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error reading configuration file: {str(e)}", exc_info=True)
                flash(f'Error reading configuration file: {str(e)}', 'danger')
        
        # If form was submitted, update the configuration
        if request.method == 'POST':
            try:
                # Extract form data
                initial_capital = float(request.form.get('initial_capital', 300))
                max_signals_per_day = int(request.form.get('max_signals_per_day', 40))
                
                # Position sizing
                tier1_threshold = float(request.form.get('tier1_threshold', 0.8))
                tier1_size = float(request.form.get('tier1_size', 0.3))
                tier2_threshold = float(request.form.get('tier2_threshold', 0.7))
                tier2_size = float(request.form.get('tier2_size', 0.2))
                tier3_threshold = float(request.form.get('tier3_threshold', 0.6))
                tier3_size = float(request.form.get('tier3_size', 0.1))
                
                # API settings
                alpaca_api_key = request.form.get('alpaca_api_key', '')
                alpaca_api_secret = request.form.get('alpaca_api_secret', '')
                paper_trading = 'paper_trading' in request.form
                
                # Advanced settings
                weekly_selection = 'weekly_selection' in request.form
                continuous_capital = 'continuous_capital' in request.form
                random_seed = int(request.form.get('random_seed', 42))
                
                # Update config dictionary
                if 'initial_capital' not in config:
                    config['initial_capital'] = initial_capital
                else:
                    config['initial_capital'] = initial_capital
                
                # Ensure backtest section exists
                if 'backtest' not in config:
                    config['backtest'] = {}
                
                config['backtest']['max_signals_per_day'] = max_signals_per_day
                config['backtest']['tier1_threshold'] = tier1_threshold
                config['backtest']['tier2_threshold'] = tier2_threshold
                config['backtest']['tier3_threshold'] = tier3_threshold
                config['backtest']['weekly_selection'] = weekly_selection
                config['backtest']['continuous_capital'] = continuous_capital
                config['backtest']['random_seed'] = random_seed
                
                # Ensure position_sizing section exists
                if 'position_sizing' not in config:
                    config['position_sizing'] = {}
                
                config['position_sizing']['tier1_size'] = tier1_size
                config['position_sizing']['tier2_size'] = tier2_size
                config['position_sizing']['tier3_size'] = tier3_size
                
                # Ensure alpaca section exists
                if 'alpaca' not in config:
                    config['alpaca'] = {}
                
                config['alpaca']['api_key'] = alpaca_api_key
                config['alpaca']['api_secret'] = alpaca_api_secret
                config['alpaca']['paper_trading'] = paper_trading
                
                # Save configuration
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                flash('Configuration saved successfully', 'success')
            except Exception as e:
                logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
                flash(f'Error saving configuration: {str(e)}', 'danger')
        
        # Add timestamp for cache-busting
        now = int(time.time())
        
        return render_template('configuration.html', config=config, now=now)
    except Exception as e:
        logger.error(f"Error viewing configuration: {str(e)}", exc_info=True)
        flash(f'Error viewing configuration: {str(e)}', 'danger')
        return redirect(url_for('index'))

# Run backtest in a separate thread
def run_backtest_thread(process_id, start_date, end_date, max_signals, initial_capital, random_seed, weekly_selection, continuous_capital, tier1_threshold, tier2_threshold, tier3_threshold):
    """Run a backtest in a separate thread"""
    try:
        logger.info(f"[DEBUG] Starting backtest thread for process {process_id}")
        logger.info(f"[DEBUG] Parameters: start_date={start_date}, end_date={end_date}, max_signals={max_signals}, initial_capital={initial_capital}")
        logger.info(f"[DEBUG] Additional parameters: random_seed={random_seed}, weekly_selection={weekly_selection}, continuous_capital={continuous_capital}")
        logger.info(f"[DEBUG] Thresholds: tier1={tier1_threshold}, tier2={tier2_threshold}, tier3={tier3_threshold}")
        
        # Check if sp500_config.yaml exists
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        if not os.path.exists(config_path):
            logger.error(f"[DEBUG] Config file not found: {config_path}")
            active_processes[process_id]['status'] = 'error'
            active_processes[process_id]['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            active_processes[process_id]['error'] = f"Config file not found: {config_path}"
            return
        
        # Check if alpaca_credentials.json exists
        credentials_path = os.path.join(script_dir, 'alpaca_credentials.json')
        if not os.path.exists(credentials_path):
            logger.error(f"[DEBUG] Credentials file not found: {credentials_path}")
            active_processes[process_id]['status'] = 'error'
            active_processes[process_id]['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            active_processes[process_id]['error'] = f"Credentials file not found: {credentials_path}"
            return
        
        # Import the backtest function
        try:
            logger.info("[DEBUG] Importing run_backtest from final_sp500_strategy")
            from final_sp500_strategy import run_backtest as strategy_run_backtest
        except ImportError as e:
            logger.error(f"[DEBUG] Error importing run_backtest: {str(e)}")
            active_processes[process_id]['status'] = 'error'
            active_processes[process_id]['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            active_processes[process_id]['error'] = f"Error importing run_backtest: {str(e)}"
            return
        
        # Update the process status
        active_processes[process_id]['status'] = 'running'
        
        # Run the backtest
        logger.info("[DEBUG] Calling strategy_run_backtest function")
        try:
            result = strategy_run_backtest(
                start_date=start_date,
                end_date=end_date,
                mode='backtest',
                max_signals=max_signals,
                initial_capital=initial_capital,
                random_seed=random_seed,
                weekly_selection=weekly_selection,
                continuous_capital=continuous_capital,
                tier1_threshold=tier1_threshold,
                tier2_threshold=tier2_threshold,
                tier3_threshold=tier3_threshold
            )
            
            logger.info(f"[DEBUG] Backtest completed successfully, result: {result}")
            
            # Update the process with results
            active_processes[process_id]['status'] = 'completed'
            active_processes[process_id]['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            active_processes[process_id]['result'] = {
                'performance': result.get('performance', {}),
                'trades': result.get('trades', []),
                'result_file': result.get('result_file', '')
            }
            
            logger.info(f"[DEBUG] Process {process_id} updated with results")
            
        except Exception as e:
            logger.error(f"[DEBUG] Error running strategy_run_backtest: {str(e)}", exc_info=True)
            active_processes[process_id]['status'] = 'error'
            active_processes[process_id]['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            active_processes[process_id]['error'] = f"Error running backtest: {str(e)}"
        
    except Exception as e:
        logger.error(f"[DEBUG] Error in backtest thread for process {process_id}: {str(e)}", exc_info=True)
        
        # Update the process with error
        active_processes[process_id]['status'] = 'error'
        active_processes[process_id]['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        active_processes[process_id]['error'] = str(e)

if __name__ == "__main__":
    # Run the Flask app
    logger.info("Running web interface on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
