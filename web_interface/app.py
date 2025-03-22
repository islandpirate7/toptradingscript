#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web Interface for S&P 500 Trading Strategy
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
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"web_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

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
    """Get list of backtest result files"""
    try:
        results_dir = os.path.join('..', 'backtest_results')
        if not os.path.exists(results_dir):
            return []
        
        result_files = []
        for file in os.listdir(results_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(results_dir, file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                result_files.append({
                    'name': file,
                    'path': file_path,
                    'date': mod_time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by date (newest first)
        result_files.sort(key=lambda x: x['date'], reverse=True)
        return result_files
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        return []

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
    """Run a Python script as a subprocess and capture output"""
    global active_processes, process_logs, process_status
    
    try:
        # Create command
        cmd = [sys.executable, os.path.join('..', script)] + args
        
        # Initialize log
        process_logs[process_name] = []
        process_status[process_name] = 'starting'
        
        # Start process
        logger.info(f"Starting process: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        active_processes[process_name] = process
        process_status[process_name] = 'running'
        
        # Read output
        for line in process.stdout:
            process_logs[process_name].append(line.strip())
            logger.info(f"[{process_name}] {line.strip()}")
        
        # Process completed
        process.wait()
        if process.returncode == 0:
            process_status[process_name] = 'completed'
            logger.info(f"Process {process_name} completed successfully")
        else:
            process_status[process_name] = 'failed'
            logger.error(f"Process {process_name} failed with return code {process.returncode}")
        
        # Remove from active processes
        if process_name in active_processes:
            del active_processes[process_name]
            
    except Exception as e:
        process_status[process_name] = 'error'
        process_logs[process_name].append(f"Error: {str(e)}")
        logger.error(f"Error running process {process_name}: {str(e)}")
        
        # Remove from active processes
        if process_name in active_processes:
            del active_processes[process_name]

def stop_process(process_name):
    """Stop a running process"""
    global active_processes, process_status
    
    if process_name in active_processes:
        try:
            process = active_processes[process_name]
            process.terminate()
            process_status[process_name] = 'terminated'
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
        # Create a simple script to close all positions
        script_path = os.path.join('..', 'handle_shutdown.py')
        if not os.path.exists(script_path):
            with open(script_path, 'w') as f:
                f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Emergency Shutdown Handler
This script closes all open positions when the system is shut down
\"\"\"

import os
import json
import yaml
import logging
from datetime import datetime
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"emergency_shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file='sp500_config.yaml'):
    \"\"\"Load configuration from YAML file\"\"\"
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def load_alpaca_credentials(mode='paper'):
    \"\"\"Load Alpaca API credentials from JSON file\"\"\"
    try:
        with open('alpaca_credentials.json', 'r') as f:
            credentials = json.load(f)
        return credentials[mode]
    except Exception as e:
        logger.error(f"Error loading Alpaca credentials: {str(e)}")
        return {}

def close_all_positions():
    \"\"\"Close all open positions\"\"\"
    try:
        # Load credentials
        credentials = load_alpaca_credentials('paper')
        
        # Initialize Alpaca API
        api = tradeapi.REST(
            credentials['api_key'],
            credentials['api_secret'],
            credentials['base_url'],
            api_version='v2'
        )
        
        # Get all positions
        positions = api.list_positions()
        logger.info(f"Found {len(positions)} open positions")
        
        # Close all positions
        if positions:
            logger.info("Closing all positions...")
            api.close_all_positions()
            logger.info("All positions closed successfully")
            
            # Save closed positions to CSV
            import pandas as pd
            positions_data = []
            for position in positions:
                positions_data.append({
                    'symbol': position.symbol,
                    'qty': position.qty,
                    'entry_price': position.avg_entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pl': position.unrealized_pl,
                    'unrealized_plpc': position.unrealized_plpc,
                    'closed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # Create directory if it doesn't exist
            config = load_config()
            trades_dir = config.get('paths', {}).get('trades', 'trades')
            os.makedirs(trades_dir, exist_ok=True)
            
            # Save to CSV
            positions_df = pd.DataFrame(positions_data)
            positions_file = os.path.join(
                trades_dir, 
                f"emergency_closed_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            positions_df.to_csv(positions_file, index=False)
            logger.info(f"Closed positions saved to {positions_file}")
        else:
            logger.info("No open positions to close")
        
        return True
    except Exception as e:
        logger.error(f"Error closing positions: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting emergency shutdown procedure")
    close_all_positions()
    logger.info("Emergency shutdown completed")
""")
        
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
                if len(parts) == 2:
                    section, param = parts
                    if section not in current_config:
                        current_config[section] = {}
                    
                    # Convert value to appropriate type
                    if value.lower() == 'true':
                        current_config[section][param] = True
                    elif value.lower() == 'false':
                        current_config[section][param] = False
                    elif value.isdigit():
                        current_config[section][param] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        current_config[section][param] = float(value)
                    else:
                        current_config[section][param] = value
            else:
                # Handle top-level keys
                if value.lower() == 'true':
                    current_config[key] = True
                elif value.lower() == 'false':
                    current_config[key] = False
                elif value.isdigit():
                    current_config[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    current_config[key] = float(value)
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

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest"""
    try:
        # Get form data
        quarters = request.form.get('quarters', '2023Q1')
        runs = request.form.get('runs', '5')
        random_seed = request.form.get('random_seed', '42')
        
        # Create process name
        process_name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start process in a thread
        args = ['--quarters', quarters, '--runs', runs, '--random_seed', random_seed]
        thread = threading.Thread(
            target=run_process,
            args=('run_comprehensive_backtest.py', args, process_name)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Backtest started with process name: {process_name}',
            'process_name': process_name
        })
    
    except Exception as e:
        logger.error(f"Error starting backtest: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

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
    if process_name in process_logs:
        return jsonify({
            'success': True,
            'logs': process_logs[process_name],
            'status': process_status.get(process_name, 'unknown')
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

@app.route('/get_backtest_results')
def get_backtest_results_route():
    """Get backtest results"""
    results = get_backtest_results()
    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    # Load configuration
    load_config()
    
    # Create required directories
    if config_data:
        for path_key in ['trades', 'performance', 'backtest_results']:
            path = config_data.get('paths', {}).get(path_key, f"../{path_key}")
            os.makedirs(path, exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=8000, debug=True)
