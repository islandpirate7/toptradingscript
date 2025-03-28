#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
S&P 500 Trading Strategy Web Interface
"""

import os
import sys
import json
import yaml
import logging
import datetime
import subprocess
import glob
import time
import threading
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_interface.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the path
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Create Flask app
app = Flask(__name__)
app.secret_key = 'sp500_trading_strategy'

# Enable CORS
CORS(app)

# Dictionary to store active processes
processes = {}

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Add cache-busting headers to prevent browser caching of static files
@app.after_request
def add_header(response):
    """Add headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Handle OPTIONS requests

# Add cache-busting headers to prevent browser caching of static files
@app.after_request
def add_header(response):
    """Add headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests"""
    return '', 200

@app.route('/')
def index():
    """Render the index page"""
    return render_template('index.html')

@app.route('/get_backtest_results')
def get_backtest_results_route():
    """Get backtest results"""
    try:
        results = get_backtest_results()
            return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_processes')
def get_processes():
    """Get list of active processes"""
    try:
        # Prepare process information
        process_list = []
        for name, process_info in processes.items():
            # Check if process is still running
            if process_info['process'].poll() is not None:
                process_info['status'] = 'Completed'
            
            # Add to list
            process_list.append({
                'name': name,
                'status': process_info['status'],
                'start_time': process_info['start_time'],
                'logs': process_info['logs']
            })
        
        return jsonify({"processes": list(processes.values())})
    except Exception as e:
        logger.error(f"Error getting processes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/stop_process', methods=['POST'])
def stop_process():
    """Stop a running process"""
    try:
        process_name = request.form.get('process_name')
        if not process_name:
            return jsonify({'success': False, 'message': 'Process name is required'}), 400
        
        if process_name not in processes:
            return jsonify({'success': False, 'message': f'Process {process_name} not found'}), 404
        
        # Terminate the process
        process_info = processes[process_name]
        if process_info['process'].poll() is None:
            process_info['process'].terminate()
            process_info['status'] = 'Terminated'
            return jsonify({'success': True, 'message': f'Process {process_name} terminated'})
        else:
            return jsonify({'success': False, 'message': f'Process {process_name} is not running'})
    except Exception as e:
        logger.error(f"Error stopping process: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop all running processes"""
    try:
        # Terminate all processes
        for name, process_info in processes.items():
            if process_info['process'].poll() is None:
                process_info['process'].terminate()
                process_info['status'] = 'Terminated'
        
        return jsonify({'success': True, 'message': 'All processes terminated'})
    except Exception as e:
        logger.error(f"Error during emergency stop: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/view_backtest_result/<filename>')
def view_backtest_result(filename):
    """View a specific backtest result"""
    try:
        # Initialize default_summary
        default_summary = {
            'total_profit_loss': 0,
            'win_rate': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'average_profit_per_trade': 0,
            'average_loss_per_trade': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'start_date': '',
            'end_date': '',
            'duration': '',
            'strategy': 'Unknown'
        }
        
        # Find the backtest result file
        results_dirs = [
            os.path.join(parent_dir, 'backtest_results'),
            os.path.join(script_dir, 'backtest_results')
        ]
        
        result_file = None
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                potential_file = os.path.join(results_dir, filename)
                if os.path.exists(potential_file):
                    result_file = potential_file
                    break
        
        if not result_file:
            flash(f'Backtest result file {filename} not found', 'danger')
            return redirect(url_for('index'))
        
        # Load the backtest result
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Extract summary and trades
        summary = result.get('summary', default_summary)
        trades = result.get('trades', [])
        
        # Render the result page
        return render_template(
            'backtest_result.html',
            filename=filename,
            summary=summary,
            trades=trades
        )
    except Exception as e:
        logger.error(f"Error viewing backtest result: {str(e)}", exc_info=True)
        flash(f'Error viewing backtest result: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run a backtest"""
    try:
        # Get form data
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        if not start_date or not end_date:
            flash('Start date and end date are required', 'danger')
            return redirect(url_for('index'))
        
        # Check if a backtest is already running
        if any(p['status'] == 'Running' and 'backtest' in p['name'].lower() for p in processes.values()):
            flash('A backtest is already running', 'warning')
            return redirect(url_for('index'))
        
        # Run the backtest
        success, message = run_process(
            'final_sp500_strategy.py',
            ['--backtest', '--start-date', start_date, '--end-date', end_date],
            f'Backtest {start_date} to {end_date}'
        )
        
        if success:
            flash(f'Backtest started: {start_date} to {end_date}', 'success')
        else:
            flash(f'Error starting backtest: {message}', 'danger')
        
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        flash(f'Error running backtest: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/run_comprehensive_backtest', methods=['POST'])
def run_comprehensive_backtest():
    """Run a comprehensive backtest"""
    try:
        # Check if this is an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        # Get form data
        years = request.form.getlist('years[]')
        quarters = request.form.getlist('quarters[]')
        
        if not years or not quarters:
            if is_ajax:
                return jsonify({'success': False, 'message': 'Years and quarters are required'}), 400
            else:
                flash('Years and quarters are required', 'danger')
                return redirect(url_for('index'))
        
        # Check if a comprehensive backtest is already running
        if any(p['status'] == 'Running' and 'comprehensive' in p['name'].lower() for p in processes.values()):
            if is_ajax:
                return jsonify({'success': False, 'message': 'A comprehensive backtest is already running'}), 400
            else:
                flash('A comprehensive backtest is already running', 'warning')
                return redirect(url_for('index'))
        
        # Prepare the quarters to test
        quarters_to_test = []
        for year in years:
            for quarter in quarters:
                quarters_to_test.append(f"{year}Q{quarter}")
        
        if not quarters_to_test:
            if is_ajax:
                return jsonify({'success': False, 'message': 'No valid quarters selected'}), 400
            else:
                flash('No valid quarters selected', 'danger')
                return redirect(url_for('index'))
        
        # Run the comprehensive backtest
        success, message = run_process(
            'final_sp500_strategy.py',
            ['--comprehensive-backtest'] + quarters_to_test,
            f'Comprehensive Backtest {", ".join(quarters_to_test)}'
        )
        
        if success:
            if is_ajax:
                return jsonify({'success': True, 'message': f'Comprehensive backtest started: {", ".join(quarters_to_test)}'})
            else:
                flash(f'Comprehensive backtest started: {", ".join(quarters_to_test)}', 'success')
        else:
            if is_ajax:
                return jsonify({'success': False, 'message': f'Error starting comprehensive backtest: {message}'}), 500
            else:
                flash(f'Error starting comprehensive backtest: {message}', 'danger')
        
        if not is_ajax:
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error running comprehensive backtest: {str(e)}", exc_info=True)
        if is_ajax:
            return jsonify({'success': False, 'message': str(e)}), 500
        else:
            flash(f'Error running comprehensive backtest: {str(e)}', 'danger')
            return redirect(url_for('index'))

def get_backtest_results():
    """Get backtest results"""
    results = []
    
    # Check multiple possible locations for backtest results
    results_dirs = [
        os.path.join(parent_dir, 'backtest_results'),
        os.path.join(script_dir, 'backtest_results')
    ]
    
    # Cache for performance
    cache_file = os.path.join(script_dir, 'backtest_results_cache.json')
    cache_valid = False
    
    # Check if cache exists and is recent
    if os.path.exists(cache_file):
        cache_mtime = os.path.getmtime(cache_file)
        if time.time() - cache_mtime < 300:  # 5 minutes
            try:
                with open(cache_file, 'r') as f:
                    results = json.load(f)
                    cache_valid = True
                    logger.info("Using cached backtest results")
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
    
    if not cache_valid:
        # Find all backtest result files
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                for result_file in glob.glob(os.path.join(results_dir, '*.json')):
                    try:
                        # Load the backtest result
                        with open(result_file, 'r') as f:
                            result = json.load(f)
                        
                        # Extract summary
                        summary = result.get('summary', {})
                        
                        # Add to results
                        results.append({
                            'filename': os.path.basename(result_file),
                            'quarter': summary.get('quarter', ''),
                            'date_range': f"{summary.get('start_date', '')} to {summary.get('end_date', '')}",
                            'profit_loss': summary.get('total_profit_loss', 0),
                            'win_rate': summary.get('win_rate', 0)
                        })
                    except Exception as e:
                        logger.warning(f"Error loading backtest result {result_file}: {str(e)}")
        
        # Sort by date (newest first)
        results.sort(key=lambda x: x.get('filename', ''), reverse=True)
        
        # Cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f)
        except Exception as e:
            logger.warning(f"Error writing cache: {str(e)}")
    
    return results

def run_process(script, args, process_name):
    """Run a Python script as a subprocess and capture its output"""
    try:
        # Check if a process with this name is already running
        if process_name in processes and processes[process_name]['process'].poll() is None:
            return False, f"Process {process_name} is already running"
        
        # Prepare the command
        cmd = [sys.executable, os.path.join(parent_dir, script)] + args
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Store process information
        processes[process_name] = {
            'name': process_name,
            'process': process,
            'status': 'Running',
            'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'logs': []
        }
        
        # Start a thread to read the output
        def read_output():
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    processes[process_name]['logs'].append(line.strip())
            
            # Update status when process completes
            processes[process_name]['status'] = 'Completed'
        
        threading.Thread(target=read_output, daemon=True).start()
        
        return True, f"Process {process_name} started"
    except Exception as e:
        logger.error(f"Error running process: {str(e)}", exc_info=True)
        return False, str(e)

# Run the app if this file is executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
