#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vercel Deployment Script

This script sets up a simple API endpoint for Vercel deployment that can run
the trading strategy in the background. It provides endpoints for:
1. Running backtests
2. Running paper trading
3. Running live trading
4. Viewing results

This allows the trading system to run without interruptions on a Vercel server.
"""

import os
import sys
import json
import yaml
import logging
import datetime
import importlib.util
from pathlib import Path
from flask import Flask, request, jsonify

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"vercel_api_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

def load_strategy_module():
    """Load the strategy module dynamically"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        strategy_path = os.path.join(script_dir, 'final_sp500_strategy.py')
        
        if not os.path.exists(strategy_path):
            logger.error(f"Strategy file not found: {strategy_path}")
            return None
            
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("final_sp500_strategy", strategy_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        logger.info(f"Successfully loaded strategy module from {strategy_path}")
        return strategy_module
    except Exception as e:
        logger.error(f"Error loading strategy module: {str(e)}", exc_info=True)
        return None

def load_config():
    """Load configuration from sp500_config.yaml"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {}
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        return {}

def ensure_directories():
    """Ensure all required directories exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_dirs = [
        'backtest_results',
        'data',
        'logs',
        'models',
        'plots',
        'results',
        'trades',
        os.path.join('performance', 'SP500Strategy')
    ]
    
    for directory in required_dirs:
        dir_path = os.path.join(script_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "status": "ok",
        "message": "S&P 500 Trading Strategy API is running",
        "endpoints": [
            "/api/backtest",
            "/api/paper",
            "/api/live",
            "/api/results"
        ]
    })

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """API endpoint for running a backtest"""
    try:
        # Load strategy module and configuration
        strategy_module = load_strategy_module()
        if not strategy_module:
            return jsonify({"error": "Failed to load strategy module"}), 500
            
        config = load_config()
        
        # Ensure all required directories exist
        ensure_directories()
        
        # Get parameters from request
        data = request.json or {}
        
        # Check if quarter is specified
        quarter = data.get('quarter')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-03-31')
        
        if quarter:
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
            
            if quarter in quarter_dates:
                start_date, end_date = quarter_dates[quarter]
                logger.info(f"Using date range for {quarter}: {start_date} to {end_date}")
            else:
                return jsonify({"error": f"Invalid quarter: {quarter}"}), 400
        
        # Get parameters from config if not specified in request
        initial_capital = data.get('initial_capital')
        if initial_capital is None and 'initial_capital' in config:
            initial_capital = config['initial_capital']
        if initial_capital is None:
            initial_capital = 300
            
        max_signals = data.get('max_signals')
        if max_signals is None and 'backtest' in config and 'max_signals_per_day' in config['backtest']:
            max_signals = config['backtest']['max_signals_per_day']
        if max_signals is None:
            max_signals = 40
            
        tier1_threshold = data.get('tier1_threshold')
        if tier1_threshold is None and 'backtest' in config and 'tier1_threshold' in config['backtest']:
            tier1_threshold = config['backtest']['tier1_threshold']
        if tier1_threshold is None:
            tier1_threshold = 0.8
            
        tier2_threshold = data.get('tier2_threshold')
        if tier2_threshold is None and 'backtest' in config and 'tier2_threshold' in config['backtest']:
            tier2_threshold = config['backtest']['tier2_threshold']
        if tier2_threshold is None:
            tier2_threshold = 0.7
            
        tier3_threshold = data.get('tier3_threshold')
        if tier3_threshold is None and 'backtest' in config and 'tier3_threshold' in config['backtest']:
            tier3_threshold = config['backtest']['tier3_threshold']
        if tier3_threshold is None:
            tier3_threshold = 0.6
            
        random_seed = data.get('random_seed', 42)
        weekly_selection = data.get('weekly_selection', False)
        continuous_capital = data.get('continuous_capital', False)
        
        # Log backtest parameters
        logger.info(f"Running backtest with parameters:")
        logger.info(f"  - Start date: {start_date}")
        logger.info(f"  - End date: {end_date}")
        logger.info(f"  - Initial capital: {initial_capital}")
        logger.info(f"  - Max signals: {max_signals}")
        logger.info(f"  - Random seed: {random_seed}")
        logger.info(f"  - Weekly selection: {weekly_selection}")
        logger.info(f"  - Continuous capital: {continuous_capital}")
        logger.info(f"  - Tier 1 threshold: {tier1_threshold}")
        logger.info(f"  - Tier 2 threshold: {tier2_threshold}")
        logger.info(f"  - Tier 3 threshold: {tier3_threshold}")
        
        # Call the run_backtest function from the strategy module
        result = strategy_module.run_backtest(
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
        
        logger.info("Backtest completed successfully")
        
        # Save the backtest parameters to the result file for future reference
        if isinstance(result, dict) and 'result_file' in result and os.path.exists(result['result_file']):
            try:
                with open(result['result_file'], 'r') as f:
                    result_data = json.load(f)
                
                # Add backtest parameters to the result file
                result_data['parameters'] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'max_signals': max_signals,
                    'random_seed': random_seed,
                    'weekly_selection': weekly_selection,
                    'continuous_capital': continuous_capital,
                    'tier1_threshold': tier1_threshold,
                    'tier2_threshold': tier2_threshold,
                    'tier3_threshold': tier3_threshold
                }
                
                with open(result['result_file'], 'w') as f:
                    json.dump(result_data, f, indent=4)
                    
                logger.info(f"Added backtest parameters to result file: {result['result_file']}")
            except Exception as e:
                logger.error(f"Error adding parameters to result file: {str(e)}")
        
        # Return the result
        return jsonify({
            "status": "success",
            "message": "Backtest completed successfully",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/paper', methods=['POST'])
def api_paper_trading():
    """API endpoint for running paper trading"""
    try:
        # Load strategy module and configuration
        strategy_module = load_strategy_module()
        if not strategy_module:
            return jsonify({"error": "Failed to load strategy module"}), 500
            
        config = load_config()
        
        # Ensure all required directories exist
        ensure_directories()
        
        # Get parameters from request
        data = request.json or {}
        
        # Get parameters from config if not specified in request
        initial_capital = data.get('initial_capital')
        if initial_capital is None and 'initial_capital' in config:
            initial_capital = config['initial_capital']
        if initial_capital is None:
            initial_capital = 300
            
        max_signals = data.get('max_signals')
        if max_signals is None and 'backtest' in config and 'max_signals_per_day' in config['backtest']:
            max_signals = config['backtest']['max_signals_per_day']
        if max_signals is None:
            max_signals = 40
            
        tier1_threshold = data.get('tier1_threshold')
        if tier1_threshold is None and 'backtest' in config and 'tier1_threshold' in config['backtest']:
            tier1_threshold = config['backtest']['tier1_threshold']
        if tier1_threshold is None:
            tier1_threshold = 0.8
            
        tier2_threshold = data.get('tier2_threshold')
        if tier2_threshold is None and 'backtest' in config and 'tier2_threshold' in config['backtest']:
            tier2_threshold = config['backtest']['tier2_threshold']
        if tier2_threshold is None:
            tier2_threshold = 0.7
            
        tier3_threshold = data.get('tier3_threshold')
        if tier3_threshold is None and 'backtest' in config and 'tier3_threshold' in config['backtest']:
            tier3_threshold = config['backtest']['tier3_threshold']
        if tier3_threshold is None:
            tier3_threshold = 0.6
            
        random_seed = data.get('random_seed', 42)
        weekly_selection = data.get('weekly_selection', False)
        
        # Log paper trading parameters
        logger.info(f"Running paper trading with parameters:")
        logger.info(f"  - Initial capital: {initial_capital}")
        logger.info(f"  - Max signals: {max_signals}")
        logger.info(f"  - Random seed: {random_seed}")
        logger.info(f"  - Weekly selection: {weekly_selection}")
        logger.info(f"  - Tier 1 threshold: {tier1_threshold}")
        logger.info(f"  - Tier 2 threshold: {tier2_threshold}")
        logger.info(f"  - Tier 3 threshold: {tier3_threshold}")
        
        # Call the run_paper_trading function from the strategy module
        result = strategy_module.run_backtest(
            start_date=datetime.datetime.now().strftime('%Y-%m-%d'),
            end_date=None,  # No end date for paper trading
            mode='paper',
            max_signals=max_signals,
            initial_capital=initial_capital,
            random_seed=random_seed,
            weekly_selection=weekly_selection,
            continuous_capital=True,  # Always use continuous capital for paper trading
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold
        )
        
        logger.info("Paper trading started successfully")
        
        # Return the result
        return jsonify({
            "status": "success",
            "message": "Paper trading started successfully",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error running paper trading: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/live', methods=['POST'])
def api_live_trading():
    """API endpoint for running live trading"""
    try:
        # Load strategy module and configuration
        strategy_module = load_strategy_module()
        if not strategy_module:
            return jsonify({"error": "Failed to load strategy module"}), 500
            
        config = load_config()
        
        # Ensure all required directories exist
        ensure_directories()
        
        # Get parameters from request
        data = request.json or {}
        
        # Get parameters from config if not specified in request
        initial_capital = data.get('initial_capital')
        if initial_capital is None and 'initial_capital' in config:
            initial_capital = config['initial_capital']
        if initial_capital is None:
            initial_capital = 300
            
        max_signals = data.get('max_signals')
        if max_signals is None and 'backtest' in config and 'max_signals_per_day' in config['backtest']:
            max_signals = config['backtest']['max_signals_per_day']
        if max_signals is None:
            max_signals = 40
            
        tier1_threshold = data.get('tier1_threshold')
        if tier1_threshold is None and 'backtest' in config and 'tier1_threshold' in config['backtest']:
            tier1_threshold = config['backtest']['tier1_threshold']
        if tier1_threshold is None:
            tier1_threshold = 0.8
            
        tier2_threshold = data.get('tier2_threshold')
        if tier2_threshold is None and 'backtest' in config and 'tier2_threshold' in config['backtest']:
            tier2_threshold = config['backtest']['tier2_threshold']
        if tier2_threshold is None:
            tier2_threshold = 0.7
            
        tier3_threshold = data.get('tier3_threshold')
        if tier3_threshold is None and 'backtest' in config and 'tier3_threshold' in config['backtest']:
            tier3_threshold = config['backtest']['tier3_threshold']
        if tier3_threshold is None:
            tier3_threshold = 0.6
            
        random_seed = data.get('random_seed', 42)
        weekly_selection = data.get('weekly_selection', False)
        
        # Log live trading parameters
        logger.info(f"Running live trading with parameters:")
        logger.info(f"  - Initial capital: {initial_capital}")
        logger.info(f"  - Max signals: {max_signals}")
        logger.info(f"  - Random seed: {random_seed}")
        logger.info(f"  - Weekly selection: {weekly_selection}")
        logger.info(f"  - Tier 1 threshold: {tier1_threshold}")
        logger.info(f"  - Tier 2 threshold: {tier2_threshold}")
        logger.info(f"  - Tier 3 threshold: {tier3_threshold}")
        
        # Call the run_live_trading function from the strategy module
        result = strategy_module.run_backtest(
            start_date=datetime.datetime.now().strftime('%Y-%m-%d'),
            end_date=None,  # No end date for live trading
            mode='live',
            max_signals=max_signals,
            initial_capital=initial_capital,
            random_seed=random_seed,
            weekly_selection=weekly_selection,
            continuous_capital=True,  # Always use continuous capital for live trading
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            tier3_threshold=tier3_threshold
        )
        
        logger.info("Live trading started successfully")
        
        # Return the result
        return jsonify({
            "status": "success",
            "message": "Live trading started successfully",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error running live trading: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/results', methods=['GET'])
def api_results():
    """API endpoint for viewing results"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'backtest_results')
        
        if not os.path.exists(results_dir):
            return jsonify({"error": "Results directory not found"}), 404
        
        # Get list of result files
        result_files = []
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                file_path = os.path.join(results_dir, file)
                file_stat = os.stat(file_path)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    result_files.append({
                        "filename": file,
                        "path": file_path,
                        "size": file_stat.st_size,
                        "modified": datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        "date": data.get('date', 'Unknown'),
                        "start_date": data.get('start_date', 'Unknown'),
                        "end_date": data.get('end_date', 'Unknown'),
                        "initial_capital": data.get('initial_capital', 0),
                        "final_value": data.get('final_value', 0),
                        "return": data.get('return', 0),
                        "sharpe_ratio": data.get('sharpe_ratio', 0),
                        "parameters": data.get('parameters', {})
                    })
                except Exception as e:
                    logger.error(f"Error reading result file {file}: {str(e)}")
        
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: x['modified'], reverse=True)
        
        # Check if a specific file is requested
        filename = request.args.get('file')
        if filename:
            for result in result_files:
                if result['filename'] == filename or filename in result['filename']:
                    # Get the full result data
                    try:
                        with open(result['path'], 'r') as f:
                            full_data = json.load(f)
                        return jsonify(full_data)
                    except Exception as e:
                        return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
            
            return jsonify({"error": f"Result file not found: {filename}"}), 404
        
        # Check if the latest result is requested
        if request.args.get('latest') == 'true' and result_files:
            # Get the full result data for the latest file
            try:
                with open(result_files[0]['path'], 'r') as f:
                    full_data = json.load(f)
                return jsonify(full_data)
            except Exception as e:
                return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
        
        # Check if results for a specific quarter are requested
        quarter = request.args.get('quarter')
        if quarter:
            quarter_files = [r for r in result_files if quarter in r['filename']]
            if not quarter_files:
                return jsonify({"error": f"No result files found for quarter: {quarter}"}), 404
            
            # Return the list of quarter files or the latest one
            if request.args.get('latest') == 'true' and quarter_files:
                # Get the full result data for the latest quarter file
                try:
                    with open(quarter_files[0]['path'], 'r') as f:
                        full_data = json.load(f)
                    return jsonify(full_data)
                except Exception as e:
                    return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
            else:
                return jsonify({"results": quarter_files})
        
        # Return the list of all result files
        return jsonify({"results": result_files})
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# For local development
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)
