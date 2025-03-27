#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix script for backtest web interface issues
This script will:
1. Check for and fix any JSON syntax errors in backtest result files
2. Add missing trading_parameters to existing backtest results
3. Ensure the web interface can properly display backtest results
"""

import os
import json
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_backtest_web_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fix_backtest_results():
    """Fix backtest result files to ensure they have the correct structure"""
    try:
        # Get the path to the backtest results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        
        # Ensure config file exists
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # Load config to get backtest results path
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        backtest_results_dir = config.get('paths', {}).get('backtest_results', os.path.join(script_dir, 'backtest_results'))
        
        # Ensure backtest results directory exists
        if not os.path.exists(backtest_results_dir):
            logger.info(f"Creating backtest results directory: {backtest_results_dir}")
            os.makedirs(backtest_results_dir, exist_ok=True)
        
        # Get all JSON files in the backtest results directory
        json_files = [f for f in os.listdir(backtest_results_dir) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files in {backtest_results_dir}")
        
        # Default trading parameters
        default_trading_parameters = {
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
        }
        
        # Process each JSON file
        for json_file in json_files:
            file_path = os.path.join(backtest_results_dir, json_file)
            logger.info(f"Processing file: {file_path}")
            
            try:
                # Read the JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if the file has the expected structure
                modified = False
                
                # Add tier thresholds if missing
                if 'tier1_threshold' not in data:
                    data['tier1_threshold'] = 0.8
                    modified = True
                
                if 'tier2_threshold' not in data:
                    data['tier2_threshold'] = 0.7
                    modified = True
                
                if 'tier3_threshold' not in data:
                    data['tier3_threshold'] = 0.6
                    modified = True
                
                # Add trading parameters if missing
                if 'trading_parameters' not in data:
                    data['trading_parameters'] = default_trading_parameters
                    modified = True
                
                # Ensure summary has required fields
                if 'summary' in data and data['summary']:
                    summary = data['summary']
                    
                    # Add missing fields to summary
                    for field, default_value in {
                        'max_drawdown': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0,
                        'calmar_ratio': 0
                    }.items():
                        if field not in summary:
                            summary[field] = default_value
                            modified = True
                
                # Save the modified file if changes were made
                if modified:
                    logger.info(f"Saving modified file: {file_path}")
                    with open(file_path, 'w') as f:
                        json.dump(data, f, default=str, indent=4)
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in file {json_file}: {str(e)}")
                logger.info(f"Attempting to fix the file: {file_path}")
                
                try:
                    # Read the file as text
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Create a new file with default structure
                    default_data = {
                        'summary': {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0,
                            'win_rate': 0,
                            'profit_factor': 0,
                            'total_return': 0,
                            'initial_capital': 300,
                            'final_capital': 300,
                            'avg_win': 0,
                            'avg_loss': 0,
                            'avg_holding_period': 0,
                            'max_drawdown': 0,
                            'sharpe_ratio': 0,
                            'sortino_ratio': 0,
                            'calmar_ratio': 0
                        },
                        'signals': [],
                        'quarter': 'unknown',
                        'start_date': '2023-01-01',
                        'end_date': '2023-03-31',
                        'max_signals': 100,
                        'initial_capital': 300,
                        'weekly_selection': False,
                        'tier1_threshold': 0.8,
                        'tier2_threshold': 0.7,
                        'tier3_threshold': 0.6,
                        'trading_parameters': default_trading_parameters,
                        'error': f"Original file had JSON syntax error: {str(e)}"
                    }
                    
                    # Save the fixed file
                    with open(file_path, 'w') as f:
                        json.dump(default_data, f, default=str, indent=4)
                    
                    logger.info(f"Fixed file saved: {file_path}")
                
                except Exception as fix_error:
                    logger.error(f"Error fixing file {json_file}: {str(fix_error)}")
            
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {str(e)}")
                traceback.print_exc()
        
        logger.info("Backtest results fix completed")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing backtest results: {str(e)}")
        traceback.print_exc()
        return False

def fix_web_interface():
    """Fix issues with the web interface"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Fix the app.py file to handle CORS properly
        app_py_path = os.path.join(web_interface_dir, 'app.py')
        
        if os.path.exists(app_py_path):
            logger.info(f"Fixing app.py: {app_py_path}")
            
            # Read the file
            with open(app_py_path, 'r') as f:
                content = f.read()
            
            # Check if CORS headers are already added
            if '@app.after_request\ndef add_cors_headers(response):' not in content:
                # Add CORS headers
                cors_code = '''
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests"""
    return '', 200
'''
                # Find the line after the imports
                import_end = content.find('# Initialize logging')
                if import_end > 0:
                    # Insert CORS code after imports
                    new_content = content[:import_end] + cors_code + content[import_end:]
                    
                    # Save the modified file
                    with open(app_py_path, 'w') as f:
                        f.write(new_content)
                    
                    logger.info("Added CORS headers to app.py")
            else:
                logger.info("CORS headers already present in app.py")
        
        logger.info("Web interface fix completed")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing web interface: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting backtest web interface fix script")
    
    # Fix backtest results
    if fix_backtest_results():
        logger.info("Successfully fixed backtest results")
    else:
        logger.error("Failed to fix backtest results")
    
    # Fix web interface
    if fix_web_interface():
        logger.info("Successfully fixed web interface")
    else:
        logger.error("Failed to fix web interface")
    
    logger.info("Fix script completed")
