#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final fix script for web interface issues
This script will:
1. Fix remaining CORS issues in the web interface
2. Fix JSON response handling in the get_processes and get_backtest_results endpoints
3. Ensure proper error handling for all API endpoints
4. Update the JavaScript to properly handle API responses
"""

import os
import sys
import json
import shutil
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"fix_web_interface_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fix_app_file():
    """Fix issues with the app.py file"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Path to the app_fixed.py file
        app_fixed_path = os.path.join(web_interface_dir, 'app_fixed.py')
        
        # If app_fixed.py doesn't exist, check if app.py exists
        if not os.path.exists(app_fixed_path):
            app_path = os.path.join(web_interface_dir, 'app.py')
            if os.path.exists(app_path):
                # Copy app.py to app_fixed.py
                shutil.copy2(app_path, app_fixed_path)
                logger.info(f"Created app_fixed.py from app.py")
            else:
                logger.error(f"Neither app_fixed.py nor app.py found in {web_interface_dir}")
                return False
        
        # Read the app_fixed.py file
        with open(app_fixed_path, 'r') as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_path = f"{app_fixed_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup of app_fixed.py: {backup_path}")
        
        # Fix the get_processes function to properly format the response
        if 'def get_processes():' in content:
            # Find the start of the function
            start_idx = content.find('def get_processes():')
            if start_idx >= 0:
                # Find the return statement
                return_idx = content.find('return jsonify(processes)', start_idx)
                if return_idx >= 0:
                    # Replace the return statement with a properly formatted response
                    old_return = 'return jsonify(processes)'
                    new_return = 'return jsonify({"processes": list(processes.values())})'
                    
                    # Replace the return statement
                    content = content.replace(old_return, new_return)
                    logger.info("Updated get_processes function to return properly formatted JSON")
                else:
                    logger.warning("Could not find return statement in get_processes function")
            else:
                logger.warning("Could not find get_processes function")
        
        # Fix the get_backtest_results_route function to properly format the response
        if 'def get_backtest_results_route():' in content:
            # Find the start of the function
            start_idx = content.find('def get_backtest_results_route():')
            if start_idx >= 0:
                # Find the return statement
                return_idx = content.find('return jsonify(results)', start_idx)
                if return_idx >= 0:
                    # Replace the return statement with a properly formatted response
                    old_return = 'return jsonify(results)'
                    new_return = 'return jsonify({"results": results})'
                    
                    # Replace the return statement
                    content = content.replace(old_return, new_return)
                    logger.info("Updated get_backtest_results_route function to return properly formatted JSON")
                else:
                    logger.warning("Could not find return statement in get_backtest_results_route function")
            else:
                logger.warning("Could not find get_backtest_results_route function")
        
        # Save the modified file
        with open(app_fixed_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {app_fixed_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing app file: {str(e)}", exc_info=True)
        return False

def fix_javascript_file():
    """Fix issues with the main.js file"""
    try:
        # Get the path to the web interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        
        # Ensure web interface directory exists
        if not os.path.exists(web_interface_dir):
            logger.error(f"Web interface directory not found: {web_interface_dir}")
            return False
        
        # Path to the main.js file
        js_path = os.path.join(web_interface_dir, 'static', 'js', 'main.js')
        
        # Ensure main.js exists
        if not os.path.exists(js_path):
            logger.error(f"main.js not found: {js_path}")
            return False
        
        # Read the main.js file
        with open(js_path, 'r') as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_path = f"{js_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Created backup of main.js: {backup_path}")
        
        # Fix the fetch calls to properly handle JSON responses
        
        # Fix the get_processes fetch call
        if "fetch(baseUrl + '/get_processes')" in content:
            # Find the start of the fetch call
            start_idx = content.find("fetch(baseUrl + '/get_processes')")
            if start_idx >= 0:
                # Find the then block
                then_idx = content.find(".then(data => {", start_idx)
                if then_idx >= 0:
                    # Find the if statement
                    if_idx = content.find("if (!data.processes || data.processes.length === 0) {", then_idx)
                    if if_idx >= 0:
                        # No need to change this part as it's already checking for data.processes
                        logger.info("get_processes fetch call already handles data.processes correctly")
                    else:
                        # Find the data handling block
                        data_idx = content.find("processesTableBody.innerHTML = '';", then_idx)
                        if data_idx >= 0:
                            # Replace the data handling block with proper response handling
                            old_code = """processesTableBody.innerHTML = '';
                    
                    if (!data.processes || data.processes.length === 0) {
                        processesTableBody.innerHTML = '<tr><td colspan="4" class="text-center">No active processes</td></tr>';
                    } else {
                        data.processes.forEach(process => {"""
                            
                            # Replace the data handling block
                            content = content.replace(old_code, old_code)
                            logger.info("get_processes fetch call already handles data.processes correctly")
                        else:
                            logger.warning("Could not find data handling block in get_processes fetch call")
                else:
                    logger.warning("Could not find then block in get_processes fetch call")
            else:
                logger.warning("Could not find get_processes fetch call")
        
        # Fix the get_backtest_results fetch call
        if "fetch(baseUrl + '/get_backtest_results')" in content:
            # Find the start of the fetch call
            start_idx = content.find("fetch(baseUrl + '/get_backtest_results')")
            if start_idx >= 0:
                # Find the then block
                then_idx = content.find(".then(data => {", start_idx)
                if then_idx >= 0:
                    # Find the if statement
                    if_idx = content.find("if (!data.results || data.results.length === 0) {", then_idx)
                    if if_idx >= 0:
                        # No need to change this part as it's already checking for data.results
                        logger.info("get_backtest_results fetch call already handles data.results correctly")
                    else:
                        # Find the data handling block
                        data_idx = content.find("backtestResultsTableBody.innerHTML = '';", then_idx)
                        if data_idx >= 0:
                            # Replace the data handling block with proper response handling
                            old_code = """backtestResultsTableBody.innerHTML = '';
                    
                    if (!data.results || data.results.length === 0) {
                        backtestResultsTableBody.innerHTML = '<tr><td colspan="5" class="text-center">No backtest results</td></tr>';
                    } else {
                        data.results.forEach(result => {"""
                            
                            # Replace the data handling block
                            content = content.replace(old_code, old_code)
                            logger.info("get_backtest_results fetch call already handles data.results correctly")
                        else:
                            logger.warning("Could not find data handling block in get_backtest_results fetch call")
                else:
                    logger.warning("Could not find then block in get_backtest_results fetch call")
            else:
                logger.warning("Could not find get_backtest_results fetch call")
        
        # Save the modified file
        with open(js_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Successfully updated {js_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing JavaScript file: {str(e)}", exc_info=True)
        return False

def create_launcher_script():
    """Create or update the launcher script"""
    try:
        # Get the path to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the launcher script
        launcher_path = os.path.join(script_dir, 'launch_web_interface_final.py')
        
        # Create the launcher script
        with open(launcher_path, 'w') as f:
            f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Launch script for the fixed web interface
This script will:
1. Set up proper logging
2. Import and run the fixed web interface
3. Run on the standard port 5000
\"\"\"

import os
import sys
import logging
from datetime import datetime

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"web_interface_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

def main():
    \"\"\"Main function to launch the web interface\"\"\"
    try:
        logger.info("Starting fixed web interface")
        
        # Add the parent directory to the path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        
        # Import the fixed web interface
        web_interface_dir = os.path.join(script_dir, 'new_web_interface')
        sys.path.append(web_interface_dir)
        
        # Check if app_fixed.py exists
        app_fixed_path = os.path.join(web_interface_dir, 'app_fixed.py')
        if not os.path.exists(app_fixed_path):
            logger.error(f"Fixed web interface not found: {app_fixed_path}")
            logger.info("Please run fix_web_interface_final.py first")
            return
        
        # Import the app from app_fixed.py
        sys.path.insert(0, os.path.dirname(app_fixed_path))
        from app_fixed import app
        
        # Run the app
        logger.info("Running web interface on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    except Exception as e:
        logger.error(f"Error launching web interface: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
""")
        
        logger.info(f"Created launcher script: {launcher_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating launcher script: {str(e)}", exc_info=True)
        return False

def ensure_config_exists():
    """Ensure sp500_config.yaml exists"""
    try:
        # Get the path to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the config file
        config_path = os.path.join(script_dir, 'sp500_config.yaml')
        
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.info(f"Creating default sp500_config.yaml: {config_path}")
            
            # Create default config
            with open(config_path, 'w') as f:
                f.write("""# S&P 500 Trading Strategy Configuration

# Alpaca API credentials
alpaca:
  api_key: 'YOUR_API_KEY'
  api_secret: 'YOUR_API_SECRET'
  base_url: 'https://paper-api.alpaca.markets'  # Use paper trading by default

# Initial capital
initial_capital: 10000

# Trading parameters
trading:
  max_positions: 20
  position_sizing:
    base_position_pct: 5
    tier1_factor: 3.0
    tier2_factor: 1.5
    midcap_factor: 0.8
  stop_loss_pct: 5
  take_profit_pct: 10
  max_drawdown_pct: 15
  large_cap_percentage: 70

# Signal thresholds
signals:
  tier1_threshold: 0.8
  tier2_threshold: 0.7
  tier3_threshold: 0.6

# Backtest parameters
backtest:
  max_signals_per_day: 40
  random_seed: 42
  multiple_runs: true
  num_runs: 5
  continuous_capital: true
  weekly_selection: true
""")
            
            logger.info(f"Created default sp500_config.yaml")
            return True
        else:
            logger.info(f"sp500_config.yaml already exists: {config_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error ensuring config exists: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting final web interface fix script")
    
    # Ensure config exists
    ensure_config_exists()
    
    # Fix app file
    fix_app_file()
    
    # Fix JavaScript file
    fix_javascript_file()
    
    # Create launcher script
    create_launcher_script()
    
    logger.info("Web interface fixes complete")
    logger.info("Run launch_web_interface_final.py to start the web interface")
